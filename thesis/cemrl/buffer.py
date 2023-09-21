from typing import Literal

import numpy as np
import torch as th
from gymnasium import spaces
from scipy.stats import binned_statistic, binned_statistic_dd
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from thesis.core.buffer import Storage

from ..core.buffer import DataTemplate, ReplayBuffer, Storage
from .policy import CemrlPolicyInput
from .task_inference import EncoderInput, TaskInference


class CemrlStorage(Storage):
    def __init__(
        self, num_episodes: int, max_episode_length: int, num_goals: int, obs_space: spaces.Dict, action_space: spaces.Box
    ) -> None:
        super().__init__(num_episodes, max_episode_length, obs_space, action_space)
        self.num_goals = num_goals
        self.goal_ep_start_stop = np.zeros((num_goals, num_episodes, 2), dtype=np.int16)
        self.goal_ep_length = np.zeros((num_goals, num_episodes), dtype=np.int16)

    def start_new_episode(self, n_env: int):
        result = super().start_new_episode(n_env)
        # remove old episodes from goal_idx_to_episode_slices
        self.goal_ep_start_stop[:, result, :] = 0
        self.goal_ep_length[:, result] = 0
        return result

    def add(self, episode_idxs: np.ndarray, data: DataTemplate, infos: list[dict[str, np.ndarray]]):
        goals = data.next_obs["goal_idx"].squeeze().astype(np.int32)
        changed_to = np.array([info.get("goal_changed", -1) for info in infos], dtype=np.int32).squeeze()
        changed = (changed_to != -1) & (changed_to != goals)

        self.goal_ep_start_stop[changed_to[changed], episode_idxs[changed]] = (
            self.episode_lengths[episode_idxs[changed], None] + 1
        )
        self.goal_ep_start_stop[goals, episode_idxs, 1] = self.episode_lengths[episode_idxs] + 1

        self.goal_ep_length[goals, episode_idxs] += 1
        super().add(episode_idxs, data, infos)
    
    def invariant(self):
        for goal_idx in np.unique(self.next_observations["goal_idx"]):
            if goal_idx == 0:
                continue
            samples = np.where(self.next_observations["goal_idx"][..., 0] == goal_idx)
            if not np.all(self.next_observations["goal"][samples] == self.next_observations["goal"][samples][0:1]):
                return False
        return True


class CemrlReplayBuffer(ReplayBuffer):
    storage: CemrlStorage
    task_inference: TaskInference
    prepared_decoder_sampling: bool = False

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        max_episode_length: int = 1000,
        num_goals: int = 1000,
        device: th.device | str = "auto",
        n_envs: int = 1,
        decoder_context_mode: Literal["random", "sequential"] = "random",
        random_normalization: Literal["none", "reward", "observation"] = "observation",
        random_normalization_mode: Literal["uniform", "skewed"] = "skewed",
        optimize_memory_usage: bool = False,
        encoder_context_length: int = 30,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            max_episode_length,
            device,
            n_envs,
            optimize_memory_usage,
            storage=CemrlStorage(
                buffer_size // max_episode_length, max_episode_length, num_goals, observation_space, action_space
            ),
        )
        self.decoder_context_mode = decoder_context_mode
        self.encoder_context_length = encoder_context_length
        self.random_normalization = random_normalization
        self.random_normalization_mode = random_normalization_mode

    def sample(self, batch_size: int, env: VecNormalize | None = None):
        if self.task_inference is None:
            return super().sample(batch_size, env)
        else:
            self.prepare_sampling_if_necessary()
            episode_idxs, sample_idxs = self.get_sample_idxs(batch_size)

            encoder_context = self._get_context(episode_idxs, sample_idxs, env, self.encoder_context_length)
            with th.no_grad():
                z, _, _ = self.task_inference(
                    EncoderInput(
                        obs=encoder_context.observations["observation"],
                        action=encoder_context.actions,
                        reward=encoder_context.rewards,
                        next_obs=encoder_context.next_observations["observation"],
                    )
                )

            policy_obs = CemrlPolicyInput(obs=encoder_context.observations["observation"][:, -1], task_encoding=z)
            policy_next_obs = CemrlPolicyInput(obs=encoder_context.next_observations["observation"][:, -1], task_encoding=z)

            return DictReplayBufferSamples(
                observations=policy_obs,  # type: ignore
                actions=encoder_context.actions[:, -1],
                next_observations=policy_next_obs,  # type: ignore
                dones=encoder_context.dones[:, -1],
                rewards=encoder_context.rewards[:, -1],
            )

    def cemrl_sample(
        self, batch_size: int, env: VecNormalize | None = None, encoder_context_length=30, decoder_context_length=400
    ):
        self.prepare_sampling_if_necessary()

        episode_idxs, sample_idxs = self.get_sample_idxs(batch_size)

        encoder_context = self._get_context(episode_idxs, sample_idxs, env, encoder_context_length)
        decoder_targets = self.get_decoder_targets(episode_idxs, sample_idxs, decoder_context_length, env)
        return encoder_context, decoder_targets

    def get_sample_idxs(self, batch_size):
        # if self.decoder_context_mode == "random":
        #     limit_to_goals_if_weighted = np.arange(self.storage.num_goals)
        #     goal_lengths = self.storage.goal_ep_length[limit_to_goals_if_weighted].sum(axis=1)
        #     available_goals = np.where(goal_lengths > 0)[0]
        #     goals = np.random.choice(available_goals, size=batch_size, replace=True)
        #     samples = []
        #     for goal, length in zip(limit_to_goals_if_weighted[goals], goal_lengths[goals]):
        #         samples.append(np.random.choice(length, 1, p=self.goal_to_sample_weights[goal, :length]))
        #     samples = np.concatenate(samples, axis=0)
        #     samples = self.goal_idx_to_episode_sample_indices[limit_to_goals_if_weighted[goals], samples]
        #     episode_idxs = samples[..., 0]
        #     sample_idxs = samples[..., 1]
        # else:
        ep_lengths = self.storage.episode_lengths[self.valid_episodes]
        ep_lengths = ep_lengths / ep_lengths.sum()
        episode_idxs = np.random.choice(self.valid_episodes, size=batch_size, p=ep_lengths)
        sample_idxs = np.random.randint(0, self.storage.episode_lengths[episode_idxs], size=batch_size)
        return episode_idxs, sample_idxs

    def get_decoder_targets(
        self, episode_idxs: np.ndarray, sample_idxs: np.ndarray, num_targets: int = 64, env: VecNormalize | None = None
    ):
        batch_size = len(episode_idxs)
        sample_goals = self.storage.next_observations["goal_idx"][episode_idxs, sample_idxs].astype(np.int32).squeeze(-1)
        goal_ep_start_stop = self.storage.goal_ep_start_stop[sample_goals]

        if self.decoder_context_mode == "sequential":
            data = self.get_empty_data_template(batch_size, num_targets)
            p_cs = self.storage.goal_ep_length[sample_goals].cumsum(axis=1)
            p_cs = p_cs / p_cs[:, -1, None]
            # fill data with episodes of same goal till full
            lengths = np.zeros(batch_size, dtype=np.int32)
            all_batch_idx = np.arange(batch_size)

            def batched_np_choice(p_cs):
                r = np.random.rand(p_cs.shape[0], 1)
                k = (p_cs < r).sum(axis=1)
                return k

            while np.any(lengths < num_targets):
                to_short = np.where(lengths < num_targets)[0]
                ep_idx = batched_np_choice(p_cs[to_short])
                ep_start_stop = goal_ep_start_stop[all_batch_idx[to_short], ep_idx]

                self.add_episode_to_data_template(to_short, lengths[to_short], ep_idx, ep_start_stop, data)
                lengths[to_short] += ep_start_stop[:, 1] - ep_start_stop[:, 0]
        elif self.decoder_context_mode == "random":
            goal_length = self.storage.goal_ep_length[sample_goals].sum(axis=1)
            samples = []
            for goal, length in zip(sample_goals, goal_length):
                samples.append(np.random.choice(length, num_targets, p=self.goal_to_sample_weights[goal, :length]))
            samples = np.stack(samples, axis=0)
            samples = self.goal_idx_to_episode_sample_indices[sample_goals[:, None], samples]

            data = DataTemplate(
                obs={k: v[samples[..., 0], samples[..., 1]] for k, v in self.storage.observations.items()},
                next_obs={k: v[samples[..., 0], samples[..., 1]] for k, v in self.storage.next_observations.items()},
                actions=self.storage.actions[samples[..., 0], samples[..., 1]],
                rewards=self.storage.rewards[samples[..., 0], samples[..., 1]],
                dones=self.storage.dones[samples[..., 0], samples[..., 1]],
                timeouts=self.storage.timeouts[samples[..., 0], samples[..., 1]],
            )
        else:
            raise ValueError(f"Unknown decoder_context_mode: {self.decoder_context_mode}")

        return self.post_process_samples(data, env)

    def prepare_sampling(self, storage: Storage):
        super().prepare_sampling(storage)
        if self.decoder_context_mode == "random":
            goal_length = self.storage.goal_ep_length.sum(axis=1)
            max_goal_ep_lenght = goal_length.max()

            self.goal_idx_to_episode_sample_indices = np.zeros((self.storage.num_goals, max_goal_ep_lenght, 2), dtype=np.int32)
            for goal_idx in np.where(goal_length > 0)[0]:
                offset = 0
                for ep_idx, goal_ep_length in enumerate(self.storage.goal_ep_length[goal_idx]):
                    ep_start, ep_stop = self.storage.goal_ep_start_stop[goal_idx, ep_idx]
                    if ep_start == ep_stop:
                        continue
                    self.goal_idx_to_episode_sample_indices[goal_idx, offset : offset + goal_ep_length, 0] = ep_idx
                    self.goal_idx_to_episode_sample_indices[goal_idx, offset : offset + goal_ep_length, 1] = np.arange(
                        ep_start, ep_stop
                    )
                    offset += self.storage.goal_ep_length[goal_idx, ep_idx]

            self.goal_to_sample_weights = np.zeros((self.storage.num_goals, max_goal_ep_lenght), dtype=np.float32)

            for goal_idx in np.where(goal_length > 0)[0]:
                if self.random_normalization == "none":
                    self.goal_to_sample_weights[goal_idx, : goal_length[goal_idx]] = 1 / goal_length[goal_idx]
                elif self.random_normalization == "reward":
                    rewards = self.storage.rewards[
                        self.goal_idx_to_episode_sample_indices[goal_idx, : goal_length[goal_idx], 0],
                        self.goal_idx_to_episode_sample_indices[goal_idx, : goal_length[goal_idx], 1],
                    ]
                    count, _, bin = binned_statistic(rewards, rewards, statistic="count")
                    sample_count = count[bin - 1]
                elif self.random_normalization == "observation":
                    obs = self.storage.observations["observation"][
                        self.goal_idx_to_episode_sample_indices[goal_idx, : goal_length[goal_idx], 0],
                        self.goal_idx_to_episode_sample_indices[goal_idx, : goal_length[goal_idx], 1],
                    ]
                    count, _, bin = binned_statistic_dd(obs, obs[:, 0], statistic="count", expand_binnumbers=True)
                    sample_count = count[tuple(zip(bin-1))]
                else:
                    raise ValueError(f"Unknown random_normalization: {self.random_normalization}")
                
                if self.random_normalization_mode == "uniform":
                    self.goal_to_sample_weights[goal_idx, : goal_length[goal_idx]] = 1 / sample_count / np.count_nonzero(count)
                elif self.random_normalization_mode == "skewed":
                    prob = count.max() / sample_count
                    self.goal_to_sample_weights[goal_idx, : goal_length[goal_idx]] = prob / prob.sum()
