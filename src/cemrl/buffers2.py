import bisect
from typing import Literal
import numpy as np
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
import torch as th
from gymnasium import spaces
from src.core.buffers import DataTemplate, Storage, ReplayBuffer
from src.cemrl.networks import Encoder
from src.cemrl.types import CEMRLPolicyInput
from scipy.stats import binned_statistic


class Slice:
    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end


class CemrlStorage(Storage):
    def __init__(
        self, num_episodes: int, max_episode_length: int, num_goals: int, obs_space: spaces.Dict, action_space: spaces.Box
    ) -> None:
        super().__init__(num_episodes, max_episode_length, obs_space, action_space)
        self.goal_idx_to_episode_slices: dict[int, dict[int, Slice]] = {}
        self.current_wip_slices: dict[int, Slice] = {}

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


class CemrlReplayBuffer(ReplayBuffer):
    storage: CemrlStorage

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        encoder: Encoder | None = None,
        encoder_context: int = 30,
        max_episode_length: int = 1000,
        num_goals: int = 1000,
        device: th.device | str = "auto",
        n_envs: int = 1,
        decoder_context_mode: Literal["random", "sequential"] = "random",
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            max_episode_length,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
            CemrlStorage(buffer_size // max_episode_length, max_episode_length, num_goals, observation_space, action_space),
        )
        self.encoder = encoder
        self.encoder_context = encoder_context
        self.decoder_context_mode = decoder_context_mode

    def sample(self, batch_size: int, env: VecNormalize | None = None):
        if self.encoder is None:
            return super().sample(batch_size, env)
        else:
            self.prepare_sampling_if_necessary()
            episode_idxs, sample_idxs = self.get_sample_idxs(batch_size)
            encoder_context = self._get_context(episode_idxs, sample_idxs, env, self.encoder_context)
            with th.no_grad():
                _, z, _ = self.encoder(self.encoder.from_samples_to_encoder_input(encoder_context))

            return DictReplayBufferSamples(
                observations=CEMRLPolicyInput(
                    observation=encoder_context.observations["observation"][:, -1], task_indicator=z
                ),  # type: ignore
                actions=encoder_context.actions[:, -1],
                next_observations=CEMRLPolicyInput(
                    observation=encoder_context.next_observations["observation"][:, -1], task_indicator=z
                ),  # type: ignore
                dones=encoder_context.dones[:, -1],
                rewards=encoder_context.rewards[:, -1],
            )

    def get_sample_idxs(self, batch_size, limit_to_goals_if_weighted=None):
        # if self.decoder_context_mode == "random":
        #     if limit_to_goals_if_weighted is None:
        #         limit_to_goals_if_weighted = np.arange(self.storage.num_goals)
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
        episode_idxs = np.random.choice(self.valid_episodes, size=batch_size)
        sample_idxs = np.random.randint(0, self.storage.episode_lengths[episode_idxs], size=batch_size)
        return episode_idxs, sample_idxs

    def cemrl_sample(
        self, batch_size: int, env: VecNormalize | None = None, encoder_batch_length=30, decoder_batch_length=400
    ):
        self.prepare_sampling_if_necessary()

        episode_idxs, sample_idxs = self.get_sample_idxs(batch_size)

        encoder_context = self._get_context(episode_idxs, sample_idxs, env, self.encoder_context)
        decoder_targets = self.get_decoder_targets(env, episode_idxs, sample_idxs, decoder_batch_length)
        return encoder_context, decoder_targets

    def get_decoder_targets(self, env, episode_idxs, sample_idxs, num_targets=64):
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
                rewards = self.storage.rewards[
                    self.goal_idx_to_episode_sample_indices[goal_idx, : goal_length[goal_idx], 0],
                    self.goal_idx_to_episode_sample_indices[goal_idx, : goal_length[goal_idx], 1],
                ]
                count, _, bin = binned_statistic(rewards, rewards, statistic="count")
                prob = 1 / count[bin - 1]
                self.goal_to_sample_weights[goal_idx, : goal_length[goal_idx]] = prob / prob.sum()

        # goal_to_indices = {}
        # not_empty_idx = np.where(self.storage.goal_ep_length > 0)
        # for goal_idx, ep_idx in zip(not_empty_idx[0], not_empty_idx[1]):
        #     ep_idx = np.full(self.storage.goal_ep_length[goal_idx, ep_idx], ep_idx, dtype=np.int32)
        #     sample_idx = np.arange(*self.storage.goal_ep_start_stop[goal_idx, ep_idx], dtype=np.int32)
        #     goal_to_indices.setdefault(goal_idx, []).append((ep_idx, sample_idx))
        # self.goal_to_indices = goal_to_indices
        super().prepare_sampling(storage)