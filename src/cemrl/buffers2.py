import bisect
import numpy as np
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
import torch as th
from gymnasium import spaces
from src.core.buffers import DataTemplate, Storage, ReplayBuffer
from src.cemrl.networks import Encoder
from src.cemrl.types import CEMRLPolicyInput


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
        changed = np.array(["goal_changed" in info for info in infos])
        changed_to = np.array([info["goal_changed"] for info in infos if "goal_changed" in info], dtype=np.int32).squeeze()

        self.goal_ep_start_stop[changed_to, episode_idxs[changed], 0] = (
            self.episode_lengths[episode_idxs[changed]] + 1
        )  # TODO: check if goal changed for this or one later
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
        encoder: Encoder,
        encoder_context: int = 30,
        max_episode_length: int = 1000,
        num_goals: int = 1000,
        device: th.device | str = "auto",
        n_envs: int = 1,
        num_multi_episode_decoder_target: int = 1,
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
        self.num_multi_episode_decoder_target = num_multi_episode_decoder_target

    def sample(self, batch_size: int, env: VecNormalize | None = None):
        self.prepare_sampling_if_necessary()

        episode_idxs = np.random.choice(self.valid_episodes, size=batch_size)
        sample_idxs = np.random.randint(0, self.storage.episode_lengths[episode_idxs], size=batch_size)
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

    def cemrl_sample(
        self, batch_size: int, env: VecNormalize | None = None, encoder_batch_length=30, decoder_batch_length=400
    ):
        self.prepare_sampling_if_necessary()

        episode_idxs = np.random.choice(self.valid_episodes, size=batch_size)
        sample_idxs = np.random.randint(0, self.storage.episode_lengths[episode_idxs], size=batch_size)

        encoder_context = self._get_context(episode_idxs, sample_idxs, env, self.encoder_context)
        decoder_targets = self.get_decoder_targets(env, episode_idxs, sample_idxs)
        return encoder_context, decoder_targets

    def get_decoder_targets(self, env, episode_idxs, sample_idxs, num_targets=64):
        batch_size = len(episode_idxs)
        data = self.get_empty_data_template(batch_size, num_targets)
        sample_goals = self.storage.observations["goal_idx"][episode_idxs, sample_idxs].astype(np.int32).squeeze(-1)
        goal_ep_start_stop = self.storage.goal_ep_start_stop[sample_goals]
        lengths = np.zeros(batch_size, dtype=np.int32)

        p_cs = self.storage.goal_ep_length[sample_goals].cumsum(axis=1)
        p_cs = p_cs / p_cs[:, -1, None]

        def batched_np_choice(p_cs):
            r = np.random.rand(p_cs.shape[0], 1)
            k = (p_cs < r).sum(axis=1)
            return k
        
        all_batch_idx = np.arange(batch_size)

        while np.any(lengths < num_targets):
            to_short = np.where(lengths < num_targets)[0]
            ep_idx = batched_np_choice(p_cs[to_short])
            ep_start_stop = goal_ep_start_stop[all_batch_idx[to_short], ep_idx]

            self.add_episode_to_data_template(
                to_short, lengths[to_short], ep_idx, ep_start_stop, data
            )
            lengths[to_short] += ep_start_stop[:, 1] - ep_start_stop[:, 0]

        return self.post_process_samples(data, env)

    def prepare_sampling(self, storage: Storage):
        super().prepare_sampling(storage)
