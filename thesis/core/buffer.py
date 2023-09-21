from typing import Any, Dict, List, TypedDict, NamedTuple

import numpy as np
import pandas as pd
import torch as th
from gymnasium import Space, spaces
from scipy.stats import binned_statistic_dd
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples, ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from .types import EncoderInput


class PolicyInput(TypedDict):
    obs: th.Tensor
    task_encoding: th.Tensor


class Output(NamedTuple):
    obs: dict[str, th.Tensor]
    next_obs: dict[str, th.Tensor]
    actions: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    timeouts: th.Tensor


class ReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: Space,
        task_encoder: th.nn.Module | None = None,
        exploration_buffer_size: int = 1_000_000,
        n_envs=1,
        use_bin_weighted_decoder_target_sampling: bool = True,
        **kwargs,
    ):
        buffer_size = buffer_size - buffer_size % n_envs
        explore_buffer_size = exploration_buffer_size - exploration_buffer_size % n_envs
        super().__init__(buffer_size + explore_buffer_size, observation_space, action_space, n_envs=1, **kwargs)
        self.n_envs = n_envs
        self.buffer_size = buffer_size
        self.task_encoder = task_encoder

        # We use two buffers in one: [NormalBuffer | ExplorationBuffer]
        self.explore_buffer_size = exploration_buffer_size - exploration_buffer_size % self.n_envs
        self.explore_pos = buffer_size
        self.explore_full = False

        self.grouped_goal_idx = None
        self.use_bin_weighted_decoder_target_sampling = use_bin_weighted_decoder_target_sampling

        self.observations = {k: v.squeeze(1) for k, v in self.observations.items()}
        self.next_observations = {k: v.squeeze(1) for k, v in self.next_observations.items()}
        self.actions = self.actions.squeeze(1)
        self.rewards = self.rewards.squeeze(1)
        self.dones = self.dones.squeeze(1)
        self.timeouts = self.timeouts.squeeze(1)
        self.goal_changes = np.zeros(buffer_size + exploration_buffer_size, bool)
        self.goal_idxs = np.full(buffer_size + exploration_buffer_size, -1, dtype=int)

    def size(self) -> int:
        """Return the current size of the buffer."""
        return self.normal_size() + self.explore_size()

    def normal_size(self) -> int:
        """Return the current size of the normal data buffer."""
        return self.buffer_size if self.full else self.pos

    def explore_size(self) -> int:
        """Return the current size of the exploration data buffer."""
        return self.explore_buffer_size if self.explore_full else self.explore_pos - self.buffer_size

    def task_inference_sample(
        self, batch_size: int, env: VecNormalize | None = None, encoder_window: int = 30, decoder_samples: int = 400
    ):
        batch_inds = np.random.choice(self.valid_indices(encoder_window), batch_size)
        enc_samples, *_ = self.get_encoder_context(batch_inds, env, encoder_window)
        dec_samples = self.get_decoder_targets(batch_inds, env, decoder_samples)

        return enc_samples, dec_samples

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Add a new transition to the replay buffer.

        Args:
            obs (Dict[str, np.ndarray]): The observation at the current time step.
            next_obs (Dict[str, np.ndarray]): The observation at the next time step.
            action (np.ndarray): The action taken at the current time step.
            reward (np.ndarray): The reward received at the next time step.
            done (np.ndarray): Whether the episode terminated at the next time step.
            infos (List[Dict[str, Any]]): Additional information about the transition.

        Returns:
            None
        """
        self.is_decoder_index_build = False
        is_exploring = np.any([info.get("is_exploration", False) for info in infos])

        pos = self.explore_pos if is_exploring else self.pos
        pos = np.arange(self.n_envs) + pos

        for k, v in self.observations.items():
            v[pos] = obs[k][:]
        for k, v in self.next_observations.items():
            v[pos] = next_obs[k][:]
        self.actions[pos] = action
        self.rewards[pos] = reward
        self.dones[pos] = done
        self.timeouts[pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
        self.goal_changes[pos] = np.array([info.get("goal_changed", False) for info in infos]) | done
        self.goal_idxs[pos] = np.array([info.get("goal_idx", -1) for info in infos])

        if is_exploring:
            self.explore_pos = (
                self.buffer_size + (self.n_envs + self.explore_pos - self.buffer_size) % self.explore_buffer_size
            )
            if self.explore_pos == self.buffer_size:
                self.explore_full = True
        else:
            self.pos = (self.n_envs + self.pos) % self.buffer_size
            if self.pos == 0:
                self.full = True

    def valid_indices(self, distance_to_buffer_pos: int = 30):
        """Return the indices of the valid data in the buffer."""
        normal = np.arange(self.normal_size())
        if self.full:
            without = (self.pos + np.arange(self.n_envs * distance_to_buffer_pos)) % self.buffer_size
            normal = np.setdiff1d(normal, without)
        explore = np.arange(self.buffer_size, self.buffer_size + self.explore_size())
        if self.explore_full:
            without = (
                self.explore_pos - self.buffer_size + np.arange(self.n_envs * distance_to_buffer_pos)
            ) % self.explore_buffer_size + self.buffer_size
            explore = np.setdiff1d(explore, without)

        return np.concatenate([normal, explore])

    def dreamer_sample(
        self, batch_size: int, env: VecNormalize | None = None, goals: np.ndarray | None = None, max_length: int = 64
    ):
        indices = np.random.choice(self.valid_indices(max_length), batch_size)
        data, indices, mask = self.get_encoder_context(indices, env, max_length)  # type:ignore

        goal_idx = self.to_torch(self.goal_idxs[indices])
        goal_idx[mask] = -1
        goal_idx = goal_idx[..., None]

        starts = {}
        for i in th.unique(mask[0]):
            start = mask[1][mask[0] == i].max() + 1
            if start < max_length:
                starts[i] = start
        is_first = th.zeros_like(goal_idx, dtype=th.bool)
        for i, start in starts.items():
            is_first[i, start] = True
        is_first = is_first.float().squeeze(-1)

        goals_tensor = self.to_torch(goals[self.goal_idxs[indices]])
        goals_tensor[mask] = 0.0
        if len(goals_tensor.shape) < 3:
            goals_tensor = goals_tensor[..., None]

        return {
            "observation": data.observations["observation"],
            "goal_idx": goal_idx,
            "goal": goals_tensor,
            "is_first": is_first,
            "is_terminal": (data.dones * (1 - data.dones.new_tensor(self.timeouts[indices]).float())).squeeze(-1),
            "reward": data.rewards,
            "action": data.actions,
        }

    def sample(self, batch_size: int, env: VecNormalize | None = None):
        """
        Sample a batch of transitions from the replay buffer.
        The data should be used for policy training (SAC, etc.).

        Args:
            batch_size (int): The number of transitions to sample.
            env (VecNormalize | None): The vectorized environment used to normalize the observations and rewards.

        Returns:
            DictReplayBufferSamples: A dictionary of tensors containing the sampled transitions.
        """
        assert self.task_encoder is not None
        indices = np.random.choice(self.valid_indices(), batch_size)
        encoder_context, *_ = self.get_encoder_context(indices, env)
        with th.no_grad():
            z, _, _ = self.task_encoder(
                EncoderInput(
                    obs=encoder_context.observations["observation"],
                    action=encoder_context.actions,
                    reward=encoder_context.rewards,
                    next_obs=encoder_context.next_observations["observation"],
                )
            )
        output = self.get_output(indices, env)
        return DictReplayBufferSamples(
            observations=PolicyInput(obs=output.obs["observation"], task_encoding=z),  # type: ignore
            next_observations=PolicyInput(obs=output.next_obs["observation"], task_encoding=z),  # type: ignore
            actions=output.actions,
            dones=output.dones * (1 - output.timeouts),
            rewards=output.rewards,
        )

    def get_encoder_context(self, indices: np.ndarray, env: VecNormalize | None = None, encoder_window=30):
        """Return the context to be used for the encoder."""
        normal_samples_mask = indices < self.buffer_size
        # flip arange and subtract from start
        indices = indices[:, None] - self.n_envs * np.flip(np.arange(encoder_window))[None]
        # take care of normal buffer overflow
        indices[normal_samples_mask] = indices[normal_samples_mask] % self.buffer_size
        # take care of exploration buffer overflow
        indices[~normal_samples_mask] = (
            indices[~normal_samples_mask] - self.buffer_size
        ) % self.explore_buffer_size + self.buffer_size

        output = self.get_output(indices, env)

        # Zero out the data of old episodes (assignment in single step is faster)
        batch_idx, time_idx, _ = th.where(output.dones)
        additional_batch_idx = [batch_idx]
        additional_time_idx = [time_idx]
        for b, i in zip(batch_idx.tolist(), time_idx.tolist()):  # type: ignore
            additional_batch_idx.append(th.full((i,), b, device=batch_idx.device))
            additional_time_idx.append(th.arange(i, device=time_idx.device))
        batch_idx = th.cat(additional_batch_idx)
        time_idx = th.cat(additional_time_idx)

        for k, v in output.obs.items():
            v[(batch_idx, time_idx)] = 0.0
        for k, v in output.next_obs.items():
            v[(batch_idx, time_idx)] = 0.0
        output.actions[(batch_idx, time_idx)] = 0.0
        output.rewards[(batch_idx, time_idx)] = 0.0
        output.dones[(batch_idx, time_idx)] = 0.0

        data = DictReplayBufferSamples(
            observations=output.obs,
            next_observations=output.next_obs,
            actions=output.actions,
            dones=output.dones * (1 - output.timeouts),
            rewards=output.rewards,
        )

        return data, indices, (batch_idx, time_idx)

    def get_decoder_targets(self, base_indices: np.ndarray, env: VecNormalize | None = None, num_decoder_targets: int = 300):
        if not self.is_decoder_index_build:
            self._build_decoder_index()
            self.is_decoder_index_build = True

        indices = self._select_decoder_target_indices(base_indices, num_decoder_targets)
        output = self.get_output(indices, env)

        return DictReplayBufferSamples(
            observations=output.obs,
            next_observations=output.next_obs,
            actions=output.actions,
            dones=output.dones * (1 - output.timeouts),
            rewards=output.rewards,
        )

    def _select_decoder_target_indices(self, indices: np.ndarray, num_decoder_targets: int) -> np.ndarray:
        group_ids = self.goal_idxs[indices]
        selected_ids = (np.random.rand(len(indices), num_decoder_targets) * self.grouped_goal_sizes[group_ids, None]).astype(
            int
        )
        indices = self.grouped_goal_idx[group_ids[:, None], selected_ids]  # type: ignore
        # assert np.all(self.goal_idxs[indices] == self.goal_idxs[indices[:, 0]][:, None])
        return indices

    def _build_decoder_index(self):
        df = pd.DataFrame({"goal_idx": self.goal_idxs})
        df = df[df.goal_idx != -1]
        groups = df.groupby(by=df.goal_idx)
        self.grouped_goal_idx = groups.groups
        self.grouped_goal_sizes = np.zeros(self.goal_idxs.max() + 1, dtype=int)
        self.grouped_goal_sizes[list(groups.groups.keys())] = groups.size().values

        if self.use_bin_weighted_decoder_target_sampling:
            indices = self.valid_indices()
            _, _, bin_index = binned_statistic_dd(self.observations["observation"][indices], indices, statistic="count", bins=10)
            _, idx_to_unique, bin_stats = np.unique(bin_index, return_inverse=True, return_counts=True)
            bin_stats = bin_stats.max() / bin_stats
            bin_stats = bin_stats / bin_stats.sum()
            weighted_bin_prob = np.zeros(self.buffer_size + self.explore_buffer_size, dtype=float)
            weighted_bin_prob[indices] = bin_stats[idx_to_unique]

            for k, v in self.grouped_goal_idx.items():
                weights = weighted_bin_prob[v] / weighted_bin_prob[v].sum()  # type: ignore
                self.grouped_goal_idx[k] = np.random.choice(v, len(v), p=weights)  # type: ignore

        grouped_goal_idx = np.zeros((self.goal_idxs.max() + 1, self.grouped_goal_sizes.max() + 1), dtype=int)
        for k, v in self.grouped_goal_idx.items():
            grouped_goal_idx[k, : len(v)] = v
        self.grouped_goal_idx = grouped_goal_idx

    def get_output(self, indices: np.ndarray, env: VecNormalize | None = None):
        obs = self._normalize_obs({k: v[indices] for k, v in self.observations.items()}, env)
        next_obs = self._normalize_obs({k: v[indices] for k, v in self.next_observations.items()}, env)

        return Output(
            obs={k: self.to_torch(v) for k, v in obs.items()},
            next_obs={k: self.to_torch(v) for k, v in next_obs.items()},
            actions=self.to_torch(self.actions[indices]),
            dones=self.to_torch(self.dones[indices])[..., None],
            rewards=self.to_torch(self._normalize_reward(self.rewards[indices], env))[..., None],
            timeouts=self.to_torch(self.timeouts[indices])[..., None],
        )
