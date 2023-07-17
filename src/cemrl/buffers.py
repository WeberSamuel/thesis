from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch as th
from gymnasium import Space, spaces
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples, ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from scipy.stats import binned_statistic_dd

from src.cemrl.types import CEMRLObsTensorDict, CEMRLSacPolicyTensorInput


class CEMRLReplayBuffer(DictReplayBuffer):
    """Replay buffer used in CEMRL."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: Space,
        encoder: th.nn.Module,
        exploration_buffer_size: int = 1_000_000,
        n_envs=1,
        use_bin_weighted_decoder_target_sampling: bool = True,
        **kwargs,
    ):
        buffer_size = buffer_size - buffer_size % n_envs
        super().__init__(buffer_size, observation_space, action_space, n_envs=n_envs, **kwargs)
        self.buffer_size = buffer_size
        self.encoder = encoder

        # Trick to include exploration data in the buffer
        self.explore_buffer_size = exploration_buffer_size - exploration_buffer_size % self.n_envs
        self.explore_pos = buffer_size
        self.explore_full = False
        self.is_exploring = False
        buffer_size = buffer_size + self.explore_buffer_size

        obs_space = observation_space["observation"]
        assert isinstance(action_space, spaces.Box)
        assert isinstance(obs_space, spaces.Box)

        obs_shape = obs_space.shape[1:]  # history will be removed
        self.observations = np.zeros((buffer_size, *obs_shape), obs_space.dtype)
        self.next_observations = np.zeros((buffer_size, *obs_shape), obs_space.dtype)
        self.goal_idxs = np.full(buffer_size, -1, int)
        self.actions = np.zeros((buffer_size, *action_space.shape), action_space.dtype)
        self.rewards = np.zeros(buffer_size, np.float32)
        self.dones = np.zeros(buffer_size, bool)
        self.goal_changes = np.zeros(buffer_size, bool)
        self.timeouts = np.zeros(buffer_size, bool)
        self.grouped_goal_idx = None

        self.use_bin_weighted_decoder_target_sampling = use_bin_weighted_decoder_target_sampling

    def size(self) -> int:
        """Return the current size of the buffer."""
        return self.normal_size() + self.explore_size()

    def normal_size(self) -> int:
        """Return the current size of the normal data buffer."""
        return self.buffer_size if self.full else self.pos

    def explore_size(self) -> int:
        """Return the current size of the exploration data buffer."""
        return self.explore_buffer_size if self.explore_full else self.explore_pos - self.buffer_size

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

        pos = self.explore_pos if self.is_exploring else self.pos
        pos = np.arange(self.n_envs) + pos

        self.observations[pos] = obs["observation"][:, -1]
        self.next_observations[pos] = next_obs["observation"][:, -1]
        self.goal_idxs[pos] = obs["goal_idx"][:, -1].squeeze(-1)
        self.actions[pos] = next_obs["action"][:, -1]
        self.rewards[pos] = next_obs["reward"][:, -1, 0]
        self.dones[pos] = done
        self.timeouts[pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
        self.goal_changes[pos] = np.array([info.get("goal_changed", False) for info in infos]) | done

        if self.is_exploring:
            self.explore_pos = (
                self.buffer_size + (self.n_envs + self.explore_pos - self.buffer_size) % self.explore_buffer_size
            )
            if self.explore_pos == self.buffer_size:
                self.explore_full = True
        else:
            self.pos = (self.n_envs + self.pos) % self.buffer_size
            if self.pos == 0:
                self.full = True

    def valid_indices(self):
        """Return the indices of the valid data in the buffer."""
        return np.concatenate(
            [np.arange(self.normal_size()), np.arange(self.buffer_size, self.buffer_size + self.explore_size())]
        )

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
        indices = np.random.choice(self.valid_indices(), batch_size)
        encoder_context = self.get_encoder_context(indices, env)
        with th.no_grad():
            _, z = self.encoder(
                CEMRLObsTensorDict(
                    observation=encoder_context.next_observations,
                    action=encoder_context.actions,
                    reward=encoder_context.rewards,
                )
            )

        obs = self.to_torch(self._normalize_obs(self.observations[indices], env))  # type: ignore
        next_obs = self.to_torch(self._normalize_obs(self.next_observations[indices], env))  # type: ignore
        rewards = self.to_torch(self._normalize_reward(self.rewards[indices], env))
        return DictReplayBufferSamples(
            observations=CEMRLSacPolicyTensorInput(observation=obs, task_indicator=z),  # type: ignore
            next_observations=CEMRLSacPolicyTensorInput(observation=next_obs, task_indicator=z),  # type: ignore
            actions=self.to_torch(self.actions[indices]),
            dones=self.to_torch(self.dones[indices] * (1 - self.timeouts[indices]))[..., None],
            rewards=rewards[..., None],
        )

    def get_encoder_context(self, indices: np.ndarray, env: VecNormalize | None = None, encoder_window=30):
        """Return the context to be used for the encoder."""
        normal_samples_mask = indices < self.buffer_size
        indices = indices[:, None] - self.n_envs * np.flip(np.arange(encoder_window))[None]
        indices[normal_samples_mask] = indices[normal_samples_mask] % self.buffer_size
        indices[~normal_samples_mask] = (
            indices[~normal_samples_mask] - self.buffer_size
        ) % self.explore_buffer_size + self.buffer_size

        obs = self.to_torch(self._normalize_obs(self.observations[indices], env))  # type: ignore
        next_obs = self.to_torch(self._normalize_obs(self.next_observations[indices], env))  # type: ignore
        actions = self.to_torch(self.actions[indices])
        dones = self.to_torch(self.dones[indices])
        rewards = self.to_torch(self._normalize_reward(self.rewards[indices], env))[..., None]

        # Zero out the data of old episodes (assignment in single step is faster)
        batch_idx, time_idx = th.where(dones)
        additional_batch_idx = [batch_idx]
        additional_time_idx = [time_idx]
        for b, i in zip(batch_idx.tolist(), time_idx.tolist()):  # type: ignore
            additional_batch_idx.append(th.full((i,), b, device=batch_idx.device))
            additional_time_idx.append(th.arange(i, device=time_idx.device))
        batch_idx = th.cat(additional_batch_idx)
        time_idx = th.cat(additional_time_idx)

        obs[(batch_idx, time_idx)] = 0.0
        next_obs[(batch_idx, time_idx)] = 0.0
        actions[(batch_idx, time_idx)] = 0.0
        rewards[(batch_idx, time_idx)] = 0.0
        dones[(batch_idx, time_idx)] = 0.0

        return ReplayBufferSamples(
            observations=obs,  # type: ignore
            next_observations=next_obs,  # type: ignore
            actions=actions,
            dones=dones,
            rewards=rewards,
        )

    def get_decoder_targets(self, base_indices: np.ndarray, env: VecNormalize | None = None, num_decoder_targets: int = 300):
        """
        Return samples to be used as targets by the decoder.
        The returned samples have the same goal as the input samples, yet they can come from different episodes.
        This is a modification of episode-linking.
        """

        if not self.is_decoder_index_build:
            self._build_decoder_index()
            self.is_decoder_index_build = True

        indices = self._select_decoder_target_indices(base_indices, num_decoder_targets)

        return ReplayBufferSamples(
            observations=self.to_torch(self._normalize_obs(self.observations[indices], env)),  # type: ignore
            next_observations=self.to_torch(self._normalize_obs(self.next_observations[indices], env)),  # type: ignore
            actions=self.to_torch(self.actions[indices]),
            dones=self.to_torch(self.dones[indices] * (1 - self.timeouts[indices])),
            rewards=self.to_torch(self._normalize_reward(self.rewards[indices], env))[..., None],
        )

    def _select_decoder_target_indices(self, indices: np.ndarray, num_decoder_targets: int) -> np.ndarray:
        group_ids = self.goal_idxs[indices]
        selected_ids = (np.random.rand(len(indices), num_decoder_targets) * self.grouped_goal_sizes[group_ids, None]).astype(
            int
        )
        indices = self.grouped_goal_idx[group_ids[:, None], selected_ids]  # type: ignore
        assert np.all(self.goal_idxs[indices] == self.goal_idxs[indices[:, 0]][:, None])
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
            _, _, bin_index = binned_statistic_dd(self.observations[indices], indices, statistic="count", bins=10)
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


class NoLinkingCemrlReplayBuffer(CEMRLReplayBuffer):
    def _build_decoder_index(self):
        self.episode_ends = np.zeros(self.buffer_size + self.explore_buffer_size, int)
        self.episode_starts = np.zeros(self.buffer_size + self.explore_buffer_size, int)
        self.episode_lengths = np.zeros(self.buffer_size + self.explore_buffer_size, int)

        for pos, offset, size, max_size in [
            [self.pos, 0, self.normal_size(), self.buffer_size],
            [self.explore_pos - self.buffer_size, self.buffer_size, self.explore_size(), self.explore_buffer_size],
        ]:
            if size == 0:
                continue

            for env in range(self.n_envs):
                goal_idxs = self.goal_idxs[offset + env : max_size + offset : self.n_envs]
                if size == max_size:
                    changes = np.diff(goal_idxs, append=goal_idxs[0]) != 0
                else:
                    changes = np.diff(goal_idxs, append=goal_idxs[-1]) != 0

                changes |= self.dones[offset + env : max_size + offset : self.n_envs]

                # roll such that pos is at the end
                pos_offset = (pos - self.n_envs + env) % max_size // self.n_envs + 1
                changes = np.roll(changes, len(changes) - pos_offset)
                changes[-1] = True

                change_idx = np.where(changes)[0]

                lengths = np.diff(change_idx, prepend=-1)
                lengths[0] -= lengths.sum() - size // self.n_envs

                # undo roll
                change_idx = change_idx - (len(changes) - pos_offset) % len(changes)
                # map to gloabl indices
                change_idx = offset + env + change_idx * self.n_envs

                for length, end in zip(lengths, change_idx):
                    ep_indices = (end - np.flip(np.arange(length)) * self.n_envs - offset) % max_size + offset
                    self.episode_ends[ep_indices] = end
                    self.episode_lengths[ep_indices] = length
                    assert np.all(self.goal_idxs[ep_indices] == self.goal_idxs[ep_indices[0]])

    def _get_episode_indices(self, indices: np.ndarray, max_length: int | None = None) -> np.ndarray:
        lengths = self.episode_lengths[indices]

        min_length = lengths.min()
        if max_length is not None:
            min_length = min(min_length, max_length)

        is_exploration = indices >= self.buffer_size
        ep_indices = self.episode_ends[indices, None] - np.flip(np.arange(min_length))[None] * self.n_envs
        ep_indices[is_exploration] = (
            ep_indices[is_exploration] - self.buffer_size
        ) % self.explore_buffer_size + self.buffer_size
        ep_indices[~is_exploration] = ep_indices[~is_exploration] % self.buffer_size
        return ep_indices

    def _select_decoder_target_indices(self, indices: np.ndarray, num_decoder_targets: int) -> np.ndarray:
        ep_indices = self._get_episode_indices(indices, num_decoder_targets)
        assert np.all(self.goal_idxs[ep_indices] == self.goal_idxs[ep_indices[:, 0]][:, None])
        return ep_indices


class EpisodeLinkingCemrlReplayBuffer(NoLinkingCemrlReplayBuffer):
    def __init__(self, *args, num_linked_episodes=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_linked_episodes = num_linked_episodes

    def _build_decoder_index(self):
        self.episode_ends: np.ndarray
        CEMRLReplayBuffer._build_decoder_index(self)  # type: ignore
        super()._build_decoder_index()

    def _select_decoder_target_indices(self, indices: np.ndarray, num_decoder_targets: int) -> np.ndarray:
        linked_samples = CEMRLReplayBuffer._select_decoder_target_indices(self, indices, self.num_linked_episodes)
        indices = linked_samples.reshape(-1)
        no_linking_targets = super()._select_decoder_target_indices(indices, num_decoder_targets // self.num_linked_episodes)  # type: ignore
        task_episode_indices = no_linking_targets.reshape(len(linked_samples), self.num_linked_episodes, -1)
        indices = task_episode_indices.reshape(len(linked_samples), -1)
        assert np.all(self.goal_idxs[indices] == self.goal_idxs[indices][:, 0, None])
        return indices
