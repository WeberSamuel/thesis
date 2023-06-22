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
        self.timeouts = np.zeros(buffer_size, bool)
        self.grouped_goal_idx = None

        self.goals = {}
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
        self.grouped_goal_idx = None
        pos = self.explore_pos if self.is_exploring else self.pos
        pos = np.arange(self.n_envs) + pos

        self.observations[pos] = obs["observation"][:, -1]
        self.next_observations[pos] = next_obs["observation"][:, -1]
        self.goal_idxs[pos] = obs["goal_idx"][:, -1].squeeze(-1)
        self.actions[pos] = next_obs["action"][:, -1]
        self.rewards[pos] = next_obs["reward"][:, -1, 0]
        self.dones[pos] = done
        self.timeouts[pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        for goal, goal_idx in zip(obs["goal"][:, -1],  obs["goal_idx"][:, -1]):
            old_goal = self.goals.setdefault(goal_idx.item(), goal)
            assert np.all(goal == old_goal), "Goal idx should always correspond to the same goal"

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
        indices = np.random.choice(self.valid_indices(), batch_size)
        encoder_context = self.get_encoder_context(indices, env)
        with th.no_grad():
            _, z = self.encoder(
                CEMRLObsTensorDict(
                    observation=encoder_context.next_observations, action=encoder_context.actions, reward=encoder_context.rewards
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
        

        obs = self.to_torch(self._normalize_obs(self.observations[indices], env)) # type: ignore
        next_obs = self.to_torch(self._normalize_obs(self.next_observations[indices], env)) # type: ignore
        actions = self.to_torch(self.actions[indices])
        dones = self.to_torch(self.dones[indices])
        rewards = self.to_torch(self._normalize_reward(self.rewards[indices], env))[..., None]

        # Zero out the data of old episodes (assignment in single step is faster)
        batch_idx, time_idx = th.where(dones)
        additional_batch_idx = [batch_idx]
        additional_time_idx = [time_idx]
        for b, i in zip(batch_idx.tolist(), time_idx.tolist()): # type: ignore
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

    def get_decoder_targets(self, indices: np.ndarray, env: VecNormalize | None = None, num_decoder_targets: int = 300):
        """
        Return samples to be used as targets by the decoder. 
        The returned samples have the same goal as the input samples, yet they can come from different episodes.
        This is a modification of episode-linking.
        """
        if self.grouped_goal_idx is None:
            self._build_decoder_index()
        rng = np.random.default_rng()

        goal_idxs = self.goal_idxs[indices]
        indices = np.zeros((len(indices), num_decoder_targets), int)
        for i, goal_idx in enumerate(goal_idxs):
            indices[i] = rng.choice(self.grouped_goal_idx[goal_idx], num_decoder_targets)  # type: ignore

        return ReplayBufferSamples(
            observations=self.to_torch(self._normalize_obs(self.observations[indices], env)),  # type: ignore
            next_observations=self.to_torch(self._normalize_obs(self.next_observations[indices], env)),  # type: ignore
            actions=self.to_torch(self.actions[indices]),
            dones=self.to_torch(self.dones[indices] * (1 - self.timeouts[indices])),
            rewards=self.to_torch(self._normalize_reward(self.rewards[indices], env))[..., None],
        )

    def _build_decoder_index(self):
        df = pd.DataFrame({"goal_idx": self.goal_idxs})
        groups = df.groupby(by=df.goal_idx)
        self.grouped_goal_idx = groups.groups
        self.grouped_goal_idx.pop(-1, -1)

        if self.use_bin_weighted_decoder_target_sampling:
            indices = self.valid_indices()
            _, _, bin_index = binned_statistic_dd(self.observations[indices], indices, statistic="count", bins=10)
            _, idx_to_unique, bin_stats = np.unique(bin_index, return_inverse=True, return_counts=True)
            bin_stats = bin_stats.max() / bin_stats
            bin_stats = bin_stats / bin_stats.sum()
            weighted_bin_prob = np.zeros(self.buffer_size + self.explore_buffer_size, dtype=float)
            weighted_bin_prob[indices] = bin_stats[idx_to_unique]

            for k,v in self.grouped_goal_idx.items():
                weights = weighted_bin_prob[v] / weighted_bin_prob[v].sum()
                self.grouped_goal_idx[k] = np.random.choice(v, len(v)*2, p=weights)
