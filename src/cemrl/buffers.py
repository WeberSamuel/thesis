from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch as th
from gymnasium import Space, spaces
from scipy.stats import binned_statistic_dd
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples, ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from .types import CEMRLPolicyInput
from ..core.buffers import ReplayBuffer


class CEMRLReplayBuffer(ReplayBuffer):
    pass

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
    def __init__(self, *args, num_linked_episodes=5, use_bin_weighted_decoder_target_sampling=False, **kwargs):
        super().__init__(*args, use_bin_weighted_decoder_target_sampling=use_bin_weighted_decoder_target_sampling, **kwargs)
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
