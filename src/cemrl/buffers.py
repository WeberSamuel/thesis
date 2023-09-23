from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch as th
from gymnasium import Space, spaces
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples, ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.policies import BasePolicy
from scipy.stats import binned_statistic_dd

from src.cemrl.types import CEMRLObsTensorDict, CEMRLPolicyInput
from .task_inference import EncoderInput, TaskInference
from .types import CEMRLPolicyInput


class CEMRLReplayBuffer(DictReplayBuffer):
    """Replay buffer used in CEMRL."""

    task_inference: TaskInference

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: Space,
        encoder: th.nn.Module | None = None,
        exploration_buffer_size: int = 1_000_000,
        n_envs=1,
        use_bin_weighted_decoder_target_sampling: bool = True,
        encoder_window: int = 30,
        num_decoder_targets: int = 400,
        **kwargs,
    ):
        buffer_size = buffer_size - buffer_size % n_envs
        super().__init__(buffer_size, observation_space, action_space, n_envs=n_envs, **kwargs)
        self.buffer_size = buffer_size

        # Trick to include exploration data in the buffer
        self.explore_buffer_size = exploration_buffer_size - exploration_buffer_size % self.n_envs
        self.explore_pos = buffer_size
        self.explore_full = False
        self.is_exploring = False
        buffer_size = buffer_size + self.explore_buffer_size

        obs_space = observation_space["observation"]
        assert isinstance(action_space, spaces.Box)
        assert isinstance(obs_space, spaces.Box)

        # obs_shape = obs_space.shape[1:]  # history will be removed
        obs_shape = obs_space.shape  # history will be removed
        self.observations = np.zeros((buffer_size, *obs_shape), obs_space.dtype)
        self.next_observations = np.zeros((buffer_size, *obs_shape), obs_space.dtype)
        self.goal_idxs = np.full(buffer_size, -1, int)
        self.actions = np.zeros((buffer_size, *action_space.shape), action_space.dtype)
        self.rewards = np.zeros(buffer_size, np.float32)
        self.dones = np.zeros(buffer_size, bool)
        self.goal_changes = np.zeros(buffer_size, bool)
        self.timeouts = np.zeros(buffer_size, bool)
        self.idxs_by_goal = None
        self.use_bin_weighted_decoder_target_sampling = use_bin_weighted_decoder_target_sampling

        self.encoder_window = encoder_window
        self.num_decoder_targets = num_decoder_targets

    def size(self) -> int:
        """Return the current size of the buffer."""
        return self.normal_size() + self.explore_size()

    def normal_size(self) -> int:
        """Return the current size of the normal data buffer."""
        return self.buffer_size if self.full else self.pos

    def explore_size(self) -> int:
        """Return the current size of the exploration data buffer."""
        return self.explore_buffer_size if self.explore_full else self.explore_pos - self.buffer_size

    def cemrl_sample(
        self,
        batch_size: int,
        env: VecNormalize | None = None,
        encoder_window: int | None = None,
        decoder_samples: int | None = None,
    ) -> tuple[ReplayBufferSamples, ReplayBufferSamples]:
        encoder_window = encoder_window or self.encoder_window
        decoder_samples = decoder_samples or self.num_decoder_targets
        batch_inds = np.random.choice(self.valid_indices(), batch_size)
        encoder_input: ReplayBufferSamples = self.get_encoder_context(batch_inds, env, encoder_window)  # type:ignore
        dec_samples = self.get_decoder_targets(batch_inds, env, decoder_samples)

        return encoder_input, dec_samples

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

        is_exploring = self.is_exploring or np.any([info.get("is_exploration", False) for info in infos])

        pos = self.explore_pos if is_exploring else self.pos
        pos = np.arange(self.n_envs) + pos

        self.observations[pos] = obs["observation"][:]
        self.next_observations[pos] = next_obs["observation"][:]
        self.goal_idxs[pos] = obs["goal_idx"][:].squeeze(-1)
        self.actions[pos] = action
        self.rewards[pos] = reward
        self.dones[pos] = done
        self.timeouts[pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
        self.goal_changes[pos] = np.array([info.get("goal_changed", False) for info in infos]) | done

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

    def valid_indices(self):
        """Return the indices of the valid data in the buffer."""
        normal = np.arange(self.normal_size())
        if self.full:
            without = (self.pos + np.arange(self.n_envs * self.encoder_window)) % self.buffer_size
            normal = np.setdiff1d(normal, without)
        explore = np.arange(self.buffer_size, self.buffer_size + self.explore_size())
        if self.explore_full:
            without = (
                self.explore_pos - self.buffer_size + np.arange(self.n_envs * self.encoder_window)
            ) % self.explore_buffer_size + self.buffer_size
            explore = np.setdiff1d(explore, without)

        return np.concatenate([normal, explore])

    def dreamer_sample(
        self, batch_size: int, env: VecNormalize | None = None, goals: np.ndarray | None = None, max_length: int = 64
    ):
        indices = np.random.choice(self.valid_indices(), batch_size)

        data, indices, mask = self.get_encoder_context(indices, env, max_length, return_indices=True)  # type:ignore

        goal_idx = self.to_torch(self.goal_idxs[indices], data.rewards.device)
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
            "observation": data.observations,
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
        indices = np.random.choice(self.valid_indices(), batch_size)
        encoder_context: ReplayBufferSamples = self.get_encoder_context(indices, env)
        with th.no_grad():
            _, z, _ = self.task_inference(
                EncoderInput(
                    obs=encoder_context.observations,
                    action=encoder_context.actions,
                    reward=encoder_context.rewards,
                    next_obs=encoder_context.next_observations,
                )
            )

        obs = self.to_torch(self._normalize_obs(self.observations[indices], env))  # type: ignore
        next_obs = self.to_torch(self._normalize_obs(self.next_observations[indices], env))  # type: ignore
        rewards = self.to_torch(self._normalize_reward(self.rewards[indices], env))
        return DictReplayBufferSamples(
            observations=CEMRLPolicyInput(observation=obs, task_indicator=z),  # type: ignore
            next_observations=CEMRLPolicyInput(observation=next_obs, task_indicator=z),  # type: ignore
            actions=self.to_torch(self.actions[indices]),
            dones=self.to_torch(self.dones[indices] * (1 - self.timeouts[indices]))[..., None],
            rewards=rewards[..., None],
        )

    def get_encoder_context(
        self, indices: np.ndarray, env: VecNormalize | None = None, encoder_window=30, return_indices: bool = False
    ):
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

        data = ReplayBufferSamples(
            observations=obs,  # type: ignore
            next_observations=next_obs,  # type: ignore
            actions=actions,
            dones=dones,
            rewards=rewards,
        )

        if return_indices:
            return data, indices, (batch_idx, time_idx)
        return data

    def get_decoder_targets(self, base_indices: np.ndarray, env: VecNormalize | None = None, num_decoder_targets: int = 300):
        """
        Return samples to be used as targets by the decoder.
        The returned samples have the same goal as the input samples, yet they can come from different episodes.
        This is a modification of episode-linking.
        """

        if not self.is_decoder_index_build:
            self._build_decoder_index(num_decoder_targets)
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
        selected_ids = np.random.rand(len(indices), num_decoder_targets) * self.grouped_goal_sizes[group_ids, None]
        selected_ids = selected_ids.astype(int)

        indices = self.padded_idxs_by_goal[group_ids[:, None], selected_ids]  # type: ignore
        assert np.all(self.goal_idxs[indices] == self.goal_idxs[indices[:, 0]][:, None])
        return indices

    def _build_decoder_index(self, num_decoder_targets: int):
        df = pd.DataFrame({"goal_idx": self.goal_idxs})
        df = df[df.goal_idx != -1]
        groups = df.groupby(by=df.goal_idx)
        self.idxs_by_goal = groups.groups
        self.grouped_goal_sizes = np.zeros(self.goal_idxs.max() + 1, dtype=int)
        self.grouped_goal_sizes[list(groups.groups.keys())] = groups.size().values
        max_group_size = max(self.grouped_goal_sizes.max(), num_decoder_targets)

        padded_idxs_by_goal = np.zeros((self.goal_idxs.max() + 1, max_group_size), dtype=int)
        if self.use_bin_weighted_decoder_target_sampling:
            self.grouped_goal_sizes[:] = max_group_size
            for k, v in self.idxs_by_goal.items():
                count, _, bins = binned_statistic_dd(self.observations[v], v, statistic="count", expand_binnumbers=True)  # type: ignore
                weights = 1 / count[tuple(bins - 1)] / np.count_nonzero(count)
                padded_idxs_by_goal[k] = np.random.choice(v, max_group_size, p=weights)  # type: ignore
        else:
            for k, v in self.idxs_by_goal.items():
                padded_idxs_by_goal[k, : len(v)] = v
        self.padded_idxs_by_goal = padded_idxs_by_goal


class NoLinkingCemrlReplayBuffer(CEMRLReplayBuffer):
    def _build_decoder_index(self, num_decoder_targets: int):
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
    def __init__(self, *args, num_linked_episodes=3, use_bin_weighted_decoder_target_sampling=False, **kwargs):
        super().__init__(*args, use_bin_weighted_decoder_target_sampling=use_bin_weighted_decoder_target_sampling, **kwargs)
        self.num_linked_episodes = num_linked_episodes

    def _build_decoder_index(self, num_decoder_targets: int):
        CEMRLReplayBuffer._build_decoder_index(self, num_decoder_targets)  # type: ignore
        super()._build_decoder_index(num_decoder_targets)

    def _select_decoder_target_indices(self, indices: np.ndarray, num_decoder_targets: int) -> np.ndarray:
        linked_samples = CEMRLReplayBuffer._select_decoder_target_indices(self, indices, self.num_linked_episodes)
        indices = linked_samples.reshape(-1)
        no_linking_targets = super()._select_decoder_target_indices(indices, num_decoder_targets // self.num_linked_episodes)  # type: ignore
        task_episode_indices = no_linking_targets.reshape(len(linked_samples), self.num_linked_episodes, -1)
        indices = task_episode_indices.reshape(len(linked_samples), -1)
        assert np.all(self.goal_idxs[indices] == self.goal_idxs[indices][:, 0, None])
        return indices


class ImagineBuffer:
    def __init__(
        self,
        imagine_horizon: int,
        policy: BasePolicy,
        replay_buffer: DictReplayBuffer,
        world_model: th.nn.Module,
        num_batches: int,
        batch_size: int,
        action_space: spaces.Box,
        env: VecNormalize | None,
    ) -> None:
        self.storage = []

        low = th.tensor(action_space.low, device=replay_buffer.device)
        high = th.tensor(action_space.high, device=replay_buffer.device)

        def scale_action(action: th.Tensor):
            return 2.0 * ((action - low) / (high - low)) - 1.0

        for _ in range(num_batches):
            samples = replay_buffer.sample(batch_size, env)
            state = samples.observations["observation"]
            task_indicator = samples.observations["task_indicator"]
            with th.no_grad():
                for _ in range(imagine_horizon):
                    action = policy._predict(state, deterministic=True)
                    action = scale_action(action)
                    next_state, reward = world_model(state, action, None, z=task_indicator)
                    self.storage.append((state, action, reward, next_state, task_indicator))
                    state = next_state

    def sample(self, batch_size: int):
        idxs = th.randint(0, len(self.storage), (batch_size,), device=self.storage[0][0].device)
        samples = [self.storage[i] for i in idxs]
        return DictReplayBufferSamples(
            observations=CEMRLPolicyInput(
                observation=th.cat([s[0] for s in samples], dim=0),
                task_indicator=th.cat([s[4] for s in samples], dim=0),
            ),  # type: ignore
            next_observations=CEMRLPolicyInput(
                observation=th.cat([s[3] for s in samples], dim=0),
                task_indicator=th.cat([s[4] for s in samples], dim=0),
            ),  # type: ignore
            actions=th.cat([s[1] for s in samples], dim=0),
            rewards=th.cat([s[2] for s in samples], dim=0),
            dones=th.zeros((len(samples), 1), device=samples[0][0].device),
        )
