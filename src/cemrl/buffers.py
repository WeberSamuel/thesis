"""This file contains the Replaybuffer Wrapper used by cemrl to train the encoder."""
from typing import Any, Dict, List, Optional, Union, cast
from gym import spaces
import numpy as np
import torch as th
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples, DictReplayBufferSamples, TensorDict
from stable_baselines3.common.vec_env import VecNormalize, VecEnv, VecEnvWrapper
from src.cemrl.types import CEMRLObsTensorDict, CEMRLSacPolicyTensorInput
from src.utils import remove_dim_from_space, apply_function_to_type, get_random_encoder_window_samples
from torch.utils.data import default_collate


class EpisodicBuffer:
    def __init__(self, buffer_size: int, obs_shape, n_envs: int, num_tasks: int, max_episode_length: int = 200) -> None:
        self.episode_lengths = np.zeros(buffer_size, dtype=int)
        self.episodes = [[] for _ in range(num_tasks)]
        self.episodes_in_progress = np.zeros((buffer_size, max_episode_length, *obs_shape))
        self.env_pos = np.zeros(buffer_size, dtype=int)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        pass


class CEMRLBuffer(DictReplayBuffer):
    def __init__(
        self,
        env: VecEnvWrapper | VecEnv,
        buffer_size: int = 10_000_000,
        device: Union[th.device, str] = "auto",
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        link_episodes=True,
    ):
        assert isinstance(env.observation_space, spaces.Dict)
        obs_spaces = {
            "observation": env.unwrapped.observation_space,
            "goal": remove_dim_from_space(env.observation_space.spaces["goal"], 0),  # type: ignore
            "goal_idx": remove_dim_from_space(env.observation_space.spaces["goal_idx"], 0),  # type: ignore
            "task": remove_dim_from_space(env.observation_space.spaces["task"], 0),  # type: ignore
        }
        super().__init__(
            buffer_size,
            spaces.Dict(obs_spaces),
            env.action_space,
            device,
            env.num_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        self.episodes = np.zeros((10_000, 2), dtype=int)
        self.episodes_env = np.zeros(10_000, dtype=int)
        self.episodes_valid = np.zeros(10_000, dtype=bool)
        self.episodes_goal_idx = np.zeros(10_000, dtype=int)
        self.env_current_episode_idx = np.arange(0, env.num_envs, dtype=int)
        self.env_episode_starting_pos = np.zeros(env.num_envs, dtype=int)
        self.num_episodes = 0
        self.all_episodes_valid = False
        self.link_episodes = link_episodes

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        obs = {
            "observation": obs["observation"][:, -1],
            "goal": obs["goal"][:, -1],
            "goal_idx": obs["goal_idx"][:, -1],
            "task": obs["task"][:, -1],
        }
        next_obs = {
            "observation": next_obs["observation"][:, -1],
            "goal": next_obs["goal"][:, -1],
            "goal_idx": next_obs["goal_idx"][:, -1],
            "task": next_obs["task"][:, -1],
        }

        if np.any(done):
            finished_idx = np.where(done)[0]
            finished_episodes_idx = self.env_current_episode_idx[finished_idx]
            self.episodes[finished_episodes_idx, 1] = self.pos
            self.episodes[finished_episodes_idx, 0] = self.env_episode_starting_pos[finished_idx]
            self.episodes_env[finished_episodes_idx] = finished_idx
            self.episodes_goal_idx[finished_episodes_idx] = obs["goal_idx"][finished_idx, 0]
            self.num_episodes = (self.num_episodes + len(finished_idx)) % len(self.episodes)
            self.env_current_episode_idx[finished_idx] = (np.arange(0, len(finished_idx)) + self.num_episodes) % len(
                self.episodes
            )
            self.env_episode_starting_pos[finished_idx] = self.pos + 1
            if self.num_episodes < len(finished_idx):
                self.all_episodes_valid = True

            episodes_for_sort = self.episodes[:, 0]
            if not self.all_episodes_valid:
                episodes_for_sort = episodes_for_sort[: self.num_episodes]
            self.sorted_episodes_start = np.argsort(episodes_for_sort)

        if self.full:
            if self.sorted_episodes_start[0] < self.pos:
                self.episodes[: len(self.episodes) if self.all_episodes_valid else self.num_episodes][
                    self.sorted_episodes_start < self.pos, 0
                ] = self.pos
                self.episodes[self.episodes[:, 1] < self.episodes[:, 0]] = self.pos

        return super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        upper_bound = len(self.episodes) if self.all_episodes_valid else self.num_episodes
        episodes_idx = np.random.randint(0, upper_bound, size=(batch_size))
        env_indices = self.episodes_env[episodes_idx]
        episodes = self.episodes[episodes_idx]

        if self.link_episodes:
            goal_idx = self.episodes_goal_idx[episodes_idx]
            episodes = np.zeros((batch_size, 4), dtype=int)
            episodes[:, 0] = episodes_idx

            for idx, linked in zip(goal_idx, episodes):
                possible = np.where(self.episodes_goal_idx[:upper_bound] == idx)[0]
                linked[1:] = np.random.choice(possible, 3)

            env_indices = self.episodes_env[episodes]
            episodes = self.episodes[episodes]
        else:
            episodes = episodes[:, None]
            env_indices = self.episodes_env[:, None]

        shortest_episode = (
            np.where(episodes[..., 0] <= episodes[..., 1], episodes[..., 1], episodes[..., 1] + self.buffer_size)
            - episodes[..., 0]
        ).min()
        dec_indices = (episodes[:, :, 0, None] + np.arange(0, shortest_episode)[None, None]) % self.buffer_size
        env_indices = np.broadcast_to(env_indices[:, :, None], dec_indices.shape)
        batch_inds = dec_indices.reshape(-1)
        env_indices = env_indices.reshape(-1)

        samples = self._get_samples(batch_inds, env_indices, env=env)
        return DictReplayBufferSamples(
            apply_function_to_type(samples.observations, th.Tensor, lambda x: x.view(batch_size, shortest_episode * 4, -1)),  # type: ignore
            samples.actions.view(batch_size, shortest_episode * 4, -1),
            apply_function_to_type(samples.next_observations, th.Tensor, lambda x: x.view(batch_size, shortest_episode * 4, -1)),  # type: ignore
            samples.dones.view(batch_size, shortest_episode * 4, -1),
            samples.rewards.view(batch_size, shortest_episode * 4, -1),
        )

    def _get_samples(
        self, batch_inds: np.ndarray, env_indices: np.ndarray, env: Optional[VecNormalize] = None
    ) -> DictReplayBufferSamples:
        batch_size = len(batch_inds)

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}, env)
        next_obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
        )

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}  # type: ignore
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}  # type: ignore
        actions = self.to_torch(self.actions[batch_inds, env_indices])
        dones = self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
            -1, 1
        )
        rewards = self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env))

        return DictReplayBufferSamples(observations, actions, next_observations, dones, rewards)


class CEMRLPolicyBuffer(DictReplayBuffer):
    """Wrapper for a CEMRL buffer that returns samples for the CEMRLPolicy.

    This means that a task encoding needs to be generated and the history, actions, and rewards has to be stripped away.
    """

    def __init__(
        self,
        env: VecEnv | VecEnvWrapper,
        encoder: th.nn.Module,
        cemrl_replay_buffer: DictReplayBuffer,
        encoder_window: int,
    ):
        """Initialize the buffer.

        Args:
            buffer_size (int): Ignored, since the wrapped buffer is used.
            observation_space (spaces.Space): Observation space. Ignored.
            action_space (spaces.Space): Action space. Ignored.
            encoder (th.nn.Module): The encoder module used to generate the task encoding.
            cemrl_replay_buffer (ReplayBuffer): CEMRL buffer that should be wrapped.
        """
        super().__init__(
            0,
            env.observation_space,
            env.action_space,
            cemrl_replay_buffer.device,
            cemrl_replay_buffer.n_envs,
            cemrl_replay_buffer.optimize_memory_usage,
            cemrl_replay_buffer.handle_timeout_termination,
        )
        self.cemrl_replay_buffer = cemrl_replay_buffer
        self.encoder = encoder
        self.encoder_window = encoder_window

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        cemrl_samples = self.cemrl_replay_buffer.sample(batch_size, env)
        obs, actions, next_obs, dones, rewards = get_random_encoder_window_samples(cemrl_samples, self.encoder_window)

        with th.no_grad():
            _, z = self.encoder(CEMRLObsTensorDict(observation=obs, reward=rewards, action=actions))

        obs = CEMRLSacPolicyTensorInput(observation=obs[:, -1], task_indicator=z)
        next_obs = CEMRLSacPolicyTensorInput(observation=next_obs[:, -1], task_indicator=z)

        return DictReplayBufferSamples(
            observations=obs,  # type: ignore
            actions=actions[:, -1],
            next_observations=next_obs,  # type: ignore
            dones=dones[:, -1],
            rewards=rewards[:, -1],
        )

    def add(self, *args, **kwargs) -> None:
        """Override to make nothing, instead of adding samples."""
        self.pos = self.cemrl_replay_buffer.pos
        self.full = self.cemrl_replay_buffer.full
        pass

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        result = self.cemrl_replay_buffer._get_samples(batch_inds, env)

        with th.no_grad():
            _, obs_z = self.encoder(result.observations)
            _, next_obs_z = self.encoder(result.next_observations)

        # add task encoding
        obs = CEMRLSacPolicyTensorInput(observation=result.observations["observation"][:, -1], task_indicator=obs_z)

        next_obs = CEMRLSacPolicyTensorInput(
            observation=result.next_observations["observation"][:, -1], task_indicator=next_obs_z
        )

        return ReplayBufferSamples(obs, result.actions, next_obs, result.dones, result.rewards)  # type: ignore


class CombinedBuffer(DictReplayBuffer):
    def __init__(self, buffers: List[CEMRLBuffer], **kwargs):
        self.buffers = buffers
        dummy_space = spaces.Dict({"obs": spaces.Box(0, 1, (1,))})
        super().__init__(0, dummy_space, spaces.Box(0, 1, (1,)))

    def add(self, *args, **kwargs) -> None:
        self.buffers[0].add(*args, **kwargs)

    def _concat_tensor_recusively(self, samples):
        if isinstance(samples[0], dict):
            return {key: self._concat_tensor_recusively([sample[key] for sample in samples]) for key, _ in samples[0].items()}
        if isinstance(samples[0], tuple):
            return tuple(self._concat_tensor_recusively([sample[i] for sample in samples]) for i, _ in enumerate(samples[0]))
        if isinstance(samples[0], th.Tensor):
            return th.cat(samples)
        return samples

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        has_no_episodes = [buffer.num_episodes == 0 and not buffer.all_episodes_valid for buffer in self.buffers]
        items_per_buffer = np.array([buffer.size() * buffer.n_envs for buffer in self.buffers])
        items_per_buffer[has_no_episodes] = 0
        sum_items = np.sum(items_per_buffer)
        num_samples = batch_size * (items_per_buffer / sum_items)
        num_samples = num_samples.astype(int)

        buffer_return = [
            buffer.sample(num_samples, env) for num_samples, buffer in zip(num_samples, self.buffers) if num_samples != 0
        ]
        return DictReplayBufferSamples(*self._concat_tensor_recusively(buffer_return))  # type: ignore
