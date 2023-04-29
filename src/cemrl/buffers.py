"""This file contains the Replaybuffer Wrapper used by cemrl to train the encoder."""
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples, ReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper, VecNormalize, unwrap_vec_wrapper

from src.cemrl.types import CEMRLObsTensorDict, CEMRLSacPolicyTensorInput
from src.cemrl.wrappers import CEMRLHistoryWrapper
from src.utils import get_random_encoder_window_samples, remove_dim_from_space


class EpisodicBuffer(DictReplayBuffer):
    def __init__(
        self,
        env: VecEnvWrapper | VecEnv,
        num_episodes_per_task: int = 100_000,
        device: Union[th.device, str] = "auto",
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        link_episodes: int = 3,
        max_episode_length: int = 200,
    ) -> None:
        assert isinstance(env.observation_space, spaces.Dict)

        num_tasks = int(env.observation_space.spaces["goal_idx"].high[0, 0])  # type: ignore
        self.episodes = np.empty((num_tasks, num_episodes_per_task), dtype=object)
        self.episode_pos = np.zeros(num_tasks, dtype=int)
        self.episode_full = np.zeros(num_tasks, dtype=bool)
        self.episode_length = np.zeros_like(self.episodes, dtype=int)
        self.max_num_episodes_per_task = num_episodes_per_task
        self.link_episodes = link_episodes

        self.history_wrapper = unwrap_vec_wrapper(env, CEMRLHistoryWrapper)
        if self.history_wrapper is None:
            observation_space = env.observation_space
        else:
            observation_space = spaces.Dict(
                {
                    "observation": self.history_wrapper.original_obs_space,
                    "goal": remove_dim_from_space(env.observation_space.spaces["goal"], 0),  # type: ignore
                    "goal_idx": remove_dim_from_space(env.observation_space.spaces["goal_idx"], 0),  # type: ignore
                    "task": remove_dim_from_space(env.observation_space.spaces["task"], 0),  # type: ignore
                }
            )

        super().__init__(
            (max_episode_length + 1) * env.num_envs,
            observation_space,
            env.action_space,
            device,
            env.num_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        self.pos = np.zeros(env.num_envs, dtype=int)

    def _strip_unnecessary_data_from_obs(self, obs: Dict[str, np.ndarray]):
        return {
            "observation": obs["observation"][:, -1],
            "goal": obs["goal"][:, -1],
            "goal_idx": obs["goal_idx"][:, -1],
            "task": obs["task"][:, -1],
        }

    def size(self) -> int:
        return self.episode_length.sum()

    def _original_add_without_updating_pos(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:  # pytype: disable=signature-mismatch
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):  # type: ignore
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])  # type: ignore
            self.observations[key][self.pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):  # type: ignore
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])  # type: ignore
            self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        assert not np.any(self.pos == self.buffer_size), "Episode max length was not kept!"

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        obs = self._strip_unnecessary_data_from_obs(obs)
        next_obs = self._strip_unnecessary_data_from_obs(next_obs)
        self._original_add_without_updating_pos(obs, next_obs, action, reward, done, infos)

        finished_indices = np.where(done)[0]
        if np.any(finished_indices):
            tasks = obs["goal_idx"].squeeze(-1).astype(int)
            for finished_idx, task in zip(finished_indices, tasks):
                obs_indicies = np.arange(self.pos[finished_idx])
                obs = {key: obs[obs_indicies, finished_idx, :].copy() for key, obs in self.observations.items()}
                next_obs = {key: obs[obs_indicies, finished_idx, :].copy() for key, obs in self.next_observations.items()}

                self.episodes[task, self.episode_pos[task]] = (
                    obs,
                    next_obs,
                    self.actions[obs_indicies, finished_idx].copy(),
                    self.rewards[obs_indicies, finished_idx].copy(),
                    self.dones[obs_indicies, finished_idx] # * (1 - self.timeouts[obs_indicies, finished_idx]),
                )
                self.episode_length[task, self.episode_pos[task]] = self.pos[finished_idx]
                self.pos[finished_idx] = 0
                self.episode_pos[task] += 1

                if self.episode_pos[task] == self.max_num_episodes_per_task:
                    self.episode_full[task] = True
                    self.episode_pos[task] = 0

    def sample(self, batch_size: int, env: VecNormalize | None = None) -> DictReplayBufferSamples:
        available_episodes = np.where(self.episode_full, self.max_num_episodes_per_task, self.episode_pos)
        available_tasks = np.where(available_episodes != 0)[0]
        tasks = np.random.choice(available_tasks, batch_size)
        episode_link_idxs = np.random.randint(0, available_episodes[tasks], (self.link_episodes + 1, len(tasks))).T
        episode_length = self.episode_length[tasks, episode_link_idxs.T]
        shortest_length = np.sum(episode_length, axis=0).min()

        obs_shape: Dict = self.obs_shape  # type: ignore
        obs_space: Dict = self.observation_space  # type: ignore

        obs = {k: np.zeros((batch_size, shortest_length, *v), dtype=obs_space[k].dtype) for k, v in obs_shape.items()}
        next_obs = {k: np.zeros((batch_size, shortest_length, *v), dtype=obs_space[k].dtype) for k, v in obs_shape.items()}
        actions = np.zeros((batch_size, shortest_length, self.action_dim), dtype=self.action_space.dtype)  # type: ignore
        rewards = np.zeros((batch_size, shortest_length), dtype=np.float32)
        dones = np.zeros((batch_size, shortest_length), dtype=np.float32)

        for task, batch_idx, link_idxs in zip(tasks, range(batch_size), episode_link_idxs):
            pos = 0
            remaining_length = shortest_length
            for link_idx in link_idxs:
                obs_, next_obs_, actions_, rewards_, dones_ = self.episodes[task, link_idx]
                take = min(self.episode_length[task, link_idx], remaining_length)
                till = pos + take
                for target, source in zip([obs, next_obs], [obs_, next_obs_]):
                    for k in target.keys():
                        target[k][batch_idx, pos:till] = source[k][:take]
                actions[batch_idx, pos:till] = actions_[:take]
                rewards[batch_idx, pos:till] = rewards_[:take]
                dones[batch_idx, pos:till] = dones_[:take]

                pos += take

        obs: Dict = self._normalize_obs(obs, env)  # type: ignore
        next_obs: Dict = self._normalize_obs(next_obs, env)  # type: ignore

        return DictReplayBufferSamples(
            {k: self.to_torch(v) for k, v in obs.items()},
            self.to_torch(actions),
            {k: self.to_torch(v) for k, v in next_obs.items()},
            self.to_torch(dones).reshape(batch_size, -1, 1),
            self.to_torch(rewards).reshape(batch_size, -1, 1),
        )


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
    def __init__(self, buffers: List[DictReplayBuffer], **kwargs):
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
        items_per_buffer = np.array([buffer.size() for buffer in self.buffers])
        sum_items = np.sum(items_per_buffer)
        num_samples = batch_size * (items_per_buffer / sum_items)
        num_samples = num_samples.astype(int)

        buffer_return = [
            buffer.sample(num_sample, env) for num_sample, buffer in zip(num_samples, self.buffers) if num_sample != 0
        ]
        return DictReplayBufferSamples(*self._concat_tensor_recusively(buffer_return))  # type: ignore
