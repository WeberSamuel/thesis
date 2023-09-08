from typing import Any, Dict, List, NamedTuple, Optional
from gymnasium import spaces
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
import torch as th
import numpy as np


class DataTemplate(NamedTuple):
    obs: Dict[str, np.ndarray]
    next_obs: Dict[str, np.ndarray]
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray


class Storage:
    def __init__(self, num_episodes: int, max_episode_length, obs_space: spaces.Dict, action_space: spaces.Box) -> None:
        self.pos = 0
        self.full = False
        self.max_episode_length = max_episode_length
        self.num_episodes = num_episodes
        num_episodes = num_episodes + 1  # last is a dummy
        self.episode_lengths = np.zeros(num_episodes, dtype=np.int64)

        self.observations: Dict[str, np.ndarray] = {key: np.zeros((num_episodes, max_episode_length, *space.shape), dtype=space.dtype) for key, space in obs_space.items()}  # type: ignore
        self.next_observations: Dict[str, np.ndarray] = {key: np.zeros((num_episodes, max_episode_length, *space.shape), dtype=space.dtype) for key, space in obs_space.items()}  # type: ignore
        self.rewards = np.zeros((num_episodes, max_episode_length), dtype=np.float32)
        self.actions = np.zeros((num_episodes, max_episode_length, *action_space.shape), dtype=action_space.dtype)
        self.dones = np.zeros((num_episodes, max_episode_length), dtype=np.float32)
        self.timeouts = np.zeros((num_episodes, max_episode_length), dtype=np.float32)
        self.first = np.zeros((num_episodes, max_episode_length), dtype=np.float32)
        self.changed = False

    def __len__(self):
        return self.num_episodes if self.full else self.pos

    def start_new_episode(self, n_env: int):
        result = np.arange(self.pos, self.pos + n_env) % self.num_episodes
        self.episode_lengths[result] = 0
        self.pos = self.pos + n_env
        if self.pos > self.num_episodes:
            self.full = True
            self.pos = self.pos % self.num_episodes
        return result

    def add(self, episode_idxs: np.ndarray, data: DataTemplate, infos: list[dict[str, np.ndarray]]):
        self.changed = True
        pos = self.episode_lengths[episode_idxs]
        for key in self.observations.keys():
            self.observations[key][episode_idxs, pos] = np.array(data.obs[key])
        for key in self.next_observations.keys():
            self.next_observations[key][episode_idxs, pos] = np.array(data.next_obs[key])
        self.actions[episode_idxs, pos] = data.actions
        self.rewards[episode_idxs, pos] = data.rewards
        self.dones[episode_idxs, pos] = data.dones
        self.timeouts[episode_idxs, pos] = data.timeouts
        self.episode_lengths[episode_idxs] += 1


class ReplayBuffer(DictReplayBuffer):
    obs_shape: Dict[str, tuple[int, ...]]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        max_episode_length: int = 1000,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        storage: Storage | None = None,
    ):
        super().__init__(0, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)

        self.pos = np.zeros_like(self.n_envs, dtype=np.int64)
        self.prepared_sampling = False
        if storage is None:
            storage = Storage(buffer_size, max_episode_length, observation_space, action_space)
        self.storage = storage
        self.storage_idxs = storage.start_new_episode(self.n_envs)

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        self.prepared_sampling = False

        for key in self.observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):  # type: ignore
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])  # type: ignore
            obs[key] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):  # type: ignore
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])  # type: ignore
            next_obs[key] = np.array(next_obs[key])

        data = DataTemplate(
            obs=obs,
            next_obs=next_obs,
            actions=np.array(action.reshape((self.n_envs, self.action_dim))).copy(),
            rewards=np.array(reward).copy(),
            dones=np.array(done).copy(),
            timeouts=np.array([info.get("TimeLimit.truncated", False) for info in infos]),
        )

        self.storage.add(self.storage_idxs, data, infos)

        env_dones = np.where(done)[0]
        self.storage_idxs[env_dones] = self.storage.start_new_episode(len(env_dones))

    def sample(self, batch_size: int, env: VecNormalize | None = None) -> DictReplayBufferSamples:
        self.prepare_sampling_if_necessary()

        episode_idxs = np.random.choice(self.valid_episodes, size=batch_size)
        sample_idxs = np.random.randint(0, self.storage.episode_lengths[episode_idxs], size=batch_size)

        return self._get_samples(episode_idxs, sample_idxs, env=env)

    def sample_context(
        self, batch_size: int, env: VecNormalize | None = None, context_length: int = 64
    ) -> DictReplayBufferSamples:
        self.prepare_sampling_if_necessary()

        episode_idxs = np.random.choice(self.valid_episodes, size=batch_size)
        return self._get_context(episode_idxs, env=env, context_length=context_length)

    def _get_context(
        self,
        episode_idx: np.ndarray,
        sample_idx: np.ndarray | None = None,
        env: VecNormalize | None = None,
        context_length: int = 64,
    ) -> DictReplayBufferSamples:
        if sample_idx is None:
            sample_idx = np.random.randint(1, self.storage.episode_lengths[episode_idx], size=len(episode_idx))
        sample_idxs = np.repeat(sample_idx, context_length)
        sample_idxs = np.tile(-np.arange(context_length)[::-1], len(episode_idx)) + sample_idxs  # type: ignore
        episode_idxs = np.repeat(episode_idx, context_length)

        # trick to load zeros for all data before the first step as last episode in storage is dummy
        episode_idxs[sample_idxs < 0] = -1

        samples = self._get_samples(episode_idxs, sample_idxs, env=env)

        return DictReplayBufferSamples(
            observations={
                key: value.reshape(len(episode_idx), context_length, *value.shape[1:])
                for key, value in samples.observations.items()
            },
            actions=samples.actions.reshape(len(episode_idx), context_length, *samples.actions.shape[1:]),
            next_observations={
                key: value.reshape(len(episode_idx), context_length, *value.shape[1:])
                for key, value in samples.next_observations.items()
            },
            dones=samples.dones.reshape(len(episode_idx), context_length, *samples.dones.shape[1:]),
            rewards=samples.rewards.reshape(len(episode_idx), context_length, *samples.rewards.shape[1:]),
        )

    def _get_samples(
        self, episode_idxs: np.ndarray, sample_idxs: np.ndarray, env: VecNormalize | None = None
    ) -> DictReplayBufferSamples:
        data = DataTemplate(
            obs={key: obs[episode_idxs, sample_idxs] for key, obs in self.storage.observations.items()},
            next_obs={key: obs[episode_idxs, sample_idxs] for key, obs in self.storage.next_observations.items()},
            rewards=self.storage.rewards[episode_idxs, sample_idxs],
            dones=self.storage.dones[episode_idxs, sample_idxs],
            timeouts=1 - self.storage.timeouts[episode_idxs, sample_idxs],
            actions=self.storage.actions[episode_idxs, sample_idxs],
        )

        return self.post_process_samples(data, env)

    def post_process_samples(self, data: DataTemplate, env: VecNormalize | None = None) -> DictReplayBufferSamples:
        obs = self._normalize_obs(data.obs, env)
        next_obs = self._normalize_obs(data.next_obs, env)

        return DictReplayBufferSamples(
            observations={k: self.to_torch(v) for k, v in obs.items()},  # type: ignore
            next_observations={k: self.to_torch(v) for k, v in next_obs.items()},  # type: ignore
            actions=self.to_torch(data.actions),
            dones=self.to_torch(data.dones * (1 - data.timeouts))[..., None],
            rewards=self.to_torch(self._normalize_reward(data.rewards[..., None], env)),
        )

    def prepare_sampling_if_necessary(self):
        if self.storage.changed:
            self.prepare_sampling(self.storage)
            self.storage.changed = False

    def prepare_sampling(self, storage: Storage):
        self.valid_episodes = np.where(storage.episode_lengths > 0)[0]

    def get_empty_data_template(self, batch_size, batch_length):
        return DataTemplate(
            obs={
                k: np.zeros((batch_size, batch_length, *v.shape[2:]), dtype=v.dtype)
                for k, v in self.storage.observations.items()
            },
            next_obs={
                k: np.zeros((batch_size, batch_length, *v.shape[2:]), dtype=v.dtype)
                for k, v in self.storage.next_observations.items()
            },
            actions=np.zeros((batch_size, batch_length, *self.storage.actions.shape[2:]), dtype=self.storage.actions.dtype),
            rewards=np.zeros((batch_size, batch_length, *self.storage.rewards.shape[2:]), dtype=self.storage.rewards.dtype),
            dones=np.zeros((batch_size, batch_length, *self.storage.dones.shape[2:]), dtype=self.storage.dones.dtype),
            timeouts=np.zeros((batch_size, batch_length, *self.storage.timeouts.shape[2:]), dtype=self.storage.timeouts.dtype),
        )

    def add_episode_to_data_template(
        self,
        to_batch_idx: np.ndarray,
        to_sample_idx: np.ndarray,
        from_episode: np.ndarray,
        from_start_stop: np.ndarray,
        data: DataTemplate,
    ):
        from_start_stop = from_start_stop.copy()
        data_length = data.actions.shape[1]

        start, stop = from_start_stop[:, 0], from_start_stop[:, 1]
        lenght = stop - start
        stop += np.minimum(data_length - to_sample_idx - lenght, 0)
        lenght = stop - start
        # numpy repeat element n times where n comes from the length array

        from_episodes = np.repeat(from_episode, lenght)
        from_samples = np.concatenate([np.arange(start, stop) for start, stop in from_start_stop])

        to_batch_idx = np.repeat(to_batch_idx, lenght)
        to_sample_idx = np.concatenate([np.arange(sample_idx, sample_idx + lenght) for sample_idx, lenght in zip(to_sample_idx, lenght)])

        for k, v in self.storage.observations.items():
            data.obs[k][(to_batch_idx, to_sample_idx)] = v[(from_episodes, from_samples)]
        for k, v in self.storage.next_observations.items():
            data.next_obs[k][(to_batch_idx, to_sample_idx)] = v[(from_episodes, from_samples)]

        data.actions[(to_batch_idx, to_sample_idx)] = self.storage.actions[(from_episodes, from_samples)]
        data.rewards[(to_batch_idx, to_sample_idx)] = self.storage.rewards[(from_episodes, from_samples)]
        data.dones[(to_batch_idx, to_sample_idx)] = self.storage.dones[(from_episodes, from_samples)]
        data.timeouts[(to_batch_idx, to_sample_idx)] = self.storage.timeouts[(from_episodes, from_samples)]


class PrepareGoalToEpisode:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def prepare_sampling(self, storage: Storage):
        if "goal_idx" not in storage.observations:
            raise ValueError("goal_idx not in storage.observations")
        self.goal_to_episode = {}
        for episode, goal_idxs in enumerate(storage.observations["goal_idx"][: len(storage)]):
            for goal_idx in np.unique(goal_idxs):
                episodes = self.goal_to_episode.setdefault(goal_idx, [])
                episodes.append(episode)
