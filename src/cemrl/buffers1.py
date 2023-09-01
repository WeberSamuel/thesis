from typing import Any, Callable, Dict, List, NamedTuple
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
import torch as th
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader

from typing import Dict, NamedTuple
import numpy as np

from src.cemrl.networks import Encoder


class StoredEpisode(NamedTuple):
    """A named tuple representing a stored episode in a replay buffer."""

    obs: Dict[str, np.ndarray]
    """A dictionary of observations."""
    next_obs: Dict[str, np.ndarray]
    """A dictionary of next observations."""
    action: np.ndarray
    """An array of actions taken in each environment."""
    reward: np.ndarray
    """An array of rewards received in each environment."""
    done: np.ndarray
    """An array of done flags."""
    goal_idx: np.ndarray
    """An array of goal indices."""


class InMemoryStorage:
    def __init__(
        self,
        capacity: int,
        tmp_capacity: int,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        max_episode_length: int,
    ) -> None:
        self.pos = 0
        self.full = False
        self.capacity = capacity
        self.tmp_capacity = tmp_capacity
        self.tmp_pos = 0
        capacity += tmp_capacity
        self.episode_lengths = np.zeros(capacity, dtype=np.int64)
        self.goal_to_episodes: dict[int, list[int]] = {}

        self.observations = {
            k: np.zeros((capacity, max_episode_length, *v.shape), dtype=v.dtype)
            for k, v in observation_space.items()
            if isinstance(v, spaces.Box)
        }
        self.actions = np.zeros((capacity, max_episode_length, *action_space.shape), dtype=action_space.dtype)
        self.rewards = np.zeros((capacity, max_episode_length), dtype=np.float32)
        self.dones = np.zeros((capacity, max_episode_length), dtype=np.float32)
        self.next_observations = {
            k: np.zeros((capacity, max_episode_length, *v.shape), dtype=v.dtype)
            for k, v in observation_space.items()
            if isinstance(v, spaces.Box)
        }
        self.goal_idx = np.zeros((capacity, max_episode_length), dtype=np.int64)
        self.is_exploration = np.zeros((capacity, max_episode_length), dtype=bool)
        self.append_to_tmp = False

    def valid_idxs(self):
        return np.concatenate(
            [np.arange(self.capacity) if self.full else np.arange(self.pos), np.arange(self.tmp_pos) + self.capacity]
        )

    def append(self, episode: StoredEpisode):
        if self.append_to_tmp:
            new_idx = self.capacity + self.tmp_pos
            self.tmp_pos = min((self.tmp_pos + 1), self.tmp_capacity - 1)
        else:
            new_idx = self.pos
            self.pos = (self.pos + 1) % self.capacity

        self.episode_lengths[new_idx] = episode_length = len(episode.done)
        for goal in np.unique(episode.goal_idx):
            self.goal_to_episodes.setdefault(goal, []).append(new_idx)

        for k, v in episode.obs.items():
            self.observations[k][new_idx, :episode_length] = v
        for k, v in episode.next_obs.items():
            self.next_observations[k][new_idx, :episode_length] = v
        self.actions[new_idx, :episode_length] = episode.action
        self.rewards[new_idx, :episode_length] = episode.reward
        self.dones[new_idx, :episode_length] = episode.done
        self.goal_idx[new_idx, :episode_length] = episode.goal_idx

    def clear_tmp(self):
        tmp_episodes = np.arange(self.capacity, self.capacity + self.tmp_pos)
        for idx in tmp_episodes:
            for goal in np.unique(self.goal_idx[idx, : self.episode_lengths[idx]]):
                self.goal_to_episodes[goal].remove(idx)
        self.tmp_pos = 0


class EpisodeDataset(IterableDataset[DictReplayBufferSamples]):
    def __init__(self, storage: InMemoryStorage, batch_size: int = 64, max_returned_length: int | None = None):
        self.storage = storage
        self.batch_size = batch_size
        self.max_returned_length = max_returned_length

    def __iter__(self):
        while True:
            episode_idxs = np.random.choice(self.storage.valid_idxs(), self.batch_size)
            data, _ = self.get_batch(episode_idxs, max_returned_length=self.max_returned_length)
            yield data

    def get_batch(self, episode_idxs: np.ndarray, max_returned_length: int | None = None):
        batch_size = len(episode_idxs)
        episode_lengths = self.storage.episode_lengths[episode_idxs]
        shortest_episode_length = episode_lengths.min()
        if max_returned_length is not None:
            shortest_episode_length = min(shortest_episode_length, max_returned_length)

        take_from = np.random.randint(0, episode_lengths - shortest_episode_length + 1, batch_size)
        take_idxs = take_from[:, None] + np.arange(shortest_episode_length)[None]
        episode_idxs = episode_idxs[:, None]

        data = DictReplayBufferSamples(
            observations={k: th.as_tensor(v[episode_idxs, take_idxs]) for k, v in self.storage.observations.items()},
            actions=th.as_tensor(self.storage.actions[episode_idxs, take_idxs]),
            next_observations={k: th.as_tensor(v[episode_idxs, take_idxs]) for k, v in self.storage.next_observations.items()},
            dones=th.as_tensor(self.storage.dones[episode_idxs, take_idxs])[..., None],
            rewards=th.as_tensor(self.storage.rewards[episode_idxs, take_idxs])[..., None],
        )

        return data, take_idxs[:, -1]


class EpisodeLinkingDataset(EpisodeDataset):
    def __init__(self, storage: InMemoryStorage, batch_size: int = 64, returned_length: int = 500):
        super().__init__(storage, batch_size)
        self.returned_length = returned_length

    def __iter__(self):
        while True:
            episode_idx = np.random.choice(self.storage.valid_idxs(), self.batch_size)
            data = self.get_batch(episode_idx, returned_length=self.returned_length)

            yield data

    def get_batch(self, episode_idx: np.ndarray, sample_idx: np.ndarray | None = None, returned_length: int = 500):
        batch_size = len(episode_idx)
        sample_idx = (
            np.random.randint(0, self.storage.episode_lengths[episode_idx], batch_size) if sample_idx is None else sample_idx
        )
        goals = self.storage.goal_idx[episode_idx, sample_idx]

        all_idxs = []
        for i, (e_idx, goal) in enumerate(zip(episode_idx, goals)):
            current_length = 0
            idxs = []
            while current_length < returned_length:
                episode_length = self.storage.episode_lengths[e_idx]
                same_goal_idx = np.where(self.storage.goal_idx[e_idx, :episode_length] == goal)[0]

                episode_length = len(same_goal_idx)
                offset = 0
                if current_length + episode_length > returned_length:
                    episode_length = returned_length - current_length
                    offset = np.random.randint(0, len(same_goal_idx) - episode_length)
                idxs.append(
                    (
                        np.full_like(same_goal_idx[offset : episode_length + offset], e_idx),
                        same_goal_idx[offset : episode_length + offset],
                    )
                )

                e_idx = np.random.choice(self.storage.goal_to_episodes[goal])
                current_length += episode_length

            all_idxs.append(tuple(np.concatenate(x) for x in zip(*idxs)))

        all_idxs = tuple(np.stack(x) for x in zip(*all_idxs))

        data = DictReplayBufferSamples(
            observations={k: th.as_tensor(v[all_idxs]) for k, v in self.storage.observations.items()},
            actions=th.as_tensor(self.storage.actions[all_idxs]),
            next_observations={k: th.as_tensor(v[all_idxs]) for k, v in self.storage.next_observations.items()},
            dones=th.as_tensor(self.storage.dones[all_idxs])[..., None],
            rewards=th.as_tensor(self.storage.rewards[all_idxs])[..., None],
        )

        # assert th.all(data.observations["goal_idx"] == data.observations["goal_idx"][:, 0, None])
        return data


class CEMRLDataset(IterableDataset[tuple[DictReplayBufferSamples, DictReplayBufferSamples]]):
    def __init__(self, storage: InMemoryStorage, batch_size=64, encoder_window: int = 30, decoder_window: int = 400) -> None:
        self.encoder_dataset = PolicyDataset(storage, batch_size, encoder_window=encoder_window)
        self.decoder_dataset = EpisodeLinkingDataset(storage, batch_size=batch_size, returned_length=decoder_window)
        self.batch_size = batch_size
        self.encoder_window = encoder_window
        self.decoder_window = decoder_window

    def __iter__(self):
        while True:
            idx = np.random.choice(self.encoder_dataset.storage.valid_idxs(), self.batch_size)
            sample_idx = np.random.randint(0, self.encoder_dataset.storage.episode_lengths[idx], self.batch_size)
            encoder_data = self.encoder_dataset.get_encoder_data(idx, sample_idx)
            decoder_data = self.decoder_dataset.get_batch(idx, sample_idx=sample_idx, returned_length=self.decoder_window)
            # assert th.all(encoder_data.actions[self.encoder_dataset.storage.episode_lengths[idx] > self.encoder_window] != 0)

            yield encoder_data, decoder_data


class PolicyDataset(IterableDataset[tuple[DictReplayBufferSamples, DictReplayBufferSamples]]):
    """A PyTorch dataset for loading individual samples."""

    def __init__(self, storage: InMemoryStorage, batch_size: int = 64, encoder_window: int = 30) -> None:
        self.storage = storage
        self.batch_size = batch_size
        self.encoder_window = encoder_window

    def __iter__(self):
        while True:
            episode_idx = np.random.choice(self.storage.valid_idxs(), self.batch_size)
            lengths = self.storage.episode_lengths[episode_idx]
            sample_idx = np.random.randint(0, lengths, self.batch_size)

            enc_data = self.get_encoder_data(episode_idx, sample_idx)

            yield DictReplayBufferSamples(
                observations={k: th.as_tensor(v[episode_idx, sample_idx]) for k, v in self.storage.observations.items()},
                actions=th.as_tensor(self.storage.actions[episode_idx, sample_idx]),
                next_observations={
                    k: th.as_tensor(v[episode_idx, sample_idx]) for k, v in self.storage.next_observations.items()
                },
                dones=th.as_tensor(self.storage.dones[episode_idx, sample_idx])[..., None],
                rewards=th.as_tensor(self.storage.rewards[episode_idx, sample_idx])[..., None],
            ), enc_data

    def get_encoder_data(self, episode_idx, sample_idx):
        start = sample_idx - self.encoder_window
        offset = np.where(start < 0, -start - 1, 0)
        start[start < 0] = 0

        observations = {
            k: np.zeros((self.batch_size, self.encoder_window, *v.shape[2:]), dtype=v.dtype)
            for k, v in self.storage.observations.items()
        }
        actions = np.zeros(
            (self.batch_size, self.encoder_window, *self.storage.actions.shape[2:]), dtype=self.storage.actions.dtype
        )
        next_observations = {
            k: np.zeros((self.batch_size, self.encoder_window, *v.shape[2:]), dtype=v.dtype)
            for k, v in self.storage.next_observations.items()
        }
        dones = np.zeros((self.batch_size, self.encoder_window), dtype=self.storage.dones.dtype)
        rewards = np.zeros((self.batch_size, self.encoder_window), dtype=self.storage.rewards.dtype)

        for i, (start, offset, idx) in enumerate(zip(start, offset, episode_idx)):
            for k, v in self.storage.observations.items():
                observations[k][i, offset:] = v[idx, start : start + self.encoder_window - offset]
            for k, v in self.storage.next_observations.items():
                next_observations[k][i, offset:] = v[idx, start : start + self.encoder_window - offset]
            actions[i, offset:] = self.storage.actions[idx, start : start + self.encoder_window - offset]
            dones[i, offset:] = self.storage.dones[idx, start : start + self.encoder_window - offset]
            rewards[i, offset:] = self.storage.rewards[idx, start : start + self.encoder_window - offset]

        return DictReplayBufferSamples(
            observations={k: th.as_tensor(v) for k, v in observations.items()},
            actions=th.as_tensor(actions),
            next_observations={k: th.as_tensor(v) for k, v in next_observations.items()},
            dones=th.as_tensor(dones)[..., None],
            rewards=th.as_tensor(rewards)[..., None],
        )


class SubReplayBuffer(DictReplayBuffer):
    """
    A replay buffer that stores transitions for multiple environments and episodes.

    This buffer extends the `DictReplayBuffer` class and adds support for handling episodes
    that may span multiple contiguous segments of the buffer. When an episode ends, the buffer
    stores only the last `max_episode_length` transitions from that episode, discarding the
    earlier ones. This allows the buffer to store more episodes without running out of memory.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        on_episode_collected: Callable[[StoredEpisode], None],
        max_episode_length: int,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        """
        Initialize a sub-buffer for collecting episodes.

        Args:
            buffer_size (int): The maximum number of episodes that can be stored in the buffer.
            observation_space (gym.Space): The observation space of the environment.
            action_space (gym.Space): The action space of the environment.
            on_episode_collected (Callable[[StoredEpisode], None]): A function to call when an episode is collected.
            max_episode_length (int): The maximum length of an episode.
            device (torch.device | str, optional): The device to use for storing the data. Defaults to "auto".
            n_envs (int, optional): The number of parallel environments. Defaults to 1.
            optimize_memory_usage (bool, optional): Whether to optimize memory usage. Defaults to False.
            handle_timeout_termination (bool, optional): Whether to handle timeout termination. Defaults to True.
        """
        super().__init__(
            n_envs * max_episode_length,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        self.max_episode_length = max_episode_length
        self.on_episode_collected = on_episode_collected
        self.start_pos = np.full(self.n_envs, self.pos)
        self.episode_length = np.zeros(self.n_envs, dtype=np.int64)
        self.is_eploration = np.zeros_like(self.dones)
        self.goal_idx = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)

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
        Add a new transition to the buffer.

        Args:
            obs (Dict[str, np.ndarray]): A dictionary containing the observations for each environment.
            next_obs (Dict[str, np.ndarray]): A dictionary containing the next observations for each environment.
            action (np.ndarray): An array containing the actions taken in each environment.
            reward (np.ndarray): An array containing the rewards obtained in each environment.
            done (np.ndarray): An array indicating whether each environment has terminated.
            infos (List[Dict[str, Any]]): A list of dictionaries containing additional information for each environment.

        Returns:
            None
        """
        super().add(obs, next_obs, action, reward, done, infos)
        self.episode_length += 1

        env_idxs = np.where(done)[0]
        self.publish_episodes(env_idxs)

        self.start_pos[env_idxs] = self.pos
        self.episode_length[env_idxs] = 0

    def publish_episodes(self, env_idxs: np.ndarray | None = None):
        if env_idxs is None:
            env_idxs = np.where(self.episode_length != 0)[0]

        for env_idx, env_start, env_length in zip(env_idxs, self.start_pos[env_idxs], self.episode_length[env_idxs]):
            episode_idxs = (env_start + np.arange(0, env_length)) % self.buffer_size
            episode_idxs = episode_idxs[-self.max_episode_length :]  # take only the last n steps from the episode

            obs = {k: v[episode_idxs, env_idx] for k, v in self.observations.items()}
            next_obs = {k: v[episode_idxs, env_idx] for k, v in self.next_observations.items()}
            actions = self.actions[episode_idxs, env_idx]
            rewards = self.rewards[episode_idxs, env_idx]
            dones = self.dones[episode_idxs, env_idx]
            goal_idxs = (
                self.observations["goal_idx"][episode_idxs, env_idx, 0].astype(int)
                if "goal_idx" in self.observations
                else np.zeros((len(episode_idxs), 1))
            )

            self.on_episode_collected(StoredEpisode(obs, next_obs, actions, rewards, dones, goal_idxs))


class EpisodicReplayBuffer(DictReplayBuffer):
    """A replay buffer that stores episodes in  in separate sub-buffers for exploration and normal data."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        max_episode_length: int = 1000,
        storage_path: str | None = None,
        device: th.device | str = "auto",
        n_envs: int = 1,
        encoder: Encoder | None = None,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        num_workers=1,
    ):
        """
        Initialize a Episodic Buffer object.

        Args:
            buffer_size (int): Ignored input passed by stable-baselines3.
            observation_space (gym.Space): The observation space of the environment.
            action_space (gym.Space): The action space of the environment.
            storage_path (str): The path where the buffer will be stored.
            max_episode_length (int): The maximum length of an episode.
            device (torch.device or str, optional): The device where the tensors will be stored. Defaults to "auto".
            n_envs (int, optional): The number of parallel environments. Defaults to 1.
            optimize_memory_usage (bool, optional): Whether to optimize memory usage. Defaults to False.
            handle_timeout_termination (bool, optional): Whether to handle timeout termination. Defaults to True.
        """
        super().__init__(
            1,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        assert isinstance(observation_space, spaces.Dict)
        assert isinstance(action_space, spaces.Box)
        self.storage = InMemoryStorage(buffer_size, n_envs * 2, observation_space, action_space, max_episode_length)
        self.num_episodes = 0
        self.exploration_buffer = SubReplayBuffer(
            max_episode_length,
            observation_space,
            action_space,
            self.storage.append,
            max_episode_length,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        self.normal_buffer = SubReplayBuffer(
            max_episode_length,
            observation_space,
            action_space,
            self.storage.append,
            max_episode_length,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )

        self.num_workers = num_workers
        self._policy_sample_loader = None
        self._cemrl_sample_loader = None
        self._episodic_sample_loader = None
        self._episodic_iterator = None
        self._cemrl_iterator = None
        self._iterator = None
        self.encoder = encoder

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
            obs (Dict[str, np.ndarray]): The current observation.
            next_obs (Dict[str, np.ndarray]): The next observation.
            action (np.ndarray): The action taken.
            reward (np.ndarray): The reward received.
            done (np.ndarray): Whether the episode is done.
            infos (List[Dict[str, Any]]): Additional information about the transition.

        Returns:
            None
        """
        self.storage.append_to_tmp = False

        if any(info.get("is_exploration", False) for info in infos):
            self.exploration_buffer.add(obs, next_obs, action, reward, done, infos)
        else:
            self.normal_buffer.add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: VecNormalize | None = None, encoder_window=30) -> DictReplayBufferSamples:
        self._store_unfinished_episode()
        if self._iterator is None:
            self._iterator = iter(self.get_policy_sample_loader(batch_size, encoder_window))

        sample, enc_data = next(self._iterator)

        sample = self._to_device(sample)
        enc_data = self._to_device(enc_data)

        if self.encoder is not None:
            with th.no_grad():
                y, z = self.encoder(enc_data.observations)
            sample = sample._replace(
                observations={"observation": sample.observations["observation"], "task_indicator": z},
                next_observations={"observation": sample.next_observations["observation"], "task_indicator": z},
            )

        return sample

    def episodic_sample(
        self, batch_size: int, env: VecNormalize | None = None, max_lenght: int | None = None
    ) -> DictReplayBufferSamples:
        self._store_unfinished_episode()
        if self._episodic_iterator is None:
            self._episodic_iterator = iter(self.get_episodic_sample_loader(batch_size, max_lenght))

        data = next(self._episodic_iterator)
        data = self._to_device(data)
        return data

    def cemrl_sample(
        self, batch_size: int, env: VecNormalize | None = None, encoder_window: int = 30, decoder_window: int = 400
    ) -> tuple[DictReplayBufferSamples, DictReplayBufferSamples]:
        self._store_unfinished_episode()
        if self._cemrl_iterator is None:
            self._cemrl_iterator = iter(self.get_cemrl_sample_loader(batch_size, encoder_window, decoder_window))

        enc_data, dec_data = next(self._cemrl_iterator)
        enc_data = self._to_device(enc_data)
        dec_data = self._to_device(dec_data)
        return enc_data, dec_data

    def _store_unfinished_episode(self):
        if self.storage.append_to_tmp:
            # no add since last call --> noop
            return
        self.storage.append_to_tmp = True
        self.storage.clear_tmp()
        self.exploration_buffer.publish_episodes()
        self.normal_buffer.publish_episodes()

    def get_policy_sample_loader(self, batch_size: int, encoder_window: int) -> DataLoader:
        if self._policy_sample_loader is None:
            self._policy_sample_loader = DataLoader(
                PolicyDataset(self.storage, batch_size, encoder_window),
                batch_size=None,
                num_workers=self.num_workers,
                # prefetch_factor=4 if self.num_workers > 0 else None,
                # persistent_workers=True if self.num_workers > 0 else False,
            )
        assert isinstance(self._policy_sample_loader.dataset, PolicyDataset)
        self._policy_sample_loader.dataset.batch_size = batch_size
        self._policy_sample_loader.dataset.encoder_window = encoder_window
        return self._policy_sample_loader

    def get_cemrl_sample_loader(self, batch_size: int, encoder_window: int, decoder_window: int) -> DataLoader:
        if self._cemrl_sample_loader is None:
            self._cemrl_sample_loader = DataLoader(
                CEMRLDataset(self.storage, batch_size, encoder_window, decoder_window),
                batch_size=None,
                num_workers=self.num_workers,
                # prefetch_factor=4 if self.num_workers > 0 else None,
                # persistent_workers=True if self.num_workers > 0 else False,
            )
        assert isinstance(self._cemrl_sample_loader.dataset, CEMRLDataset)
        self._cemrl_sample_loader.dataset.batch_size = batch_size
        self._cemrl_sample_loader.dataset.encoder_window = encoder_window
        self._cemrl_sample_loader.dataset.decoder_window = decoder_window
        return self._cemrl_sample_loader

    def get_episodic_sample_loader(self, batch_size: int, max_lenght: int | None = None) -> DataLoader:
        if self._episodic_sample_loader is None:
            self._episodic_sample_loader = DataLoader(
                EpisodeDataset(self.storage, batch_size, max_lenght),
                batch_size=None,
                num_workers=self.num_workers,
                # prefetch_factor=4 if self.num_workers > 0 else None,
                # persistent_workers=True if self.num_workers > 0 else False,
            )
        assert isinstance(self._episodic_sample_loader.dataset, EpisodeDataset)
        self._episodic_sample_loader.dataset.batch_size = batch_size
        self._episodic_sample_loader.dataset.max_returned_length = max_lenght
        return self._episodic_sample_loader

    def _to_device(self, data: DictReplayBufferSamples):
        return DictReplayBufferSamples(
            observations={k: v.to(self.device) for k, v in data.observations.items()},
            next_observations={k: v.to(self.device) for k, v in data.next_observations.items()},
            actions=data.actions.to(self.device),
            rewards=data.rewards.to(self.device),
            dones=data.dones.to(self.device),
        )
