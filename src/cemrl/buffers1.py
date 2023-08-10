from enum import Enum
import os
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
import torch as th
import pickle
from torch.utils.data import Dataset
from glob import glob
from torch.utils.data import default_collate
from torch.utils.data import DataLoader

from typing import Dict, NamedTuple
import numpy as np
import shutil

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
    is_exploration: np.ndarray
    """An array of flags indicating whether the episode was collected during exploration."""


class EpisodeDataset(Dataset):
    def __init__(self, storage_path: str, device: th.device):
        self.storage_path = storage_path
        self.files = glob(os.path.join(self.storage_path, "**", "*.pkl"), recursive=True)
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as f:
            episode: StoredEpisode = pickle.load(f)
        return DictReplayBufferSamples(
            observations={key: th.as_tensor(episode.obs[key], device=self.device) for key in episode.obs.keys()},
            actions=th.as_tensor(episode.action, device=self.device),
            next_observations={key: th.as_tensor(episode.next_obs[key], device=self.device) for key in episode.next_obs.keys()},
            dones=th.as_tensor(episode.done, device=self.device).reshape(-1, 1),
            rewards=th.as_tensor(episode.reward, device=self.device).reshape(-1, 1),
        )


class EpisodeLinkingDataset(EpisodeDataset):
    def __init__(self, storage_path: str, device: th.device, returned_length=500):
        super().__init__(storage_path, device)
        self.returned_length = returned_length
        self.goals: dict[str, list[int]] = {}
        self.lengths: dict[int, int] = {}
        for idx, file in enumerate(self.files):
            goal, length, _ = file.split("/")[:-3]
            self.goals.setdefault(goal, []).append(idx)
            self.lengths[idx] = int(length)

    def __getitem__(self, idx):
        file = self.files[idx]
        left_to_get = self.returned_length
        episodes = [idx]
        goal = file.split("/")[-3]
        data: list[StoredEpisode] = []

        while 0 < left_to_get:
            left_to_get -= self.lengths[episodes[-1]]
            episodes.append(np.random.choice(self.goals[goal]))

        for episode in episodes:
            data.append(super().__getitem__(episode))

        return StoredEpisode(*default_collate(data))


class CEMRLDataset(Dataset):
    def __init__(self, storage_path: str, device: th.device) -> None:
        self.encoder_dataset = EpisodeDataset(storage_path, device)
        self.decoder_dataset = EpisodeLinkingDataset(storage_path, device)

    def __len__(self):
        return len(self.encoder_dataset)

    def __getitem__(self, idx):
        return self.encoder_dataset[idx], self.decoder_dataset[idx]


class SampleDataset(Dataset[DictReplayBufferSamples]):
    """A PyTorch dataset for loading individual samples."""

    def __init__(self, storage_path: str, device: th.device) -> None:
        self.storage_path = storage_path
        self.device = device
        self.files = glob(os.path.join(self.storage_path, "**", "*.pkl"), recursive=True)
        self.lengths_cum_sum = np.cumsum([int(file.split(os.sep)[-2]) for file in self.files])

    def __len__(self):
        return self.lengths_cum_sum[-1]

    def __getitem__(self, index) -> DictReplayBufferSamples:
        file_idx = self.lengths_cum_sum.searchsorted(index)
        file = self.files[file_idx]
        episode_idx = index - self.lengths_cum_sum[file_idx]
        with open(file, "rb") as f:
            episode: StoredEpisode = pickle.load(f)

        return DictReplayBufferSamples(
            observations={key: th.as_tensor(episode.obs[key][episode_idx], device=self.device) for key in episode.obs.keys()},
            actions=th.as_tensor(episode.action[episode_idx], device=self.device),
            next_observations={key: th.as_tensor(episode.next_obs[key][episode_idx], device=self.device) for key in episode.next_obs.keys()},
            dones=th.as_tensor(episode.done[episode_idx], device=self.device).reshape(-1, 1),
            rewards=th.as_tensor(episode.reward[episode_idx], device=self.device).reshape(-1, 1),
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
            n_envs * max_episode_length, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination
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
        for env_idx, env_start, env_length in zip(env_idxs, self.start_pos[env_idxs], self.episode_length[env_idxs]):
            episode_idxs = (env_start + np.arange(0, env_length)) % self.buffer_size
            episode_idxs = episode_idxs[-self.max_episode_length :]  # take only the last n steps from the episode

            obs = {k: v[episode_idxs, env_idx] for k, v in self.observations.items()}
            next_obs = {k: v[episode_idxs, env_idx] for k, v in self.next_observations.items()}
            actions = self.actions[episode_idxs, env_idx]
            rewards = self.rewards[episode_idxs, env_idx]
            dones = self.dones[episode_idxs, env_idx]
            goal_idxs = self.goal_idx[episode_idxs, env_idx]
            is_exploration = self.is_eploration[episode_idxs, env_idx]

            self.on_episode_collected(StoredEpisode(obs, next_obs, actions, rewards, dones, goal_idxs, is_exploration))

        self.start_pos[env_idxs] = self.pos
        self.episode_length[env_idxs] = 0


class BufferModes(Enum):
    Policy = 1
    Episode = 2
    CEMRL = 3


class EpisodicReplayBuffer(DictReplayBuffer):
    """A replay buffer that stores episodes in  in separate sub-buffers for exploration and normal data."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        storage_path: str,
        max_episode_length: int,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        mode: BufferModes = BufferModes.Policy,
        num_workers=16,
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
        self.is_exploring = False
        self.storage_path = storage_path
        shutil.rmtree(storage_path, ignore_errors=True)
        self.num_episodes = 0
        self.exploration_buffer = SubReplayBuffer(
            max_episode_length,
            observation_space,
            action_space,
            self.store_episode,
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
            self.store_episode,
            max_episode_length,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )

        self.num_workers = num_workers
        self._policy_sample_loader = None
        self._cemrl_sample_loader = None
        self._mode = mode
        self._iterator = None

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
        is_exploring = self.is_exploring or np.any([info.get("is_exploration", False) for info in infos])
        if is_exploring:
            self.exploration_buffer.add(obs, next_obs, action, reward, done, infos)
        else:
            self.normal_buffer.add(obs, next_obs, action, reward, done, infos)

    def store_episode(self, data: StoredEpisode):
        """
        Store the given episode data in a pickle file. This is used as callback from the sub-buffers.

        Args:
            data (StoredEpisode): The episode data to store.
        """
        self.num_episodes += 1
        path = os.path.join(self.storage_path, str(data.goal_idx[0]), str(len(data.done)))
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, f"{self.num_episodes}.pkl"), "wb") as f:
            pickle.dump(data, f)

    def sample(self, batch_size: int, env: VecNormalize | None = None) -> DictReplayBufferSamples:
        self._batch_size = batch_size

        if self.mode == BufferModes.Policy:
            dataloader = self.policy_sample_loader.__iter__()
        elif self.mode == BufferModes.CEMRL:
            dataloader = self.cemrl_sample_loader.__iter__()
        elif self.mode == BufferModes.Episode:
            dataloader = self.episodic_sample_loader.__iter__()
        else:
            raise NotImplementedError()

        if self._iterator is None:
            self._iterator = iter(dataloader)

        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(dataloader)
            return next(self._iterator)
        
    def decoder_encoder_sample(self, batch_size: int, env: VecNormalize|None = None, encoder_window:int, decoder_window:int) -> DictReplayBufferSamples:
        self._batch_size = batch_size

        if self.mode == BufferModes.Policy:
            dataloader = self.policy_sample_loader.__iter__()
        elif self.mode == BufferModes.CEMRL:
            dataloader = self.cemrl_sample_loader.__iter__()
        elif self.mode == BufferModes.Episode:
            dataloader = self.episodic_sample_loader.__iter__()
        else:
            raise NotImplementedError()

        if self._cemrl_iterator is None:
            self._iterator = iter(dataloader)

        try:
            return next(self._cemrl_iterator)
        except StopIteration:
            self._cemrl_iterator = iter(dataloader)
            return next(self._cemrl_iterator)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def set_mode(self, mode: BufferModes):
        if self.mode != mode:
            self._iterator = None
        self._mode = mode

    @property
    def policy_sample_loader(self):
        if self._policy_sample_loader is None or self._policy_sample_loader.batch_size != self._batch_size:
            self._policy_sample_loader = DataLoader(
                SampleDataset(self.storage_path, self.device),
                batch_size=self._batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                drop_last=True,
                persistent_workers=True,
            )
        return self._policy_sample_loader

    @property
    def cemrl_sample_loader(self):
        if self._cemrl_sample_loader is None or self._cemrl_sample_loader.batch_size != self._batch_size:
            self._cemrl_sample_loader = DataLoader(
                CEMRLDataset(self.storage_path, self.device),
                batch_size=self._batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                drop_last=True,
                persistent_workers=True,
            )
        return self._cemrl_sample_loader

    @property
    def episodic_sample_loader(self):
        if self._cemrl_sample_loader is None or self._cemrl_sample_loader.batch_size != self._batch_size:
            self._cemrl_sample_loader = DataLoader(
                EpisodeDataset(self.storage_path, self.device),
                batch_size=self._batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                drop_last=True,
                persistent_workers=True,
            )
        return self._cemrl_sample_loader
