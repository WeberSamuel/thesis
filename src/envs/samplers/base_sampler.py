from abc import ABC, abstractmethod
from typing import Optional, Tuple
import gym.spaces as spaces
import numpy as np
from stable_baselines3.common.vec_env import VecEnv


class BaseSampler(ABC):
    def __init__(self, num_tasks: int, num_goals: int) -> None:
        self.num_tasks = num_tasks
        self.num_goals = num_goals

    def _init_sampler(self, env: VecEnv):
        self.env = env
        self.goal_space = self._get_goal_space()
        self.goals, self.tasks = self.sample(self.num_goals)

    @abstractmethod
    def _get_goal_space(self) -> spaces.Space:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, num_goals: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
