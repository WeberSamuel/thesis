from abc import ABC, abstractmethod
from typing import Tuple
from gymnasium import Env, spaces
import numpy as np


class BaseSampler(ABC):
    def __init__(self, available_tasks: list[int], num_goals: int) -> None:
        self.available_tasks = available_tasks
        self.num_goals = num_goals
        self.initialized = False

    def _init_sampler(self, env: Env):
        self.env = env
        self.goal_space = self._get_goal_space()
        self.goals, self.tasks = self.sample(self.num_goals, self.available_tasks)
        self.initialized = True

    @abstractmethod
    def _get_goal_space(self) -> spaces.Space:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, num_goals: int, available_tasks: list[int]) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
