from abc import ABC, abstractmethod
from typing import Tuple
from gymnasium import Env, spaces
import numpy as np


class BaseSampler(ABC):
    def __init__(self, num_tasks: int, num_goals: int) -> None:
        self.num_tasks = num_tasks
        self.num_goals = num_goals

    def _init_sampler(self, env: Env):
        self.env = env
        self.goal_space = self._get_goal_space()
        self.goals, self.tasks = self.sample(self.num_goals, self.num_tasks)

    @abstractmethod
    def _get_goal_space(self) -> spaces.Space:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, num_goals: int, num_tasks) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
