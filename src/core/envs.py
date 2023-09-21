from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from gymnasium import Env, spaces


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
    def sample(self, num_goals: int, available_tasks: list[int]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()


class MetaMixin:
    def __init__(self, goal_sampler: BaseSampler, *args, **kwargs) -> None:
        assert isinstance(self, Env), "MetaMixin must be inherited by an Env"

        self.goal_sampler = goal_sampler
        self.neutral_action: Optional[np.ndarray] = None
        super().__init__(*args, **kwargs)
        if isinstance(goal_sampler, BaseSampler) and not self.goal_sampler.initialized:
            self.goal_sampler._init_sampler(self)

    def change_goal(self):
        self.goal_idx = np.random.randint(0, len(self.goal_sampler.goals))
        self.goal = self.goal_sampler.goals[self.goal_idx]
        self.task = self.goal_sampler.tasks[self.goal_idx]

    def add_meta_info(self, info: dict) -> dict:
        info["goal_idx"] = self.goal_idx
        info["goal"] = self.goal
        info["task"] = self.task
        return info

    def reset(self, *args, **kwargs):
        self.change_goal()
        return super().reset(*args, **kwargs) # type: ignore