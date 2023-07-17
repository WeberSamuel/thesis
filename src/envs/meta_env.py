from typing import Optional
import numpy as np
from gymnasium import Env
from src.envs.samplers.base_sampler import BaseSampler


class MetaMixin:
    def __init__(self, goal_sampler: BaseSampler, *args, **kwargs) -> None:
        assert isinstance(self, Env), "MetaMixin must be inherited by an Env"

        self.goal_sampler = goal_sampler
        self.neutral_action: Optional[np.ndarray] = None
        super().__init__(*args, **kwargs)
        if isinstance(goal_sampler, BaseSampler):
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
        return super().reset(*args, **kwargs)  # type: ignore