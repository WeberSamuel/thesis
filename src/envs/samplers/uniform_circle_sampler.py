from typing import Tuple
import numpy as np
import gymnasium.spaces as spaces
from src.envs.samplers.base_sampler import BaseSampler


class UniformCircleSampler(BaseSampler):
    def __init__(self, radius: float = 0.9, num_tasks=1, num_goals=16) -> None:
        super().__init__(num_tasks=num_tasks, num_goals=num_goals)
        self.radius = radius

    def _get_goal_space(self) -> spaces.Space:
        return spaces.flatten_space(self.env.observation_space)

    def sample(self, num_goals: int, num_tasks: int) -> Tuple[np.ndarray, np.ndarray]:
        assert isinstance(self.goal_space, spaces.Box)
        assert self.goal_space.shape[0] == 2
        high = self.goal_space.high
        low = self.goal_space.low
        center = (high + low) / 2
        center = np.nan_to_num(center, nan=0, posinf=0, neginf=0)

        if self.radius <= 1.0:
            radius = (np.min(high) - np.min(low)) / 2 * self.radius
        else:
            radius = self.radius

        theta = np.linspace(0.0, 1.0, num_goals) * 2 * np.pi
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        goals = np.stack([x, y], axis=-1)
        return np.tile(goals, (num_tasks, 1)), np.arange(num_tasks).repeat(num_goals)
