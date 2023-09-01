from typing import Tuple
import numpy as np
from src.envs.samplers.base_sampler import BaseSampler
import gymnasium.spaces as spaces

class RandomBoxSampler(BaseSampler):
    def __init__(self, available_tasks=[0], num_goals=100) -> None:
        super().__init__(available_tasks=available_tasks, num_goals=num_goals)

    def _get_goal_space(self) -> spaces.Space:
        return spaces.flatten_space(self.env.observation_space)

    def sample(self, num_goals: int, available_tasks: list[int]) -> Tuple[np.ndarray, np.ndarray]:
        assert isinstance(self.goal_space, spaces.Box)
        high = self.goal_space.high
        low = self.goal_space.low
        goals = np.random.random((num_goals * len(available_tasks), 2)) * (high - low) + low
        goals = goals.astype(np.float16)
        return goals, np.array(available_tasks).repeat(num_goals)
