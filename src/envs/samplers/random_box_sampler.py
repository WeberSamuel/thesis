from typing import Tuple
import numpy as np
from src.envs.samplers.base_sampler import BaseSampler
from stable_baselines3.common.vec_env import VecEnv
import gym.spaces as spaces


class RandomBoxSampler(BaseSampler):
    def __init__(self, num_tasks=1, num_goals=100) -> None:
        super().__init__(num_tasks=num_tasks, num_goals=num_goals)

    def _get_goal_space(self) -> spaces.Space:
        return spaces.flatten_space(self.env.observation_space)

    def sample(self, num_goals: int) -> Tuple[np.ndarray, np.ndarray]:
        assert isinstance(self.goal_space, spaces.Box)
        high = self.goal_space.high
        low = self.goal_space.low
        goals = np.random.random((num_goals, 2)) * (high - low) + low
        goals = goals.astype(np.float16)
        return goals, np.full((num_goals,), 0, np.int64)
