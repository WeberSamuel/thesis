import numpy as np
from gymnasium import spaces
from src.envs.samplers.base_sampler import BaseSampler


class RandomSampler(BaseSampler):
    def __init__(self, available_tasks: list[int] = [0], num_goals: int = 100, max_goal_radius=25.0, one_sided:bool = False) -> None:
        super().__init__(available_tasks, num_goals)
        self.max_goal_radius = max_goal_radius
        self.one_sided = one_sided

    def _get_goal_space(self):
        if self.one_sided:
            return spaces.Box(low=0, high=self.max_goal_radius, shape=(1,), dtype=np.float32)
        return spaces.Box(low=-self.max_goal_radius, high=self.max_goal_radius, shape=(1,), dtype=np.float32)

    def sample(self, num_goals:int, available_tasks:list[int]):
        if self.one_sided:
            goals = np.random.random(num_goals) * self.max_goal_radius
        else:
            goals = np.random.random(num_goals) * 2 * self.max_goal_radius - self.max_goal_radius
        tasks = np.random.choice(available_tasks, num_goals)
        goals[tasks == 1] = np.sign(goals[tasks == 1])
        return goals.astype(np.float32), tasks