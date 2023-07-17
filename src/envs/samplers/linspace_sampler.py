import numpy as np
from gymnasium import spaces
from src.envs.samplers.base_sampler import BaseSampler


class LinspaceSampler(BaseSampler):
    def __init__(self, available_tasks: list[int] = [2], num_goals: int = 100, max_goal_radius=25.0) -> None:
        super().__init__(available_tasks, num_goals)
        self.max_goal_radius = max_goal_radius

    def _get_goal_space(self):
        return spaces.Box(low=0, high=1, shape=(1,))

    def sample(self, num_goals:int, available_tasks:list[int]):
        num_task = len(available_tasks)
        goals = np.linspace(-self.max_goal_radius, self.max_goal_radius, num_goals)
        goals = np.repeat(goals, num_task)
        tasks = np.array(available_tasks * num_goals)
        goals[tasks == 1] = np.sign(goals[tasks == 1])
        return goals, tasks