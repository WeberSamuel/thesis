from enum import Enum
from typing import Optional
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv as GymHalfCheetahEnv
import numpy as np
from src.envs.samplers.base_sampler import BaseSampler
from src.envs.meta_env import MetaMixin
from src.envs.samplers import RandomBoxSampler
from gymnasium import spaces


class HalfCheetahMetaClasses(Enum):
    VELOCITY = 0
    DIRECTION = 1


class RandomSampler(BaseSampler):
    def _get_goal_space(self):
        return spaces.Box(low=0, high=1, shape=(1,))

    def sample(self, num_goals):
        goals = np.random.rand(num_goals)
        tasks = np.random.randint(0, 2, num_goals)
        goals[tasks == 1] = np.sign(goals[tasks == 1] - 0.5)
        return goals, tasks


class HalfCheetahEnv(MetaMixin, GymHalfCheetahEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.task == HalfCheetahMetaClasses.VELOCITY:
            reward = np.abs(info["x_velocity"]) + info["reward_ctrl"]
        elif self.task == HalfCheetahMetaClasses.DIRECTION:
            reward = (np.sign(info["x_velocity"]) == np.sign(info["x_velocity"])) + info["reward_ctrl"]
        return obs, reward, terminated, truncated, info
