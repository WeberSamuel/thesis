from enum import Enum
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv as GymHalfCheetahEnv
import numpy as np
from gymnasium.spaces import Box
from src.envs.samplers.base_sampler import BaseSampler
from src.envs.meta_env import MetaMixin

class HalfCheetahMetaClasses(Enum):
    VELOCITY = 0
    DIRECTION = 1
    GOAL = 2

class HalfCheetahEnv(MetaMixin, GymHalfCheetahEnv):
    def __init__(self, goal_sampler: BaseSampler, *args, **kwargs) -> None:
        super().__init__(goal_sampler, *args, **kwargs)
        if self._exclude_current_positions_from_observation:
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
            )
        else:
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
            )


    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.task == HalfCheetahMetaClasses.VELOCITY:
            reward = np.abs(info["x_velocity"] - self.goal) + info["reward_ctrl"]
        elif self.task == HalfCheetahMetaClasses.DIRECTION:
            reward = (np.sign(info["x_velocity"]) == np.sign(self.goal)) + info["reward_ctrl"]
        elif self.task == HalfCheetahMetaClasses.GOAL:
            reward = -np.abs(info["x_position"] - self.goal) + info["reward_ctrl"]
        self.add_meta_info(info)
        return obs, reward, terminated, truncated, info
