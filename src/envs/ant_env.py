from enum import Enum
from gymnasium.envs.mujoco.ant_v4 import AntEnv as GymAntEnv
import numpy as np
from src.envs.meta_env import MetaMixin


class AntMetaClasses(Enum):
    POSITION = 0
    VELOCITY = 1


class AntEnv(MetaMixin, GymAntEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.task == 0:
            xy_pos = np.array([info["x_position"], info["y_position"]])
            distance = np.linalg.norm(xy_pos - self.goal)
            reward = -distance + info["reward_ctrl"] + info["reward_survive"]
        elif self.task == 1:
            reward = np.sqrt(info["x_velocity"] ** 2 + info["y_velocity"] ** 2) + info["reward_ctrl"] + info["reward_survive"]
        return obs, reward, terminated, truncated, info
