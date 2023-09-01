from enum import Enum
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv as GymHalfCheetahEnv
import numpy as np
from gymnasium.spaces import Box
from src.envs.samplers.base_sampler import BaseSampler
from src.envs.meta_env import MetaMixin
import cv2


class HalfCheetahMetaClasses(Enum):
    VELOCITY = 0
    DIRECTION = 1
    GOAL = 2


class HalfCheetahEnv(MetaMixin, GymHalfCheetahEnv):
    def __init__(self, goal_sampler: BaseSampler, *args, width: int = 256, height: int = 256, success_threshold:float = 2, **kwargs) -> None:
        super().__init__(goal_sampler, width=width, height=height, exclude_current_positions_from_observation=False, *args, **kwargs)
        self.success_threshold = success_threshold
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.observation_space.shape, dtype=np.float32) # type: ignore

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.add_meta_info(info)
        if self.task == HalfCheetahMetaClasses.VELOCITY.value:
            reward = np.abs(info["x_velocity"] - self.goal) + info["reward_ctrl"]
        elif self.task == HalfCheetahMetaClasses.DIRECTION.value:
            reward = (np.sign(info["x_velocity"]) == np.sign(self.goal)) + info["reward_ctrl"]
        elif self.task == HalfCheetahMetaClasses.GOAL.value:
            reward = -np.abs(info["x_position"] - self.goal) + info["reward_ctrl"]
            info["is_success"] = np.linalg.norm(info["x_position"] - self.goal) < 2
        self.last_info = info
        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self.last_info = None
        return super().reset(*args, **kwargs)

    def render(self):
        if self.render_mode == "rgb_array":
            img = super().render().copy()
            img = cv2.putText(
                img,
                f"Task {HalfCheetahMetaClasses(self.task).name}: {self.goal:.2f}",
                (10, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
            if self.last_info is not None:
                if self.task == HalfCheetahMetaClasses.DIRECTION.value:
                    img = cv2.putText(
                        img,
                        f"Velocity: {self.last_info['x_velocity']:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                elif self.task == HalfCheetahMetaClasses.GOAL.value:
                    img = cv2.putText(
                        img,
                        f"Position: {self.last_info['x_position']:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                elif self.task == HalfCheetahMetaClasses.VELOCITY.value:
                    img = cv2.putText(
                        img,
                        f"Velocity: {self.last_info['x_velocity']:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
            return img
