from enum import Enum
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.ant_v4 import AntEnv as GymAntEnv
import numpy as np
import cv2
from src.envs.samplers.base_sampler import BaseSampler
from src.envs.meta_env import MetaMixin


class AntMetaClasses(Enum):
    POSITION = 0
    VELOCITY = 1


class AntEnv(MetaMixin, GymAntEnv):
    def __init__(
        self, goal_sampler: BaseSampler, *args, width: int = 256, height: int = 256, success_threshold: float = 2, **kwargs
    ) -> None:
        super().__init__(
            goal_sampler, width=width, height=height, exclude_current_positions_from_observation=False, *args, **kwargs
        )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.observation_space.shape, dtype=np.float32)  # type: ignore
        self.success_threshold = success_threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.add_meta_info(info)
        if self.task == AntMetaClasses.POSITION.value:
            xy_pos = np.array([info["x_position"], info["y_position"]])
            distance = np.linalg.norm(xy_pos - self.goal)
            reward = -distance + info["reward_ctrl"] + info["reward_survive"]
            info["is_success"] = distance < self.success_threshold
        elif self.task == AntMetaClasses.VELOCITY.value:
            velocity = np.sqrt(info["x_velocity"] ** 2 + info["y_velocity"] ** 2)
            reward = velocity + info["reward_ctrl"] + info["reward_survive"]
            info["is_success"] = np.linalg.norm(velocity - self.goal) > self.success_threshold
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
                f"Task {AntMetaClasses(self.task).name}: {self.goal[0]:.2f}|{self.goal[1]:.2f}",
                (10, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
            if self.last_info is not None:
                if self.task == AntMetaClasses.POSITION.value:
                    img = cv2.putText(
                        img,
                        f"Position: {self.last_info['x_position']:.2f}|{self.last_info['y_position']:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                elif self.task == AntMetaClasses.VELOCITY.value:
                    img = cv2.putText(
                        img,
                        f"Velocity: {np.sqrt(self.last_info['x_velocity'] ** 2 + self.last_info['y_velocity'] ** 2):.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
            return img
