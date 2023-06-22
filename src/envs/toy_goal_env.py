"""Code for the ToyGoal environment used in the Thesis."""
from enum import Enum
import cv2
from gymnasium import Env, spaces
import numpy as np
from typing import Any, Optional, SupportsFloat
from src.envs.samplers.base_sampler import BaseSampler
from src.envs.meta_env import MetaMixin

class ToyGoalMetaClasses(Enum):
    REACH = 0
    FLEE = 1
    DONT_MOVE = 2

class ToyGoalEnv(MetaMixin, Env[np.ndarray, np.ndarray]):
    """Simple 2D toy goal environment, where the agent has to reach a goal position.

    Actions space is in [-step_size, step_size]
    Observation space is in [-1, 1]
    The environment is automatically timelimited to 200 timesteps.
    The reward the agent receives is the L2 distance to the goal position.
    """

    def __init__(
        self,
        goal_sampler: BaseSampler,
        step_size=0.25,
        distance_norm=2,
        success_distance=2,
        random_position=False,
        render_mode: Optional[str] = "rgb_array",
    ):
        """Initialize the environment.

        Args:
            num_envs (int): Number of environments to simulate
            step_size (float, optional): Step size of the agent. Defaults to 0.1.
            distance_norm (int, optional): Norm used to compute the reward. Defaults to 2.
        """
        self.observation_space: spaces.Box = spaces.Box(-20, 20, (2,))
        self.action_space: spaces.Box = spaces.Box(-step_size, step_size, (2,))
        super().__init__(goal_sampler=goal_sampler)
        self.render_mode = render_mode

        self.random_position = random_position
        self.step_size = step_size
        self.distance_norm = distance_norm
        self.neutral_action = np.zeros(2)
        self.state = np.zeros(2)
        self.success_distance = success_distance

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment.

        Returns:
            VecEnvObs: The new obersavation after resetting the environment
        """
        super().reset(seed=seed, options=options)

        if self.random_position:
            self.state = (np.random.random(2).astype(np.float32) - 0.5) * 2 * self.observation_space.low
        else:
            self.state = np.zeros(2)
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        self.state = self.state + action
        self.state = np.clip(
            self.state,
            self.observation_space.low[:2],
            self.observation_space.high[:2],
        )
        obs = self._get_obs()
        reward = self.get_reward(action)
        info = self._get_info()

        if ToyGoalMetaClasses(self.task) == ToyGoalMetaClasses.REACH:
            info["is_success"] = reward > -self.success_distance
        elif ToyGoalMetaClasses(self.task) == ToyGoalMetaClasses.FLEE:
            info["is_success"] = reward > self.success_distance
        elif ToyGoalMetaClasses(self.task) == ToyGoalMetaClasses.DONT_MOVE:
            info["is_success"] = reward > -0.1

        if self.render_mode == "human":
            self.render()

        return obs, reward, False, False, info

    def _get_info(self):
        return {"goal": self.goal, "goal_idx": self.goal_idx, "task": self.task}

    def _get_obs(self):
        return np.copy(self.state)

    def get_reward(self, action: np.ndarray):
        """Get the reward for the current state.

        The reward is the negative 'distance_norm' distance from the agent to the goal.

        Returns:
            np.ndarray: Reward as [n_envs, 1]
        """
        distance = np.linalg.norm(self.state - self.goal, ord=self.distance_norm)
        if ToyGoalMetaClasses(self.task) == ToyGoalMetaClasses.REACH:
            return -distance
        elif ToyGoalMetaClasses(self.task) == ToyGoalMetaClasses.FLEE:
            return distance
        elif ToyGoalMetaClasses(self.task) == ToyGoalMetaClasses.DONT_MOVE:
            return -np.linalg.norm(action, ord=self.distance_norm)
        else:
            raise NotImplementedError()

    def render(self):
        if self.render_mode == "human":
            cv2.imshow("ToyGoal", self.get_image())
        elif self.render_mode == "rgb_array":
            return self.get_image()
        else:
            raise NotImplementedError()

    def close(self) -> None:
        cv2.destroyAllWindows()

    def get_image(self, image_size=256) -> np.ndarray:
        """Get a list of images representing the environment.

        Args:
            image_size (int, optional): Size of the individual images. Defaults to 256.

        Returns:
            Sequence[np.ndarray]: Sequence of images with lenght of num_envs
        """

        def scale_cord_to_img(x, low: float, high: float):
            return (x - low) / (high - low) * image_size

        x, y = self.state
        x_goal, y_goal = self.goal

        assert (
            isinstance(self.observation_space, spaces.Box)
            and isinstance(self.observation_space.low, np.ndarray)
            and isinstance(self.observation_space.high, np.ndarray)
        )
        x = scale_cord_to_img(x, self.observation_space.low[0], self.observation_space.high[0])
        y = scale_cord_to_img(y, self.observation_space.low[1], self.observation_space.high[1])
        x_goal = scale_cord_to_img(
            x_goal,
            self.observation_space.low[0],
            self.observation_space.high[0],
        )
        y_goal = scale_cord_to_img(
            y_goal,
            self.observation_space.low[1],
            self.observation_space.high[1],
        )
        img = np.ones([image_size, image_size, 3], dtype=np.uint8)
        img = cv2.rectangle(img, (0, 0), (image_size, image_size), (255, 255, 255), 2)
        img = cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
        if ToyGoalMetaClasses(self.task) == ToyGoalMetaClasses.REACH:
            img = cv2.circle(img, (int(x_goal), int(y_goal)), 5, (255, 0, 0))
        elif ToyGoalMetaClasses(self.task) == ToyGoalMetaClasses.FLEE:
            img = cv2.circle(img, (int(x_goal), int(y_goal)), 5, (0, 0, 255))
        elif ToyGoalMetaClasses(self.task) == ToyGoalMetaClasses.DONT_MOVE:
            img = cv2.circle(img, (int(x_goal), int(y_goal)), 5, (255, 0, 255))

        return img

class ToyGoal1DEnv(ToyGoalEnv):
    def step(self, action: np.ndarray):
        action[1] = 0.0
        return super().step(action)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.state[1] = 0.0
        obs[..., 1] = 0.0
        return obs, info

    def change_goal(self):
        super().change_goal()
        self.goal[1] = 0.0
