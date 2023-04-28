"""Code for the ToyGoal environment used in the Thesis."""
import cv2
import gym.spaces as spaces
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvObs,
    VecEnvStepReturn,
)
from typing import Sequence
from stable_baselines3.common.vec_env import VecEnvWrapper

from src.envs.samplers.base_sampler import BaseSampler
from src.envs.meta_env import MetaVecEnv


class ToyGoalEnv(MetaVecEnv):
    """Simple 2D toy goal environment, where the agent has to reach a goal position.

    Actions space is in [-step_size, step_size]
    Observation space is in [-1, 1]
    The environment is automatically timelimited to 200 timesteps.
    The reward the agent receives is the L2 distance to the goal position.
    """

    def __init__(self, num_envs: int, goal_sampler: BaseSampler, step_size=0.25, distance_norm=2, random_position=False):
        """Initialize the environment.

        Args:
            num_envs (int): Number of environments to simulate
            step_size (float, optional): Step size of the agent. Defaults to 0.1.
            distance_norm (int, optional): Norm used to compute the reward. Defaults to 2.
        """
        super().__init__(
            num_envs,
            observation_space=spaces.Box(-20, 20, (2,)),
            action_space=spaces.Box(-step_size, step_size, (2,)),
            goal_sampler=goal_sampler,
        )
        self.random_position = random_position
        self.step_size = step_size
        self.distance_norm = distance_norm
        self.neutral_action = np.zeros((num_envs, 2))

    def reset(self) -> VecEnvObs:
        """Reset the environment.

        Returns:
            VecEnvObs: The new obersavation after resetting the environment
        """
        self.num_episode_steps = 0
        if self.random_position:
            self.state = (np.random.random((self.num_envs, 2)).astype(np.float32) - 0.5) * 2 * self.observation_space.low
        else:
            self.state = np.zeros((self.num_envs, 2))
        self.reset_current_goals()
        return self.get_obs()

    def step_async(self, actions: np.ndarray) -> None:
        """Register the next action to be performed in the simulation.

        Args:
            actions (np.ndarray): Array of [n_envs, 2] containing the actions
        """
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        """Perform the stored actions in the environment.

        Info will include the 'terminal_observation' and 'TimeLimit.truncated' key, when the timelimed is reached.
        Also dones will be set to True.

        Returns:
            VecEnvStepReturn: Tuple of (obs, rewards, dones, infos)
        """
        self.state = self.state + self.actions
        self.state = np.clip(
            self.state,
            self.observation_space.low[:2],
            self.observation_space.high[:2],
        )

        rewards = self.get_reward()
        self.num_episode_steps += 1
        infos = [{"goal": goal, "goal_idx": goal_idx, "task": task} for goal, goal_idx, task in zip(self.goals, self.goals_idx, self.tasks)]
        if self.num_episode_steps > 200:
            dones = np.ones((self.num_envs,), dtype=bool)
            obs = self.get_obs()
            for observation, reward, info in zip(obs, rewards, infos):
                info["TimeLimit.truncated"] = True
                info["terminal_observation"] = observation
                
                info["is_success"] = reward > -2.0
            obs = self.reset()
        else:
            dones = np.zeros((self.num_envs,), dtype=bool)
            obs = self.get_obs()
        return obs, rewards, dones, infos

    def get_obs(self):
        """Get a copy of the current state.

        Returns:
            np.ndarray: Array of [n_envs, 2|4] containing the state and maybe the goal in the environment, depending if 'include_goal' is set.
        """
        return np.copy(self.state)

    def get_reward(self):
        """Get the reward for the current state.

        The reward is the negative 'distance_norm' distance from the agent to the goal.

        Returns:
            np.ndarray: Reward as [n_envs, 1]
        """
        return -np.linalg.norm(self.state - self.goals, ord=self.distance_norm, axis=-1)

    def get_images(self, image_size=256) -> Sequence[np.ndarray]:
        """Get a list of images representing the environment.

        Args:
            image_size (int, optional): Size of the individual images. Defaults to 256.

        Returns:
            Sequence[np.ndarray]: Sequence of images with lenght of num_envs
        """

        def scale_cord_to_img(x, low: float, high: float):
            return (x - low) / (high - low) * image_size

        imgs = []

        for (x, y), (x_goal, y_goal) in zip(self.state, self.goals):
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
            img = cv2.circle(img, (int(x_goal), int(y_goal)), 5, (255, 0, 0))
            imgs.append(img)

        return imgs


class ToyGoal1DEnv(VecEnvWrapper):
    """Wrapper to turn the ToyGoalEnv into a 1d ToyGoal Problem."""

    def __init__(
        self,
        venv: ToyGoalEnv,
    ):
        """Initialize the wrapper.

        Args:
            venv (ToyGoalEnv): Toygoal Environment to wrap.
        """
        super().__init__(venv)

    def step_async(self, actions: np.ndarray) -> None:
        actions[..., 1] = 0
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        assert isinstance(self.unwrapped, ToyGoalEnv)
        self.unwrapped.goals[..., 1] = 0.0
        self.unwrapped.state[..., 1] = 0.0
        obs[..., 1] = 0.0
        return obs
