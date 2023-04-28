from typing import Dict
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv
import gym.spaces as spaces

from src.envs.meta_env import MetaVecEnv


class IncludeGoalWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv):
        assert isinstance(venv.unwrapped, MetaVecEnv)
        obs_space = (
            venv.observation_space.spaces
            if isinstance(venv.observation_space, spaces.Dict)
            else {"observation": venv.observation_space}
        )
        obs_space = spaces.Dict(
            {
                **obs_space,
                "goal": venv.unwrapped.goal_sampler.goal_space,
                "goal_idx": spaces.Box(0, venv.unwrapped.goal_sampler.num_goals, (1,)),
                "task": spaces.Box(0, venv.unwrapped.goal_sampler.num_tasks, (1,)),
            }
        )
        super().__init__(venv, obs_space)

    def step_wait(self):
        assert isinstance(self.venv.unwrapped, MetaVecEnv)
        obs, rewards, dones, infos = self.venv.step_wait()

        for info in infos:
            if "terminal_observation" in info:
                terminal_obs = info["terminal_observation"]
                terminal_obs = terminal_obs if isinstance(obs, dict) else {"observation": terminal_obs}
                info["terminal_observation"] = {
                    **terminal_obs,
                    "goal_idx": np.array([info["goal_idx"]]).astype(np.float32),
                    "goal": info["goal"],
                    "task": np.array([info["task"]]).astype(np.float32),
                }

        obs = obs if isinstance(obs, dict) else {"observation": obs}
        obs = {
            **obs,
            "goal_idx": self.venv.unwrapped.goals_idx[:, None].astype(np.float32),
            "goal": self.venv.unwrapped.goals,
            "task": self.venv.unwrapped.tasks[:, None].astype(np.float32),
        }

        return obs, rewards, dones, infos

    def reset(self):
        assert isinstance(self.venv.unwrapped, MetaVecEnv)
        obs = self.venv.reset()
        obs = obs if isinstance(obs, dict) else {"observation": obs}
        obs = {**obs, "goal": self.venv.unwrapped.goals, "task": self.venv.unwrapped.tasks.astype(np.float32)}
        return obs
