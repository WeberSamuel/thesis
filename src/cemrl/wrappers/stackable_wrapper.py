from typing import Dict, List
import numpy as np
import gym.spaces as spaces
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvWrapper,
    VecEnvObs,
    VecEnvStepReturn,
)


class StackableWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv):
        def add_dimension_to_space(space: spaces.Space):
            if isinstance(space, spaces.Dict):
                return spaces.Dict({key: add_dimension_to_space(space[key]) for key in space})
            assert isinstance(space, spaces.Box), "Only Dict or Box spaces are supported!"
            return spaces.Box(space.low[None], space.high[None])

        extended_observation_space = add_dimension_to_space(venv.observation_space)
        super().__init__(venv, observation_space=extended_observation_space)

    def add_dimension_to_obs(self, obs: VecEnvObs, single_obs=False):
        if isinstance(obs, dict):
            return {key: self.add_dimension_to_obs(value, single_obs) for key, value in obs.items()}
        assert isinstance(obs, (np.ndarray, np.float32)), "Only Dict or Box observations are supported!"
        if isinstance(obs, float):
            return np.array([obs])
        if single_obs:
            return obs[None]
        return obs[:, None]

    def add_dimension_to_terminal_obs(self, infos: List[Dict]):
        for info in infos:
            terminal_obs = info.get("terminal_observation")
            if terminal_obs is not None:
                info["terminal_observation"] = self.add_dimension_to_obs(terminal_obs, single_obs=True)
        return infos

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return self.add_dimension_to_obs(obs)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        return self.add_dimension_to_obs(obs), rewards, dones, self.add_dimension_to_terminal_obs(infos)
