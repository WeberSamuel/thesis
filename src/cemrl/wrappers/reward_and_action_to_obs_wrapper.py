from typing import Optional
import numpy as np
import gym.spaces as spaces
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvWrapper,
    VecEnvObs,
    VecEnvStepReturn,
)
from src.envs.meta_env import MetaVecEnv


class RewardAndActionToObsWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, use_box=True):
        self.use_box = use_box
        obs_space = {
            "observation": venv.observation_space,
            "reward": spaces.Box(-np.inf, np.inf, shape=(1,)),
            "action": venv.action_space,
        }

        self.dict_observation_space = spaces.Dict(obs_space)

        if use_box:
            observation_space = spaces.flatten_space(self.dict_observation_space)
        else:
            observation_space = self.dict_observation_space

        super().__init__(venv, observation_space, venv.action_space)

    def step_async(self, actions: np.ndarray) -> None:
        self.next_action = actions
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        obs = self.update_obs(obs, rewards, self.next_action)

        for info, reward, actions in zip(infos, rewards, self.next_action):
            if "terminal_observation" in info:
                info["terminal_observation"] = self.update_obs(info["terminal_observation"], reward, actions, True)
        return obs, rewards, dones, infos

    def update_obs(self, obs, rewards, actions, single_env=False):
        obs = {
            "observation": obs,
            "reward": rewards[..., None],
            "action": actions,
        }

        if self.use_box:
            if single_env:
                obs = spaces.flatten(self.dict_observation_space, obs)
            else:
                obs = np.stack(
                    [
                        spaces.flatten(
                            self.dict_observation_space,
                            {
                                "observation": obs["observation"][i],
                                "reward": obs["reward"][i],
                                "action": obs["action"][i],
                            },
                        )
                        for i in range(self.num_envs)
                    ]
                )
        return obs

    def reset(self) -> VecEnvObs:
        self.venv.reset()

        # action needed for observation
        action = getattr(self.venv.unwrapped, "neutral_action", None)
        if action is None:
            action = np.stack([self.action_space.sample() for _ in range(self.num_envs)])
        return self.step(action)[0]
