import numpy as np
import gym.spaces as spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

from src.envs.wrappers.include_goal import IncludeGoalWrapper
from .stackable_wrapper import StackableWrapper
from .reward_and_action_to_obs_wrapper import RewardAndActionToObsWrapper


class CEMRLHistoryWrapper(VecFrameStack):
    def __init__(self, venv: VecEnv, n_stack: int, use_box=False):
        self.original_obs_space = venv.observation_space
        venv = RewardAndActionToObsWrapper(venv, use_box=use_box)
        venv = IncludeGoalWrapper(venv)
        venv = StackableWrapper(venv)
        super().__init__(venv, n_stack, "first")

    def step_wait(self):
        obs, reward, done, infos = super().step_wait()
        
        if True in ["terminal_observation" in info for info in infos]:
            # episode ended but reset was not called
            obs = self.reset()

        return obs, reward, done, infos

    def reset(self):
        obs = self.venv.reset()
        for _ in range(self.stacked_obs.n_stack):
            obs, *_ = self.step(np.array([self.action_space.sample() for _ in range(self.num_envs)]))
        return obs
