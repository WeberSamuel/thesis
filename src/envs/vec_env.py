from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from copy import deepcopy
from gymnasium import Env

from src.envs.meta_env import MetaMixin

class SequentialVecEnv(DummyVecEnv):
    def __init__(self, env: Env, n_envs: int):
        def copy_env():
            result = deepcopy(env)
            if isinstance(env.unwrapped, MetaMixin):
                assert isinstance(result.unwrapped, MetaMixin)
                result.unwrapped.goal_sampler = env.unwrapped.goal_sampler
            return result
        
        super().__init__([copy_env]*n_envs)

class SubprocessVecEnv(SubprocVecEnv):
    def __init__(self, env: Env, n_envs: int, start_method: str | None = None):
        def copy_env():
            result = deepcopy(env)
            if isinstance(env.unwrapped, MetaMixin):
                assert isinstance(result.unwrapped, MetaMixin)
                result.unwrapped.goal_sampler = env.unwrapped.goal_sampler
            return result
        
        super().__init__([copy_env]*n_envs, start_method)