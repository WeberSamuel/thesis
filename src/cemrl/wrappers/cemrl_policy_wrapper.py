import torch as th
import numpy as np
import gym.spaces as spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper, unwrap_vec_wrapper
from src.cemrl.policies import CEMRLPolicy
from .cemrl_history_wrapper import CEMRLHistoryWrapper


class CEMRLPolicyWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, latent_dim: int):
        cemrl_history = unwrap_vec_wrapper(venv, CEMRLHistoryWrapper)
        assert cemrl_history is not None, "CEMRLPolicyWrapper requires an environment that uses CEMRLHistoryWrapper"
        assert isinstance(cemrl_history, CEMRLHistoryWrapper)

        original_obs_space = cemrl_history.original_obs_space
        super().__init__(
            venv,
            spaces.Dict(
                {
                    "observation": original_obs_space,
                    "task_indicator": spaces.Box(-np.inf, np.inf, (latent_dim,)),
                }
            ),
        )

    def reset(self) -> VecEnvObs:
        raise NotImplementedError()

    def step_wait(self) -> VecEnvStepReturn:
        raise NotImplementedError()