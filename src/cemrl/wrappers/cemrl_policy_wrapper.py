from typing import Optional
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from gymnasium import spaces, ObservationWrapper, Env
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn


class CEMRLPolicyWrapper(ObservationWrapper):
    def __init__(self, env: Env|VecEnv, latent_dim: int):
        super().__init__(env)
        original_obs_space = getattr(env, "original_obs_space", None)
        if original_obs_space is None:
            raise ValueError("CEMRL policy must have access to original observation space")
        
        self.observation_space = spaces.Dict(
            {"observation": original_obs_space, "task_indicator": spaces.Box(-np.inf, np.inf, (latent_dim,))}
        )

class CEMRLPolicyVecWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, latent_dim: int):
        original_obs_space:spaces.Space = venv.get_attr("original_obs_space")[0]
        observation_space = spaces.Dict(
            {"observation": original_obs_space, "task_indicator": spaces.Box(-np.inf, np.inf, (latent_dim,))}
        )
        super().__init__(venv, observation_space=observation_space)

    def reset(self) -> VecEnvObs:
        return super().reset()
    
    def step_wait(self) -> VecEnvStepReturn:
        return super().step_wait()