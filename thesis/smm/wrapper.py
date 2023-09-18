import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

class AddSmmMetaToObservationWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, meta_size: int):
        if not isinstance(venv.observation_space, spaces.Dict):
            observation_space = spaces.Dict({"observation": venv.observation_space})
        else:
            observation_space = venv.observation_space
        observation_space["smm_meta"] = spaces.Box(0, 1, shape=(meta_size,), dtype=np.float32)
        super().__init__(venv, observation_space)

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        if not isinstance(obs, dict):
            obs = {"observation": obs}
        return obs  # type: ignore

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, dones, infos = self.venv.step_wait()
        if not isinstance(obs, dict):
            obs = {"observation": obs}
        return obs, reward, dones, infos  # type: ignore
