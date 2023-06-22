from typing import Any, Dict, Optional, Type
from gymnasium import spaces
import numpy as np
import torch as th
from gymnasium import Env
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from src.cemrl.types import CEMRLObsTensorDict
from src.cemrl.wrappers.cemrl_policy_wrapper import CEMRLPolicyVecWrapper, CEMRLPolicyWrapper
from src.plan2explore.networks import Ensemble, WorldModel
from src.cemrl.networks import Encoder


class CEMRLPolicy(BasePolicy):
    def __init__(
        self,
        *args,
        env: Env|VecEnv,
        sub_policy_algorithm_class: Type[OffPolicyAlgorithm],
        sub_policy_algorithm_kwargs: Dict[str, Any],
        num_classes: int,
        latent_dim: int,
        decoder_ensemble_size: int = 5,
        net_complexity: float = 40.0,
        **kwargs,
    ):
        super().__init__(env.observation_space, env.action_space, *args, **kwargs)
        o_obs_space = env.get_attr("original_obs_space")[0] if isinstance(env, VecEnv) else getattr(env, "original_obs_space")
        self.encoder = Encoder(
            num_classes,
            latent_dim,
            spaces.flatdim(o_obs_space),
            spaces.flatdim(self.action_space),
            net_complexity,
        )
        self.decoder = Ensemble(
            th.nn.ModuleList(
                [
                    WorldModel(
                        spaces.flatdim(o_obs_space),
                        spaces.flatdim(self.action_space),
                        latent_dim,
                        net_complexity,
                    )
                    for i in range(decoder_ensemble_size)
                ]
            )
        )
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)

        wrapper = CEMRLPolicyVecWrapper(env, latent_dim) if isinstance(env, VecEnv) else CEMRLPolicyWrapper(env, latent_dim)
        self.sub_policy_algorithm = sub_policy_algorithm_class(
            "MultiInputPolicy", wrapper, buffer_size=0, **sub_policy_algorithm_kwargs
        )

        assert isinstance(self.action_space, spaces.Box)

    @th.no_grad()
    def _predict(
        self,
        observation: CEMRLObsTensorDict,
        deterministic: bool = False,
    ) -> th.Tensor:
        with th.no_grad():
            y, z = self.encoder(observation)
        policy_obs = {
            "observation": observation["observation"][:, -1].to(self.sub_policy_algorithm.device),
            "task_indicator": z.to(self.sub_policy_algorithm.device),
        }
        action = self.sub_policy_algorithm.policy._predict(policy_obs, deterministic)  # type: ignore
        return action