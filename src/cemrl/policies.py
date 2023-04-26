from typing import Any, Dict, Optional, Type
from gym import spaces
import numpy as np
import torch as th
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from src.cemrl.types import CEMRLObsTensorDict
from src.plan2explore.networks import Ensemble, WorldModel
from src.cemrl.networks import Encoder


class CEMRLPolicy(BasePolicy):
    def __init__(
        self,
        env: VecEnvWrapper | VecEnv | GymEnv,
        num_classes: int,
        latent_dim: int,
        optimizer_type: Type[th.optim.Optimizer] = th.optim.AdamW,
        decoder_ensemble_size: int = 5,
        net_complexity: float = 40.0,
        optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3},
        scheduler: Optional[th.optim.lr_scheduler.LRScheduler] = None,
    ):
        super().__init__(env.observation_space, env.action_space)
        self.encoder = Encoder(
            num_classes,
            latent_dim,
            spaces.flatdim(env.unwrapped.observation_space),
            spaces.flatdim(self.action_space),
            net_complexity,
        )
        self.decoder = Ensemble(
            th.nn.ModuleList(
                [
                    WorldModel(
                        spaces.flatdim(env.unwrapped.observation_space),
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

        self.reconstruction_optimizer = optimizer_type(list(self.parameters()), **optimizer_kwargs)

        assert isinstance(self.action_space, spaces.Box)
        self._action_low = th.nn.Parameter(th.from_numpy(self.action_space.low))  # type: ignore
        self._action_high = th.nn.Parameter(th.from_numpy(self.action_space.high))  # type: ignore

    def _build(self, policy_algorithm: OffPolicyAlgorithm):
        self.policy_algorithm = policy_algorithm
        self.policy = policy_algorithm.policy

    @th.no_grad()
    def _predict(
        self,
        observation: CEMRLObsTensorDict,
        deterministic: bool = False,
    ) -> th.Tensor:
        with th.no_grad():
            y, z = self.encoder(observation)
        policy_obs = {
            "observation": observation["observation"][:, -1].to(self.policy.device),
            "task_indicator": z.to(self.policy.device),
        }
        action = self.policy._predict(policy_obs, deterministic)  # type: ignore
        return action

    def scale_action(self, action: np.ndarray | th.Tensor) -> np.ndarray | th.Tensor:
        if isinstance(action, th.Tensor):
            return 2.0 * ((action - self._action_low) / (self._action_high - self._action_low)) - 1.0
        return super().scale_action(action)
