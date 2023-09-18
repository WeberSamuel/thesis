from typing import Any, Dict, List, Optional, Type, Union

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import SACPolicy

from core.policy import BasePolicy
from core.world_model import WorldModel


class MPC(th.nn.Module):
    def __init__(self, lr: float = 1e-3, horizon: int = 10) -> None:
        super().__init__()
        self.lr = lr
        self.horizon = horizon

    def forward(self, world_model: th.nn.Module, state: th.Tensor, action: th.Tensor):
        action = action.clone().detach().requires_grad_(True)
        optimizer = th.optim.AdamW([action], lr=self.lr)
        for _ in range(self.horizon):
            optimizer.zero_grad()
            loss = -world_model(state, action).reward
            loss.backward()
            optimizer.step()
        return action


class MetaDreamerPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        features_extractor_class: type[BaseFeaturesExtractor] = ...,
        features_extractor_kwargs: Dict[str, Any] | None = None,
        features_extractor: BaseFeaturesExtractor | None = None,
        squash_output: bool = False,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Dict[str, Any] | None = None,
        mpc_lr: float = 1e-3,
        mpc_horizon: int = 10,
        sac_net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        sac_activation_fn: Type[th.nn.Module] = th.nn.ReLU,
        sac_use_sde: bool = False,
        sac_log_std_init: float = -3,
        sac_use_expln: bool = False,
        sac_clip_mean: float = 2.0,
        sac_n_critics: int = 2,
        sac_share_features_extractor: bool = False,
        world_model: WorldModel|None = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            features_extractor=features_extractor,
            squash_output=squash_output,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.mpc = MPC(lr=mpc_lr, horizon=mpc_horizon)
        self.sac = SACPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            net_arch=sac_net_arch,
            activation_fn=sac_activation_fn,
            use_sde=sac_use_sde,
            log_std_init=sac_log_std_init,
            use_expln=sac_use_expln,
            clip_mean=sac_clip_mean,
            n_critics=sac_n_critics,
            share_features_extractor=sac_share_features_extractor,
        )
        self.world_model = world_model

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
       sac_action = self.sac._predict(observation, deterministic)
       mpc_action = self.mpc(self.world_model, observation, sac_action)
       return mpc_action