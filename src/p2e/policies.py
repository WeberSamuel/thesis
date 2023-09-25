from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from jsonargparse import lazy_instance
from stable_baselines3.common.policies import BasePolicy

from src.cemrl.policies import CEMRLPolicy
from src.cemrl.task_inference import EncoderInput
from src.cemrl.types import CEMRLPolicyInput
from src.core.state_aware_algorithm import StateAwarePolicy
from src.p2e.networks import OneStepModel
from src.plan2explore.networks import Ensemble
from src.utils import apply_function_to_type, build_network

from .config import P2EConfig


class P2EPolicy(StateAwarePolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Any,
        main_policy: CEMRLPolicy,
        sub_policy: BasePolicy,
        optimizer_class: type[th.optim.Optimizer] = th.optim.AdamW,
        optimizer_kwargs: dict[str, Any] | None = None,
        config: P2EConfig = lazy_instance(P2EConfig),
        **kwargs
    ):
        super().__init__(
            observation_space, action_space, optimizer_class=optimizer_class, optimizer_kwargs=optimizer_kwargs, **kwargs
        )
        self.config = config
        self.sub_policy = sub_policy
        self.task_inference = main_policy.task_inference
        self.encoder_context_length = main_policy.config.training.encoder_context_length

        if config.use_world_model_as_ensemble:
            self.one_step_models = None
        else:
            self.input_size = (
                spaces.flatdim(observation_space["observation"])
                + spaces.flatdim(action_space)
                + main_policy.config.task_inference.encoder.latent_dim
            )
            self.one_step_models = Ensemble(
                th.nn.ModuleList(
                    [
                        OneStepModel(
                            spaces.flatdim(observation_space["observation"]),
                            spaces.flatdim(action_space),
                            main_policy.config.task_inference.encoder.latent_dim,
                            config.one_step_model,
                        )
                        for _ in range(config.one_step_model.ensemble_size)
                    ]
                )
            )
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore

    @th.no_grad()
    def _predict(
        self,
        observation: dict,
        deterministic: bool = False,
        task_encoding: th.Tensor | None = None,
    ) -> th.Tensor:
        if task_encoding is None:
            prev_observation = self.state  # type: ignore
            next_obs = {}
            for k, v in prev_observation.items():  # type:ignore
                v: th.Tensor
                next_obs[k] = v.clone()
                next_obs[k][:, :-1] = v[:, 1:]
                next_obs[k][:, -1] = observation[k]

            with th.no_grad():
                y, z, encoder_state = self.task_inference.forward(
                    EncoderInput(
                        obs=prev_observation["observation"],
                        next_obs=next_obs["observation"],
                        action=next_obs["action"],
                        reward=next_obs["reward"],
                    )
                )
            self.state = next_obs
        else:
            z = task_encoding

        policy_obs = CEMRLPolicyInput(observation=observation["observation"], task_indicator=z)
        action = self.sub_policy._predict(policy_obs, deterministic)  # type: ignore
        return action
    
    def online_disagreement(self, obs: th.Tensor, task_encoding: th.Tensor) -> th.Tensor:
        pass

    def _reset_states(self, size: int) -> tuple[np.ndarray, ...]:
        return apply_function_to_type(
            self.observation_space.sample(),
            np.ndarray,
            lambda x: th.zeros((size, self.encoder_context_length, *x.shape), device=self.device),
        )
