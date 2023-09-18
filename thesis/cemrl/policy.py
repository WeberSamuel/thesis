from typing import TypedDict

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import MultiInputPolicy as SACPolicy
from torch._tensor import Tensor

from ..core.policy import BasePolicy
from .config import CemrlConfig
from .task_inference import EncoderInput, TaskInference


class CemrlPolicyInput(TypedDict):
    obs: th.Tensor
    task_encoding: th.Tensor


class CemrlPolicy(BasePolicy[tuple[th.Tensor, th.Tensor]]):
    observation_space: spaces.Dict

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        config: CemrlConfig = CemrlConfig(),
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
        )
        self.config = config
        self.task_inference = TaskInference(
            spaces.flatdim(observation_space.spaces["observation"]), spaces.flatdim(action_space), config.task_inference
        )
        self.sac_policy = SACPolicy(
            observation_space=spaces.Dict(
                CemrlPolicyInput(
                    obs=observation_space.spaces["observation"],  # type: ignore
                    task_encoding=spaces.Box(-np.inf, np.inf, shape=(config.task_inference.encoder.latent_dim,)),  # type: ignore
                )
            ),
            action_space=action_space,
            lr_schedule=lr_schedule,
        )

    def _predict(self, observation: dict[str, th.Tensor], deterministic: bool = False) -> th.Tensor:
        prev_obs, encoder_state = self.state
        with th.no_grad():
            z, _, encoder_state = self.task_inference.encoder(
                EncoderInput(
                    prev_obs[:, None],
                    observation["action"][:, None],
                    observation["observation"][:, None],
                    observation["reward"][:, None],
                ),
                encoder_state.transpose(0, 1)
            )
        self.state = (observation["observation"], encoder_state.transpose(0, 1))

        action = self.sac_policy._predict(CemrlPolicyInput(obs=observation["observation"], task_encoding=z), deterministic)  # type: ignore
        return action

    def _reset_states(self, size: int) -> tuple[Tensor, Tensor]:
        return th.zeros(size, *self.observation_space["observation"].shape, device=self.device), self.task_inference.get_init_state(size)  # type: ignore
