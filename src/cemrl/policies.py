from typing import Any, Optional
from gymnasium import spaces
import numpy as np
import torch as th
from src.core.state_aware_algorithm import StateAwarePolicy
from src.cemrl.types import CEMRLObsTensorDict, CEMRLPolicyInput
from src.utils import apply_function_to_type
from .task_inference import EncoderInput, TaskInference
from .config import CemrlConfig
from jsonargparse import lazy_instance
from stable_baselines3.common.policies import BasePolicy


class CEMRLPolicy(StateAwarePolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Any,
        sub_policy: BasePolicy,
        config: CemrlConfig = lazy_instance(CemrlConfig),
        **kwargs,
    ):
        super().__init__(observation_space, action_space, **kwargs)
        self.config = config
        self.task_inference = TaskInference(
            spaces.flatdim(observation_space["observation"]), spaces.flatdim(action_space), config=config.task_inference
        )
        self.sub_policy = sub_policy

    @th.no_grad()
    def _predict(
        self,
        observation: CEMRLObsTensorDict,
        deterministic: bool = False,
        task_encoding: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        if task_encoding is None:
            prev_observation: CEMRLObsTensorDict = self.state  # type: ignore
            next_obs: CEMRLObsTensorDict = {}
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

    def _reset_states(self, size: int) -> tuple[np.ndarray, ...]:
        return apply_function_to_type(
            self.observation_space.sample(),
            np.ndarray,
            lambda x: th.zeros((size, self.config.training.encoder_context_length, *x.shape), device=self.device),
        )
