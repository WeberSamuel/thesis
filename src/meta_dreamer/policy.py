from stable_baselines3.sac.policies import SACPolicy
from gymnasium import spaces
import torch as th
from ..core.policies import StateAwarePolicy
from .task_inference import TaskInference
from .mpc import MPC

from submodules.dreamer.dreamer import Dreamer

class MetaDreamerPolicy(StateAwarePolicy):
    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Box, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self.world_model = Dreamer(observation_space, action_space, **kwargs)
        self.exploration_policy = SACPolicy(observation_space, action_space, **kwargs)
        self.policy = SACPolicy(observation_space, action_space, **kwargs)
        self.policy_mpc = MPC(self.world_model, **kwargs)
        self.task_inference = TaskInference(observation_space, action_space)
        self.is_evaluating = False

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        if self.is_evaluating:
            action_raw = self.policy._predict(observation, deterministic)
            action_refined = self.policy_mpc(observation, action_raw)
            return action_refined
        else:
            return self.exploration_policy._predict(observation, deterministic)