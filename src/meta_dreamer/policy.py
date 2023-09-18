from stable_baselines3.sac.policies import SACPolicy
from gymnasium import spaces
import torch as th
from src.core.state_aware_algorithm import StateAwarePolicy
from src.meta_dreamer.task_inference import TaskEncoder

class MetaDreamerPolicy(StateAwarePolicy):
    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Box, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self.exploration_policy = SACPolicy(observation_space, action_space, **kwargs)
        self.policy = SACPolicy(observation_space, action_space, **kwargs)
        self.policy_mpc = MPC()
        self.task_inference = TaskEncoder(observation_space, action_space)
        self.is_training = False

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        if self.is_training:
            return self.exploration_policy._predict(observation, deterministic)
        else:
            return self.policy._predict(observation, deterministic)
        
    def _train(self) -> th.Tensor:
        raise NotImplementedError()