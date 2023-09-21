from dataclasses import dataclass
from typing import Literal

import torch as th


@dataclass
class MPCConfig:
    optim_class: type[th.optim.Optimizer] = th.optim.AdamW
    lr: float = 1e-3
    delta_loss_threshold: float = 0.1
    max_num_iterations = 10
    horizon: int = 10
    target_on: Literal["value", "reward"] = "reward"

class TaskInference(th.nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def forward(self, observation, action):
        pass