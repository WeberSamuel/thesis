import torch as th
from gymnasium import spaces

class ExplorationPolicy(th.nn.Module):
    def __init__(self, observation_space:spaces.Dict, action_space:spaces.Box) -> None:
        super().__init__()

    def forward(self, observation:th.Tensor) -> th.Tensor:
        raise NotImplementedError()