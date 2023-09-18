import torch as th

class TaskEncoder(th.nn.Module):
    def __init__(self, observation_space, action_space) -> None:
        super().__init__()

    def forward(self, observation:th.Tensor) -> th.Tensor:
        raise NotImplementedError()
    
    def _train(self) -> th.Tensor:
        raise NotImplementedError()