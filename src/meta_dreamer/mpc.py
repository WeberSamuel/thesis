import torch as th

class MPC(th.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, observation:th.Tensor) -> th.Tensor:
        raise NotImplementedError()
    
    def _train(self) -> th.Tensor:
        raise NotImplementedError()