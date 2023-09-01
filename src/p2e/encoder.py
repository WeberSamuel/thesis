import torch as th
import torch.nn as nn

from src.p2e.utils import build_network, horizontal_forward, initialize_weights

class Encoder(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        embedded_state_size: int,
        activation: str = "ReLU",
        depth: int = 32,
        kernel_size: int = 4,
        stride: int = 2,
    ):
        super().__init__()
        activation_fn: th.nn.Module = getattr(nn, activation)()
        self.observation_shape = observation_shape

        if len(self.observation_shape) == 3:
            # Image observation
            self.network = nn.Sequential(
                nn.Conv2d(
                    self.observation_shape[0],
                    depth * 1,
                    kernel_size,
                    stride,
                ),
                activation_fn,
                nn.Conv2d(
                    depth * 1,
                    depth * 2,
                    kernel_size,
                    stride,
                ),
                activation_fn,
                nn.Conv2d(
                    depth * 2,
                    depth * 4,
                    kernel_size,
                    stride,
                ),
                activation_fn,
                nn.Conv2d(
                    depth * 4,
                    depth * 8,
                    kernel_size,
                    stride,
                ),
                activation_fn,
            )
        else:
            # Vector observation
            self.network = build_network(self.observation_shape[0], 400, 3, "ELU", embedded_state_size)
        self.network.apply(initialize_weights)

    def forward(self, x: th.Tensor):
        return horizontal_forward(self.network, x, input_shape=self.observation_shape)
