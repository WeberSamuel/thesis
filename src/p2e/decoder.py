import torch.nn as nn

from src.p2e.utils import build_network, create_normal_dist, initialize_weights, horizontal_forward


class Decoder(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        stochastic_size: int,
        deterministic_size: int,
        activation: str = "ReLU",
        depth: int = 32,
        kernel_size: int = 5,
        stride: int = 2,
    ):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size

        activation_fn = getattr(nn, activation)()
        self.observation_shape = observation_shape

        if len(self.observation_shape) == 3:
            # image observation
            self.network = nn.Sequential(
                nn.Linear(self.deterministic_size + self.stochastic_size, depth * 32),
                nn.Unflatten(1, (depth * 32, 1)),
                nn.Unflatten(2, (1, 1)),
                nn.ConvTranspose2d(
                    depth * 32,
                    depth * 4,
                    kernel_size,
                    stride,
                ),
                activation_fn,
                nn.ConvTranspose2d(
                    depth * 4,
                    depth * 2,
                    kernel_size,
                    stride,
                ),
                activation_fn,
                nn.ConvTranspose2d(
                    depth * 2,
                    depth * 1,
                    kernel_size + 1,
                    stride,
                ),
                activation_fn,
                nn.ConvTranspose2d(
                    depth * 1,
                    self.observation_shape[0],
                    kernel_size + 1,
                    stride,
                ),
            )
        else:
            # vector observation
            self.network = build_network(self.deterministic_size + self.stochastic_size, 400, 4, "ELU", self.observation_shape[0])
        self.network.apply(initialize_weights)

    def forward(self, posterior, deterministic):
        x = horizontal_forward(self.network, posterior, deterministic, output_shape=self.observation_shape)
        dist = create_normal_dist(x, std=1, event_shape=len(self.observation_shape))
        return dist
