from typing import Any, Callable
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution

from dataclasses import fields

# dataclasses asdict is usually recursive, but we don't want that
def asdict(dc) -> dict[str, Any]:
    return {field.name:getattr(dc, field.name) for field in fields(dc)}

def compute_lambda_values(
    rewards: th.Tensor, values: th.Tensor, continues: th.Tensor, horizon_length: int, device: th.device, lambda_: float
):
    """
    rewards : (batch_size, time_step, hidden_size)
    values : (batch_size, time_step, hidden_size)
    continue flag will be added
    """
    rewards = rewards[:, :-1]
    continues = continues[:, :-1]
    next_values = values[:, 1:]
    last = next_values[:, -1]
    inputs = rewards + continues * next_values * (1 - lambda_)

    outputs = []
    # single step
    for index in reversed(range(horizon_length - 1)):
        last = inputs[:, index] + continues[:, index] * lambda_ * last
        outputs.append(last)
    returns = torch.stack(list(reversed(outputs)), dim=1).to(device)
    return returns


def horizontal_forward(
    network: th.nn.Module,
    x: th.Tensor,
    y: th.Tensor | None = None,
    input_shape: tuple[int, ...] = (-1,),
    output_shape: tuple[int, ...] = (-1,),
):
    batch_with_horizon_shape = x.shape[: -len(input_shape)]
    if not batch_with_horizon_shape:
        batch_with_horizon_shape = (1,)
    if y is not None:
        x = torch.cat((x, y), -1)
        input_shape = (x.shape[-1],)  #
    x = x.reshape(-1, *input_shape)
    x = network(x)

    x = x.reshape(*batch_with_horizon_shape, *output_shape)
    return x


def initialize_weights(m: th.nn.Module):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def build_network(input_size: int, hidden_size: int, num_layers: int, activation: str, output_size: int):
    assert num_layers >= 2, "num_layers must be at least 2"
    activation_fn = getattr(nn, activation)()
    layers = []
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(activation_fn)

    for i in range(num_layers - 2):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(activation_fn)

    layers.append(nn.Linear(hidden_size, output_size))

    network = nn.Sequential(*layers)
    network.apply(initialize_weights)
    return network


def create_normal_dist(
    x: th.Tensor,
    std: float | th.Tensor | None = None,
    mean_scale: float = 1,
    init_std: float = 0,
    min_std: float = 0.1,
    activation: Callable[[th.Tensor], th.Tensor] | None = None,
    event_shape=None,
):
    if std == None:
        mean, std = torch.chunk(x, 2, -1)
        mean = mean / mean_scale
        if activation:
            mean = activation(mean)
        mean = mean_scale * mean
        std = F.softplus(std + init_std) + min_std
    else:
        mean = x
    dist: Distribution = torch.distributions.Normal(mean, std)
    if event_shape:
        dist = torch.distributions.Independent(dist, event_shape)
    return dist
