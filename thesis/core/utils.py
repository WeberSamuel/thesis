import torch as th
from stable_baselines3.common.torch_layers import create_mlp


def initialize_weights(m):
    if isinstance(m, (th.nn.Conv2d, th.nn.ConvTranspose2d)):
        th.nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            th.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, th.nn.Linear):
        th.nn.init.kaiming_uniform_(m.weight.data)
        th.nn.init.constant_(m.bias.data, 0)


def build_network(input_size: int, layers: tuple[int, ...] | list[int], activation: type[th.nn.Module], output_size: int):
    network = th.nn.Sequential(*create_mlp(input_size, output_size, list(layers), activation))
    network.apply(initialize_weights)
    return network
