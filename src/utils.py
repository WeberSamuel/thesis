"""This module contains utility functions."""
from typing import Any, Callable

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


class DeviceAwareModuleMixin:
    """
    Mixin class that makes a module aware of the device it is on.
    """

    def __init__(self, *args, **kwargs):
        if not isinstance(self, th.nn.Module):
            raise ValueError("DeviceAwareModuleMixin can only be used with torch.nn.Module subclasses.")
        super().__init__(*args, **kwargs)

    @property
    def device(self) -> th.device:
        """
        The device the module is on.

        :return: The device.
        """
        if isinstance(self, th.nn.Module):
            return next(self.parameters()).device
        raise NotImplementedError()


def apply_function_to_type(data: Any, apply_on_type: type, function: Callable) -> Any:
    """
    Recursively applies a function to all elements of a nested data structure of a given type.

    :param data: The data structure to apply the function to.
    :param apply_on_type: The type of elements to apply the function to.
    :param function: The function to apply.
    :return: The modified data structure.
    """
    if isinstance(data, apply_on_type):
        return function(data)
    elif isinstance(data, dict):
        return {key: apply_function_to_type(value, apply_on_type, function) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [apply_function_to_type(value, apply_on_type, function) for value in data]

    return data


def function_to_iterator(f: Callable):
    while True:
        yield f()
