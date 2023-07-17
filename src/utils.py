"""This module contains utility functions."""
import numpy as np
from gym import spaces
from typing import Any, Callable


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


def remove_dim_from_space(space: spaces.Space, dim: int = 0) -> spaces.Space:
    """
    Removes a dimension from a gym.spaces.Space object.

    :param space: The space to remove the dimension from.
    :param dim: The index of the dimension to remove.
    :return: A new space with the specified dimension removed.
    """
    if isinstance(space, spaces.Dict):
        return spaces.Dict({key: remove_dim_from_space(value, dim) for key, value in space.spaces.items()})
    elif isinstance(space, spaces.Box):
        return spaces.Box(low=np.take(space.low, dim, axis=dim), high=np.take(space.high, dim, axis=dim))
    elif isinstance(space, spaces.Tuple):
        return spaces.Tuple(remove_dim_from_space(s, dim) for s in space.spaces)
    else:
        raise NotImplementedError()