"""Utility functions used in different parts of the project."""
from typing import Any, Callable, Type
from gym import spaces
import numpy as np
import torch as th
from stable_baselines3.common.type_aliases import ReplayBufferSamples, DictReplayBufferSamples


def apply_function_to_type(data: Any, apply_on_type: Type, function: Callable):
    """Apply a function to each element of type apply_on_type in data.

    Args:
        data (Any): data to process.
                    If an list, tuple, dict is passed, each element is processed recursively.
        apply_on_type (Type): Type on which to apply the function.
        function (Callable): Function that is applied on each entry, that is  the apply_on_type

    Returns:
        Any: data with function applied to all elements of type apply_on_type
    """
    if isinstance(data, apply_on_type):
        return function(data)

    if isinstance(data, dict):
        return {key: apply_function_to_type(value, apply_on_type, function) for key, value in data.items()}

    if isinstance(data, tuple):
        return tuple(apply_function_to_type(value, apply_on_type, function) for value in data)

    if isinstance(data, list):
        return [apply_function_to_type(value, apply_on_type, function) for value in data]

    return data


def remove_dim_from_space(space: spaces.Dict | spaces.Box, dim: int):
    if isinstance(space, spaces.Dict):
        return {key: remove_dim_from_space(value) for key, value in space.spaces.items()}
    return spaces.Box(low=np.take(space.low, 0, axis=dim), high=np.take(space.high, 0, axis=dim))


def get_random_encoder_window_samples(samples: DictReplayBufferSamples, encoder_window:int):
    device = samples.actions.device
    episode_length = samples.actions.shape[1]

    # filter out encoder windows that contain other epochs
    dones = samples.dones.squeeze(-1)
    dones = dones.roll(1)
    invalid = th.where(dones)
    invalid_idx = invalid[1][None] + th.arange(0, encoder_window, device=device)[:, None]
    invalid_idx = invalid_idx.clamp_max(episode_length - 1).view(-1)
    invalid = (invalid[0].repeat(encoder_window),  invalid_idx)
    idx_probs = th.ones_like(dones)
    idx_probs[invalid] = 0
    
    encoder_timestep_idx = th.multinomial(idx_probs, 1, replacement=True).squeeze(-1)
    encoder_timestep_idx = encoder_timestep_idx[:, None] - th.arange(0, encoder_window, device=device)[None]
    encoder_timestep_idx = encoder_timestep_idx.flip(1)

    obs_gather = encoder_timestep_idx[..., None].expand(-1, -1, *samples.observations["observation"].shape[2:])
    action_gather = encoder_timestep_idx[..., None].expand(-1, -1, *samples.actions.shape[2:])
    reward_gather = encoder_timestep_idx[..., None].expand(-1, -1, *samples.rewards.shape[2:])
    dones_gather = encoder_timestep_idx[..., None].expand(-1, -1, *samples.dones.shape[2:])
    # Forward pass through encoder
    return ReplayBufferSamples(
        observations=th.gather(samples.observations["observation"], 1, obs_gather),
        actions=th.gather(samples.actions, 1, action_gather),
        rewards=th.gather(samples.rewards, 1, reward_gather),
        next_observations=th.gather(samples.next_observations["observation"], 1, obs_gather),
        dones=th.gather(samples.dones, 1, dones_gather)
    )
