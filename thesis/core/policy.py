from typing import Any, Dict, Generic, Optional, Type, TypeVar
import numpy as np

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy as sb3BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule

State = TypeVar("State")

def recursivly_apply(fn, x):
    if isinstance(x, dict):
        return {k: recursivly_apply(fn, v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(recursivly_apply(fn, v) for v in x)
    return fn(x)

def recursivly_apply_on_two(fn, x, y):
    if isinstance(x, dict):
        return {k: recursivly_apply_on_two(fn, v, y[k]) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(recursivly_apply_on_two(fn, v, w) for v, w in zip(x, y))
    return fn(x, y)

class BasePolicy(sb3BasePolicy, Generic[State]):
    state: State
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
        squash_output: bool = False,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            features_extractor=features_extractor,
            squash_output=squash_output,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.is_evaluating = False

    def predict(
        self,
        observation: dict[str, np.ndarray],
        state: State | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, State | None]:
        n_env = len(observation) if not isinstance(observation, dict) else observation["observation"].shape[0]
        self.state = state or self.reset_states(n_env)
        self.state = self.reset_states(None, observation["is_first"], self.state)

        action, _ = super().predict(observation, state, episode_start, deterministic) # type: ignore
        return action, self.state

    def reset_states(
        self,
        n_env: int | None = None,
        dones: np.ndarray | None = None,
        state: State | None = None,
    ) -> State:
        if dones is not None:
            if state is None:
                raise ValueError("dones was provided but not state")
            done_idx = np.where(dones)[0]
            if len(done_idx) == 0:
                return state

            state = recursivly_apply(lambda x: x.clone() if isinstance(x, th.Tensor) else x.copy(), state) # type: ignore

            reset_states = self._reset_states(len(done_idx))

            def copy_into_state(x, y):
                x[done_idx] = y
                return x

            state = recursivly_apply_on_two(copy_into_state, state, reset_states) # type: ignore
            return state # type: ignore
        if n_env is None:
            raise ValueError("n_env was not provided")
        return self._reset_states(n_env)

    def _reset_states(self, size: int) -> State:
        raise NotImplementedError()
