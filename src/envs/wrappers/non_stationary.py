from enum import Enum
from typing import Any, List, Optional, SupportsFloat
from gymnasium import Env, Wrapper, spaces
import numpy as np
from src.envs.meta_env import MetaMixin

class ChangeModes(Enum):
    AFTER_TIMESTEP = 1
    AFTER_LOCATION = 2
    PROBABILITY = 3
    AFTER_TIMESTEP_WITH_PROBABILITY = 4
    AFTER_LOCATION_WITH_PROBABILITY = 5
    AFTER_LOCATION_AND_TIMESTEP = 6
    AFTER_LOCATION_AND_TIMESTEP_WITH_PROBABILITY = 7

class NonStationaryWrapper(Wrapper):
    def __init__(
        self,
        env: Env,
        change_after_timestep: Optional[int] = None,
        change_probability: Optional[float] = None,
        change_after_location: Optional[List[List[float]]] = None,
    ):
        super().__init__(env)
        assert isinstance(self.unwrapped, MetaMixin)

        self.mode = {
            (False, False, False): ChangeModes.PROBABILITY,
            (False, False, True): ChangeModes.AFTER_LOCATION,
            (False, True, False): ChangeModes.PROBABILITY,
            (False, True, True): ChangeModes.AFTER_LOCATION_WITH_PROBABILITY,
            (True, False, False): ChangeModes.AFTER_TIMESTEP,
            (True, False, True): ChangeModes.AFTER_LOCATION_AND_TIMESTEP,
            (True, True, False): ChangeModes.AFTER_TIMESTEP_WITH_PROBABILITY,
            (True, True, True): ChangeModes.AFTER_LOCATION_AND_TIMESTEP_WITH_PROBABILITY,
        }[(change_after_timestep is not None, change_probability is not None, change_after_location is not None)] # type: ignore

        self.unwrapped: MetaMixin # type: ignore

        if self.mode == ChangeModes.PROBABILITY:
            self.change_probability = 0.0 if change_probability is None else change_probability
        else:
            self.change_probability = 1.0 if change_probability is None else change_probability

        self.change_after_timestep = np.inf if change_after_timestep is None else change_after_timestep

        self.timestep = 0

        self.change_after_location = None
        if change_after_location is not None:
            box = np.array(change_after_location)
            self.change_after_timestep = 0 if change_after_timestep is None else change_after_timestep
            self.change_after_location = spaces.Box(box[:, 0], box[:, 1])

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self.timestep += 1
        obs, reward, terminated, truncated, info = super().step(action)

        if self.mode in [ChangeModes.AFTER_TIMESTEP, ChangeModes.AFTER_TIMESTEP_WITH_PROBABILITY]:
            if np.random.random() < self.change_probability and self.change_after_timestep <= self.timestep:
                self.unwrapped.change_goal()
                info["goal_changed"] = self.unwrapped.goal_idx
                self.timestep = 0

        if self.mode == ChangeModes.PROBABILITY and np.random.random() < self.change_probability:
            self.unwrapped.change_goal()
            info["goal_changed"] = self.unwrapped.goal_idx

        if self.change_after_location is not None and not self.change_after_location.contains(obs):
            if np.random.random() < self.change_probability:
                if self.timestep > self.change_after_timestep:
                    self.unwrapped.change_goal()
                    info["goal_changed"] = self.unwrapped.goal_idx
                    self.timestep = 0

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self.timestep = 0
        return super().reset(seed=seed, options=options)
