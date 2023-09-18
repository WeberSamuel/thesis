from abc import ABC, abstractmethod
from typing import Any, Optional, SupportsFloat, Tuple
from gymnasium.core import Env

import numpy as np
from gymnasium import Env, spaces, ObservationWrapper, Wrapper


class BaseSampler(ABC):
    def __init__(self, available_tasks: list[int], num_goals: int) -> None:
        self.available_tasks = available_tasks
        self.num_goals = num_goals
        self.initialized = False

    def _init_sampler(self, env: Env):
        self.env = env
        self.goal_space = self._get_goal_space()
        self.goals, self.tasks = self.sample(self.num_goals, self.available_tasks)
        self.initialized = True

    @abstractmethod
    def _get_goal_space(self) -> spaces.Space:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, num_goals: int, available_tasks: list[int]) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()


class MetaMixin:
    def __init__(self, goal_sampler: BaseSampler, *args, **kwargs) -> None:
        assert isinstance(self, Env), "MetaMixin must be inherited by an Env"

        self.goal_sampler = goal_sampler
        self.neutral_action: Optional[np.ndarray] = None
        super().__init__(*args, **kwargs)
        if isinstance(goal_sampler, BaseSampler) and not self.goal_sampler.initialized:
            self.goal_sampler._init_sampler(self)

    def change_goal(self):
        self.goal_idx = np.random.randint(0, len(self.goal_sampler.goals))
        self.goal = self.goal_sampler.goals[self.goal_idx]
        self.task = self.goal_sampler.tasks[self.goal_idx]

    def add_meta_info(self, info: dict) -> dict:
        info["goal_idx"] = self.goal_idx
        info["goal"] = self.goal
        info["task"] = self.task
        return info

    def reset(self, *args, **kwargs):
        self.change_goal()
        return super().reset(*args, **kwargs)  # type: ignore


class AddMetaToObservationWrapper(Wrapper):
    unwrapped: MetaMixin

    def __init__(self, env: Env):
        super().__init__(env)

        obs_space = (
            self.observation_space
            if isinstance(self.observation_space, spaces.Dict)
            else spaces.Dict({"observation": self.observation_space})
        )

        self.observation_space = spaces.Dict(
            {
                **obs_space.spaces,
                "goal": self.unwrapped.goal_sampler.goal_space,
                "goal_idx": spaces.Box(0, self.unwrapped.goal_sampler.num_goals, (1,)),
                "task": spaces.Box(0, len(self.unwrapped.goal_sampler.available_tasks), (1,)),
            }
        )

    def observation(self, observation):
        obs = observation
        if not isinstance(obs, dict):
            obs = {"observation": obs}
        obs["goal"] = self.unwrapped.goal_sampler.goals[self.unwrapped.goal_idx]
        obs["goal_idx"] = self.unwrapped.goal_idx
        obs["task"] = self.unwrapped.task
        return obs


class AddAdditionalToObservationWrapper(Wrapper):
    action_space: spaces.Box

    def __init__(self, env: Env, normalize_observed_action = True):
        super().__init__(env)
        self.normalize_observed_action = normalize_observed_action

        obs_space = (
            self.observation_space
            if isinstance(self.observation_space, spaces.Dict)
            else spaces.Dict({"observation": self.observation_space})
        )

        self.observation_space = spaces.Dict(
            {
                "observation": obs_space,
                "action": self.action_space,
                "reward": spaces.Box(-np.inf, np.inf, (1,)),
                "done": spaces.Box(0, 1, (1,)),
                "is_first": spaces.Box(0, 1, (1,)),
            }
        )

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        obs = self.observation(obs, action, reward, terminated or truncated)

        return obs, reward, terminated, truncated, info
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        
        obs = self.observation(obs, np.zeros_like(self.action_space.low), 0.0, False)
        obs["is_first"] = True

        return obs, info
    
    def observation(self, obs, action, reward, done) -> dict[str, Any]:
        if self.normalize_observed_action:
            assert isinstance(self.action_space, spaces.Box)
            low, high = self.action_space.low, self.action_space.high
            obs["action"] = 2.0 * ((action - low) / (high - low)) - 1.0

        return {"observation": obs, "action": action, "reward": reward, "done": done, "is_first": False}