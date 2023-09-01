from typing import Any, Dict
from gymnasium import Env, ObservationWrapper, spaces
import numpy as np
from src.envs.meta_env import MetaMixin


def add_dimension_to_space(space: spaces.Space, dim_size: int):
    if isinstance(space, spaces.Dict):
        return spaces.Dict({k: add_dimension_to_space(v, dim_size) for k, v in space.spaces.items()})
    elif isinstance(space, spaces.Box):
        low = np.repeat(space.low[np.newaxis, ...], dim_size, axis=0)
        high = np.repeat(space.high[np.newaxis, ...], dim_size, axis=0)
        return spaces.Box(low=low, high=high, dtype=space.dtype)  # type: ignore
    else:
        raise NotImplementedError()


class CEMRLWrapper(ObservationWrapper):
    def __init__(self, env: Env, n_stack: int, normalize_obs_action: bool = True, disabled: bool = False):
        super().__init__(env)
        self.disabled = disabled

        if not self.disabled:
            self.original_obs_space = env.observation_space
            self.n_stack = n_stack
            self.normalize_obs_action = normalize_obs_action

            assert isinstance(env.unwrapped, MetaMixin)
            assert isinstance(env.action_space, spaces.Box)
            self.unwrapped: MetaMixin
            self.action_space: spaces.Box

            self.observation_space: spaces.Dict = self._get_obs_space()

    def _get_obs_space(self) -> spaces.Dict:
        if isinstance(self.observation_space, spaces.Box):
            obs_space = spaces.Dict({"observation": self.observation_space})
        elif isinstance(self.observation_space, spaces.Dict):
            obs_space = self.observation_space
        else:
            raise NotImplementedError()

        if self.normalize_obs_action:
            assert isinstance(self.action_space, spaces.Box)
            action_obs_space = spaces.Box(-np.ones_like(self.action_space.low), np.ones_like(self.action_space.high))
        else:
            action_obs_space = self.action_space
        obs_space = spaces.Dict(
            {
                **obs_space.spaces,
                "goal": self.unwrapped.goal_sampler.goal_space,
                "goal_idx": spaces.Box(0, self.unwrapped.goal_sampler.num_goals, (1,)),
                "task": spaces.Box(0, len(self.unwrapped.goal_sampler.available_tasks), (1,)),
                "action": action_obs_space,
                "reward": spaces.Box(-np.inf, np.inf, (1,)),
                "is_first": spaces.Box(0, 1, (1,)),
            }
        )

        obs_space = add_dimension_to_space(obs_space, self.n_stack)

        return obs_space  # type: ignore

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.disabled:
            return obs, reward, terminated, truncated, info
        return self.observation(obs, action, reward, info), reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.disabled:
            return self.env.reset(**kwargs)

        self.frames = {
            k: np.zeros((*v.shape,), dtype=v.dtype)
            for k, v in self.observation_space.spaces.items()
            if isinstance(v, spaces.Box)
        }

        obs, info = self.env.reset(**kwargs)
        info.setdefault("is_first", True)
        action = np.zeros_like(self.action_space.low)
        obs = self.observation(obs, action, 0.0, info)
        return obs, info

    def observation(self, obs: Any, action: Any, reward: Any, info: dict[str, Any]) -> dict[str, Any]:
        obs = {
            "observation": obs,
            "goal": info.get("goal", np.zeros_like(self.unwrapped.goal_sampler.goal_space.low)),
            "goal_idx": info.get("goal_idx", 0),
            "task": info.get("task", 0),
            "action": action,
            "reward": reward,
            "is_first": info.get("is_first", False),
        }

        if self.normalize_obs_action:
            assert isinstance(self.action_space, spaces.Box)
            low, high = self.action_space.low, self.action_space.high
            obs["action"] = 2.0 * ((action - low) / (high - low)) - 1.0

        for k, v in self.frames.items():
            v[:-1] = v[1:]
            v[-1] = obs[k]

        return self.frames
