from typing import Any
from gymnasium import Env, ObservationWrapper, spaces
import numpy as np
from ...core.envs import MetaMixin

class CEMRLWrapper(ObservationWrapper):
    def __init__(self, env: Env, normalize_obs_action: bool = True, disabled: bool = False, n_stack:int=30):
        super().__init__(env)
        self.disabled = disabled

        if not self.disabled:
            self.original_obs_space = env.observation_space
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
                "is_terminal": spaces.Box(0, 1, (1,)),
            }
        )

        return obs_space  # type: ignore

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        info.setdefault("step_count", self.step_count)
        return self.observation(obs, action, reward, terminated, info), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info.setdefault("is_first", True)
        action = np.zeros_like(self.action_space.low)
        obs = self.observation(obs, action, 0.0, False, info)
        self.step_count = 0
        return obs, info

    def observation(self, obs: Any, action: Any, reward: Any, terminated: Any, info: dict[str, Any]) -> dict[str, Any]:
        obs = {
            "observation": obs,
            "goal": info.get("goal", np.zeros_like(self.unwrapped.goal_sampler.goal_space.low)),
            "goal_idx": info.get("goal_idx", 0),
            "task": info.get("task", 0),
            "action": action,
            "reward": reward,
            "is_first": info.get("is_first", False),
            "is_terminal": terminated,
        }

        if self.normalize_obs_action:
            assert isinstance(self.action_space, spaces.Box)
            low, high = self.action_space.low, self.action_space.high
            obs["action"] = 2.0 * ((action - low) / (high - low)) - 1.0

        return obs
