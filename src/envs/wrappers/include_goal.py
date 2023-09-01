from typing import Any, Dict, SupportsFloat
import numpy as np
from gymnasium import spaces, Wrapper, Env
from src.envs.meta_env import MetaMixin


class IncludeGoalWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space = self._get_obs_space()

        assert isinstance(self.unwrapped, MetaMixin)
        self.unwrapped: MetaMixin # type: ignore

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        return self.update_obs(obs, info), reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        return self.update_obs(obs, info), info

    def update_obs(self, observation: Dict[str, Any] | Any, info: Dict[str, Any]) -> Dict[str, Any]:
        observation = observation if isinstance(observation, dict) else {"observation": observation}
        observation = {
            **observation,
            "goal_idx": np.ndarray([self.unwrapped.goal_idx]).astype(np.float32),
            "goal": np.ndarray(self.unwrapped.goal).astype(np.float32),
            "task": np.ndarray([self.unwrapped.task]).astype(np.float32),
        }
        return observation

    def _get_obs_space(self):
        obs_space = (
            self.observation_space.spaces
            if isinstance(self.observation_space, spaces.Dict)
            else {"observation": self.observation_space}
        )

        assert "goal" not in obs_space
        assert "goal_idx" not in obs_space
        assert "task" not in obs_space

        obs_space = spaces.Dict(
            {
                **obs_space,
                "goal": self.unwrapped.goal_sampler.goal_space,
                "goal_idx": spaces.Box(0, self.unwrapped.goal_sampler.num_goals, (1,)),
                "task": spaces.Box(0, len(self.unwrapped.goal_sampler.available_tasks), (1,)),
            }
        )

        return obs_space
