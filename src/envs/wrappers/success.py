from typing import Any, SupportsFloat

from gymnasium import Env, Wrapper


class SuccessWrapper(Wrapper):
    def is_success(self) -> bool:
        return self.success_buffer[-1]

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        self.success_buffer.append(info.get("is_success", False))
        info["is_success"] = self.is_success()
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Any:
        obs, info = super().reset(**kwargs)
        self.success_buffer = []
        self.success_buffer.append(info.get("is_success", False))
        info["is_success"] = self.is_success()
        return obs, info


class PercentageSuccessWrapper(SuccessWrapper):
    def __init__(self, env: Env, success_threshold: float = 0.5, episode_window: float = 0.1):
        super().__init__(env)
        self.success_threshold = success_threshold
        self.episode_window = episode_window

    def is_success(self) -> bool:
        if self.episode_window > 1.0:
            last_steps = int(self.episode_window)
        else:
            last_steps = max(int(len(self.success_buffer) * self.episode_window), 1)

        return sum(self.success_buffer[-last_steps:]) / last_steps >= self.success_threshold
