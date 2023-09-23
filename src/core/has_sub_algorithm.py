from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Any, cast
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces
from typing import Callable
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.logger import Logger


class HasSubAlgorithm:
    env: VecEnv
    policy_kwargs: dict[str, Any]
    replay_buffer: ReplayBuffer
    logger: Logger

    def __init__(
        self,
        *args,
        sub_algorithm_class: type[OffPolicyAlgorithm],
        sub_algorithm_kwargs: dict[str, Any] | None = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sub_algorithm_kwargs = sub_algorithm_kwargs or {}
        self.sub_algorithm_class = sub_algorithm_class

    def _setup_sub_algorithm(self):
        policy = "MultiInputPolicy" if isinstance(self.env.observation_space, spaces.Dict) else "MlpPolicy"
        self.sub_algorithm = self.sub_algorithm_class(
            policy, self.env, buffer_size=0, **self.sub_algorithm_kwargs
        )

    def _setup_model(self) -> None:
        self.policy_kwargs.setdefault("sub_policy", self.sub_algorithm.policy)
        super()._setup_model()
        self.sub_algorithm.replay_buffer = self.replay_buffer

    def _setup_learn(self, *args, **kwargs) -> tuple[int, BaseCallback]:
        result = super()._setup_learn(*args, **kwargs)
        self.sub_algorithm.set_logger(self.logger)
        return result

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["sub_algorithm"]
