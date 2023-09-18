from typing import Any, Dict, Tuple
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from core.algorithm import BaseAlgorithm
from meta_dreamer.policy import MetaDreamerPolicy


class MetaDreamer(BaseAlgorithm):
    policy: MetaDreamerPolicy

    def __init__(
        self,
        policy: str | type[BasePolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | Tuple[int, str] = ...,
        gradient_steps: int = 1,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: Dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Dict[str, Any] | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        verbose: int = 0,
        device: th.device | str = "auto",
        monitor_wrapper: bool = True,
        seed: int | None = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
        )

