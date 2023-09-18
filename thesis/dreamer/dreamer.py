from typing import Any, Dict, Tuple

import torch as th
from gymnasium import spaces
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from torch._tensor import Tensor

from ..core.algorithm import BaseAlgorithm
from ..core.buffer import ReplayBuffer
from ..core.policy import BasePolicy

from .actor_critic import ActorCritic
from .config import DreamerConfig
from .world_model import Deterministic, DreamerWorldModel, Stochastic


class DreamerPolicy(BasePolicy[tuple[th.Tensor, Stochastic, Deterministic]]):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        features_extractor_class: type[BaseFeaturesExtractor] = ...,
        features_extractor_kwargs: Dict[str, Any] | None = None,
        features_extractor: BaseFeaturesExtractor | None = None,
        squash_output: bool = False,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Dict[str, Any] | None = None,
        config: DreamerConfig = DreamerConfig(),
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class,
            features_extractor_kwargs,
            features_extractor,
            squash_output,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self.config = config
        self.action_size = spaces.flatdim(action_space)
        self.world_model = DreamerWorldModel(
            config.stochastic_size,
            config.deterministic_size,
            config.embedded_size,
            config.task_size,
            observation_space,
            self.action_size,
            config.world_model_config,
        )
        self.actor_critic = ActorCritic(
            isinstance(self.action_space, spaces.Discrete),
            self.action_size,
            config.stochastic_size,
            config.deterministic_size,
            self.world_model,
            config.actor_critic_config,
        )

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        last_action, posterior, deter = self.state
        embedded_observation = self.world_model.encoder(observation)
        deter = self.world_model.rssm.sequence_model(posterior, last_action, deter)
        embedded_observation = embedded_observation.reshape(1, -1)
        _, posterior = self.world_model.rssm.representation_model(embedded_observation, deter)
        action = self.actor_critic.actor(posterior, deter)
        self.state = (action, posterior, deter)
        return action

    def _reset_states(self, size: int) -> tuple[Tensor, Stochastic, Deterministic]:
        posterior, deterministic = self.world_model.rssm.get_init_input(size)
        action = th.zeros(size, self.action_size, device=self.device)
        return (action, posterior, deterministic)


class Dreamer(BaseAlgorithm):
    policy: DreamerPolicy

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
        train_freq: int | Tuple[int, str] = 1,
        gradient_steps: int = 1,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = ReplayBuffer,
        replay_buffer_kwargs: Dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Dict[str, Any] | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        verbose: int = 0,
        device: th.device | str = "auto",
        monitor_wrapper: bool = True,
        seed: int | None = None,
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            policy_kwargs,
            stats_window_size,
            tensorboard_log,
            verbose,
            device,
            monitor_wrapper,
            seed,
        )

        if _init_setup_model:
            self._setup_model()

    def train(self, gradient_steps: int, batch_size: int) -> None:
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample_context(batch_size, self.get_vec_normalize_env(), self.policy.config.batch_length)
            posterior, deterministic, world_model_metrics = self.policy.world_model.training_step(data)
            actor_critic_metrics = self.policy.actor_critic.training_step(posterior, deterministic)

            for key, value in world_model_metrics.items():
                self.logger.record_mean(f"world_model/{key}", value) 

            for key, value in actor_critic_metrics.items():
                self.logger.record_mean(f"actor_critic/{key}", value)

        super().train(gradient_steps, batch_size)