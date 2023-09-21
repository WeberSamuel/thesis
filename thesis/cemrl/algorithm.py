from typing import Any, Dict, Tuple

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name

from ..core.algorithm import BaseAlgorithm, StateStorage
from ..core.policy import BasePolicy
from ..core.buffer import ReplayBuffer
from .policy import CemrlPolicy


class Cemrl(BaseAlgorithm):
    policy: CemrlPolicy
    replay_buffer: ReplayBuffer

    def __init__(
        self,
        policy: str | type[BasePolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 1e-3,
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
        self.sac = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=0,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=train_freq,
            gradient_steps=self.gradient_steps,
            action_noise=self.action_noise,
            device=self.device,
            seed=self.seed,
            _init_setup_model=True,
        )
        self.state_storage = CemrlStateStorage()

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # copied from SAC._setup_model
        def internal_sac_setup():
            self.sac._create_aliases()
            self.sac.batch_norm_stats = get_parameters_by_name(self.sac.critic, ["running_"])
            self.sac.batch_norm_stats_target = get_parameters_by_name(self.sac.critic_target, ["running_"])
            if self.sac.target_entropy == "auto":
                self.sac.target_entropy = float(-np.prod(self.sac.env.action_space.shape).astype(np.float32))  # type: ignore
            else:
                self.sac.target_entropy = float(self.sac.target_entropy)

            if isinstance(self.sac.ent_coef, str) and self.sac.ent_coef.startswith("auto"):
                init_value = 1.0
                if "_" in self.sac.ent_coef:
                    init_value = float(self.sac.ent_coef.split("_")[1])
                    assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

                self.sac.log_ent_coef = th.log(th.ones(1, device=self.sac.device) * init_value).requires_grad_(True)
                self.sac.ent_coef_optimizer = th.optim.Adam([self.sac.log_ent_coef], lr=self.sac.lr_schedule(1))
            else:
                self.sac.ent_coef_tensor = th.tensor(float(self.sac.ent_coef), device=self.sac.device)

        super()._setup_model()
        self.sac.policy = self.policy.sac_policy
        self.sac.replay_buffer = self.replay_buffer
        internal_sac_setup()
        self.replay_buffer.task_encoder = self.policy.task_inference
        self.config = self.policy.config.training

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        result = super()._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar)
        self.sac.set_logger(self.logger)
        return result

    def train(self, gradient_steps: int, batch_size: int):
        self.policy.train(True)
        config = self.policy.config.training
        for i in range(gradient_steps):
            for j in range(config.task_inference_gradient_steps):
                encoder_samples, decoder_samples = self.replay_buffer.task_inference_sample(
                    batch_size, self.get_vec_normalize_env(), config.encoder_context_length, config.decoder_context_length
                )
                metrics = self.policy.task_inference.training_step(encoder_samples, decoder_samples)
                for key, value in metrics.items():
                    self.logger.record_mean(f"task_inference/{key}", value)
            self.sac.train(config.policy_gradient_steps, batch_size)

        super().train(gradient_steps, batch_size)
        self.policy.train(False)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["sac"]

class CemrlStateStorage(StateStorage):
    def _on_step(self) -> bool:
        self.policy_state = self.policy.obs_to_tensor(self.model._last_obs)[0]["observation"], self.policy_state[1] # type: ignore
        return super()._on_step()