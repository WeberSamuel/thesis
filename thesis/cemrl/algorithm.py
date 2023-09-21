from typing import Any, Dict, Tuple

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

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
        sub_algorithm_class: type[OffPolicyAlgorithm] | None = None,
        sub_algorithm_kwargs: Dict[str, Any] | None = None,
        explore_algorithm_class: type[OffPolicyAlgorithm] | None = None,
        explore_algorithm_kwargs: Dict[str, Any] | None = None,
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
            sub_algorithm_class,
            sub_algorithm_kwargs,
            explore_algorithm_class,
            explore_algorithm_kwargs,
        )
        self.state_storage = CemrlStateStorage()

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # copied from SAC._setup_model
        def sac_setup(sac: SAC):
            sac._create_aliases()
            sac.batch_norm_stats = get_parameters_by_name(sac.critic, ["running_"])
            sac.batch_norm_stats_target = get_parameters_by_name(sac.critic_target, ["running_"])
            if sac.target_entropy == "auto":
                sac.target_entropy = float(-np.prod(sac.env.action_space.shape).astype(np.float32))  # type: ignore
            else:
                sac.target_entropy = float(sac.target_entropy)

            if isinstance(sac.ent_coef, str) and sac.ent_coef.startswith("auto"):
                init_value = 1.0
                if "_" in sac.ent_coef:
                    init_value = float(sac.ent_coef.split("_")[1])
                    assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

                sac.log_ent_coef = th.log(th.ones(1, device=sac.device) * init_value).requires_grad_(True)
                sac.ent_coef_optimizer = th.optim.Adam([sac.log_ent_coef], lr=sac.lr_schedule(1))
            else:
                sac.ent_coef_tensor = th.tensor(float(sac.ent_coef), device=sac.device)

        super()._setup_model()
        if isinstance(self.sub_algorithm, SAC):
            sac_setup(self.sub_algorithm)
        self.replay_buffer.task_encoder = self.policy.task_inference
        self.config = self.policy.config.training

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

        super().train(gradient_steps, batch_size)


class CemrlStateStorage(StateStorage):
    def _on_step(self) -> bool:
        self.policy_state = self.policy.obs_to_tensor(self.model._last_obs)[0]["observation"], self.policy_state[1]  # type: ignore
        return super()._on_step()
