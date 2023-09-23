"""This file contains the CEMRL algorithm for stable-baselines3."""
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from src.cemrl.buffers import CEMRLReplayBuffer
from src.cemrl.policies import CEMRLPolicy
from src.core.has_sub_algorithm import HasSubAlgorithm
from src.core.state_aware_algorithm import StateAwareOffPolicyAlgorithm
from .config import CemrlConfig
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.sac import SAC
from src.cemrl.wrappers.cemrl_policy_wrapper import CEMRLPolicyVecWrapper
from .buffers import ImagineBuffer


class CEMRL(HasSubAlgorithm, StateAwareOffPolicyAlgorithm):
    """CEMRL algorithm."""

    replay_buffer: CEMRLReplayBuffer
    policy: CEMRLPolicy
    env: VecEnv

    def __init__(
        self,
        policy: Type[CEMRLPolicy],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 20000,
        learning_starts: int = 1000,
        batch_size: int = 256,
        train_freq: Union[int, Tuple[int, str]] = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[CEMRLReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        gradient_steps=1,
        sub_algorithm_class: Type[OffPolicyAlgorithm] = SAC,
        sub_algorithm_kwargs: Dict[str, Any] | None = None,
        _init_setup_model=True,
        **kwargs,
    ):
        super().__init__(
            policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class or CEMRLReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            support_multi_env=True,
            sde_support=False,
            sub_algorithm_class=sub_algorithm_class,
            sub_algorithm_kwargs=sub_algorithm_kwargs,
            **kwargs,
        )
        self.config: CemrlConfig = self.policy_kwargs.get("config", CemrlConfig())

        self.replay_buffer_kwargs.setdefault("encoder_window", self.config.training.encoder_context_length)
        self.replay_buffer_kwargs.setdefault("num_decoder_targets", self.config.training.num_decoder_targets)

        if _init_setup_model:
            self._setup_model()

        self.replay_buffer.task_inference = self.policy.task_inference

    def _setup_sub_algorithm(self):
        latent_dim = self.config.task_inference.encoder.latent_dim
        wrapper = CEMRLPolicyVecWrapper(self.env, latent_dim)
        self.sub_algorithm = self.sub_algorithm_class(
            "MultiInputPolicy", wrapper, buffer_size=0, **self.sub_algorithm_kwargs
        )

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """Train the model ``gradient_steps`` times.

        First the encoder and decoder is trained via the CEMRL ELBO ``num_encoder_gradient_steps`` times.
        Second is the policy training with ``num_policy_gradient_steps```updates.
        Finally the exploration method is trained ``num_exploration_gradient_steps`` times.

        Args:
            gradient_steps (int): How often the training should be applied
            batch_size (int): Batch size used in the training
        """
        self.policy.set_training_mode(True)
        for _ in range(gradient_steps):
            for _ in range(self.config.training.task_inference_gradient_steps):
                metrics = self.policy.task_inference.training_step(
                    *self.replay_buffer.cemrl_sample(batch_size, self.get_vec_normalize_env())
                )

                for k, v in metrics.items():
                    self.logger.record_mean("reconstruction/" + k, v)

            if self.config.training.imagination_horizon == 0:
                self.sub_algorithm.replay_buffer = self.replay_buffer
            else:
                self.sub_algorithm.replay_buffer = ImagineBuffer(
                    self.config.training.imagination_horizon,
                    self.policy,
                    self.replay_buffer,
                    self.policy.task_inference.decoder.ensemble[0],  # using only one ensemble member for now
                    self.config.training.policy_gradient_steps,
                    self.sub_algorithm.batch_size,
                    self.action_space,  # type: ignore
                    self.get_vec_normalize_env(),
                )  # type: ignore
            self.sub_algorithm.train(self.config.training.policy_gradient_steps, batch_size)
        self.dump_logs_if_neccessary()

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["sub_algorithm"]
