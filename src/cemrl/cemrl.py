"""This file contains the CEMRL algorithm for stable-baselines3."""
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from src.cemrl.trainer import train_encoder
from src.cemrl.buffers2 import CemrlReplayBuffer
from src.cemrl.buffers1 import EpisodicReplayBuffer
from src.cemrl.buffers import CEMRLReplayBuffer
from src.cemrl.policies import CEMRLPolicy
from src.cli import DummyPolicy
from src.core.state_aware_algorithm import StateAwareOffPolicyAlgorithm


class CEMRL(StateAwareOffPolicyAlgorithm):
    """CEMRL algorithm."""

    def __init__(
        self,
        policy: CEMRLPolicy,
        env: Union[GymEnv, str],
        decoder_samples: int = 400,
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000, # 20000,
        learning_starts: int = 1000,
        batch_size: int = 256,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        encoder_grad_steps: int = 40,
        policy_grad_steps: int = 10,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[EpisodicReplayBuffer|CEMRLReplayBuffer|CemrlReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        _init_setup_model=True,
    ):
        super().__init__(
            DummyPolicy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            train_freq=train_freq,
            gradient_steps=1,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class or EpisodicReplayBuffer,
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
        )
        self.decoder_samples = decoder_samples
        self.encoder_grad_steps = encoder_grad_steps
        self.policy_grad_steps = policy_grad_steps

        self.replay_buffer_kwargs["encoder"] = policy.encoder
        if _init_setup_model:
            self._setup_model()
        self.policy: CEMRLPolicy = policy.to(self.device)
        self.policy.sub_policy_algorithm.replay_buffer = self.replay_buffer
        self.replay_buffer: EpisodicReplayBuffer|CEMRLReplayBuffer|CemrlReplayBuffer

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        result = super()._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar)
        self.policy.sub_policy_algorithm.replay_buffer = self.replay_buffer
        self.policy.sub_policy_algorithm.set_logger(self.logger)

        return result

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """Train the model ``gradient_steps`` times.

        First the encoder and decoder is trained via the CEMRL ELBO ``num_encoder_gradient_steps`` times.
        Second is the policy training with ``num_policy_gradient_steps```updates.
        Finally the exploration method is trained ``num_exploration_gradient_steps`` times.

        Args:
            gradient_steps (int): How often the training should be applied
            batch_size (int): Batch size used in the training
        """
        for _ in range(self.encoder_grad_steps):
            loss, state_loss, reward_loss = train_encoder(self.policy.encoder, self.policy.decoder, *self.replay_buffer.cemrl_sample(batch_size, self.get_vec_normalize_env()), self.policy.optimizer)

            self.logger.record_mean("reconstruction/loss", loss.item())
            self.logger.record_mean("reconstruction/state_loss", state_loss.item())
            self.logger.record_mean("reconstruction/reward_loss", reward_loss.item())

        self.policy.sub_policy_algorithm.train(self.policy_grad_steps, batch_size)

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["extension"]
