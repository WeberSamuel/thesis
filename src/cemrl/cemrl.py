"""This file contains the CEMRL algorithm for stable-baselines3."""
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
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
        replay_buffer_class: Optional[Type[EpisodicReplayBuffer|CEMRLReplayBuffer]] = None,
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
        self.replay_buffer: EpisodicReplayBuffer|CEMRLReplayBuffer

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
        for j in range(self.encoder_grad_steps):
            self.reconstruction_training_step(batch_size)
        self.policy.sub_policy_algorithm.train(self.policy_grad_steps, batch_size)

    def reconstruction_training_step(self, batch_size: int):
        """Perform a training step for the encoder and decoder.

        The overall objective due to the generative model is:
        parameter* = arg max ELBO
        ELBO = sum_k q(y=k | x) *     [log p(x|z_k)             - KL( q(z, x,y=k)    || p(z|y=k))]    -     KL(q(y | x)|| p(y))
        ELBO ≈ sum_k log q(y = k | c) [log p(x_out | x_in, z_k) - a  KL( q(z | c, y = k) ∥ p(z | c, y))] - β * KL(q(y | c) ∥ p(y | c))
        Args:
            batch_size (int): Size of the batches to sample from the replay buffer
        """
        assert isinstance(self.policy, CEMRLPolicy)
        assert isinstance(self.replay_buffer, CEMRLReplayBuffer|EpisodicReplayBuffer)

        self.policy.set_training_mode(True)
        
        enc_input, dec_input = self.replay_buffer.cemrl_sample(batch_size, self._vec_normalize_env, self.policy.encoder_window, self.decoder_samples)
        
        dec_observations = dec_input.observations["observation"]
        dec_next_observations = dec_input.next_observations["observation"]

        y_distribution, z_distributions = self.policy.encoder.encode(
            enc_input.observations["observation"],
            enc_input.actions,
            enc_input.rewards,
            enc_input.next_observations["observation"],
        )

        kl_qz_pz = th.zeros(batch_size, self.policy.num_classes, device=self.device)
        state_losses = th.zeros(batch_size, self.policy.num_classes, device=self.device)
        state_vars = 0.0
        reward_vars = 0.0
        reward_losses = th.zeros(batch_size, self.policy.num_classes, device=self.device)
        nll_px = th.zeros(batch_size, self.policy.num_classes, device=self.device)

        # every y component (see ELBO formula)
        for y in range(self.policy.num_classes):
            _, z = self.policy.encoder.sample(y_distribution, z_distributions, y=y)
            z = z.unsqueeze(1).repeat(1, dec_observations.shape[1], 1)
            # put in decoder to get likelihood
            state_estimate, reward_estimate = self.policy.decoder(
                dec_observations, dec_input.actions, dec_next_observations, z, return_raw=True
            )

            # state_vars += th.var(state_estimate, dim=0).sum().item()
            # reward_vars += th.var(reward_estimate, dim=0).sum().item()

            reward_loss = th.sum((reward_estimate - dec_input.rewards[None].expand(reward_estimate.shape)) ** 2, dim=-1)
            reward_loss = th.mean(reward_loss, dim=-1)
            reward_loss = th.mean(reward_loss, dim=0)
            reward_losses[:, y] = reward_loss

            state_loss = th.sum((state_estimate - dec_next_observations[None].expand(state_estimate.shape)) ** 2, dim=-1)
            state_loss = th.mean(state_loss, dim=-1)
            state_loss = th.mean(state_loss, dim=0)
            state_losses[:, y] = state_loss

            # p(x|z_k)
            nll_px[:, y] = 0.3333 * state_loss + 0.6666 * reward_loss

            # KL ( q(z | x,y=k) || p(z|y=k))
            ones = th.ones(batch_size, self.policy.latent_dim, device=self.device)
            prior_pz = th.distributions.normal.Normal(ones * y, ones * 0.5)
            kl_qz_pz[:, y] = th.sum(th.distributions.kl.kl_divergence(z_distributions[y], prior_pz), dim=-1)

        # KL ( q(y | x) || p(y) )
        ones = th.ones(batch_size, self.policy.num_classes, device=self.device)
        prior_py = th.distributions.categorical.Categorical(probs=ones * (1.0 / self.policy.num_classes))
        kl_qy_py = th.distributions.kl.kl_divergence(y_distribution, prior_py)

        alpha_kl_z = 1e-3  # weighting factor KL loss of z distribution vs prior
        beta_kl_y = 1e-3  # weighting factor KL loss of y distribution vs prior
        y_dist_probs = cast(th.Tensor, y_distribution.probs)
        elbo = th.sum(th.sum(th.mul(y_dist_probs, -nll_px - alpha_kl_z * kl_qz_pz), dim=-1) - beta_kl_y * kl_qy_py)
        loss = -elbo
        # loss = encoder_loss

        self.policy.optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        # self.scaler.scale(loss).backward()  # type: ignore
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        # self.scaler.step(self.policy.optimizer)
        self.policy.optimizer.step()
        # self.scaler.update()

        loss = loss / batch_size
        state_loss = th.mean(state_losses)
        reward_loss = th.mean(reward_losses)

        self.logger.record_mean("reconstruction/loss", loss.item())
        self.logger.record_mean("reconstruction/state_loss", state_loss.item())
        # self.logger.record_mean("reconstruction/state_var", state_vars / self.policy.num_classes)
        self.logger.record_mean("reconstruction/reward_loss", reward_loss.item())
        # self.logger.record_mean("reconstruction/reward_var", reward_vars / self.policy.num_classes)

        return loss.item()

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["extension"]
