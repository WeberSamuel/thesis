"""This file contains the CEMRL algorithm for stable-baselines3."""
from typing import List, Optional, Tuple, Union, cast
from stable_baselines3 import SAC
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import unwrap_vec_wrapper, VecEnv, VecEnvWrapper
from stable_baselines3.common.type_aliases import Schedule
from src.cemrl.networks import Encoder
from src.cli import DummyPolicy, DummyReplayBuffer
from src.cemrl.extensions import CEMRLExtension
from src.cemrl.buffers import CEMRLPolicyBuffer
from src.cemrl.policies import CEMRLPolicy
from src.cemrl.types import CEMRLObsTensorDict
from src.cemrl.wrappers import CEMRLPolicyWrapper, CEMRLHistoryWrapper
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
import torch as th
from src.utils import get_random_encoder_window_samples
from src.cemrl.buffers import CEMRLBuffer, CEMRLPolicyBuffer


class CEMRLSACPolicy(SAC):
    def __init__(
        self,
        env: VecEnv | VecEnvWrapper,
        cemrl_policy_encoder: Encoder,
        cemrl_replay_buffer: CEMRLBuffer,
        encoder_window: int,
        **kwargs
    ):
        super().__init__("MultiInputPolicy", CEMRLPolicyWrapper(env, cemrl_policy_encoder.latent_dim), buffer_size=0, **kwargs)

        self.replay_buffer = CEMRLPolicyBuffer(env, cemrl_policy_encoder, cemrl_replay_buffer, encoder_window)


class CEMRL(OffPolicyAlgorithm):
    """CEMRL algorithm."""

    def __init__(
        self,
        policy: CEMRLPolicy,
        policy_algorithm: OffPolicyAlgorithm,
        env: GymEnv | VecEnv | VecEnvWrapper,
        replay_buffer: ReplayBuffer,
        encoder_window: int = 30,
        encoder_gradient_steps: float | Schedule = 0.5,
        policy_gradient_steps: float | Schedule = 0.5,
        extension: Optional[CEMRLExtension] = None,
        learning_starts: int = 100,
        batch_size: int = 256,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        learning_rate: float | Schedule = 1e-3,
        _init_setup_model=True,
    ):
        super().__init__(
            DummyPolicy,
            env,
            learning_rate,
            buffer_size=0,
            learning_starts=learning_starts,
            batch_size=batch_size,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=DummyReplayBuffer,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            sde_support=sde_support,
            support_multi_env=True,
        )
        assert self.env is not None
        cemrl_wrapper = unwrap_vec_wrapper(self.env, CEMRLHistoryWrapper)
        assert isinstance(
            cemrl_wrapper, CEMRLHistoryWrapper
        ), "CEMRL requires the usage of the CEMRLVecEnvWrapper, such that the decoder / encoder has access to earlier data"
        self.encoder_gradient_steps = get_schedule_fn(encoder_gradient_steps)
        self.policy_gradient_steps = get_schedule_fn(policy_gradient_steps)
        self.extension = extension if extension is not None else CEMRLExtension()
        self.encoder_window = encoder_window
        self.scaler = th.cuda.amp.GradScaler()
        self.replay_buffer = replay_buffer
        if _init_setup_model:
            self._setup_model()
        self.policy = policy
        self.policy._build(policy_algorithm)
        self.policy.to(self.device)

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        result = super()._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar)

        # replay_buffer may have changed --> update policy buffer as well
        assert isinstance(self.policy, CEMRLPolicy)
        assert isinstance(self.policy.policy_algorithm.replay_buffer, CEMRLPolicyBuffer)
        assert isinstance(self.replay_buffer, DictReplayBuffer)
        self.policy.policy_algorithm.replay_buffer.cemrl_replay_buffer = self.replay_buffer
        self.policy.policy_algorithm.set_logger(self.logger)

        self.extension._init_extension(self)

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
        assert isinstance(self.policy, CEMRLPolicy)
        encoder_steps, policy_steps = self._get_gradient_steps(gradient_steps)

        for j in range(encoder_steps):
            self.reconstruction_training_step(batch_size, 20)
        self.policy.policy_algorithm.train(policy_steps, batch_size)

    def _get_gradient_steps(self, gradient_steps: int):
        encoder_steps_factor = self.encoder_gradient_steps(self._current_progress_remaining)
        policy_steps_factor = self.policy_gradient_steps(self._current_progress_remaining)
        normalization_factor = encoder_steps_factor + policy_steps_factor

        encoder_steps_factor = encoder_steps_factor / normalization_factor
        policy_steps_factor = policy_steps_factor / normalization_factor

        encoder_steps = int(gradient_steps * encoder_steps_factor)
        policy_steps = int(gradient_steps * policy_steps_factor)

        self.logger.record("encoder_steps", encoder_steps)
        self.logger.record("policy_steps", policy_steps)
        return encoder_steps, policy_steps

    def reconstruction_training_step(self, batch_size: int, reuse_steps: int):
        """Perform a training step for the encoder and decoder.

        The overall objective due to the generative model is:
        parameter* = arg max ELBO
        ELBO = sum_k q(y=k | x) *     [log p(x|z_k)             - KL( q(z, x,y=k)    || p(z|y=k))]    -     KL(q(y | x)|| p(y))
        ELBO ≈ sum_k log q(y = k | c) [log p(x_out | x_in, z_k) - a  KL( q(z | c, y = k) ∥ p(z | c, y))] - β * KL(q(y | c) ∥ p(y | c))
        Args:
            batch_size (int): Size of the batches to sample from the replay buffer
        """
        assert isinstance(self.policy, CEMRLPolicy)
        assert isinstance(self.replay_buffer, DictReplayBuffer)

        self.policy.set_training_mode(True)
        (
            observations,
            actions,
            next_observations,
            dones,
            rewards,
        ) = samples = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        batch_size = len(actions)  # buffer may return different number of batches
        observations = cast(CEMRLObsTensorDict, observations)
        next_observations = cast(CEMRLObsTensorDict, next_observations)
        observations = observations["observation"]
        next_observations = next_observations["observation"]

        # Forward pass through encoder
        for i in range(reuse_steps):
            with th.autocast("cuda"):
                encoder_input = get_random_encoder_window_samples(samples, self.encoder_window)
                y_distribution, z_distributions = self.policy.encoder.encode(
                    encoder_input.observations, encoder_input.actions, encoder_input.rewards, encoder_input.next_observations
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
                    z = z.unsqueeze(1).repeat(1, observations.shape[1], 1)
                    # put in decoder to get likelihood
                    state_estimate, reward_estimate = self.policy.decoder(
                        observations, actions, next_observations, z, return_raw=True
                    )

                    state_vars += th.var(state_estimate, dim=0).sum().item()
                    reward_vars += th.var(reward_estimate, dim=0).sum().item()

                    reward_loss = th.sum((reward_estimate - rewards[None].expand(reward_estimate.shape)) ** 2, dim=-1)
                    reward_loss = th.mean(reward_loss, dim=-1)
                    reward_loss = th.mean(reward_loss, dim=0)
                    reward_losses[:, y] = reward_loss

                    state_loss = th.sum((state_estimate - next_observations[None].expand(state_estimate.shape)) ** 2, dim=-1)
                    state_loss = th.mean(state_loss, dim=-1)
                    state_loss = th.mean(state_loss, dim=0)
                    state_losses[:, y] = state_loss

                    # p(x|z_k)
                    nll_px[:, y] = 0.3333 * state_loss + 0.6666 * reward_loss

                    # KL ( q(z | x,y=k) || p(z|y=k))
                    ones = th.ones(batch_size, self.policy.latent_dim, device=self.device)
                    prior_pz = th.distributions.normal.Normal(ones * y, ones * 0.5)
                    kl_qz_pz[:, y] = th.sum(th.distributions.kl.kl_divergence(z_distributions[y], prior_pz), dim=-1)
                    self.extension.after_reconstruction_class_loss(locals())

                # KL ( q(y | x) || p(y) )
                ones = th.ones(batch_size, self.policy.num_classes, device=self.device)
                prior_py = th.distributions.categorical.Categorical(probs=ones * (1.0 / self.policy.num_classes))
                kl_qy_py = th.distributions.kl.kl_divergence(y_distribution, prior_py)

                alpha_kl_z = 1e-3  # weighting factor KL loss of z distribution vs prior
                beta_kl_y = 1e-3  # weighting factor KL loss of y distribution vs prior
                y_dist_probs = cast(th.Tensor, y_distribution.probs)
                elbo = th.sum(th.sum(th.mul(y_dist_probs, -nll_px - alpha_kl_z * kl_qz_pz), dim=-1) - beta_kl_y * kl_qy_py)
                loss = -elbo
                self.extension.after_reconstruction_loss_calculation(locals())
                # loss = encoder_loss

            self.policy.reconstruction_optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            self.scaler.scale(loss).backward()
            self.extension.after_reconstruction_backward(locals())

            # Calling the step function on an Optimizer makes an update to its parameters
            self.scaler.step(self.policy.reconstruction_optimizer)
            self.scaler.update()

            loss = loss / batch_size
            state_loss = th.mean(state_losses)
            reward_loss = th.mean(reward_losses)

            self.extension.after_reconstruction_step(locals())

            self.logger.record_mean("reconstruction/loss", loss.item())
            self.logger.record_mean("reconstruction/state_loss", state_loss.item())
            self.logger.record_mean("reconstruction/state_var", state_vars / self.policy.num_classes)
            self.logger.record_mean("reconstruction/reward_loss", reward_loss.item())
            self.logger.record_mean("reconstruction/reward_var", reward_vars / self.policy.num_classes)

        return loss.item()

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["extension"]
