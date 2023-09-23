from typing import NamedTuple

import torch as th
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from torch.distributions import Categorical, Normal, kl

from src.plan2explore.networks import Ensemble

from ..utils import build_network, DeviceAwareModuleMixin
from .config import DecoderConfig, EncoderConfig, TaskInferenceConfig

REWARD_DIM = 1


def process_gaussian_parameters(mu_sigma, latent_dim):
    mus, sigmas = th.split(mu_sigma, split_size_or_sections=latent_dim, dim=-1)
    sigmas = th.nn.functional.softplus(sigmas)
    return th.cat([mus, sigmas], dim=-1)


class StatePreprocessor(th.nn.Module):
    def __init__(
        self, state_dim: int, state_preprocessing_dim: int, net_complex_enc_dec: float, simplified_state_preprocessor: bool
    ):
        super().__init__()
        self.output_dim = state_preprocessing_dim if state_preprocessing_dim != 0 else state_dim
        if state_preprocessing_dim != 0:
            hidden_dim = int(net_complex_enc_dec * state_dim)
            if simplified_state_preprocessor:
                self.layers = build_network(state_dim, [], th.nn.Identity, state_preprocessing_dim)
            else:
                self.layers = build_network(state_dim, [hidden_dim], th.nn.ReLU, state_preprocessing_dim)
        else:
            self.layers = th.nn.Identity()

    def forward(self, m):
        return self.layers(m)


class EncoderInput(NamedTuple):
    obs: th.Tensor
    action: th.Tensor
    reward: th.Tensor
    next_obs: th.Tensor


class Encoder(DeviceAwareModuleMixin, th.nn.Module):
    def __init__(self, state_size: int, action_size: int, config: EncoderConfig):
        super().__init__()
        self.config = config

        self.state_preprocessor = StatePreprocessor(
            state_size, config.preprocessed_state_size, config.complexity, config.use_simplified_state_preprocessor
        )

        encoder_input_dim = 2 * self.state_preprocessor.output_dim + action_size + REWARD_DIM
        shared_dim = int(config.complexity * encoder_input_dim)
        self.shared_encoder = th.nn.GRU(input_size=encoder_input_dim, hidden_size=shared_dim, batch_first=True)

        self.class_encoder = th.nn.Sequential(th.nn.Linear(shared_dim, config.num_classes), th.nn.Softmax(dim=-1))

        self.gauss_encoder_list = th.nn.ModuleList(
            [th.nn.Linear(shared_dim, config.latent_dim * 2) for _ in range(config.num_classes)]
        )

    def forward(self, x: EncoderInput, encoder_state=None, return_distributions=False):
        """
        Encode the provided context
        :param x: a context list containing [obs, action, reward, next_obs], respectively
                of the form (batch_size, time_steps, <obs/action/reward dim>)
        :param return_distributions: If true, also return the distribution objects, not just a sampled data point
        :return: z - task indicator [batch_size, latent_dim]
                 y - base task indicator [batch_size]
        """
        y_distribution, z_distributions, encoder_state = self.encode(x, encoder_state=encoder_state)
        y, z = self.sample_z(y_distribution, z_distributions, y_usage="most_likely", sampler="mean")
        if return_distributions:
            distribution = {
                "y_probs": y_distribution.probs.detach().cpu().numpy().squeeze(),  # type: ignore
                "z_means": [z.loc.detach().cpu().numpy().squeeze() for z in z_distributions],
                "z_stds": [z.scale.detach().cpu().numpy().squeeze() for z in z_distributions],
            }
            return y, z, distribution, encoder_state
        else:
            return y, z, encoder_state

    def encode(self, x: EncoderInput, encoder_state=None):
        # Compute shared encoder forward pass
        # Just encode everything at once; if necessary, the shared_encoder must take the padding_mask into account.
        cat_x = th.cat([self.state_preprocessor(x.obs), x.action, x.reward, self.state_preprocessor(x.next_obs)], dim=-1)
        _, encoder_state = self.shared_encoder(cat_x, encoder_state)
        m = encoder_state.squeeze(dim=0)

        # Compute class probabilities
        y = self.class_encoder(m)

        # Compute every gauss_encoder forward pass
        all_mu_sigma = []
        for net in self.gauss_encoder_list:
            all_mu_sigma.append(net(m))

        final_mu_sigma = [process_gaussian_parameters(mu_sigma, self.config.latent_dim) for mu_sigma in all_mu_sigma]

        # Construct the categorical and Normal distributions
        y_distribution = Categorical(probs=y)
        z_distributions = [
            Normal(*th.split(final_mu_sigma[i], split_size_or_sections=self.config.latent_dim, dim=-1))
            for i in range(self.config.num_classes)
        ]
        return y_distribution, z_distributions, encoder_state

    def sample_z(self, y_distribution, z_distributions, y_usage="specific", y=None, sampler="random"):
        """
        Sample from the latent Gaussian mixture model

        :param y_distribution: Categorical distribution of the classes
        :param z_distributions: List of Gaussian distributions
        :param y_usage: 'most_likely' to sample from the most likely class per batch
                'specific' to sample from the class specified in param y for all batches
        :param y: class to sample from if y_usage=='specific'
        :param sampler: 'random' for actual sampling, 'mean' to return the mean of the Gaussian
        :return: z - task indicator [batch_size, latent_dim]
                 y - base task indicator [batch_size]
        """
        batch_size = y_distribution.probs.shape[0]
        # Select from which Gaussian to sample
        # Used for individual sampling when computing ELBO
        if y_usage == "specific":
            y = th.ones(batch_size, dtype=th.long, device=self.device) * y
        # Used while inference
        elif y_usage == "most_likely":
            y = th.argmax(y_distribution.probs, dim=1)
        else:
            raise RuntimeError("Sampling strategy not specified correctly")

        mask = y.view(-1, 1).unsqueeze(2).repeat(1, 1, self.config.latent_dim)

        if sampler == "random":
            # Sample from specified Gaussian using reparametrization trick
            # this operation samples from each Gaussian for every class first
            # (diag_embed not possible for distributions), put it back to tensor with shape [class, batch, latent]
            sampled = th.cat([th.unsqueeze(z_distributions[i].rsample(), 0) for i in range(self.config.num_classes)], dim=0)

        elif sampler == "mean":
            sampled = th.cat([th.unsqueeze(z_distributions[i].mean, 0) for i in range(self.config.num_classes)], dim=0)
        else:
            raise RuntimeError("Sampler not specified correctly")

        # tensor with shape [batch, class, latent]
        permute = sampled.permute(1, 0, 2)
        z = th.squeeze(th.gather(permute, 1, mask), 1)
        return y, z


class Decoder(th.nn.Module):
    """
    Uses data (state, action, reward, task_hypothesis z) from the replay buffer or online
    and computes estimates for the next state and reward.
    Through that it reconstructs the MDP and gives gradients back to the task hypothesis.
    """

    def __init__(
        self, state_size: int, action_size: int, z_size: int, state_preprocessor: StatePreprocessor, config: DecoderConfig
    ):
        super().__init__()

        self.config = config
        self.state_preprocessor = state_preprocessor

        self.state_decoder_input_size = self.state_preprocessor.output_dim + action_size + z_size

        self.reward_decoder_input_size = self.state_preprocessor.output_dim + action_size + z_size
        if config.use_next_state_for_reward:
            self.reward_decoder_input_size += self.state_preprocessor.output_dim

        if config.use_state_decoder:
            self.net_state_decoder = build_network(
                input_size=self.state_decoder_input_size,
                layers=[int(self.state_decoder_input_size * config.complexity)] * config.num_layers,
                output_size=state_size,
                activation=config.activation,
            )
        else:
            self.net_state_decoder = None

        self.net_reward_decoder = build_network(
            input_size=self.reward_decoder_input_size,
            layers=[int(self.reward_decoder_input_size * config.complexity)] * config.num_layers,
            output_size=REWARD_DIM,
            activation=config.activation,
        )

    def forward(self, state:th.Tensor, action:th.Tensor, next_state:th.Tensor|None, z:th.Tensor):
        state = self.state_preprocessor(state)
        if self.config.use_state_decoder:
            assert self.net_state_decoder is not None
            state_estimate = self.net_state_decoder(th.cat([state, action, z], dim=-1))
        else:
            state_estimate = None

        if self.config.use_next_state_for_reward:
            if next_state is not None:
                next_state = self.state_preprocessor(next_state)
            elif state_estimate is not None:
                next_state = state_estimate
            else:
                raise RuntimeError("No next state available, but use next state for reward is set")
            reward_estimate = self.net_reward_decoder(th.cat([state, action, next_state, z], dim=-1)) # type: ignore
        else:
            reward_estimate = self.net_reward_decoder(th.cat([state, action, z], dim=-1))

        return state_estimate, reward_estimate


class TaskInference(DeviceAwareModuleMixin, th.nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: TaskInferenceConfig,
        decoder: Ensemble | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(state_size=obs_dim, action_size=action_dim, config=config.encoder)
        self.decoder = decoder or Ensemble(
            th.nn.ModuleList(
                [
                    Decoder(
                        state_size=obs_dim,
                        action_size=action_dim,
                        z_size=config.encoder.latent_dim,
                        state_preprocessor=self.encoder.state_preprocessor,
                        config=config.decoder,
                    )
                    for i in range(self.config.decoder.ensemble_size)
                ]
            )
        )
        self.prior_pz_layer = th.nn.Linear(config.encoder.num_classes, config.encoder.latent_dim * 2)

        self.encoder_optimizer = config.training.optimizer_class(
            list(self.encoder.parameters()) + list(self.prior_pz_layer.parameters()), lr=config.encoder.lr  # type: ignore
        )
        self.decoder_optimizer = config.training.optimizer_class(self.decoder.parameters(), lr=config.decoder.lr)  # type: ignore
        self.reward_loss_factor = obs_dim / (REWARD_DIM + obs_dim)
        self.state_loss_factor = REWARD_DIM / (REWARD_DIM + obs_dim)

    def forward(self, encoder_obs: EncoderInput, encoder_state: th.Tensor | None = None):
        return self.encoder(encoder_obs, encoder_state)

    def get_init_state(self, batch_size: int):
        return th.zeros(batch_size, 1, self.encoder.shared_encoder.hidden_size, device=self.device)

    def training_step(
        self,
        encoder_context: ReplayBufferSamples,
        decoder_context: ReplayBufferSamples,
    ):
        config = self.config.training
        batch_size = encoder_context.rewards.shape[0]

        decoder_action = decoder_context.actions
        decoder_state = decoder_context.observations
        decoder_next_state = decoder_context.next_observations
        reward_target = decoder_context.rewards

        if not config.reconstruct_all_steps:
            # Reconstruct only the current timestep
            decoder_action = decoder_action[:, -1, :]
            decoder_state = decoder_state[:, -1, :]
            decoder_next_state = decoder_next_state[:, -1, :]
            reward_target = reward_target[:, -1, :]

        if config.use_state_diff:
            state_target = decoder_next_state - decoder_state
        else:
            state_target = decoder_next_state

        # Forward pass through encoder
        enc_input = EncoderInput(
            obs=encoder_context.observations,
            action=encoder_context.actions,
            reward=encoder_context.rewards,
            next_obs=encoder_context.next_observations,
        )
        y_distribution, z_distributions, _ = self.encoder.encode(enc_input)
        if config.alpha_kl_z_query is not None:
            y_distribution_query, z_distributions_query, _ = self.encoder.encode(
                EncoderInput(
                    obs=decoder_context.observations,
                    action=decoder_context.actions,
                    reward=decoder_context.rewards,
                    next_obs=decoder_context.next_observations,
                )
            )
            kl_qz_qz_query = th.zeros(batch_size, self.config.encoder.num_classes, device=self.device)
        else:
            y_distribution_query, z_distributions_query = None, None
            kl_qz_qz_query = None

        kl_qz_pz = th.zeros(batch_size, self.config.encoder.num_classes, device=self.device)
        state_losses = th.zeros(batch_size, self.config.encoder.num_classes, device=self.device)
        reward_losses = th.zeros(batch_size, self.config.encoder.num_classes, device=self.device)
        nll_px = th.zeros(batch_size, self.config.encoder.num_classes, device=self.device)

        # every y component (see ELBO formula)
        for y in range(self.config.encoder.num_classes):
            _, z = self.encoder.sample_z(y_distribution, z_distributions, y_usage="specific", y=y)
            if config.reconstruct_all_steps:
                z = z.unsqueeze(1).repeat(1, decoder_state.shape[1], 1)

            # put in decoder to get likelihood
            state_estimate, reward_estimate = self.decoder(decoder_state, decoder_action, decoder_next_state, z, return_raw=True)
            reward_loss = th.sum((reward_estimate - reward_target[None]) ** 2, dim=-1)
            reward_loss = th.mean(reward_loss, dim=0)
            if config.reconstruct_all_steps:
                reward_loss = th.mean(reward_loss, dim=1)
            reward_losses[:, y] = reward_loss

            if self.config.decoder.use_state_decoder:
                state_loss = th.sum((state_estimate - state_target[None]) ** 2, dim=-1)
                state_loss = th.mean(state_loss, dim=0)
                if config.reconstruct_all_steps:
                    state_loss = th.mean(state_loss, dim=1)
                state_losses[:, y] = state_loss
                nll_px[:, y] = self.state_loss_factor * state_loss + self.reward_loss_factor * reward_loss
            else:
                nll_px[:, y] = self.reward_loss_factor * reward_loss

            # KL ( q(z | x,y=k) || p(z|y=k) )
            prior = self.prior_pz(batch_size, y)
            kl_qz_pz[:, y] = th.sum(kl.kl_divergence(z_distributions[y], prior), dim=-1)
            # KL ( q(z | x_decoder) || q(z | x_encoder) )
            if config.alpha_kl_z_query is not None:
                assert kl_qz_qz_query is not None and z_distributions_query is not None
                kl_qz_qz_query[:, y] = th.sum(kl.kl_divergence(z_distributions_query[y], z_distributions[y]), dim=-1)

        # KL ( q(y | x) || p(y) )
        kl_qy_py = kl.kl_divergence(y_distribution, self.prior_py(batch_size))
        # KL ( q(y | x_decoder) || q(y | x_encoder) )
        if config.alpha_kl_z_query is not None:
            assert y_distribution_query is not None
            kl_qy_qy_query = kl.kl_divergence(y_distribution_query, y_distribution)
        else:
            kl_qy_qy_query = None

        y_probs: th.Tensor = y_distribution.probs  # type: ignore

        elbo = th.sum(
            th.sum(th.mul(y_probs, (-1) * nll_px - config.alpha_kl_z * kl_qz_pz), dim=-1) - config.beta_kl_y * kl_qy_py
        )
        if config.alpha_kl_z_query is not None:
            assert kl_qz_qz_query is not None and kl_qy_qy_query is not None
            elbo += th.sum(
                th.sum(th.mul(y_probs, -config.alpha_kl_z_query * kl_qz_qz_query), dim=-1)
                - config.beta_kl_y_query * kl_qy_qy_query
            )

        # but elbo should be maximized, and backward function assumes minimization
        loss = (-1) * elbo

        # Optimization strategy:
        # Decoder: the two head loss functions backpropagate their gradients into corresponding parts
        # of the network, then ONE common optimizer compute all weight updates
        # Encoder: the KLs and the likelihood from the decoder backpropagate their gradients into
        # corresponding parts of the network, then ONE common optimizer computes all weight updates
        # This is not done explicitly but all within the elbo loss.

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        self.decoder_optimizer.step()
        self.encoder_optimizer.step()

        return {
            "loss": loss.item() / batch_size,
            "kl_qz_pz": th.sum(kl_qz_pz, dim=0).item() / batch_size,
            "kl_qy_py": kl_qy_py.sum().item() / batch_size,
            "nll_px": th.sum(nll_px, dim=0).item() / batch_size,
            "reward_loss": th.sum(reward_losses, dim=0).item() / batch_size,
            "state_loss": th.sum(state_losses, dim=0).item() / batch_size,
        }

    def prior_pz(self, batch_size: int, y: int):
        """
        As proposed in the CURL paper: use linear layer, that conditioned on y gives Gaussian parameters
        OR
        Gaussian with N(y, 0.5)
        IF z not used:
        Just give back y with 0.01 variance
        """

        if self.config.training.prior_mode == "fixedOnY":
            return Normal(
                th.ones(batch_size, self.config.encoder.latent_dim, device=self.device) * y,
                th.ones(batch_size, self.config.encoder.latent_dim, device=self.device) * self.config.training.prior_sigma,
            )

        elif self.config.training.prior_mode == "network":
            one_hot = th.zeros(batch_size, self.config.encoder.num_classes, device=self.device)
            one_hot[:, y] = 1
            mu_sigma = self.prior_pz_layer(one_hot)  # .detach() # we do not want to backprop into prior
            mu_sigma = process_gaussian_parameters(mu_sigma, self.config.encoder.latent_dim)
            return Normal(*th.split(mu_sigma, split_size_or_sections=self.config.encoder.latent_dim, dim=-1))
        else:
            raise ValueError("Prior mode not specified correctly")

    def prior_py(self, batch_size: int):
        """
        Categorical uniform distribution
        """
        return Categorical(
            probs=th.ones(batch_size, self.config.encoder.num_classes, device=self.device)
            * (1.0 / self.config.encoder.num_classes)
        )
