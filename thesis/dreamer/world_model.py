import re
from typing import Any, NamedTuple

import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from torch.distributions import Distribution as Dist

from ..core.module import BaseModule
from ..core.utils import build_network, initialize_weights
from .config import (ContinueModelConfig, DecoderConfig,
                     DreamerWorldModelConfig, EncoderConfig,
                     RecurrentStateSpaceModelConfig, RepresentationModelConfig,
                     RewardModelConfig, SequenceModelConfig,
                     TransitionModelConfig)

Stochastic = th.Tensor
Deterministic = th.Tensor
EmbeddedObs = th.Tensor


class DynamicsInfo(NamedTuple):
    priors: Stochastic
    prior_dist_means: th.Tensor
    prior_dist_stds: th.Tensor
    posteriors: Stochastic
    posterior_dist_means: th.Tensor
    posterior_dist_stds: th.Tensor
    deterministics: Deterministic


def create_normal_dist(
    x: th.Tensor,
    std: th.Tensor | float | None = None,
    mean_scale: float = 1.0,
    init_std: float = 0.0,
    min_std: float = 0.1,
    activation: th.nn.Module | None = None,
    event_shape: th.Size | int | None = None,
):
    if std == None:
        mean, std = th.chunk(x, 2, -1)
        mean = mean / mean_scale
        if activation:
            mean = activation(mean)
        mean = mean_scale * mean
        std = th.nn.functional.softplus(std + init_std) + min_std
    else:
        mean = x
    dist = th.distributions.Normal(mean, std)
    if event_shape:
        dist = th.distributions.Independent(dist, event_shape)
    return dist


def horizontal_forward(
    network,
    x: th.Tensor,
    y: th.Tensor | None = None,
    input_shape: tuple[int, ...] = (-1,),
    output_shape: tuple[int, ...] = (-1,),
):
    batch_with_horizon_shape = x.shape[: -len(input_shape)]
    if not batch_with_horizon_shape:
        batch_with_horizon_shape = (1,)
    if y is not None:
        x = th.cat((x, y), -1)
        input_shape = (x.shape[-1],)  #
    x = x.reshape(-1, *input_shape)
    x = network(x)

    x = x.reshape(*batch_with_horizon_shape, *output_shape)
    return x


class SequenceModel(BaseModule):
    def __init__(
        self, stochastic_size: int, deterministic_size: int, action_size: int, task_size: int, config: SequenceModelConfig
    ) -> None:
        super().__init__()
        self.deterministic_size = deterministic_size
        self.preprocess = th.nn.Sequential(
            th.nn.Linear(task_size + stochastic_size + action_size, config.hidden_size), config.activation()
        )
        self.gru = th.nn.GRUCell(config.hidden_size, deterministic_size)

    def forward(self, embedded_state: Stochastic, action: th.Tensor, deterministic: Deterministic) -> Deterministic:
        input = th.cat((embedded_state, action), dim=1)
        gru_input = self.preprocess(input)
        deterministic = self.gru(gru_input, deterministic)
        return deterministic

    def get_init_input(self, batch_size) -> Deterministic:
        return th.zeros(batch_size, self.deterministic_size, device=self.device)


class TransitionModel(BaseModule):
    def __init__(self, stochastic_size: int, deterministic_size: int, config: TransitionModelConfig) -> None:
        super().__init__()
        self.config = config
        self.stochastic_size = stochastic_size

        self.model = build_network(deterministic_size, config.layers, config.activation, 2 * stochastic_size)

    def forward(self, deterministic: Deterministic) -> tuple[Dist, Stochastic]:
        deterministic = self.model(deterministic)
        prior_dist = create_normal_dist(deterministic, min_std=self.config.min_std)
        prior = prior_dist.rsample()
        return prior_dist, prior

    def get_init_input(self, batch_size) -> Stochastic:
        return th.zeros(batch_size, self.stochastic_size, device=self.device)


class RepresentationModel(BaseModule):
    def __init__(
        self, embedded_size: int, stochastic_size: int, deterministic_size: int, config: RepresentationModelConfig
    ) -> None:
        super().__init__()
        self.config = config

        self.model = build_network(embedded_size + deterministic_size, config.layers, config.activation, 2 * stochastic_size)

    def forward(self, embedded_obs: EmbeddedObs, deterministic: Deterministic) -> tuple[Dist, Stochastic]:
        x = th.cat([embedded_obs, deterministic], dim=1)
        x = self.model(x)
        posterior_dist = create_normal_dist(x, min_std=self.config.min_std)
        posterior = posterior_dist.rsample()
        return posterior_dist, posterior


class RecurrentSpaceStateModel(BaseModule):
    def __init__(
        self,
        stochastic_size: int,
        deterministic_size: int,
        embedded_size: int,
        task_size: int,
        action_size: int,
        config: RecurrentStateSpaceModelConfig,
    ) -> None:
        super().__init__()
        self.transition_model = TransitionModel(
            stochastic_size=stochastic_size, deterministic_size=deterministic_size, config=config.transition_model
        )
        self.representation_model = RepresentationModel(
            embedded_size=embedded_size,
            stochastic_size=stochastic_size,
            deterministic_size=deterministic_size,
            config=config.representation_model,
        )
        self.sequence_model = SequenceModel(
            stochastic_size=stochastic_size,
            deterministic_size=deterministic_size,
            task_size=task_size,
            action_size=action_size,
            config=config.sequence_model,
        )

    def get_init_input(self, batch_size):
        return (self.transition_model.get_init_input(batch_size), self.sequence_model.get_init_input(batch_size))


class ContinueModel(BaseModule):
    def __init__(self, stochastic_size: int, deterministic_size: int, config: ContinueModelConfig):
        super().__init__()
        self.model = build_network(stochastic_size + deterministic_size, config.layers, config.activation, 1)

    def forward(self, posterior: Stochastic, deterministic: Deterministic) -> Dist:
        x = horizontal_forward(self.network, posterior, deterministic, output_shape=(1,))
        dist = th.distributions.Bernoulli(logits=x)
        return dist


class RewardModel(BaseModule):
    def __init__(self, stochastic_size: int, deterministic_size: int, config: RewardModelConfig) -> None:
        super().__init__()
        self.model = build_network(stochastic_size + deterministic_size, config.layers, config.activation, 1)

    def forward(self, posterior: Stochastic, deterministic: Deterministic) -> Dist:
        x = horizontal_forward(self.model, posterior, deterministic, output_shape=(1,))
        dist = create_normal_dist(x, std=1, event_shape=1)
        return dist


class Encoder(th.nn.Module):
    def __init__(self, observation_space: spaces.Dict, embedded_size: int, config: EncoderConfig):
        super().__init__()
        self.mlp_observation_space = spaces.Dict(
            {k: v for k, v in observation_space.spaces.items() if re.match(config.mlp_filter, k)}
        )
        self.cnn_observation_space = spaces.Dict(
            {k: v for k, v in observation_space.spaces.items() if re.match(config.cnn_filter, k)}
        )

        self.cnn_network = th.nn.ModuleDict(
            {
                k: th.nn.Sequential(
                    th.nn.Conv2d(
                        v.shape[0],
                        config.cnn_config.depth * 1,
                        config.cnn_config.kernel_size,
                        config.cnn_config.stride,
                    ),
                    config.cnn_config.activation(),
                    th.nn.Conv2d(
                        config.cnn_config.depth * 1,
                        config.cnn_config.depth * 2,
                        config.cnn_config.kernel_size,
                        config.cnn_config.stride,
                    ),
                    config.cnn_config.activation(),
                    th.nn.Conv2d(
                        config.cnn_config.depth * 2,
                        config.cnn_config.depth * 4,
                        config.cnn_config.kernel_size,
                        config.cnn_config.stride,
                    ),
                    config.cnn_config.activation(),
                    th.nn.Conv2d(
                        config.cnn_config.depth * 4,
                        config.cnn_config.depth * 8,
                        config.cnn_config.kernel_size,
                        config.cnn_config.stride,
                    ),
                    config.cnn_config.activation(),
                    th.nn.LazyLinear(config.cnn_config.extracted_size),
                )
                for k, v in self.cnn_observation_space.spaces.items()
                if v.shape != None
            }
        )
        self.cnn_network.apply(initialize_weights)

        input_size = spaces.flatdim(self.mlp_observation_space)
        input_size += len(self.cnn_observation_space.spaces) * config.cnn_config.extracted_size
        self.mlp = build_network(input_size, config.mlp_config.layers, config.mlp_config.activation, embedded_size)
        self.mlp.apply(initialize_weights)

    def forward(self, x: dict) -> EmbeddedObs:
        mlp_input = th.cat([x[k] for k in self.mlp_observation_space.spaces.keys()], dim=-1)

        if len(self.cnn_observation_space.spaces) != 0:
            cnn_embed = []
            for k, v in self.cnn_network.items():
                cnn_embed.append(v(x[k]))
            cnn_embed = th.cat(cnn_embed, dim=-1)
            mlp_input = th.cat([cnn_embed, mlp_input], dim=-1)
        return self.mlp(mlp_input)


class Decoder(th.nn.Module):
    def __init__(self, stochastic_size: int, deterministic_size: int, observation_space: spaces.Dict, config: DecoderConfig):
        super().__init__()
        self.config = config

        self.mlp_observation_space = spaces.Dict(
            {k: v for k, v in observation_space.spaces.items() if re.match(config.mlp_filter, k)}
        )
        self.cnn_observation_space = spaces.Dict(
            {k: v for k, v in observation_space.spaces.items() if re.match(config.cnn_filter, k)}
        )

        self.mlp_network = build_network(
            stochastic_size + deterministic_size,
            config.mlp_config.layers,
            config.mlp_config.activation,
            2 * spaces.flatdim(self.mlp_observation_space)
            + len(self.cnn_observation_space.spaces) * config.cnn_config.extracted_size,
        )
        self.mlp_network.apply(initialize_weights)

        self.cnn_network = th.nn.ModuleDict(
            {
                k: th.nn.Sequential(
                    th.nn.Linear(deterministic_size + stochastic_size, config.cnn_config.depth * 32),
                    th.nn.Unflatten(1, (config.cnn_config.depth * 32, 1)),
                    th.nn.Unflatten(2, (1, 1)),
                    th.nn.ConvTranspose2d(
                        config.cnn_config.depth * 32,
                        config.cnn_config.depth * 4,
                        config.cnn_config.kernel_size,
                        config.cnn_config.stride,
                    ),
                    config.cnn_config.activation(),
                    th.nn.ConvTranspose2d(
                        config.cnn_config.depth * 4,
                        config.cnn_config.depth * 2,
                        config.cnn_config.kernel_size,
                        config.cnn_config.stride,
                    ),
                    config.cnn_config.activation(),
                    th.nn.ConvTranspose2d(
                        config.cnn_config.depth * 2,
                        config.cnn_config.depth * 1,
                        config.cnn_config.kernel_size + 1,
                        config.cnn_config.stride,
                    ),
                    config.cnn_config.activation(),
                    th.nn.ConvTranspose2d(
                        config.cnn_config.depth * 1,
                        v.shape[0],
                        config.cnn_config.kernel_size + 1,
                        config.cnn_config.stride,
                    ),
                )
                for k, v in self.cnn_observation_space.spaces.items()
                if v.shape != None
            }
        )
        for v in self.cnn_network.values():
            v.apply(initialize_weights)

    def forward(self, posterior: Stochastic, deterministic: Deterministic):
        x = th.cat([posterior, deterministic], dim=-1)
        x = self.mlp_network(x)
        output: dict[str, Dist] = {}
        idx = 0
        for k, v in self.cnn_network.items():
            cnn_x = v(x[..., idx : idx + self.config.cnn_config.extracted_size])
            dist = create_normal_dist(cnn_x, std=1, event_shape=len(self.cnn_observation_space[k].shape))  # type: ignore
            output[k] = dist
            idx += self.config.cnn_config.extracted_size
        for k, v in self.mlp_observation_space.spaces.items():
            space_dim = 2 * spaces.flatdim(v)
            mlp_x = x[..., idx : idx + space_dim].reshape(*x.shape[:-1], *v.shape[:-1], v.shape[-1] * 2) # type: ignore
            dist = create_normal_dist(mlp_x, event_shape=len(v.shape))  # type: ignore
            output[k] = dist
            idx += space_dim
        return output


class DreamerWorldModel:
    def __init__(
        self,
        stochastic_size: int,
        deterministic_size: int,
        embedded_size: int,
        task_size: int,
        observation_space: spaces.Dict,
        action_size: int,
        config: DreamerWorldModelConfig,
    ) -> None:
        super().__init__()
        self.training_config = config.world_model_training_config

        self.rssm = RecurrentSpaceStateModel(
            stochastic_size,
            deterministic_size,
            embedded_size,
            task_size,
            action_size,
            config.recurrent_state_space_model_config,
        )
        self.continue_predictor = ContinueModel(stochastic_size, deterministic_size, config.continue_config)
        self.reward_predictor = RewardModel(stochastic_size, deterministic_size, config.reward_config)
        self.decoder = Decoder(stochastic_size, deterministic_size, observation_space, config.decoder_config)
        self.encoder = Encoder(observation_space, embedded_size, config.encoder_config)
        self.continue_criterion = th.nn.BCELoss()

        self.optimizer = self.training_config.optimizer_class(self.parameters(), lr=self.training_config.lr)  # type: ignore

    def training_step(self, data: DictReplayBufferSamples):
        prior, deterministic = self.rssm.get_init_input(len(data.actions))
        posterior_info = self.observe_data(data, prior, deterministic)

        reconstructed_observation_dist: dict[str, Dist] = self.decoder(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reconstruction_observation_losses = {
            k: v.log_prob(data.observations[k][:, 1:]) for k, v in reconstructed_observation_dist.items()
        }
        reconstruction_observation_loss = th.sum(
            th.cat([v for v in reconstruction_observation_losses.values()], dim=-1), dim=-1
        )

        if self.training_config.use_continue_flag:
            continue_dist = self.continue_predictor(posterior_info.posteriors, posterior_info.deterministics)
            continue_loss = self.continue_criterion(continue_dist.probs, 1 - data.dones[:, 1:])
        else:
            continue_loss = th.zeros_like(reconstruction_observation_loss)

        reward_dist: Dist = self.reward_predictor(posterior_info.posteriors, posterior_info.deterministics)
        reward_loss = reward_dist.log_prob(data.rewards[:, 1:])

        prior_dist = create_normal_dist(
            posterior_info.prior_dist_means,
            posterior_info.prior_dist_stds,
            event_shape=1,
        )
        posterior_dist = create_normal_dist(
            posterior_info.posterior_dist_means,
            posterior_info.posterior_dist_stds,
            event_shape=1,
        )
        kl_divergence_loss = th.mean(th.distributions.kl.kl_divergence(posterior_dist, prior_dist))
        kl_divergence_loss = th.max(th.tensor(self.training_config.free_nats, device=self.device), kl_divergence_loss)
        model_loss = (
            self.training_config.kl_divergence_scale * kl_divergence_loss
            - reconstruction_observation_loss.mean()
            - reward_loss.mean()
        )
        if self.training_config.use_continue_flag:
            model_loss += continue_loss.mean()

        self.optimizer.zero_grad()
        model_loss.backward()
        th.nn.utils.clip_grad_norm_(  # type: ignore
            self.parameters(),
            self.training_config.clip_grad,
            norm_type=self.training_config.grad_norm_type,
        )
        self.optimizer.step()

        return (
            posterior_info.posteriors.detach(),
            posterior_info.deterministics.detach(),
            {
                "model_loss": model_loss.item(),
                "reconstruction_observation_loss": -reconstruction_observation_loss.mean().item(),
                "reward_loss": -reward_loss.mean().item(),
                "continue_loss": continue_loss.mean().item(),
                "kl_divergence_loss": kl_divergence_loss.item(),
            },
        )

    def observe_data(self, data: DictReplayBufferSamples, prior: Stochastic, deterministic: Deterministic):
        batch_length = data.actions.shape[1]
        embedded_observation = self.encoder(data.observations)
        cache = []

        for t in range(1, batch_length):
            deterministic = self.rssm.sequence_model(prior, data.actions[:, t - 1], deterministic)
            prior_dist, prior = self.rssm.transition_model(deterministic)
            posterior_dist, posterior = self.rssm.representation_model(embedded_observation[:, t], deterministic)

            cache.append(
                dict(
                    priors=prior,
                    prior_dist_means=prior_dist.mean,
                    prior_dist_stds=prior_dist.scale,
                    posteriors=posterior,
                    posterior_dist_means=posterior_dist.mean,
                    posterior_dist_stds=posterior_dist.scale,
                    deterministics=deterministic,
                )
            )

            prior = posterior

        infos = {k: th.stack([c[k] for c in cache], dim=1) for k in cache[0].keys()}
        return DynamicsInfo(**infos)
