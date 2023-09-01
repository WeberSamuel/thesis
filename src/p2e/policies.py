from typing import Dict, Tuple

import numpy as np
import torch
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from src.cemrl.networks import Encoder as CEMRLEncoder
from src.core.state_aware_algorithm import StateAwarePolicy
from src.p2e.decoder import Decoder
from src.p2e.encoder import Encoder
from src.p2e.networks import (
    RSSM,
    Actor,
    ActorKwargs,
    ContinueModel,
    ContinueModelKwargs,
    Critic,
    CriticKwargs,
    DecoderKwargs,
    EncoderKwargs,
    OneStepModel,
    OneStepModelKwargs,
    RewardModel,
    RewardModelKwargs,
    RSSMKwargs,
)
from src.p2e.utils import asdict


class P2EPolicy(StateAwarePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: float | Schedule,
        encoder: CEMRLEncoder,
        encoder_kwargs: EncoderKwargs = EncoderKwargs(),
        decoder_kwargs: DecoderKwargs = DecoderKwargs(),
        rssm_kwargs: RSSMKwargs = RSSMKwargs(),
        reward_model_kwargs: RewardModelKwargs = RewardModelKwargs(),
        continue_model_kwargs: ContinueModelKwargs = ContinueModelKwargs(),
        actor_kwargs: ActorKwargs = ActorKwargs(),
        critic_kwargs: CriticKwargs = CriticKwargs(),
        one_step_model_kwargs: OneStepModelKwargs = OneStepModelKwargs(),
        one_step_model_learning_rate: float = 0.003,
        actor_learning_rate: float = 0.008,
        critic_learning_rate: float = 0.008,
        model_learning_rate: float = 0.003,
        num_ensemble: int = 10,
        use_continue_flag: bool = True,
        stochastic_size: int = 30,
        deterministic_size: int = 100,
        embedded_state_size: int = 200,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, **kwargs)
        self.use_continue_flag = use_continue_flag
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.embedded_state_size = embedded_state_size
        self.cemrl_encoder = encoder
        if isinstance(observation_space, spaces.Dict):
            obs_shape = observation_space["observation"].shape[1:]  # type: ignore
        else:
            obs_shape = observation_space.shape
        assert obs_shape is not None
        obs_shape = (self.cemrl_encoder.latent_dim + obs_shape[0], *obs_shape[1:])

        action_size = spaces.flatdim(self.action_space)

        self.encoder = Encoder(obs_shape, embedded_state_size, **asdict(encoder_kwargs))
        self.decoder = Decoder(obs_shape, stochastic_size, deterministic_size, **asdict(decoder_kwargs))
        self.rssm = RSSM(action_size, stochastic_size, deterministic_size, embedded_state_size, **asdict(rssm_kwargs))
        self.reward_predictor = RewardModel(stochastic_size, deterministic_size, **asdict(reward_model_kwargs))
        if use_continue_flag:
            self.continue_predictor = ContinueModel(stochastic_size, deterministic_size, **asdict(continue_model_kwargs))

        self.actor = Actor(
            isinstance(self.action_space, spaces.Discrete),
            action_size,
            stochastic_size,
            deterministic_size,
            **asdict(actor_kwargs),
        )
        self.actor.intrinsic = False
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=actor_learning_rate)

        self.critic = Critic(stochastic_size, deterministic_size, **asdict(critic_kwargs))
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=critic_learning_rate)

        # optimizer
        self.model_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_predictor.parameters())
            + list(self.continue_predictor.parameters())
            if use_continue_flag
            else []
        )
        self.model_optimizer = torch.optim.AdamW(self.model_params, lr=model_learning_rate)

        self.intrinsic_actor = Actor(
            isinstance(action_space, spaces.Discrete), action_size, stochastic_size, deterministic_size, **asdict(actor_kwargs)
        )
        self.intrinsic_actor.intrinsic = True
        self.intrinsic_actor_optimizer = torch.optim.AdamW(self.intrinsic_actor.parameters(), lr=actor_learning_rate)

        self.intrinsic_critic = Critic(stochastic_size, deterministic_size, **asdict(critic_kwargs))
        self.intrinsic_critic_optimizer = torch.optim.AdamW(self.intrinsic_critic.parameters(), lr=critic_learning_rate)

        self.one_step_models = th.nn.ModuleList(
            [
                OneStepModel(
                    action_size, embedded_state_size, stochastic_size, deterministic_size, **asdict(one_step_model_kwargs)
                )
                for _ in range(num_ensemble)
            ]
        )
        self.one_step_models_optimizer = torch.optim.AdamW(self.one_step_models.parameters(), lr=one_step_model_learning_rate)

        self.use_intrinsic = True

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        if isinstance(observation, dict):
            with th.no_grad():
                y, z = self.cemrl_encoder(observation)
            observation = th.cat([observation["observation"][:, -1], z], dim=-1)  # type: ignore

        if self.state is None:
            n_envs = observation.shape[0]
            self.state = self.reset_states(n_envs)
        posterior, deter, action = self.state
        embedded_observation = self.encoder(observation)

        deter = self.rssm.recurrent_model(posterior, action, deter)
        _, posterior = self.rssm.representation_model(embedded_observation, deter)
        action = (self.intrinsic_actor if self.use_intrinsic else self.actor)(posterior, deter)

        self.state = posterior, deter, action
        return action

    def _reset_states(self, size: int) -> Tuple[th.Tensor, ...]:
        rsmm_init = self.rssm.recurrent_model_input_init(size)
        action_init = th.zeros((size, spaces.flatdim(self.action_space)), device=self.device)
        return *rsmm_init, action_init