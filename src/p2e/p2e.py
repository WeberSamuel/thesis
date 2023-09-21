import math
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, cast

import numpy as np
import torch
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, ReplayBufferSamples, RolloutReturn, TrainFreq
from stable_baselines3.common.vec_env import VecEnv
from torch.distributions import Bernoulli, Distribution, Normal

from from ..core.algorithms import BaseAlgorithm
from src.p2e.networks import Actor, Critic
from src.p2e.policies import P2EPolicy
from src.p2e.utils import compute_lambda_values, create_normal_dist
from src.p2e.buffers import P2EBuffer


class DynamicLearningInfo(NamedTuple):
    priors: th.Tensor
    prior_dist_means: th.Tensor
    prior_dist_stds: th.Tensor
    posteriors: th.Tensor
    posterior_dist_means: th.Tensor
    posterior_dist_stds: th.Tensor
    deterministics: th.Tensor


class BehaviorLearningInfo(NamedTuple):
    priors: th.Tensor
    deterministics: th.Tensor
    actions: th.Tensor



class P2E(StateAwareOffPolicyAlgorithm):
    def __init__(
        self,
        policy: str | type[BasePolicy],
        env: GymEnv | str,
        batch_size: int = 64,
        buffer_size: int = 5_000_000,
        clip_grad: float = 100,
        discount: float = 0.99,
        free_nats: float = 3.0,
        gamma: float = 0.99,
        grad_norm_type: int = 2,
        gradient_steps: int = 10,
        horizon_length: int = 15,
        kl_divergence_scale: float = 1.0,
        lambda_: float = 0.95,
        replay_buffer_class: type[ReplayBuffer] | None = P2EBuffer,
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            buffer_size=buffer_size,
            env=env,
            gamma=gamma,
            gradient_steps=gradient_steps,
            learning_rate=1e-3,
            policy=policy,
            sde_support=False,
            support_multi_env=True,
            replay_buffer_class=replay_buffer_class,
            **kwargs,
        )
        self.replay_buffer_kwargs["encoder"] = self.policy_kwargs["encoder"]
        super()._setup_model()
        self.policy: P2EPolicy
        self.free_nats = free_nats
        self.kl_divergence_scale = kl_divergence_scale
        self.clip_grad = clip_grad
        self.grad_norm_type = grad_norm_type
        self.horizon_length = horizon_length
        self.discount = discount
        self.lambda_ = lambda_

        self.continue_criterion = torch.nn.BCELoss()
        self.action_size = spaces.flatdim(self.action_space)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        assert self.replay_buffer is not None

        for step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self.get_vec_normalize_env())
            if data.observations.shape[1] < 2:
                continue

            posteriors, deterministics = self.dynamic_learning(data)
            self.behavior_learning(
                self.policy.actor,
                self.policy.critic,
                self.policy.actor_optimizer,
                self.policy.critic_optimizer,
                posteriors,
                deterministics,
            )

            self.behavior_learning(
                self.policy.intrinsic_actor,
                self.policy.intrinsic_critic,
                self.policy.intrinsic_actor_optimizer,
                self.policy.intrinsic_critic_optimizer,
                posteriors,
                deterministics,
            )

    def dynamic_learning(self, data: ReplayBufferSamples):
        chunk_length = data.actions.shape[1]
        prior, deterministic = self.policy.rssm.recurrent_model_input_init(len(data.actions))

        embedded_observation: th.Tensor = self.policy.encoder(data.observations)

        dynamic_learning_infos = []
        for t in range(1, chunk_length):
            deterministic = cast(th.Tensor, self.policy.rssm.recurrent_model(prior, data.actions[:, t - 1], deterministic))
            prior_dist, prior = cast(Tuple[Normal, th.Tensor], self.policy.rssm.transition_model(deterministic))
            posterior_dist, posterior = cast(
                Tuple[Normal, th.Tensor], self.policy.rssm.representation_model(embedded_observation[:, t], deterministic)
            )

            dynamic_learning_infos.append(
                DynamicLearningInfo(
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

        infos = DynamicLearningInfo(*map(lambda x: th.stack(x, dim=1), zip(*dynamic_learning_infos)))
        self._model_update(data, infos, embedded_observation)
        return infos.posteriors.detach(), infos.deterministics.detach()

    def _model_update(self, data: ReplayBufferSamples, posterior_info: DynamicLearningInfo, embedded_observation: th.Tensor):
        continue_loss: th.Tensor = torch.zeros(1).to(self.device)
        reconstructed_observation_dist: Distribution = self.policy.decoder(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reconstruction_observation_loss: th.Tensor = reconstructed_observation_dist.log_prob(data.observations[:, 1:])
        reconstruction_observation_loss = -reconstruction_observation_loss.mean()
        if self.policy.use_continue_flag:
            continue_dist: Bernoulli = self.policy.continue_predictor(posterior_info.posteriors, posterior_info.deterministics)
            continue_loss = self.continue_criterion(continue_dist.probs, 1 - data.dones[:, 1:]).mean()

        reward_dist: Distribution = self.policy.reward_predictor(
            posterior_info.posteriors.detach(), posterior_info.deterministics.detach()
        )
        reward_loss: th.Tensor = reward_dist.log_prob(data.rewards[:, 1:])
        reward_loss = -reward_loss.mean()

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
        kl_divergence_loss = torch.mean(torch.distributions.kl.kl_divergence(posterior_dist, prior_dist))
        kl_divergence_loss = torch.max(torch.tensor(self.free_nats).to(self.device), kl_divergence_loss)
        model_loss: th.Tensor = (
            self.kl_divergence_scale * kl_divergence_loss + reconstruction_observation_loss + reward_loss
        )
        if self.policy.use_continue_flag:
            model_loss += continue_loss

        self.policy.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.policy.model_params, self.clip_grad, norm_type=self.grad_norm_type)
        self.policy.model_optimizer.step()

        predicted_feature_dists: list[Distribution] = [
            x.forward(
                data.actions[:, :-1],
                posterior_info.priors.detach(),
                posterior_info.deterministics.detach(),
            )
            for x in self.policy.one_step_models
        ]
        one_step_model_loss = cast(
            th.Tensor, -sum([x.log_prob(embedded_observation[:, 1:].detach()).mean() for x in predicted_feature_dists])
        )

        self.policy.one_step_models_optimizer.zero_grad()
        one_step_model_loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(
            self.policy.one_step_models.parameters(), self.clip_grad, norm_type=self.grad_norm_type
        )
        self.policy.one_step_models_optimizer.step()

        self.logger.record("train/reconstruction_observation_loss", reconstruction_observation_loss.item())
        self.logger.record("train/reward_loss", reward_loss.item())
        self.logger.record("train/kl_divergence_loss", kl_divergence_loss.item())
        self.logger.record("train/model_loss", model_loss.item())
        self.logger.record("train/one_step_model_loss", one_step_model_loss.item())
        self.logger.record("train/continue_loss", continue_loss.item())

    def behavior_learning(
        self,
        actor: Actor,
        critic: Critic,
        actor_optimizer: th.optim.Optimizer,
        critic_optimizer: th.optim.Optimizer,
        states: th.Tensor,
        deterministics: th.Tensor,
    ):
        """
        #TODO : last posterior truncation(last can be last step)
        posterior shape : (batch, timestep, stochastic)
        """
        state = states.reshape(-1, self.policy.stochastic_size)
        deterministic = deterministics.reshape(-1, self.policy.deterministic_size)

        behavior_learning_infos = []
        # continue_predictor reinit
        for t in range(self.horizon_length):
            action = actor.forward(state, deterministic)
            deterministic = self.policy.rssm.recurrent_model.forward(state, action, deterministic)
            _, state = self.policy.rssm.transition_model.forward(deterministic)
            behavior_learning_infos.append(BehaviorLearningInfo(priors=state, deterministics=deterministic, actions=action))
        infos = BehaviorLearningInfo(*map(lambda x: th.stack(x, dim=1), zip(*behavior_learning_infos)))

        self._agent_update(actor, critic, actor_optimizer, critic_optimizer, infos)

    def _agent_update(
        self,
        actor: Actor,
        critic: Critic,
        actor_optimizer: th.optim.Optimizer,
        critic_optimizer: th.optim.Optimizer,
        behavior_learning_infos: BehaviorLearningInfo,
    ):
        if actor.intrinsic:
            predicted_feature_means: list[th.Tensor] = [
                x(
                    behavior_learning_infos.actions,
                    behavior_learning_infos.priors,
                    behavior_learning_infos.deterministics,
                ).mean
                for x in self.policy.one_step_models
            ]
            predicted_feature_mean_stds = torch.stack(predicted_feature_means, 0).std(0)

            predicted_rewards: th.Tensor = predicted_feature_mean_stds.mean(-1, keepdim=True)
        else:
            predicted_rewards = self.policy.reward_predictor(
                behavior_learning_infos.priors, behavior_learning_infos.deterministics
            ).mean
        values: th.Tensor = critic(behavior_learning_infos.priors, behavior_learning_infos.deterministics).mean

        if self.policy.use_continue_flag:
            continues: th.Tensor = self.policy.continue_predictor(
                behavior_learning_infos.priors, behavior_learning_infos.deterministics
            ).mean
        else:
            continues = self.discount * torch.ones_like(values)

        lambda_values = compute_lambda_values(
            predicted_rewards, values, continues, self.horizon_length, self.device, self.lambda_
        )

        actor_loss = -torch.mean(lambda_values)

        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), self.clip_grad, norm_type=self.grad_norm_type)
        actor_optimizer.step()

        value_dist: Distribution = critic(
            behavior_learning_infos.priors.detach()[:, :-1],
            behavior_learning_infos.deterministics.detach()[:, :-1],
        )
        value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))

        critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(critic.parameters(), self.clip_grad, norm_type=self.grad_norm_type)
        critic_optimizer.step()

        self.logger.record("train/actor_loss", actor_loss.item())
        self.logger.record("train/critic_loss", value_loss.item())

