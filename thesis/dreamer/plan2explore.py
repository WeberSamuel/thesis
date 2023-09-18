from typing import Any

import torch as th
from torch.distributions import Distribution as Dist

from .actor_critic import ActorCritic, AgentInfos
from .world_model import (
    Deterministic,
    DreamerWorldModel,
    DynamicsInfo,
    Stochastic,
    create_normal_dist,
    horizontal_forward,
)
from ..core.utils import build_network


class OneStepModel(th.nn.Module):
    def __init__(
        self,
        stochastic_size: int,
        deterministic_size: int,
        embedded_state_size: int,
        action_size: int,
        layers: list[int],
        activation: type[th.nn.Module],
    ):
        """
        For plan2explore
        There are several variations, but in our implementation,
        we use stochastic and deterministic actions as input and embedded observations as output
        """
        super().__init__()
        self.embedded_state_size = embedded_state_size

        self.network = build_network(
            deterministic_size + stochastic_size + action_size,
            layers,
            activation,
            embedded_state_size,
        )

    def forward(self, action: th.Tensor, stochastic: Stochastic, deterministic: Deterministic):
        stoch_deter = th.concat((stochastic, deterministic), dim=-1)
        x = horizontal_forward(
            self.network,
            action,
            stoch_deter,
            output_shape=(self.embedded_state_size,),
        )
        dist = create_normal_dist(x, std=1, event_shape=1)
        return dist


class Plan2ExploreActorCritic(ActorCritic):
    def __init__(
        self,
        ensemble: th.nn.ModuleList,
        discrete_action_bool: bool,
        action_size: int,
        stochastic_size: int,
        deterministic_size: int,
        mean_scale: float,
        init_std: float,
        min_std: float,
        actor_layers: list[int],
        actor_activation: type[th.nn.Module],
        critic_layers: list[int],
        critic_activation: type[th.nn.Module],
        world_model: DreamerWorldModel,
        optimizer_class: type[th.optim.Optimizer],
        discount: float,
        lambda_: float,
        critic_optimizer_kwargs: dict[str, Any] = ...,
        actor_optimizer_kwargs: dict[str, Any] = ...,
        clip_grad: float = 100,
        horizon_length: int = 15,
        grad_norm_type: int = 2,
        use_continue_flag: bool = False,
    ) -> None:
        super().__init__(
            discrete_action_bool,
            action_size,
            stochastic_size,
            deterministic_size,
            mean_scale,
            init_std,
            min_std,
            actor_layers,
            actor_activation,
            critic_layers,
            critic_activation,
            world_model,
            optimizer_class,
            discount,
            lambda_,
            critic_optimizer_kwargs,
            actor_optimizer_kwargs,
            clip_grad,
            horizon_length,
            grad_norm_type,
            use_continue_flag,
        )
        self.ensemble = ensemble

    def get_reward_prediction(self, infos: AgentInfos):
        predicted_feature_means = [
            one_step_model(infos.actions, infos.priors, infos.deterministics).mean for one_step_model in self.ensemble
        ]
        predicted_feature_mean_stds = th.stack(predicted_feature_means, 0).std(0)

        predicted_rewards = predicted_feature_mean_stds.mean(-1, keepdim=True)
        return predicted_rewards


class Plan2Explore(th.nn.Module):
    def __init__(
        self,
        stochastic_size: int,
        deterministic_size: int,
        embedded_state_size: int,
        action_size: int,
        ensemble_size: int,
        one_step_layers: list[int],
        world_model: DreamerWorldModel,
        one_step_activation: type[th.nn.Module],
        discrete_action_bool: bool,
        mean_scale: float,
        init_std: float,
        min_std: float,
        actor_layers: list[int],
        actor_activation: type[th.nn.Module],
        critic_layers: list[int],
        critic_activation: type[th.nn.Module],
        optimizer_class: type[th.optim.Optimizer],
        discount: float,
        lambda_: float,
        ensemlbe_optimizer_kwargs: dict[str, Any] = dict(lr=1e-3),
        critic_optimizer_kwargs: dict[str, Any] = dict(lr=1e-3),
        actor_optimizer_kwargs: dict[str, Any] = dict(lr=1e-3),
        clip_grad: float = 100,
        horizon_length: int = 15,
        grad_norm_type: int = 2,
        use_continue_flag: bool = False,
    ):
        super().__init__()
        self.clip_grad = clip_grad
        self.grad_norm_type = grad_norm_type
        self.ensemble = th.nn.ModuleList(
            [
                OneStepModel(
                    stochastic_size=stochastic_size,
                    deterministic_size=deterministic_size,
                    embedded_state_size=embedded_state_size,
                    action_size=action_size,
                    layers=one_step_layers,
                    activation=one_step_activation,
                )
                for _ in range(ensemble_size)
            ]
        )
        self.ensemble_optimizer = optimizer_class(self.ensemble.parameters(), **ensemlbe_optimizer_kwargs)

        self.intrinsic_actor_critic = Plan2ExploreActorCritic(
            ensemble=self.ensemble,
            discrete_action_bool=discrete_action_bool,
            action_size=action_size,
            stochastic_size=stochastic_size,
            deterministic_size=deterministic_size,
            mean_scale=mean_scale,
            init_std=init_std,
            min_std=min_std,
            actor_layers=actor_layers,
            actor_activation=actor_activation,
            critic_layers=critic_layers,
            critic_activation=critic_activation,
            world_model=world_model,
            optimizer_class=optimizer_class,
            discount=discount,
            lambda_=lambda_,
            critic_optimizer_kwargs=critic_optimizer_kwargs,
            actor_optimizer_kwargs=actor_optimizer_kwargs,
            clip_grad=clip_grad,
            horizon_length=horizon_length,
            grad_norm_type=grad_norm_type,
            use_continue_flag=use_continue_flag,
        )

        self.extrinsic_actor_critic = ActorCritic(
            discrete_action_bool=discrete_action_bool,
            action_size=action_size,
            stochastic_size=stochastic_size,
            deterministic_size=deterministic_size,
            mean_scale=mean_scale,
            init_std=init_std,
            min_std=min_std,
            actor_layers=actor_layers,
            actor_activation=actor_activation,
            critic_layers=critic_layers,
            critic_activation=critic_activation,
            world_model=world_model,
            optimizer_class=optimizer_class,
            discount=discount,
            lambda_=lambda_,
            critic_optimizer_kwargs=critic_optimizer_kwargs,
            actor_optimizer_kwargs=actor_optimizer_kwargs,
            clip_grad=clip_grad,
            horizon_length=horizon_length,
            grad_norm_type=grad_norm_type,
            use_continue_flag=use_continue_flag,
        )

    def training_step(self, actions: th.Tensor, embedded_obs: th.Tensor, posterior_info: DynamicsInfo):
        metrics = {}
        predicted_feature_dists: list[Dist] = [
            one_step_model(
                actions[:, :-1],
                posterior_info.priors.detach(),
                posterior_info.deterministics.detach(),
            )
            for one_step_model in self.ensemble
        ]
        ensemble_loss = -th.stack([x.log_prob(embedded_obs[:, 1:].detach()).mean() for x in predicted_feature_dists]).sum()

        self.ensemble_optimizer.zero_grad()
        ensemble_loss.backward()
        metrics["ensemble_loss"] = ensemble_loss.item()

        th.nn.utils.clip_grad_norm_(  # type: ignore
            self.ensemble.parameters(),
            self.clip_grad,
            norm_type=self.grad_norm_type,
        )
        self.ensemble_optimizer.step()

        self.intrinsic_actor_critic.training_step(posterior_info.posteriors.detach(), posterior_info.deterministics.detach())
        self.extrinsic_actor_critic.training_step(posterior_info.posteriors.detach(), posterior_info.deterministics.detach())

    def forward(self, posterior: Stochastic, deterministic: Deterministic, is_evaluating: bool = False):
        if is_evaluating:
            return self.extrinsic_actor_critic.actor(posterior, deterministic)
        else:
            return self.intrinsic_actor_critic.actor(posterior, deterministic)
