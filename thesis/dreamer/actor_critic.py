from typing import Any, NamedTuple

import torch as th
from torch.distributions import TanhTransform

from ..core.module import BaseModule
from ..core.utils import build_network
from .config import ActorConfig, ActorCriticConfig, CriticConfig
from .world_model import Deterministic, DreamerWorldModel, Stochastic, create_normal_dist, horizontal_forward


def compute_lambda_values(rewards: th.Tensor, values: th.Tensor, continues: th.Tensor, horizon_length: int, lambda_: float):
    """
    rewards : (batch_size, time_step, hidden_size)
    values : (batch_size, time_step, hidden_size)
    continue flag will be added
    """
    rewards = rewards[:, :-1]
    continues = continues[:, :-1]
    next_values = values[:, 1:]
    last = next_values[:, -1]
    inputs = rewards + continues * next_values * (1 - lambda_)

    outputs = []
    # single step
    for index in reversed(range(horizon_length - 1)):
        last = inputs[:, index] + continues[:, index] * lambda_ * last
        outputs.append(last)
    returns = th.stack(list(reversed(outputs)), dim=1).to(rewards.device)
    return returns


class AgentInfos(NamedTuple):
    priors: Stochastic
    deterministics: Deterministic
    actions: th.Tensor


class Actor(th.nn.Module):
    def __init__(
        self,
        discrete_action_bool: bool,
        action_size: int,
        stochastic_size: int,
        deterministic_size: int,
        config: ActorConfig,
    ):
        super().__init__()
        self.config = config
        self.discrete_action_bool = discrete_action_bool
        action_size = action_size if discrete_action_bool else 2 * action_size

        self.model = build_network(stochastic_size + deterministic_size, config.layers, config.activation, action_size)

    def forward(self, posterior: Stochastic, deterministic: Deterministic):
        x = th.cat((posterior, deterministic), -1)
        x = self.model(x)
        if self.discrete_action_bool:
            dist = th.distributions.OneHotCategoricalStraightThrough(logits=x)
            action = dist.rsample()
        else:
            dist = create_normal_dist(
                x,
                mean_scale=self.config.mean_scale,
                init_std=self.config.init_std,
                min_std=self.config.min_std,
                activation=th.nn.Tanh(),
            )
            dist = th.distributions.TransformedDistribution(dist, TanhTransform())
            action = th.distributions.Independent(dist, 1).rsample()
        return action


class Critic(th.nn.Module):
    def __init__(self, stochastic_size: int, deterministic_size: int, config: CriticConfig):
        super().__init__()
        self.network = build_network(stochastic_size + deterministic_size, config.layers, config.activation, 1)

    def forward(self, posterior: Stochastic, deterministic: Deterministic):
        x = horizontal_forward(self.network, posterior, deterministic, output_shape=(1,))
        dist = create_normal_dist(x, std=1, event_shape=1)
        return dist


class ActorCritic(BaseModule):
    def __init__(
        self,
        discrete_action_bool: bool,
        action_size: int,
        stochastic_size: int,
        deterministic_size: int,
        world_model: DreamerWorldModel,
        config: ActorCriticConfig,
    ) -> None:
        super().__init__()
        self.training_config = config.actor_critic_training_config
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.world_model = world_model

        self.actor = Actor(
            discrete_action_bool=discrete_action_bool,
            action_size=action_size,
            stochastic_size=stochastic_size,
            deterministic_size=deterministic_size,
            config=config.actor_config,
        )

        self.critic = Critic(
            stochastic_size=stochastic_size,
            deterministic_size=deterministic_size,
            config=config.critic_config,
        )

        optimizer_class = self.training_config.optimizer_class
        self.actor_optimizer = optimizer_class(self.actor.parameters(), lr=self.training_config.actor_optimizer_lr)  # type: ignore
        self.critic_optimizer = optimizer_class(self.critic.parameters(), lr=self.training_config.critic_optimizer_lr)  # type: ignore

    def training_step(self, posteriors: Stochastic, deterministics: Deterministic):
        """
        #TODO : last posterior truncation(last can be last step)
        posterior shape : (batch, timestep, stochastic)
        """
        infos = self.imagine(posteriors, deterministics)

        predicted_rewards = self.get_reward_prediction(infos)
        values = self.critic(infos.priors, infos.deterministics).mean

        if self.training_config.use_continue_flag:
            continues = self.world_model.continue_predictor(infos.priors, infos.deterministics).mean
        else:
            continues = self.training_config.discount * th.ones_like(values)

        lambda_values = compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            self.training_config.horizon_length,
            self.training_config.lambda_,
        )

        actor_loss = -th.mean(lambda_values)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        th.nn.utils.clip_grad_norm_(  # type: ignore
            self.actor.parameters(),
            self.training_config.clip_grad,
            norm_type=self.training_config.grad_norm_type,
        )
        self.actor_optimizer.step()

        value_dist = self.critic(
            infos.priors.detach()[:, :-1],
            infos.deterministics.detach()[:, :-1],
        )
        value_loss = -th.mean(value_dist.log_prob(lambda_values.detach()))

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        th.nn.utils.clip_grad_norm_(  # type: ignore
            self.critic.parameters(),
            self.training_config.clip_grad,
            norm_type=self.training_config.grad_norm_type,
        )
        self.critic_optimizer.step()
        return {
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
        }

    def get_reward_prediction(self, infos: AgentInfos):
        predicted_rewards = self.world_model.reward_predictor(infos.priors, infos.deterministics).mean
        return predicted_rewards

    def imagine(self, posteriors: Stochastic, deterministics: Deterministic):
        state = posteriors.reshape(-1, self.stochastic_size)
        deterministic = deterministics.reshape(-1, self.deterministic_size)

        cache = []
        # continue_predictor reinit
        for t in range(self.training_config.horizon_length):
            action = self.actor(state, deterministic)
            deterministic = self.world_model.rssm.sequence_model.forward(state, action, deterministic)
            _, state = self.world_model.rssm.transition_model(deterministic)
            cache.append(dict(priors=state, deterministics=deterministic, actions=action))

        infos = {k: th.stack([c[k] for c in cache], 1) for k in cache[0].keys()}
        return AgentInfos(**infos)
