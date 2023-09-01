from typing import Callable
from torch import nn
import torch as th

from src.dreamer import models, networks, tools


class Plan2Explore(nn.Module):
    def __init__(
        self,
        world_model: models.WorldModel,
        num_actions: int,
        disagrement_target: str = "stoch",
        disagrement_action_cond: bool = False,
        disagrement_layers: int = 4,
        disagrement_units: int = 400,
        disagrement_models: int = 10,
        use_amp: bool = False,
        dynamics_discrete: int = 32,
        dynamics_stoch: int = 32,
        dynamics_deter: int = 512,
        actor_layers: int = 2,
        units: int = 512,
        activation: str = "SiLU",
        normalization: str = "LayerNorm",
        actor_distribution: str = "normal",
        actor_init_std: float = 1.0,
        actor_min_std: float = 0.1,
        actor_max_std: float = 1.0,
        actor_temp: float = 0.1,
        action_unimix_ratio: float = 0.01,
        critic_head: str = "symlog_disc",
        critic_layers: int = 2,
        slow_value_target: bool = True,
        actor_lr: float = 3e-5,
        actor_optimizer_eps: float = 1e-5,
        actor_grad_clip: int = 100,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        critic_lr: float = 3e-5,
        critic_grad_clip: int = 100,
        reward_EMA: bool = True,
        imagine_sample: bool = True,
        actor_entropy: float = 3e-4,
        actor_state_entropy: float = 0.0,
        discount_lambda: float = 0.95,
        model_lr: float = 1e-4,
        optimizer_eps: float = 1e-8,
        grad_clip: int = 100,
        imagine_gradient: str = "dynamics",
        future_entropy: bool = False,
        discount: float = 0.997,
        reward: Callable[[th.Tensor, dict[str, th.Tensor]], th.Tensor] | None = None,
    ):
        super(Plan2Explore, self).__init__()

        self.behavior = models.ImagitiveBehavior(
            world_model=world_model,
            num_actions=num_actions,
            dynamics_discrete=dynamics_discrete,
            dynamics_stoch=dynamics_stoch,
            dynamics_deter=dynamics_deter,
            actor_layers=actor_layers,
            units=units,
            activation=activation,
            normalization=normalization,
            actor_distribution=actor_distribution,
            actor_init_std=actor_init_std,
            actor_min_std=actor_min_std,
            actor_max_std=actor_max_std,
            actor_temp=actor_temp,
            action_unimix_ratio=action_unimix_ratio,
            critic_head=critic_head,
            critic_layers=critic_layers,
            slow_value_target=slow_value_target,
            actor_lr=actor_lr,
            actor_optimizer_eps=actor_optimizer_eps,
            actor_grad_clip=actor_grad_clip,
            weight_decay=weight_decay,
            optimizer=optimizer,
            critic_lr=critic_lr,
            critic_grad_clip=critic_grad_clip,
            reward_EMA=reward_EMA,
            imagine_sample=imagine_sample,
            actor_entropy=actor_entropy,
            actor_state_entropy=actor_state_entropy,
            discount_lambda=discount_lambda,
            reward=reward,
            imagine_gradient=imagine_gradient,
            future_entropy=future_entropy,
            discount=discount,
        )

        if dynamics_discrete:
            feat_size = dynamics_stoch * dynamics_discrete + dynamics_deter
            stoch = dynamics_stoch * dynamics_discrete
        else:
            feat_size = dynamics_stoch + dynamics_deter
            stoch = dynamics_stoch

        size = {
            "embed": world_model.embed_size,
            "stoch": stoch,
            "deter": dynamics_deter,
            "feat": dynamics_stoch + dynamics_deter,
        }[disagrement_target]

        self._networks = nn.ModuleList(
            [
                networks.MLP(
                    inp_dim=feat_size + (num_actions if disagrement_action_cond else 0),  # pytorch version
                    shape=size,
                    num_layers=disagrement_layers,
                    hidden_units=disagrement_units,
                    activation=activation,
                )
                for _ in range(disagrement_models)
            ]
        )

        self._model_opt = tools.Optimizer(
            name="explorer",
            parameters=self._networks.parameters,
            lr=model_lr,
            eps=optimizer_eps,
            grad_clip=grad_clip,
            weight_decay=weight_decay,
            optimizer=optimizer,
            use_amp=use_amp,
        )
