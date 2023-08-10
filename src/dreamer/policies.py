import os
from typing import Tuple, cast

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from torch import distributions as torchd

from src.dreamer.networks import State
import src.dreamer.exploration as expl
import src.dreamer.models as models
from src.core.state_aware_algorithm import StateAwarePolicy
from src.dreamer.config import Config


class DreamerPolicy(StateAwarePolicy):
    """A policy class that implements the Dreamer algorithm for model-based reinforcement learning."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        learning_rate: Schedule,
        compile=True,
        use_amp=False,
        reward_EMA=True,
        dynamics_cell="gru_layer_norm",
        dynamics_hidden=512,
        dynamics_deter=512,
        dynamics_stoch=32,
        dynamics_discrete=32,
        dynamics_input_layers=1,
        dynamics_output_layers=1,
        dynamics_rec_depth=1,
        dynamics_shared=False,
        dynamics_mean_activation="none",
        dynamics_std_activation="sigmoid2",
        dynamics_min_std=0.1,
        dynamics_temp_post=True,
        dynamics_grad_heads=["decoder", "reward", "cont"],
        units=512,
        reward_layers=2,
        continuation_layers=2,
        critic_layers=2,
        actor_layers=2,
        activation="SiLU",
        normalization="LayerNorm",
        enc_mlp_keys="observation",
        enc_cnn_keys="image",
        enc_activation="SiLU",
        enc_normalization="LayerNorm",
        enc_cnn_depth=32,
        enc_kernel_size=4,
        enc_min_res=4,
        enc_mlp_layers=2,
        enc_mlp_units=512,
        enc_symlog_inputs=True,
        dec_mlp_keys="observation",
        dec_cnn_keys="image",
        dec_activation="SiLU",
        dec_normalization="LayerNorm",
        dec_cnn_depth=32,
        dec_kernel_size=4,
        dec_min_res=4,
        dec_mlp_layers=2,
        dec_mlp_units=512,
        dec_cnn_sigmoid=False,
        dec_image_dist="mse",
        dec_vector_dist="symlog_mse",
        critic_head="symlog_disc",
        reward_head="symlog_disc",
        dynamics_scale=0.5,
        representation_scale=0.1,
        kl_free=1.0,
        continuation_scale=1.0,
        reward_scale=1.0,
        weight_decay=0.0,
        unimix_ratio=0.01,
        action_unimix_ratio=0.01,
        dynamics_initial="learned",
        batch_size=16,
        batch_length=64,
        train_ratio=512,
        pretrain=100,
        model_lr=1e-4,
        optimizer_eps=1e-8,
        grad_clip=1000,
        critic_lr=3e-5,
        actor_lr=3e-5,
        actor_optimizer_eps=1e-5,
        critic_grad_clip=100,
        actor_grad_clip=100,
        slow_value_target=True,
        slow_target_update=1,
        slow_target_fraction=0.02,
        optimizer="adam",
        # Behavior.,
        discount=0.997,
        discount_lambda=0.95,
        imagine_horizon=15,
        imagine_gradient="dynamics",
        imagine_gradient_mix=0.0,
        imagine_sample=True,
        actor_distribution="normal",
        actor_entropy=3e-4,
        actor_state_entropy=0.0,
        actor_init_std=1.0,
        actor_min_std=0.1,
        actor_max_std=1.0,
        actor_temp=0.1,
        explore_amount=0.0,
        eval_state_mean=False,
        collect_dynamics_sample=True,
        behavior_stop_grad=True,
        value_decay=0.0,
        future_entropy=False,
        # Exploration,
        explore_until=0,
        explore_extr_scale=0.0,
        explore_intr_scale=1.0,
        disagrement_target="stoch",
        use_log_disagrement=True,
        disagrement_models=10,
        disagrement_offset=1,
        disagrement_layers=4,
        disagrement_units=400,
        disagrement_action_cond=False,
        **kwargs,
    ):
        """
        Initialize a Dreamer policy object.

        Args:
            observation_space (gym.spaces.Dict): The observation space of the environment.
            action_space (gym.spaces.Box): The action space of the environment.
            learning_rate (Schedule): The learning rate schedule for the optimizer.
            config (Config): The configuration object for the policy.
            **kwargs: Additional keyword arguments.

        Attributes:
            config (Config): The configuration object for the policy.
            num_actions (int): The number of actions in the action space.
            world_model (models.WorldModel): The world model used by the policy.
            task_behavior (models.ImagitiveBehavior): The task behavior model used by the policy.
            expl_behavior (expl.Exploration): The exploration behavior used by the policy.
            use_intrinsic (bool): Whether to use intrinsic rewards or not.
        """
        super().__init__(observation_space, action_space, **kwargs)
        self.kl_free = kl_free
        self.representation_scale = representation_scale
        self.dynamics_scale = dynamics_scale
        self.dynamics_discrete = dynamics_discrete

        self.disagrement_target = disagrement_target
        self.disagrement_offset = disagrement_offset
        self.disagrement_action_cond = disagrement_action_cond
        self.use_log_disagrement = use_log_disagrement
        self.exploration_intrinsic_scale = explore_intr_scale
        self.exploration_extrinsic_scale = explore_extr_scale
        self.actor_distribution = actor_distribution

        self.slow_value_target = slow_value_target
        self.slow_target_update = slow_target_update
        self.slow_target_fraction = slow_target_fraction
        self.behavior_stop_grad = behavior_stop_grad
        self.value_decay = value_decay
        self.imagine_horizon = imagine_horizon
        self.dynamics_grad_heads = dynamics_grad_heads

        self.num_actions = spaces.flatdim(self.action_space)

        self.world_model = models.WorldModel(
            observation_space,
            action_space,
            use_amp=use_amp,
            dynamics_stoch=dynamics_stoch,
            dynamics_deter=dynamics_deter,
            dynamics_hidden=dynamics_hidden,
            dynamics_input_layers=dynamics_input_layers,
            dynamics_output_layers=dynamics_output_layers,
            dynamics_rec_depth=dynamics_rec_depth,
            dynamics_shared=dynamics_shared,
            dynamics_discrete=dynamics_discrete,
            dynamics_mean_activation=dynamics_mean_activation,
            dynamics_std_activation=dynamics_std_activation,
            dynamics_temp_post=dynamics_temp_post,
            dynamics_min_std=dynamics_min_std,
            dynamics_cell=dynamics_cell,
            dynamics_grad_heads=dynamics_grad_heads,
            unimix_ratio=unimix_ratio,
            dynamics_initial=dynamics_initial,
            activation=activation,
            normalization=normalization,
            reward_head=reward_head,
            reward_layers=reward_layers,
            continuation_layers=continuation_layers,
            units=units,
            reward_scale=reward_scale,
            continuation_scale=continuation_scale,
            model_lr=model_lr,
            optimizer_eps=optimizer_eps,
            grad_clip=grad_clip,
            weight_decay=weight_decay,
            optimizer=optimizer,
            enc_mlp_keys=enc_mlp_keys,
            enc_cnn_keys=enc_cnn_keys,
            enc_activation=enc_activation,
            enc_normalization=enc_normalization,
            enc_cnn_depth=enc_cnn_depth,
            enc_kernel_size=enc_kernel_size,
            enc_min_res=enc_min_res,
            enc_mlp_layers=enc_mlp_layers,
            enc_mlp_units=enc_mlp_units,
            enc_symlog_inputs=enc_symlog_inputs,
            dec_mlp_keys=dec_mlp_keys,
            dec_cnn_keys=dec_cnn_keys,
            dec_activation=dec_activation,
            dec_normalization=dec_normalization,
            dec_cnn_depth=dec_cnn_depth,
            dec_kernel_size=dec_kernel_size,
            dec_min_res=dec_min_res,
            dec_mlp_layers=dec_mlp_layers,
            dec_mlp_units=dec_mlp_units,
            dec_cnn_sigmoid=dec_cnn_sigmoid,
            dec_image_dist=dec_image_dist,
            dec_vector_dist=dec_vector_dist,
        )

        self.task_behavior = models.ImagitiveBehavior(
            self.world_model,
            self.num_actions,
            use_amp=use_amp,
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
            discount=discount,
            future_entropy=future_entropy,
            imagine_gradient=imagine_gradient
        )

        self.expl_behavior = expl.Plan2Explore(
            self.world_model,
            self.num_actions,
            disagrement_target=disagrement_target,
            disagrement_action_cond=disagrement_action_cond,
            disagrement_layers=disagrement_layers,
            disagrement_units=disagrement_units,
            disagrement_models=disagrement_models,
            use_amp=use_amp,
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
            model_lr=model_lr,
            optimizer_eps=optimizer_eps,
            grad_clip=grad_clip,
            discount=discount,
            future_entropy=future_entropy,
            imagine_gradient=imagine_gradient
        )

        if compile and os.name != "nt":  # compilation is not supported on windows
            self.world_model = th.compile(self._wm)  # type: ignore
            self.task_behavior = th.compile(self._task_behavior)  # type: ignore

        self.use_intrinsic = False
        self.collect_dynamics_sample = collect_dynamics_sample

    def _predict(self, observation: dict[str, th.Tensor], deterministic: bool = False) -> th.Tensor:
        logit, stoch, deter, action = self.state
        latent = State(stoch=stoch, deter=deter)
        latent["logit"] = logit  # type: ignore
        latent, action, logprob = self.policy_step(observation, latent, action)
        self.state = (latent["logit"], latent["stoch"], latent["deter"], action)
        return action

    def policy_step(self, observation: dict[str, th.Tensor], latent: State, action: th.Tensor):
        """
        Run a single step of the policy, taking in an observation, latent state, and previous action, and returning
        the updated latent state, new action, and the log probability of the action.

        Args:
            observation (dict): A dictionary containing the current observation.
            latent (dict): A dictionary containing the current latent state.
            action (torch.Tensor): The previous action taken.

        Returns:
            Tuple[dict, torch.Tensor, torch.Tensor]: A tuple containing the updated latent state, new action, and the
            log probability of the action.
        """
        embed = self.world_model.encoder(observation)
        latent, _ = self.world_model.dynamics.obs_step(
            latent,
            action,
            embed,
            observation.get("is_first", th.zeros_like(action, dtype=th.bool)),
            self.collect_dynamics_sample,
        )
        feat = self.world_model.dynamics.get_features(latent)

        if self.use_intrinsic:
            actor: torchd.Distribution = self.expl_behavior.behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self.task_behavior.actor(feat)
            action = actor.sample() if self.training else actor.mode

        logprob = actor.log_prob(action)

        # detach state variables
        latent = cast(State, {k: cast(th.Tensor, v).detach() for k, v in latent.items()})
        action = action.detach()
        return latent, action, logprob

    def _reset_states(self, size: int) -> Tuple[np.ndarray, ...]:
        latent = self.world_model.dynamics.initial(size)
        action = th.zeros(size, self.num_actions).to(self.device)
        return (latent["logit"], latent["stoch"], latent["deter"], action)  # type: ignore
