import copy
from dataclasses import asdict
from typing import Callable, cast

import numpy as np
import torch
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from torch import nn

from src.dreamer import networks, tools
from src.dreamer.config import Config
from src.dreamer.tools import to_np


class DreamReplayBufferSamples(DictReplayBufferSamples):
    """Samples returned by the DreamReplayBuffer."""

    is_first: th.Tensor
    """A boolean tensor indicating the start of an episode."""


class RewardEMA(th.nn.Module):
    """
    Exponential Moving Average (EMA) of the quantiles of a tensor.

    This module computes the EMA of the quantiles of a tensor, which can be used
    as a reward scaling factor in reinforcement learning. The EMA is updated
    using a fixed alpha parameter and the current quantiles of the input tensor.

    Args:
        alpha (float): EMA smoothing factor (default: 1e-2).
    """

    values: th.Tensor
    range: th.Tensor

    def __init__(self, alpha=1e-2):
        super().__init__()
        self.alpha = alpha

        self.register_buffer("values", torch.zeros((2,)))
        self.register_buffer("range", torch.tensor([0.05, 0.95]))

    def forward(self, x: th.Tensor):
        """
        Forward pass of the QuantileTransformer module.

        Args:
            x (torch.Tensor): Input tensor to be transformed.

        Returns:
            tuple: A tuple containing the transformed tensor's offset and scale.
        """
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    """A PyTorch module that represents the world model used in the Dreamer algorithm."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        use_amp: bool = False,
        dynamics_stoch: int = 32,
        dynamics_deter: int = 512,
        dynamics_hidden: int = 512,
        dynamics_input_layers: int = 1,
        dynamics_output_layers: int = 1,
        dynamics_rec_depth: int = 1,
        dynamics_shared: bool = False,
        dynamics_discrete: int = 32,
        dynamics_mean_activation: str = "none",
        dynamics_std_activation: str = "sigmoid2",
        dynamics_temp_post: bool = True,
        dynamics_min_std: float = 0.1,
        dynamics_cell: str = "gru_layer_norm",
        dynamics_grad_heads=["decoder", "reward", "cont"],
        unimix_ratio: float = 0.01,
        dynamics_initial: str = "learned",
        activation: str = "SiLU",
        normalization: str = "LayerNorm",
        reward_head: str = "symlog_disc",
        reward_layers: int = 2,
        continuation_layers: int = 2,
        units: int = 512,
        reward_scale: float = 1.0,
        continuation_scale: float = 1.0,
        model_lr: float = 1e-4,
        optimizer_eps: float = 1e-8,
        grad_clip: int = 1000,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        enc_mlp_keys="$^",
        enc_cnn_keys="image",
        enc_activation="SiLU",
        enc_normalization="LayerNorm",
        enc_cnn_depth=32,
        enc_kernel_size=4,
        enc_min_res=4,
        enc_mlp_layers=2,
        enc_mlp_units=512,
        enc_symlog_inputs=True,
        dec_mlp_keys="$^",
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
    ):
        """
        Initializes a WorldModel object.

        Args:
            obs_space (gym.spaces.Dict): The observation space of the environment.
            act_space (gym.spaces.Box): The action space of the environment.
            config (dreamer.config.Config): The configuration object used to initialize the world model.
        """
        super(WorldModel, self).__init__()
        self.use_amp = use_amp
        """Whether to use automatic mixed precision for training."""

        shapes = {k: tuple(v.shape) for k, v in observation_space.spaces.items() if isinstance(v, (spaces.Box))}
        self.encoder = networks.MultiEncoder(
            shapes=shapes,
            mlp_keys=enc_mlp_keys,
            cnn_keys=enc_cnn_keys,
            activation=enc_activation,
            normalization=enc_normalization,
            cnn_depth=enc_cnn_depth,
            kernel_size=enc_kernel_size,
            min_res=enc_min_res,
            mlp_layers=enc_mlp_layers,
            mlp_units=enc_mlp_units,
            symlog_inputs=enc_symlog_inputs,
        )
        """A multi-encoder network that encodes the observations."""

        self.embed_size = self.encoder.outdim
        """The size of the encoded observations."""

        self.dynamics = networks.RSSM(
            num_actions=spaces.flatdim(action_space),
            stoch=dynamics_stoch,
            deter=dynamics_deter,
            hidden=dynamics_hidden,
            layers_input=dynamics_input_layers,
            layers_output=dynamics_output_layers,
            rec_depth=dynamics_rec_depth,
            shared=dynamics_shared,
            discrete=dynamics_discrete,
            activation=activation,
            normalization=normalization,
            mean_act=dynamics_mean_activation,
            std_activation=dynamics_std_activation,
            temp_post=dynamics_temp_post,
            min_std=dynamics_min_std,
            cell=dynamics_cell,
            unimix_ratio=unimix_ratio,
            initial=dynamics_initial,
            embed=self.embed_size,
        )
        """A recurrent state-space model that models the dynamics of the environment."""

        self.heads = nn.ModuleDict()
        """A dictionary of the different heads (decoder, reward, cont) of the world model."""

        if dynamics_discrete:
            feat_size = dynamics_stoch * dynamics_discrete + dynamics_deter
        else:
            feat_size = dynamics_stoch + dynamics_deter
        self.decoder = self.heads["decoder"] = networks.MultiDecoder(
            feat_size=feat_size,
            shapes=shapes,
            mlp_keys=dec_mlp_keys,
            activation=dec_activation,
            normalization=dec_normalization,
            cnn_depth=dec_cnn_depth,
            kernel_size=dec_kernel_size,
            min_res=dec_min_res,
            mlp_layers=dec_mlp_layers,
            mlp_units=dec_mlp_units,
            cnn_keys=dec_cnn_keys,
            cnn_sigmoid=dec_cnn_sigmoid,
            image_dist=dec_image_dist,
            vector_dist=dec_vector_dist,
        )
        """A multi-decoder network that decodes the features from the dynamics model."""

        if reward_head == "symlog_disc":
            reward = self.heads["reward"] = networks.MLP(
                feat_size,  # pytorch version
                (255,),
                reward_layers,
                units,
                activation,
                normalization,
                distribution=reward_head,
                outscale=0.0,
            )
        else:
            reward = self.heads["reward"] = networks.MLP(
                feat_size,  # pytorch version
                (1,),
                reward_layers,
                units,
                activation,
                normalization,
                distribution=reward_head,
                outscale=0.0,
            )
        self.reward = reward
        """A network that predicts the reward given the features from the dynamics model."""

        self.cont = self.heads["cont"] = networks.MLP(
            feat_size,  # pytorch version
            (1,),
            continuation_layers,
            units,
            activation,
            normalization,
            distribution="binary",
        )
        """A network that predicts the termination probability given the features from the dynamics model."""

        for name in dynamics_grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters,
            model_lr,
            optimizer_eps,
            grad_clip,
            weight_decay,
            optimizer=optimizer,
            use_amp=self.use_amp,
        )
        """The optimizer used to train the world model."""

        self._scales = dict(reward=reward_scale, cont=continuation_scale)
        """The scales used to scale the reward and termination probability."""

    def video_pred(self, data: DreamReplayBufferSamples) -> torch.Tensor:
        """
        Predicts the next frames of a video sequence given a batch of observations and actions.

        Args:
            data (DreamReplayBufferSamples): A batch of samples from the replay buffer, containing
                observations, actions, rewards, and other metadata.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_frames, 3*3, height, width) containing the
                predicted frames, ground truth frames, and the absolute pixel-wise error between them.
        """
        embed = self.encoder.forward(data.observations)

        states, _ = self.dynamics.observe(embed[:6, :5], data.actions[:6, :5], data.is_first)
        recon = self.decoder.forward(self.dynamics.get_features(states))["image"].mode[:6]
        init = {k: cast(th.Tensor, v)[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine(data.actions[:6, 5:], cast(networks.State, init))
        openl = self.decoder.forward(self.dynamics.get_features(prior))["image"].mode

        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data.observations["image"][:6] + 0.5
        model = model + 0.5
        error = th.abs(model - truth) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagitiveBehavior(nn.Module):
    """A module containing the behavior methods for the Dreamer algorithm."""

    def __init__(
        self,
        world_model: WorldModel,
        num_actions: int,
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
        stop_grad_actor: bool = True,
        imagine_gradient: str = "dynamics",
        future_entropy: bool = False,
        discount: float = 0.997,
        reward: Callable[[th.Tensor, dict[str, th.Tensor]], th.Tensor] | None = None,
    ):
        """
        Initializes an instance of the ImagitiveBehavior class.

        Args:
            config (Config): A configuration object.
            world_model (WorldModel): The world model this agent will be trained against.
            num_actions (int): The number of actions.
            reward (Callable[[th.Tensor, dict[str, th.Tensor]], th.Tensor], optional): A callable object that computes the reward. Defaults to None.
        """
        super(ImagitiveBehavior, self).__init__()
        self._stop_grad_actor = stop_grad_actor
        self._use_amp = use_amp
        """Whether or not to use automatic mixed precision (AMP)."""

        self._world_model = world_model
        """The world model this agent will be trained against."""

        self._reward = reward
        """A callable object that computes the reward."""

        self.imagine_sample = imagine_sample
        self.actor_entropy = actor_entropy
        self.actor_state_entropy = actor_state_entropy
        self.discount_lambda = discount_lambda
        self.discount = discount
        self.future_entropy = future_entropy
        self.reward_EMA = reward_EMA
        self.imag_gradient = imagine_gradient

        if dynamics_discrete:
            feat_size = dynamics_stoch * dynamics_discrete + dynamics_deter
        else:
            feat_size = dynamics_stoch + dynamics_deter
        self.actor = networks.ActionHead(
            feat_size,
            num_actions,
            actor_layers,
            units,
            activation,
            normalization,
            actor_distribution,
            actor_init_std,
            actor_min_std,
            actor_max_std,
            actor_temp,
            outscale=1.0,
            unimix_ratio=action_unimix_ratio,
        )
        """A network that predicts the action given the features from the dynamics model."""

        if critic_head == "symlog_disc":
            critic = networks.MLP(
                feat_size,
                (255,),
                critic_layers,
                units,
                activation,
                normalization,
                critic_head,
                outscale=0.0,
            )
        else:
            critic = networks.MLP(
                feat_size,
                (1,),
                critic_layers,
                units,
                activation,
                normalization,
                critic_head,
                outscale=0.0,
            )
        self.critic = critic
        """A network that predicts the value given the features from the dynamics model."""

        if slow_value_target:
            self._slow_value = copy.deepcopy(self.critic)
            """A copy of the critic network used for the slow value target."""

        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters,
            actor_lr,
            actor_optimizer_eps,
            actor_grad_clip,
            weight_decay=weight_decay,
            optimizer=optimizer,
            use_amp=self._use_amp,
        )
        """The optimizer used to train the actor."""

        self._critic_opt = tools.Optimizer(
            "critic",
            self.critic.parameters,
            critic_lr,
            actor_optimizer_eps,
            critic_grad_clip,
            weight_decay=weight_decay,
            optimizer=optimizer,
            use_amp=self._use_amp,
        )
        """The optimizer used to train the critic."""

        if reward_EMA:
            self.reward_ema = RewardEMA()
            """EMA of the reward."""

    def _imagine(
        self, start: networks.State, policy: networks.ActionHead, horizon: int
    ) -> tuple[th.Tensor, networks.State, th.Tensor]:
        """
        Runs the imagination rollouts for a given number of time steps.

        Args:
            start (networks.State): The initial state of the environment.
            policy (networks.ActionHead): The policy network used to generate actions.
            horizon (int): The number of time steps to simulate.

        Returns:
            A tuple containing the following elements:
            - features (th.Tensor): The features extracted from the imagined states.
            - states (networks.State): The imagined states.
            - actions (th.Tensor): The actions taken in each time step.
        """
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = cast(networks.State, {k: flatten(v) for k, v in start.items()})

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_features(state)
            inp = feat.detach() if self._stop_grad_actor else feat
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action, sample=self.imagine_sample)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(step, (torch.arange(horizon),), (start, None, None))
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, cast(networks.State, states), actions

    def _compute_target(
        self,
        imagine_feat: th.Tensor,
        imagine_state: networks.State,
        reward: th.Tensor,
        actor_ent: th.Tensor,
        state_ent: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Computes the target value for the critic network, along with the weights and the value estimates.

        Args:
            imagine_feat (th.Tensor): The feature tensor obtained from the world model.
            imagine_state (networks.State): The state tensor obtained from the world model.
            reward (th.Tensor): The reward tensor.
            actor_ent (th.Tensor): The entropy of the actor policy.
            state_ent (th.Tensor): The entropy of the state distribution.

        Returns:
            Tuple[th.Tensor, th.Tensor, th.Tensor]: A tuple containing the target tensor, the weights tensor and the value tensor.
        """
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_features(imagine_state)
            discount = self.discount * self._world_model.cont.forward(inp).mean
        else:
            discount = self.discount * torch.ones_like(reward)
        if self.future_entropy and self.actor_entropy > 0:
            reward += self.actor_entropy * actor_ent
        if self.future_entropy and self.actor_state_entropy > 0:
            reward += self.actor_state_entropy * state_ent
        value = self.critic.forward(imagine_feat).mode
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imagine_features: th.Tensor,
        imagine_action: th.Tensor,
        actor_entropy: th.Tensor,
        target: th.Tensor,
        state_entropy: th.Tensor,
        weights: th.Tensor,
        base: th.Tensor,
    ) -> tuple[th.Tensor, dict[str, float | np.ndarray]]:
        """
        Computes the actor loss for the Dreamer model.

        Args:
            imagine_feat (th.Tensor): The imagined features tensor.
            imagine_action (th.Tensor): The imagined actions tensor.
            actor_entropy (th.Tensor): The entropy of the actor policy.
            target (th.Tensor): The target tensor.
            state_entropy (th.Tensor): The entropy of the state.
            weights (th.Tensor): The weights tensor.
            base (th.Tensor): The base tensor.

        Returns:
            Tuple[th.Tensor, Dict[str, Union[float, np.ndarray]]]: A tuple containing the actor loss tensor and a dictionary
            of metrics computed during the loss computation.
        """
        metrics = {}
        # Q-val for actor is not transformed using symlog
        target = torch.transpose(target, 0, 1)
        adv = None
        if self.reward_EMA:
            offset, scale = self.reward_ema(target)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            values = self.reward_ema.values
            metrics["EMA_005"] = to_np(values[0])
            metrics["EMA_095"] = to_np(values[1])

        if self.imag_gradient == "dynamics":
            actor_target = adv
        else:
            raise NotImplementedError(self.imag_gradient)

        if not self.future_entropy and (self.actor_entropy > 0):
            actor_entropy = self.actor_entropy * actor_entropy[:-1][:, :, None]
            actor_target += actor_entropy
        if not self.future_entropy and (self.actor_state_entropy > 0):
            state_entropy = self.actor_state_entropy * state_entropy[:-1]
            actor_target += state_entropy
            metrics["actor_state_entropy"] = to_np(torch.mean(state_entropy))
        actor_loss = -torch.mean(weights[:-1] * actor_target)
        return actor_loss, metrics
