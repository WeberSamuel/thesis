from dataclasses import asdict, replace

import torch as th, torch
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from submodules.dreamer.dreamer import Dreamer as ExternalDreamer

from src.core.state_aware_algorithm import StateAwarePolicy
from src.dreamer.config import DecoderConfig, DreamerConfig, EncoderConfig

class DreamerPolicy(StateAwarePolicy):
    state: dict[str, th.Tensor]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        dreamer_config=DreamerConfig(),
        **kwargs,
    ):
        super().__init__(observation_space, action_space)
        dreamer_config.encoder=EncoderConfig(
            mlp_keys=dreamer_config.encoder_mlp_keys,
            cnn_keys=dreamer_config.encoder_cnn_keys,
            act=dreamer_config.encoder_act,
            norm=dreamer_config.encoder_norm,
            cnn_depth=dreamer_config.encoder_cnn_depth,
            kernel_size=dreamer_config.encoder_kernel_size,
            minres=dreamer_config.encoder_minres,
            mlp_layers=dreamer_config.encoder_mlp_layers,
            mlp_units=dreamer_config.encoder_mlp_units,
            symlog_inputs=dreamer_config.encoder_symlog_inputs,
        )
        dreamer_config.decoder=DecoderConfig(
            mlp_keys=dreamer_config.decoder_mlp_keys,
            cnn_keys=dreamer_config.decoder_cnn_keys,
            act=dreamer_config.decoder_act,
            norm=dreamer_config.decoder_norm,
            cnn_depth=dreamer_config.decoder_cnn_depth,
            kernel_size=dreamer_config.decoder_kernel_size,
            minres=dreamer_config.decoder_minres,
            mlp_layers=dreamer_config.decoder_mlp_layers,
            mlp_units=dreamer_config.decoder_mlp_units,
            cnn_sigmoid=dreamer_config.decoder_cnn_sigmoid,
            image_dist=dreamer_config.decoder_image_dist,
            vector_dist=dreamer_config.decoder_vector_dist,
        )
        dreamer_config = replace(dreamer_config)  # necessary because save config otherwise would have to dump instances
        dreamer_config.encoder=EncoderConfig(
            mlp_keys=dreamer_config.encoder_mlp_keys,
            cnn_keys=dreamer_config.encoder_cnn_keys,
            act=dreamer_config.encoder_act,
            norm=dreamer_config.encoder_norm,
            cnn_depth=dreamer_config.encoder_cnn_depth,
            kernel_size=dreamer_config.encoder_kernel_size,
            minres=dreamer_config.encoder_minres,
            mlp_layers=dreamer_config.encoder_mlp_layers,
            mlp_units=dreamer_config.encoder_mlp_units,
            symlog_inputs=dreamer_config.encoder_symlog_inputs,
        )
        dreamer_config.decoder=DecoderConfig(
            mlp_keys=dreamer_config.decoder_mlp_keys,
            cnn_keys=dreamer_config.decoder_cnn_keys,
            act=dreamer_config.decoder_act,
            norm=dreamer_config.decoder_norm,
            cnn_depth=dreamer_config.decoder_cnn_depth,
            kernel_size=dreamer_config.decoder_kernel_size,
            minres=dreamer_config.decoder_minres,
            mlp_layers=dreamer_config.decoder_mlp_layers,
            mlp_units=dreamer_config.decoder_mlp_units,
            cnn_sigmoid=dreamer_config.decoder_cnn_sigmoid,
            image_dist=dreamer_config.decoder_image_dist,
            vector_dist=dreamer_config.decoder_vector_dist,
        )
        dreamer_config.grad_heads = ["decoder", "reward", "cont"]  # type: ignore
        dreamer_config.encoder = asdict(dreamer_config.encoder)  # type: ignore
        dreamer_config.decoder = asdict(dreamer_config.decoder)  # type: ignore
        dreamer_config.num_actions = spaces.flatdim(action_space)  # type: ignore
        self.dreamer = ExternalDreamer(observation_space, action_space, dreamer_config)
        self.is_evaluating = False

    def _predict(self, observation: dict[str, th.Tensor], deterministic: bool = False) -> th.Tensor:
        latent = self.state
        action = latent.pop("_action_state")
        observation["is_first"] = observation["is_first"].squeeze(-1)
        observation["is_first"] = observation["is_terminal"].squeeze(-1)
        action, (latent, _) = self.dreamer._policy(observation, (latent, action), not self.is_evaluating)
        latent["_action_state"] = action["action"]
        self.state = latent
        return action["action"]

    def _reset_states(self, size: int) -> dict[str, th.Tensor]:
        latent = self.dreamer._wm.dynamics.initial(size)
        action = torch.zeros(size, self.dreamer._config.num_actions, device=self.device)  # type: ignore
        latent["_action_state"] = action
        return latent
