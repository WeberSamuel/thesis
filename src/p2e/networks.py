import torch as th
from src.cemrl.task_inference import REWARD_DIM
from src.plan2explore.networks import Ensemble
from src.utils import build_network
from .config import OneStepModelConfig


class LatentDisagreementEnsemble(th.nn.Module):
    def __init__(self, ensemble: Ensemble) -> None:
        super().__init__()
        self.ensemble = ensemble

    def forward(self, *args, **kwargs):
        (state_var, state_mean), (reward_var, reward_mean) = self.ensemble(*args, **kwargs)
        return state_mean, th.cat([state_var, reward_var], dim=-1).mean(dim=-1)


class OneStepModel(th.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, config: OneStepModelConfig) -> None:
        super().__init__()
        input_size = obs_dim + action_dim + latent_dim
        self.net = build_network(
            input_size, [int(input_size * config.complexity)] * config.layers, th.nn.ReLU, obs_dim + REWARD_DIM
        )

    def forward(self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor | None, z: th.Tensor):
        x = self.net(th.cat([state, action, z], dim=-1))
        return x[..., :-REWARD_DIM], x[..., -REWARD_DIM:]
