from src.dreamer.config import DreamerConfig
from src.plan2explore.networks import WorldModel
import torch as th
from submodules.dreamer.dreamer import Dreamer
from gymnasium import spaces


class CemrlWorldModel(th.nn.Module):
    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Box, config) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config

    def _build(self, cemrl_decoder: WorldModel | None = None):
        self.cemrl_decoder = cemrl_decoder or WorldModel(
            spaces.flatdim(self.observation_space),
            spaces.flatdim(self.action_space),
            self.config.latent_dim,
            self.config.net_complexity,
        )

    def forward(self, obs: th.Tensor, action: th.Tensor, task_encoding: th.Tensor):
        obs, reward = self.cemrl_decoder.forward(obs, action, z=task_encoding)
        return obs, reward, None


class DreamerWorldModel(th.nn.Module):
    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Box, config: DreamerConfig) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config

    def _build(self, dreamer: Dreamer | None = None):
        self.dreamer = dreamer or Dreamer(self.observation_space, self.action_space, self.config)

    def forward_latent(self, latent: dict[str, th.Tensor], action: th.Tensor, task_encoding: th.Tensor):
        latent = self.dreamer._wm.dynamics.img_step(latent, action, None, self.dreamer._config.collect_dyn_sample)
        if self.dreamer._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self.dreamer._wm.dynamics.get_feat(latent)
        reward = self.dreamer._wm.heads["reward"](feat)
        return latent, reward, self.dreamer._task_behavior.value(feat)

    def forward(self, observation: dict[str, th.Tensor], action: th.Tensor, task_encoding: th.Tensor):
        embed = self.dreamer._wm.encoder(observation)
        latent, _ = self.dreamer._wm.dynamics.observe(embed, action, observation["is_first"])
        latent, reward, value = self.forward_latent(latent, action, task_encoding)
        obs = self.dreamer._wm.heads["decoder"](latent)
        return obs, reward, value
