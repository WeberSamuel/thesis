from random import sample
from typing import List
import torch as th
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.torch_layers import create_mlp
from gymnasium.spaces import flatdim
from src.cemrl.cemrl import CEMRL
from src.envs.meta_env import MetaVecEnv
from src.cemrl.policies import CEMRLPolicy
from stable_baselines3.common.buffers import DictReplayBuffer


class LatentToGoalCallback(BaseCallback):
    def __init__(self, gradient_steps=5, verbose: int = 0):
        super().__init__(verbose)
        self.gradient_steps = gradient_steps

    def _init_callback(self) -> None:
        assert isinstance(self.model.policy, CEMRLPolicy)
        assert self.training_env is not None and isinstance(self.training_env.unwrapped, MetaVecEnv)
        latent_dim = self.model.policy.latent_dim
        goal_dim = flatdim(self.training_env.unwrapped.goal_sampler.goal_space)
        self.network = th.nn.Sequential(*create_mlp(latent_dim, goal_dim, [64, 64]))
        self.network = self.network.to(self.model.device)
        self.optimizer = th.optim.Adam(self.network.parameters())

    def _on_rollout_start(self) -> None:
        assert isinstance(self.model, CEMRL)
        assert isinstance(self.model.policy, CEMRLPolicy)
        assert isinstance(self.model.replay_buffer, DictReplayBuffer)
        assert self.model.replay_buffer is not None
        batch_size = self.model.batch_size

        if not self.model.replay_buffer.full and self.model.replay_buffer.pos < batch_size:
            return

        for _ in range(self.gradient_steps):
            samples = self.model.replay_buffer.sample(batch_size, self.model._vec_normalize_env)
            with th.no_grad():
                y, z = self.model.policy.encoder(samples.observations)
            goal = samples.next_observations["goal"][:, -1]

            prediction = self.network(z)
            loss = th.nn.functional.mse_loss(prediction, goal)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.logger.record_mean("reconstruction/latent_to_goal_loss", loss.item())

    def _on_step(self) -> bool:
        return super()._on_step()
