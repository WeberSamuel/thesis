"""Plan2Explore Algorithm."""
from typing import Type
import numpy as np

import torch as th
from gymnasium import Env
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.vec_env import VecEnv

from src.core.state_aware_algorithm import StateAwareOffPolicyAlgorithm
from src.cemrl.buffers1 import EpisodicReplayBuffer
from src.cemrl.buffers import CEMRLReplayBuffer
from src.cemrl.types import CEMRLObsTensorDict
from src.plan2explore.policies import Plan2ExplorePolicy


class Plan2Explore(StateAwareOffPolicyAlgorithm):
    """Uses a world model for exploration via uncertainty and reward prediction during evaluation."""

    def __init__(
        self, policy: Type[Plan2ExplorePolicy], env: Env | VecEnv, learning_rate=1e-3, _init_setup_model=True, learning_starts=1024, **kwargs
    ):
        """Initialize the Algorithm."""
        super().__init__(policy, env, learning_rate, learning_starts=learning_starts, support_multi_env=True, **kwargs, sde_support=False)

        if self.replay_buffer_class == CEMRLReplayBuffer:
            self.replay_buffer_kwargs["encoder"] = self.policy_kwargs["encoder"]
        if _init_setup_model:
            self._setup_model()

        self.policy: Plan2ExplorePolicy
        self.scaler = th.cuda.amp.GradScaler()
        self.log_prefix = "p2e/"

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """Train the policy .

        Args:
            gradient_steps (int): Defines how many steps to perform
            batch_size (int): Defines how large the batch_size should be
        """
        assert (
            isinstance(self.policy, Plan2ExplorePolicy)
            and self.policy.optimizer is not None
            and self.replay_buffer is not None
        )
        self.policy.set_training_mode(True)

        for _ in range(gradient_steps):
            if isinstance(self.replay_buffer, EpisodicReplayBuffer):
                enc_input, dec_input = self.replay_buffer.cemrl_sample(batch_size)
                obs = dec_input.observations["observation"]
                next_obs = dec_input.next_observations["observation"]
                _, z = self.replay_buffer.encoder(enc_input.observations)
                dec_timesteps = dec_input.actions.shape[1]
                z = th.broadcast_to(z[:, None], (batch_size, dec_timesteps, *z.shape[1:]))
                z = z.reshape(batch_size * dec_timesteps, *z.shape[2:])
                actions = dec_input.actions.view(batch_size * dec_timesteps, *dec_input.actions.shape[2:])
                rewards = dec_input.rewards.view(batch_size * dec_timesteps, *dec_input.rewards.shape[2:])
                next_obs = {"observation": next_obs.view(batch_size * dec_timesteps, *obs.shape[2:]), "task_indicator": z}
                obs = {"observation": obs.view(batch_size * dec_timesteps, *obs.shape[2:]), "task_indicator": z}

            elif isinstance(self.replay_buffer, DictReplayBuffer):
                (obs, actions, next_obs, dones, rewards) = self.replay_buffer.sample(batch_size)
            else:
                (obs, actions, next_obs, dones, rewards) = self.replay_buffer.sample(batch_size)
                obs = {"observation": obs}
                next_obs = {"observation": next_obs}

            z = None
            z = next_obs.get("goal", z)

            if isinstance(self.replay_buffer, (CEMRLReplayBuffer, EpisodicReplayBuffer)):
                z = obs["task_indicator"]
            elif self.policy.latent_generator is not None:
                latent_input = CEMRLObsTensorDict(observation=obs["observation"], reward=rewards, action=actions)
                z = self.policy.latent_generator(latent_input)

            obs = obs["observation"]
            next_obs = next_obs["observation"]

            pred_next_obs, pred_reward = self.policy.ensemble(obs, actions, z=z, return_raw=True)
            obs_loss = th.nn.functional.mse_loss(pred_next_obs, next_obs[None].expand_as(pred_next_obs))
            reward_loss = th.nn.functional.mse_loss(pred_reward, rewards[None].expand_as(pred_reward))
            loss = obs_loss + reward_loss

            self.logger.record_mean(f"{self.log_prefix}obs_loss", obs_loss.detach().item())
            self.logger.record_mean(f"{self.log_prefix}reward_loss", reward_loss.detach().item())
            self.logger.record_mean(f"{self.log_prefix}loss", loss.detach().item())

            self.policy.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.policy.optimizer)
            self.scaler.update()

        self._n_updates += gradient_steps
        self.logger.record(f"{self.log_prefix}n_updates", self._n_updates, exclude="tensorboard")
