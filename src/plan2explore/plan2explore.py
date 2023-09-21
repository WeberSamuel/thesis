"""Plan2Explore Algorithm."""
from typing import Type
import numpy as np

import torch as th
from gymnasium import Env
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.vec_env import VecEnv

from thesis.core.algorithm import BaseAlgorithm
from src.cemrl.types import CEMRLObsTensorDict
from src.plan2explore.policies import Plan2ExplorePolicy
from thesis.core.types import EncoderInput
from thesis.cemrl.algorithm import Cemrl

class Plan2Explore(BaseAlgorithm):
    """Uses a world model for exploration via uncertainty and reward prediction during evaluation."""

    def __init__(
        self, policy: Type[Plan2ExplorePolicy], env: Env | VecEnv, learning_rate=1e-3, _init_setup_model=True, learning_starts=1024, gradient_steps=1, train_freq=1, main_algorithm: BaseAlgorithm|None = None, **kwargs
    ):
        if isinstance(main_algorithm, Cemrl):
            kwargs.setdefault("policy_kwargs", {})["encoder"] = main_algorithm.policy.task_inference.encoder
        """Initialize the Algorithm."""
        super().__init__(policy, env, learning_rate, learning_starts=learning_starts, gradient_steps=gradient_steps, train_freq=train_freq, **kwargs)

        if _init_setup_model:
            self._setup_model()

        self.policy: Plan2ExplorePolicy
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
            task_encoder = getattr(self.replay_buffer, "task_encoder", None)
            assert task_encoder is not None
            enc_input, dec_input = self.replay_buffer.task_inference_sample(batch_size)
            obs = dec_input.observations["observation"]
            next_obs = dec_input.next_observations["observation"]
            z, _, *_ = task_encoder(EncoderInput(
                obs=enc_input.observations["observation"],
                action=enc_input.actions,
                reward=enc_input.rewards,
                next_obs=enc_input.next_observations["observation"],
            ))
            dec_timesteps = dec_input.actions.shape[1]
            z = th.broadcast_to(z[:, None], (batch_size, dec_timesteps, *z.shape[1:]))
            z = z.reshape(batch_size * dec_timesteps, *z.shape[2:])
            actions = dec_input.actions.view(batch_size * dec_timesteps, *dec_input.actions.shape[2:])
            rewards = dec_input.rewards.view(batch_size * dec_timesteps, *dec_input.rewards.shape[2:])
            next_obs = {"observation": next_obs.view(batch_size * dec_timesteps, *obs.shape[2:]), "task_indicator": z}
            obs = {"observation": obs.view(batch_size * dec_timesteps, *obs.shape[2:]), "task_indicator": z}

            z = None
            z = obs["task_indicator"]

            obs = obs["observation"]
            next_obs = next_obs["observation"]

            pred_next_obs, pred_reward = self.policy.ensemble(th.cat([obs, actions, z], dim=-1), return_raw=True)
            obs_loss = th.nn.functional.mse_loss(pred_next_obs, next_obs[None].expand_as(pred_next_obs))
            reward_loss = th.nn.functional.mse_loss(pred_reward, rewards[None].expand_as(pred_reward))
            loss = obs_loss + reward_loss

            self.logger.record_mean(f"{self.log_prefix}obs_loss", obs_loss.detach().item())
            self.logger.record_mean(f"{self.log_prefix}reward_loss", reward_loss.detach().item())
            self.logger.record_mean(f"{self.log_prefix}loss", loss.detach().item())

            self.policy.optimizer.zero_grad()

            loss.backward()

            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record(f"{self.log_prefix}n_updates", self._n_updates, exclude="tensorboard")

        super().train(gradient_steps, batch_size)
