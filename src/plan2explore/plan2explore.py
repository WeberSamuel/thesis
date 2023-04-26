"""Plan2Explore Algorithm."""
from typing import Type
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import RolloutReturn
import torch as th
from gym import Env
from src.cli import DummyPolicy
from src.env.wrappers.include_goal import IncludeGoalWrapper
from src.cemrl.wrappers.cemrl_history_wrapper import CEMRLHistoryWrapper
from src.plan2explore.policies import Plan2ExplorePolicy
from stable_baselines3.common.vec_env import is_vecenv_wrapped, VecEnv
from stable_baselines3.common.buffers import ReplayBuffer


class Plan2Explore(OffPolicyAlgorithm):
    """Uses a world model for exploration via uncertainty and reward prediction during evaluation."""

    def __init__(self, policy: Plan2ExplorePolicy, env: VecEnv|Env, replay_buffer: ReplayBuffer, learning_rate=1e-3, _init_setup_model=True, **kwargs):
        """Initialize the Algorithm."""
        super().__init__(DummyPolicy, env, learning_rate, policy_kwargs={}, buffer_size=0, support_multi_env=True, **kwargs)
        self.replay_buffer = replay_buffer
        
        if _init_setup_model:
            self._setup_model()
        
        self.policy = policy

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
        contains_history = is_vecenv_wrapped(self.env, CEMRLHistoryWrapper)
        contains_goals = is_vecenv_wrapped(self.env, IncludeGoalWrapper)

        for _ in range(gradient_steps):
            (observations, actions, next_observations, dones, rewards) = self.replay_buffer.sample(batch_size)
            z = None
            if contains_goals:
                z = next_observations["goal"]
            if contains_history:
                if contains_goals:
                    z = next_observations["goal"][:, -1]
                observations = observations["observation"][:, -1]
                next_observations = next_observations["observation"][:, -1]
            assert th.allclose(next_observations - observations, actions, atol=1e-5)
            assert th.allclose(-th.norm(z - next_observations, dim=-1), rewards[:, 0])
            pred_next_obs, pred_reward = self.policy.ensemble(observations, actions, z=z, return_raw=True)
            obs_loss = th.nn.functional.mse_loss(pred_next_obs, next_observations[None].expand_as(pred_next_obs))
            reward_loss = th.nn.functional.mse_loss(pred_reward, rewards[None].expand_as(pred_reward))
            loss = obs_loss + reward_loss

            self.logger.record_mean("obs_loss", obs_loss.detach().item())
            self.logger.record_mean("reward_loss", reward_loss.detach().item())
            self.logger.record_mean("loss", loss.detach().item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
