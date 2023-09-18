from gymnasium import spaces

from thesis.cemrl.task_inference import EncoderInput
from .buffer import CemrlReplayBuffer
from .cemrl import Cemrl
from ..core.callback import ExplorationCallback
from ..core.utils import build_network
import torch as th
from stable_baselines3.common.callbacks import BaseCallback


class CemrlExplorationCallback(ExplorationCallback):
    model: Cemrl

    def _link_replay_buffers(self):
        super()._link_replay_buffers()
        if isinstance(self.exploration_algorithm.replay_buffer, CemrlReplayBuffer):
            self.exploration_algorithm.replay_buffer.task_inference = self.model.replay_buffer.task_inference


class TaskEncodingCheckerCallback(BaseCallback):
    model: Cemrl

    def _init_callback(self) -> None:
        self.mlp = build_network(
            self.model.policy.config.task_inference.encoder.latent_dim,
            [100, 100, 100],
            th.nn.ELU,
            spaces.flatdim(self.model.policy.observation_space["goal"]),
        )
        self.mlp.to(self.model.device)
        self.optimizer = th.optim.Adam(self.mlp.parameters(), lr=1e-3)

    def _on_rollout_start(self) -> None:
        if self.num_timesteps < self.model.learning_starts:
            return
        encoder_samples = self.model.replay_buffer.sample_context(
            self.model.batch_size, self.model.get_vec_normalize_env(), self.model.config.encoder_context_length
        )
        with th.no_grad():
            z, *_ = self.model.policy.task_inference.encoder(
                EncoderInput(
                    obs=encoder_samples.observations["observation"],
                    action=encoder_samples.actions,
                    next_obs=encoder_samples.next_observations["observation"],
                    reward=encoder_samples.rewards,
                )
            )
        goals = self.mlp(z)
        loss = th.nn.functional.mse_loss(goals, encoder_samples.observations["goal"][:, -1])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger.record("task_inference/task_encoding_to_goal_loss", loss.item())

    def _on_step(self) -> bool:
        return super()._on_step()