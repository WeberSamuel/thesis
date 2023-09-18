from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from ..core import BasePolicy
from .config import SmmAgentConfig, SmmConfig
from .ddpg import DDPGAgent
from .networks import SMM
from .utils import soft_update_params, to_torch


class SMMAgent(DDPGAgent):
    config: SmmAgentConfig

    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...], meta_size: int, config: SmmAgentConfig):
        super().__init__(obs_shape, action_shape, meta_size, config)
        self.meta_size = meta_size

        # self.obs_dim is now the real obs_dim (or repr_dim) + z_dim
        self.smm = SMM(self.obs_dim - meta_size, meta_size, hidden_dim=meta_size, vae_beta=config.vae_beta)
        self.pred_optimizer = torch.optim.Adam(self.smm.z_pred_net.parameters(), lr=config.sp_lr)
        self.vae_optimizer = torch.optim.Adam(self.smm.vae.parameters(), lr=config.vae_lr)

        # fine tuning SMM agent
        self.ft_returns = np.zeros(meta_size, dtype=np.float32)
        self.ft_not_finished = [True for z in range(meta_size)]

    def init_meta(self):
        z = np.zeros(self.meta_size, dtype=np.float32)
        z[np.random.choice(self.meta_size)] = 1.0
        meta = OrderedDict()
        meta["z"] = z
        return meta

    def update_vae(self, obs_z):
        metrics = dict()
        loss, h_s_z = self.smm.vae.loss(obs_z)
        self.vae_optimizer.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.vae_optimizer.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        metrics["loss_vae"] = loss.cpu().item()

        return metrics, h_s_z

    def update_pred(self, obs, z):
        metrics = dict()
        logits = self.smm.predict_logits(obs)
        h_z_s = self.smm.loss(logits, z).unsqueeze(-1)
        loss = h_z_s.mean()
        self.pred_optimizer.zero_grad()
        loss.backward()
        self.pred_optimizer.step()

        metrics["loss_pred"] = loss.cpu().item()

        return metrics, h_z_s

    def update(self, replay_iter, step: int):
        metrics = {}
        for batch in replay_iter:
            obs, actions, next_obs, z, dones, extr_reward = to_torch(batch, self.device)
            discount = (1 - dones) * self.config.gamma

            obs = self.aug_and_encode(obs)
            with torch.no_grad():
                next_obs = self.aug_and_encode(next_obs)
            obs_z = torch.cat([obs, z], dim=1)  # do not learn encoder in the VAE
            next_obs_z = torch.cat([next_obs, z], dim=1)

            vae_metrics = None
            pred_metrics = None
            intr_reward = None

            if self.config.reward_free:
                vae_metrics, h_s_z = self.update_vae(obs_z)
                pred_metrics, h_z_s = self.update_pred(obs.detach(), z)

                h_z = np.log(self.meta_size)  # One-hot z encoding
                h_z *= torch.ones_like(extr_reward).to(self.device)

                pred_log_ratios = (
                    self.config.state_ent_coef * h_s_z.detach()
                )  # p^*(s) is ignored, as state space dimension is inaccessible from pixel input
                intr_reward = (
                    pred_log_ratios + self.config.latent_ent_coef * h_z + self.config.latent_cond_ent_coef * h_z_s.detach()
                )
                reward = intr_reward
            else:
                reward = extr_reward

            if self.config.use_tb or self.config.use_wandb:
                if vae_metrics is not None:
                    metrics.update(vae_metrics)
                if pred_metrics is not None:
                    metrics.update(pred_metrics)
                if intr_reward is not None:
                    metrics["intr_reward"] = intr_reward.mean().item()
                metrics["extr_reward"] = extr_reward.mean().item()
                metrics["batch_reward"] = reward.mean().item()

            if not self.config.update_encoder:
                obs_z = obs_z.detach()
                next_obs_z = next_obs_z.detach()

            # update critic
            metrics.update(self.update_critic(obs_z.detach(), actions, reward, discount, next_obs_z.detach(), step))

            # update actor
            metrics.update(self.update_actor(obs_z.detach(), step))

            # update critic target
            soft_update_params(self.critic, self.critic_target, self.config.critic_target_tau)

        return metrics


class SmmPolicy(BasePolicy[tuple[np.ndarray, np.ndarray]]):
    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        config: SmmConfig = SmmConfig(),
    ):
        super().__init__(observation_space, action_space, lr_schedule)
        self.config = config

        obs_size: tuple[int, ...] = observation_space["observation"].shape  # type: ignore
        self.smm_ensemble = th.nn.ModuleList(
            SMMAgent(obs_size, action_space.shape, config.meta_size, config.smm_agent_config)
            for _ in range(config.ensemble_size)
        )

    def _predict(self, observation: dict, deterministic: bool = False) -> th.Tensor:
        model_idx, meta = self.state

        observation = observation["observation"]  # type: ignore

        unique, inverse = np.unique(model_idx, return_inverse=True)
        action = th.zeros((len(observation), self.action_space.shape[0]), device=self.device)
        for i in unique:
            idx = np.where(inverse == i)[0]
            model: SMMAgent = self.smm_ensemble[i]  # type: ignore
            action[idx] = model.act(observation[idx], {"meta": meta[idx]}, 0, eval_mode=False)
        return action

    def _reset_states(self, size: int) -> Tuple[np.ndarray, ...]:
        return (
            np.random.randint(len(self.smm_ensemble), size=size),
            np.eye(self.config.meta_size, dtype=np.float32)[np.random.choice(self.config.meta_size, size)],
        )
