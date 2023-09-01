from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule
from src.core.state_aware_algorithm import StateAwarePolicy

import src.smm.utils as utils
from src.smm.ddpg import DDPGAgent
from src.smm.networks import SMM


class SMMAgent(DDPGAgent):
    def __init__(
        self,
        reward_free: bool,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        device: torch.device,
        z_dim: int = 4,
        sp_lr: float = 1e-3,
        vae_lr: float = 1e-2,
        vae_beta: float = 0.5,
        state_ent_coef: float = 1.0,
        latent_ent_coef: float = 1.0,
        latent_cond_ent_coef: float = 1.0,
        update_encoder: bool = True,
        lr: float = 1e-4,
        critic_target_tau: float = 0.01,
        hidden_dim: int = 1024,
        feature_dim: int = 50,
        stddev_schedule: float = 0.2,
        stddev_clip: float = 0.3,
        init_critic: bool = True,
        gamma=0.99,
    ):
        self.z_dim = z_dim

        self.state_ent_coef = state_ent_coef
        self.latent_ent_coef = latent_ent_coef
        self.latent_cond_ent_coef = latent_cond_ent_coef
        self.update_encoder = update_encoder
        self.gamma = gamma

        super().__init__(
            reward_free,
            obs_shape,
            action_shape,
            device,
            meta_dim=self.z_dim,
            lr=lr,
            critic_target_tau=critic_target_tau,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            stddev_schedule=stddev_schedule,
            stddev_clip=stddev_clip,
            init_critic=init_critic,
        )
        # self.obs_dim is now the real obs_dim (or repr_dim) + z_dim
        self.smm = SMM(self.obs_dim - z_dim, z_dim, hidden_dim=self.z_dim, vae_beta=vae_beta, device=device)
        self.smm = self.smm.to(device)
        self.pred_optimizer = torch.optim.Adam(self.smm.z_pred_net.parameters(), lr=sp_lr)
        self.vae_optimizer = torch.optim.Adam(self.smm.vae.parameters(), lr=vae_lr)

        self.smm.train()

        # fine tuning SMM agent
        self.ft_returns = np.zeros(z_dim, dtype=np.float32)
        self.ft_not_finished = [True for z in range(z_dim)]

    def init_meta(self):
        z = np.zeros(self.z_dim, dtype=np.float32)
        z[np.random.choice(self.z_dim)] = 1.0
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
            obs, actions, next_obs, z, dones, extr_reward = utils.to_torch(batch, self.device)
            discount = (1 - dones) * self.gamma

            obs = self.aug_and_encode(obs)
            with torch.no_grad():
                next_obs = self.aug_and_encode(next_obs)
            obs_z = torch.cat([obs, z], dim=1)  # do not learn encoder in the VAE
            next_obs_z = torch.cat([next_obs, z], dim=1)

            vae_metrics = None
            pred_metrics = None
            intr_reward = None

            if self.reward_free:
                vae_metrics, h_s_z = self.update_vae(obs_z)
                pred_metrics, h_z_s = self.update_pred(obs.detach(), z)

                h_z = np.log(self.z_dim)  # One-hot z encoding
                h_z *= torch.ones_like(extr_reward).to(self.device)

                pred_log_ratios = (
                    self.state_ent_coef * h_s_z.detach()
                )  # p^*(s) is ignored, as state space dimension is inaccessible from pixel input
                intr_reward = pred_log_ratios + self.latent_ent_coef * h_z + self.latent_cond_ent_coef * h_z_s.detach()
                reward = intr_reward
            else:
                reward = extr_reward

            if self.use_tb or self.use_wandb:
                if vae_metrics is not None:
                    metrics.update(vae_metrics)
                if pred_metrics is not None:
                    metrics.update(pred_metrics)
                if intr_reward is not None:
                    metrics["intr_reward"] = intr_reward.mean().item()
                metrics["extr_reward"] = extr_reward.mean().item()
                metrics["batch_reward"] = reward.mean().item()

            if not self.update_encoder:
                obs_z = obs_z.detach()
                next_obs_z = next_obs_z.detach()

            # update critic
            metrics.update(self.update_critic(obs_z.detach(), actions, reward, discount, next_obs_z.detach(), step))

            # update actor
            metrics.update(self.update_actor(obs_z.detach(), step))

            # update critic target
            utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics


class SMMPolicy(StateAwarePolicy):
    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        z_dim=4,
        ensemble_size=1,
        **kwargs
    ):
        super().__init__(observation_space=observation_space, action_space=action_space)
        self.z_dim = z_dim
        if isinstance(observation_space, spaces.Dict):
            obs_shape = observation_space["observation"].shape  # type: ignore
        else:
            obs_shape = observation_space.shape

        assert obs_shape is not None
        assert isinstance(action_space, spaces.Box)

        self.smm_ensemble = th.nn.ModuleList(
            SMMAgent(True, obs_shape, action_space.shape, th.device("cuda"), z_dim=z_dim, **kwargs)
            for _ in range(ensemble_size)
        )

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        model_idx, z = self.state

        if isinstance(observation, dict):
            observation = observation["observation"]  # type: ignore

        unique, inverse = np.unique(model_idx, return_inverse=True)
        action = th.zeros((len(observation), self.action_space.shape[0]), device=self.device)
        for i in unique:
            idx = np.where(inverse == i)[0]
            model: SMMAgent = self.smm_ensemble[i]  # type: ignore
            action[idx] = model.act(observation[idx], {"z": z[idx]}, 0, eval_mode=False)
        return action

    def _reset_states(self, size: int) -> Tuple[np.ndarray, ...]:
        return np.random.randint(len(self.smm_ensemble), size=size), np.eye(self.z_dim, dtype=np.float32)[np.random.choice(self.z_dim, size)]
