from collections import OrderedDict

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..core import BaseModule
from .config import DdpgAgentConfig
from .networks import Actor, Critic, Encoder
from .utils import RandomShiftsAug, hard_update_params, schedule, soft_update_params, to_torch


class DDPGAgent(BaseModule):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        meta_size: int,
        config: DdpgAgentConfig = DdpgAgentConfig(),
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.solved_meta = None
        self.config = config

        # models
        if config.obs_type == "pixels":
            self.aug = RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape)
            self.obs_dim = self.encoder.repr_dim + meta_size
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_size

        self.actor = Actor(config.obs_type, self.obs_dim, self.action_dim, config.feature_dim, config.hidden_dim)

        self.critic = Critic(config.obs_type, self.obs_dim, self.action_dim, config.feature_dim, config.hidden_dim)
        self.critic_target = Critic(config.obs_type, self.obs_dim, self.action_dim, config.feature_dim, config.hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers

        if config.obs_type == "pixels":
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=config.lr)
        else:
            self.encoder_opt = None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def init_from(self, other):
        # copy parameters over
        hard_update_params(other.encoder, self.encoder)
        hard_update_params(other.actor, self.actor)
        if self.init_critic:
            hard_update_params(other.critic.trunk, self.critic.trunk)

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta: dict, global_step: int, time_step: int, finetune: bool = False):
        return meta

    def act(self, obs: np.ndarray | th.Tensor, meta: dict, step: int, eval_mode: bool):
        obs = torch.as_tensor(obs, device=self.device)
        h = self.encoder(obs)
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        # assert obs.shape[-1] == self.obs_shape[-1]
        stddev = schedule(self.config.stddev_schedule, step)
        dist = self.actor(inpt, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action.detach()

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = schedule(self.config.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.config.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.config.use_tb or self.config.use_wandb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = schedule(self.config.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.config.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.config.use_tb or self.config.use_wandb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = to_torch(batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.config.use_tb or self.config.use_wandb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        soft_update_params(self.critic, self.critic_target, self.config.critic_target_tau)

        return metrics
