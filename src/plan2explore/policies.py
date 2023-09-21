"""Policies for the Plan2Explore Algorithm."""
from typing import Dict, Optional, Tuple
import numpy as np
import torch as th
from gymnasium import spaces
from src.plan2explore.networks import Ensemble, OneStepModel
from thesis.core.types import EncoderInput
from ..core.policies import BasePolicy
from thesis.cemrl.task_inference import Encoder
from src.utils import apply_function_to_type
from stable_baselines3.common.type_aliases import Schedule


class Plan2ExplorePolicy(BasePolicy):
    """Base Policy of the Plan2Explore Algorithm.

    It uses an ensemble of world models to steer the agent into undiscovered areas at training time.
    During evaluation it tries to maximize the future reward.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        num_actions=3,
        num_timesteps=1,
        repeat_action=10,
        latent_dim=0,
    ):
        """Initialize the class.

        Args:
            ensemble (Ensemble): Ensemble used as world model
            num_actions (int, optional): Number of actions to sample. Defaults to 5.
            num_timesteps (int, optional): Number of timesteps to predict the future. Defaults to 20.
        """
        super().__init__(observation_space, action_space, lr_schedule)
        self.num_actions = num_actions
        self.num_timesteps = num_timesteps
        self.ensemble = Ensemble(
            th.nn.ModuleList(
                [
                    OneStepModel(
                        latent_dim + spaces.flatdim(observation_space["observation"]) + spaces.flatdim(action_space),
                        spaces.flatdim(observation_space["observation"]),
                    )
                    for i in range(5)
                ]
            )
        )
        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        self.repeat_action = repeat_action
        self.action = []
        self._is_collecting = True

    @th.no_grad()
    def _predict(
        self, observation: th.Tensor | Dict[str, th.Tensor], deterministic: bool = False, z: Optional[th.Tensor] = None
    ) -> th.Tensor:
        if len(self.action) != 0:
            return self.action.pop()

        if isinstance(observation, Dict):
            observation = th.cat(list(observation.values()), dim=-1)
        assert isinstance(observation, th.Tensor)

        n_envs = len(observation)

        total_num_actions = n_envs * self.num_actions * self.num_timesteps

        # actions = np.random.choice([-0.25, 0, 2.5], (total_num_actions * 2)).reshape(-1, 2).astype(np.float32)
        actions = np.stack([self.action_space.sample() for _ in range(total_num_actions)])
        actions = self.scale_action(actions)
        actions = th.from_numpy(actions).to(self.device)

        total_num_timesteps = self.num_timesteps * self.repeat_action
        timestep_actions = actions.repeat_interleave(self.repeat_action, 0)
        timestep_actions = timestep_actions.reshape(self.num_actions, n_envs, total_num_timesteps, *actions.shape[1:]).movedim(
            2, 0
        )

        observation = observation[None].expand(self.num_actions, *observation.shape)
        if z is not None:
            z = z[None].expand(self.num_actions, *z.shape)
        state_vars = th.zeros(total_num_timesteps, *observation.shape)
        reward_means = th.zeros(total_num_timesteps, self.num_actions, n_envs, 1)

        for timestep, actions in enumerate(timestep_actions):
            (state_var, state_mean), (reward_var, reward_mean) = self.ensemble(th.cat([observation, actions, z], dim=-1))
            observation = state_mean
            state_vars[timestep] = state_var
            reward_means[timestep] = reward_mean

        if self._is_collecting:
            action_ensemble_var = state_vars.sum(dim=(0, 3))
            highest_var_index = action_ensemble_var.argmax(dim=0)
            self.action = [timestep_actions[0, highest_var_index, th.arange(n_envs)]] * self.repeat_action
        else:
            future_max_reward = reward_means.squeeze(dim=-1).max(dim=0)[0]
            highest_reward_index = future_max_reward.argmax(dim=0)
            self.action = [timestep_actions[0, highest_reward_index, th.arange(n_envs)]]
        return self.action.pop()


class CEMRLExplorationPolicy(Plan2ExplorePolicy):
    def __init__(self, *args, encoder: Encoder, **kwargs):
        super().__init__(*args, **kwargs, latent_dim=encoder.config.latent_dim)
        self.encoder = encoder
        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)

    def _predict(self, observation: Dict[str, th.Tensor], deterministic: bool = False) -> th.Tensor:  # type: ignore
        prev_observation = self.state  # type: ignore
        next_obs = {}
        for k, v in prev_observation.items():  # type:ignore
            v: th.Tensor
            next_obs[k] = v.clone()
            next_obs[k][:, :-1] = v[:, 1:]
            next_obs[k][:, -1] = observation[k]

        with th.no_grad():
            z, _, _ = self.encoder(
                EncoderInput(
                    obs=prev_observation["observation"],
                    action=next_obs["action"],
                    next_obs=next_obs["observation"],
                    reward=next_obs["reward"],
                )
            )
        return super()._predict(observation["observation"], deterministic, z=z)

    def _reset_states(self, size: int) -> dict[str, th.Tensor | np.ndarray] | Tuple[np.ndarray | th.Tensor, ...]:
        return apply_function_to_type(
            self.observation_space.sample(),
            np.ndarray,
            lambda x: th.zeros((size, 30, *x.shape), device=self.device),
        )


class Plan2ExploreMPCPolicy(Plan2ExplorePolicy):
    def __init__(
        self,
        *args,
        num_actions=3,
        num_timesteps=3,
        mpc_loss_threshold=-0.1,
        mpc_max_optim_steps=15,
        mpc_lr=1e-1,
        reduce_loss_threshold_after_num_threshold=3,
        max_timesteps=10,
    ):
        super().__init__(*args, num_actions=num_actions, num_timesteps=num_timesteps)
        self.mpc_loss_threshold = mpc_loss_threshold
        self.mpc_max_optim_steps = mpc_max_optim_steps
        self.mpc_lr = mpc_lr
        self.times_threshold_reached = 0
        self.reduce_loss_threshold_after_num_threshold = reduce_loss_threshold_after_num_threshold
        self.max_timesteps = max_timesteps
        self.action_low = th.tensor(self.action_space.low).cuda()  # type: ignore
        self.action_high = th.tensor(self.action_space.high).cuda()  # type: ignore

    @th.enable_grad()
    def _predict(
        self, observation: th.Tensor | Dict[str, th.Tensor], deterministic: bool = False, z: th.Tensor | None = None
    ) -> th.Tensor:
        n_envs = len(observation)

        if isinstance(observation, Dict):
            observation = th.cat(list(observation.values()), dim=-1)
        assert isinstance(observation, th.Tensor)
        observation = observation[None].expand(self.num_actions, *observation.shape)
        if z is not None:
            z = z[None].expand(self.num_actions, *z.shape)
        device = observation.device

        if len(self.action) == 0 or self.action.shape[-2] != n_envs:
            self.action = th.tensor(
                [
                    [[self.action_space.sample() for _ in range(n_envs)] for _ in range(self.num_actions)]
                    for _ in range(self.num_timesteps)
                ],
                device=device,
            )
        else:
            self.action = self.action.roll(-1, 0)
            self.action[-1] = th.tensor(
                [[self.action_space.sample() for _ in range(n_envs)] for _ in range(self.num_actions)], device=device
            )
        self.action = self.action.detach()
        self.action.requires_grad_(True)
        optim = th.optim.AdamW([self.action], lr=self.mpc_lr)

        loss = 0
        original_obs = observation.clone()
        state_vars = th.zeros(self.num_timesteps, *observation.shape)
        reward_means = th.zeros(self.num_timesteps, self.num_actions, n_envs, 1)

        for step in range(self.mpc_max_optim_steps):
            observation = original_obs
            state_vars = th.zeros(self.num_timesteps, *observation.shape)
            reward_means = th.zeros(self.num_timesteps, self.num_actions, n_envs, 1)

            for timestep, actions in enumerate(self.action.clip(self.action_low, self.action_high)):
                (state_var, state_mean), (reward_var, reward_mean) = self.ensemble(observation, actions, z=z)
                observation = state_mean
                state_vars[timestep] = state_var
                reward_means[timestep] = reward_mean

            if self._is_collecting:
                var = state_vars.mean()
                loss = -var
            else:
                reward = reward_means.mean()
                loss = -reward

            if loss < self.mpc_loss_threshold:
                break

            optim.zero_grad()
            loss.backward()
            optim.step()

        if self._is_collecting:
            self.times_threshold_reached = self.times_threshold_reached + (1 if loss < self.mpc_loss_threshold else -1)
            self.times_threshold_reached = np.clip(
                self.times_threshold_reached,
                -self.reduce_loss_threshold_after_num_threshold,
                self.reduce_loss_threshold_after_num_threshold,
            )
            if self.times_threshold_reached <= -self.reduce_loss_threshold_after_num_threshold:
                self.times_threshold_reached = self.reduce_loss_threshold_after_num_threshold
                self.mpc_loss_threshold /= 2
            elif self.times_threshold_reached >= self.reduce_loss_threshold_after_num_threshold:
                self.times_threshold_reached = -self.reduce_loss_threshold_after_num_threshold
                self.mpc_loss_threshold *= 2

        if self._is_collecting:
            action_ensemble_var = state_vars.sum(dim=(0, 3))
            highest_index = action_ensemble_var.argmax(dim=0)
        else:
            future_max_reward = reward_means.squeeze(dim=-1).max(dim=0)[0]
            highest_index = future_max_reward.argmax(dim=0)
        return self.action[0, highest_index, th.arange(n_envs)].detach()


class CEMRLMPCExplorationPolicy(Plan2ExploreMPCPolicy):
    def __init__(self, *args, encoder: Encoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)

    def _predict(self, observation: Dict[str, th.Tensor], deterministic: bool = False) -> th.Tensor:  # type: ignore
        with th.no_grad():
            _, z = self.encoder(observation)
        obs = observation["observation"][:, -1]  # strip history
        return super()._predict(obs, deterministic, z=z)
