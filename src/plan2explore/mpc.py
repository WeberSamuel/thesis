"""Policies for the Plan2Explore Algorithm."""
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch as th
from stable_baselines3.common.policies import BasePolicy
from src.cemrl.networks import Encoder
from src.plan2explore.networks import Ensemble


class Plan2ExplorePolicy(BasePolicy):
    """Base Policy of the Plan2Explore Algorithm.

    It uses an ensemble of world models to steer the agent into undiscovered areas at training time.
    During evaluation it tries to maximize the future reward.
    """

    def __init__(
        self,
        *args,
        best_selection_strategy=th.sum,
        num_actions=5,
        num_timesteps=20,
        ensemble: Ensemble = None,
        latent_generator: th.nn.Module = None,
        **kwargs
    ):
        """Initialize the class.

        Args:
            ensemble (Ensemble): Ensemble used as world model
            num_actions (int, optional): Number of actions to sample. Defaults to 5.
            num_timesteps (int, optional): Number of timesteps to predict the future. Defaults to 20.
        """
        del kwargs["use_sde"]
        super().__init__(*args, **kwargs)
        self.ensemble = ensemble
        self.latent_generator = latent_generator
        self.num_actions = num_actions
        self.num_timesteps = num_timesteps
        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        self.is_collecting = True
        self.best_selection_strategy = best_selection_strategy

    def _predict(
        self, observation: th.Tensor | Dict[str, th.Tensor], deterministic: bool = False, z: Optional[th.Tensor] = None
    ) -> th.Tensor:
        if isinstance(observation, Dict):
            observation = th.cat(list(observation.values()), dim=-1)
        assert isinstance(observation, th.Tensor)

        n_envs = len(observation)

        total_num_actions = n_envs * self.num_actions * self.num_timesteps

        actions = np.random.choice([-0.25, 0, 2.5], (total_num_actions * 2)).reshape(-1, 2).astype(np.float32)
        # actions = np.stack([self.action_space.sample() for _ in range(total_num_actions)])
        actions = th.from_numpy(actions).to(self.device)
        timestep_actions = actions.reshape(self.num_timesteps, self.num_actions, n_envs, *actions.shape[1:])

        observation = observation[None].expand(self.num_actions, *observation.shape)
        if z is not None:
            z = z[None].expand(self.num_actions, *z.shape)
        self.mpc(observation, timestep_actions, z=z, n_envs=n_envs)
        return timestep_actions[0]
        # state_vars = th.zeros(self.num_timesteps, *observation.shape)
        # reward_means = th.zeros(self.num_timesteps, self.num_actions, n_envs, 1)

        # for timestep, actions in enumerate(timestep_actions):
        #     (state_var, state_mean), (reward_var, reward_mean) = self.ensemble(observation, actions, z=z)
        #     observation = state_mean
        #     state_vars[timestep] = state_var
        #     reward_means[timestep] = reward_mean

        # if self.is_collecting:
        #     action_ensemble_var = state_vars.sum(dim=(0, 3))
        #     highest_var_index = action_ensemble_var.argmax(dim=0)
        #     return timestep_actions[0, highest_var_index, th.arange(n_envs)]
        # else:
        #     future_max_reward = reward_means.squeeze(dim=-1).max(dim=0)[0]
        #     highest_reward_index = future_max_reward.argmax(dim=0)
        #     return timestep_actions[0, highest_reward_index, th.arange(n_envs)]

    def mpc(self, state: th.Tensor, action: th.Tensor, z: th.Tensor, n_envs, gradient_steps=5, lr=1e-1):
        action.requires_grad_()
        self.ensemble.requires_grad_(False)
        optim = th.optim.Adam([action], lr=lr)

        for i in range(gradient_steps):
            tmp_state = state.clone()

            state_vars = th.zeros(self.num_timesteps, *state.shape)
            reward_means = th.zeros(self.num_timesteps, self.num_actions, n_envs, 1)
            for timestep, actions in enumerate(action):
                (state_var, state_mean), (reward_var, reward_mean) = self.ensemble(tmp_state, actions, z=z)
                tmp_state = state_mean
                state_vars[timestep] = state_var
                reward_means[timestep] = reward_mean

            if self.is_collecting:
                loss = -state_vars.sum()
            else:
                loss = -reward_means.sum()

            optim.zero_grad()
            loss.backward()
            optim.step()

        action.requires_grad_(False)
        self.ensemble.requires_grad_()


class CEMRLExplorationPolicy(Plan2ExplorePolicy):
    def __init__(self, *args, encoder: Optional[Encoder] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder

    def _predict(self, observation: Dict[str, th.Tensor], deterministic: bool = False) -> th.Tensor:
        y, z = self.encoder(observation)
        # z = observation["goal"][:, -1]
        observation = observation["observation"][:, -1]  # strip history
        return super()._predict(observation, deterministic, z=z)
