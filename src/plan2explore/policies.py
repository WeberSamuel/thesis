"""Policies for the Plan2Explore Algorithm."""
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch as th
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv
from gym import Env
from src.cemrl.networks import Encoder
from src.plan2explore.networks import Ensemble

class Plan2ExplorePolicy(BasePolicy):
    """Base Policy of the Plan2Explore Algorithm.

    It uses an ensemble of world models to steer the agent into undiscovered areas at training time.
    During evaluation it tries to maximize the future reward.
    """

    def __init__(
        self,
        env: VecEnv|Env,
        ensemble: Ensemble,
        num_actions=3,
        num_timesteps=1,
        repeat_action = 10,
    ):
        """Initialize the class.

        Args:
            ensemble (Ensemble): Ensemble used as world model
            num_actions (int, optional): Number of actions to sample. Defaults to 5.
            num_timesteps (int, optional): Number of timesteps to predict the future. Defaults to 20.
        """
        super().__init__(env.observation_space, env.action_space)
        self.ensemble = ensemble
        self.num_actions = num_actions
        self.num_timesteps = num_timesteps
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
        timestep_actions = timestep_actions.reshape(self.num_actions, n_envs, total_num_timesteps, *actions.shape[1:]).movedim(2, 0)

        observation = observation[None].expand(self.num_actions, *observation.shape)
        if z is not None:
            z = z[None].expand(self.num_actions, *z.shape)
        state_vars = th.zeros(total_num_timesteps, *observation.shape)
        reward_means = th.zeros(total_num_timesteps, self.num_actions, n_envs, 1)

        for timestep, actions in enumerate(timestep_actions):
            (state_var, state_mean), (reward_var, reward_mean) = self.ensemble(observation, actions, z=z)
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
        super().__init__(*args, **kwargs)
        self.encoder = encoder

    def _predict(self, observation: Dict[str, th.Tensor], deterministic: bool = False) -> th.Tensor: # type: ignore
        with th.no_grad():
            _, z = self.encoder(observation)
        obs = observation["observation"][:, -1]  # strip history
        return super()._predict(obs, deterministic, z=z)
