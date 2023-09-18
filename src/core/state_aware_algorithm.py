from typing import Dict, List, Optional, Tuple, Union, cast
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, ReplayBufferSamples, RolloutReturn, TrainFreq
from stable_baselines3.common.policies import BasePolicy
import torch


class StateAwarePolicy(BasePolicy):
    def predict(
        self,
        observation: np.ndarray | Dict[str, np.ndarray],
        state: Tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, dict[str, np.ndarray | torch.Tensor] | Tuple[np.ndarray | torch.Tensor, ...] | None]:
        n_env = len(observation) if not isinstance(observation, dict) else observation["observation"].shape[0]
        self.state = state or self.reset_states(n_env)
        action, _ = super().predict(observation, state, episode_start, deterministic)
        return action, self.state

    def reset_states(
        self,
        n_env: int | None = None,
        dones: np.ndarray | None = None,
        state: dict[str, np.ndarray | torch.Tensor] | Tuple[np.ndarray | torch.Tensor, ...] | None = None,
    ) -> dict[str, np.ndarray | torch.Tensor] | Tuple[np.ndarray | torch.Tensor, ...]:
        if dones is not None:
            if state is None:
                raise ValueError("dones was provided but not state")
            done_idx = np.where(dones)[0]
            if len(done_idx) == 0:
                return state

            if isinstance(state, dict):
                state = {k: v.clone() if isinstance(v, torch.Tensor) else v.copy() for k, v in state.items()}
            else:
                state = tuple(s.clone() if isinstance(s, torch.Tensor) else s.copy() for s in state)
            reset_states = self._reset_states(len(done_idx))
            if isinstance(state, dict):
                assert isinstance(reset_states, dict)
                for k, v in state.items():
                    v[done_idx] = reset_states[k] # type: ignore
            else:
                for s, r in zip(state, reset_states):
                    s[done_idx] = r # type: ignore
            return state
        if n_env is None:
            raise ValueError("n_env was not provided")
        return self._reset_states(n_env)

    def _reset_states(self, size: int) -> dict[str, torch.Tensor | np.ndarray] | Tuple[np.ndarray | torch.Tensor, ...]:
        raise NotImplementedError()


class StateAwareOffPolicyAlgorithm(OffPolicyAlgorithm):
    policy: StateAwarePolicy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_storage = StateStorage()

    # original function adjusted to pass state to policy
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, self.state_storage.policy_state = self.predict(self._last_obs, deterministic=False, state=self.state_storage.policy_state)  # type: ignore

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        total_timesteps, orig_callback = super()._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )
        callback = CallbackList([self.state_storage])
        callback.init_callback(self)
        callback.callbacks.append(orig_callback)
        return total_timesteps, callback
    
    def learn(self, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 4, tb_log_name: str = "run", reset_num_timesteps: bool = True, progress_bar: bool = False):
        self.log_interval = log_interval
        return super().learn(total_timesteps, callback, 999999999, tb_log_name, reset_num_timesteps, progress_bar)
    
    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["state_storage"]
    
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts, tensors = super()._get_torch_save_params()
        tensors += ["state_storage.policy_state"]
        return state_dicts, tensors
    
    def dump_logs_if_neccessary(self):
        if (self.num_timesteps // self.n_envs) % max(1, self.log_interval // self.train_freq.frequency) == 0:
            self._dump_logs()


class StateStorage(BaseCallback):
    model: StateAwareOffPolicyAlgorithm

    def __init__(self, verbose: int = 0, reset_state_every: int | None = None):
        super().__init__(verbose)
        self.reset_state_every = reset_state_every

    def _init_callback(self) -> None:
        self.policy_state = self.model.policy.reset_states(self.model.n_envs)
        self._change_obs(self.model._last_obs)  # type: ignore

    def _on_step(self) -> bool:
        if self.reset_state_every is not None and self.num_timesteps % self.reset_state_every == 0:
            self.policy_state = self.model.policy.reset_states(self.model.n_envs)
        infos = self.locals["infos"]
        terminal_obs = [info.get("terminal_observation", None) for info in infos]
        self._change_terminal_obs(terminal_obs)
        dones = self.locals["dones"]
        self.policy_state = self.model.policy.reset_states(self.model.n_envs, dones, self.policy_state)
        new_obs = self.locals["new_obs"]
        self._change_obs(new_obs)
        return super()._on_step()

    def _change_obs(self, obs: Dict[str, np.ndarray]):
        pass

    def _change_terminal_obs(self, terminals: list[Dict[str, np.ndarray]]):
        pass
