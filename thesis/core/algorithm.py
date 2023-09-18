from typing import Any, Dict, Optional, Tuple

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecEnv
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq

from thesis.core.buffer import ReplayBuffer
from thesis.core.policy import BasePolicy

from .buffer import ReplayBuffer
from .policy import BasePolicy


class BaseAlgorithm(OffPolicyAlgorithm):
    policy: BasePolicy
    replay_buffer: ReplayBuffer

    def __init__(
        self,
        policy: str | type[BasePolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | Tuple[int, str] = ...,
        gradient_steps: int = 1,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: Dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Dict[str, Any] | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        verbose: int = 0,
        device: th.device | str = "auto",
        monitor_wrapper: bool = True,
        seed: int | None = None,
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=False,
            sde_support=False,
        )
        self.state_storage = StateStorage()

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        self.log_interval = log_interval
        return super().learn(total_timesteps, callback, 999999999, tb_log_name, reset_num_timesteps, progress_bar)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        if (self.num_timesteps // self.n_envs) % max(1, self.log_interval // self.train_freq.frequency) == 0: # type: ignore
            self._dump_logs()

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

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["state_storage"]

    def _get_torch_save_params(self) -> Tuple[list[str], list[str]]:
        state_dicts, tensors = super()._get_torch_save_params()
        tensors += ["state_storage.policy_state"]
        return state_dicts, tensors


class StateStorage(BaseCallback):
    model: BaseAlgorithm

    def __init__(self, reset_state_every: int | None = None):
        super().__init__()
        self.reset_state_every = reset_state_every

    def _init_callback(self) -> None:
        self.policy = self.model.policy
        self.policy_state = self.model.policy.reset_states(self.model.n_envs)

    def _on_step(self) -> bool:
        if self.reset_state_every is not None and self.num_timesteps % self.reset_state_every == 0:
            self.policy_state = self.model.policy.reset_states(self.model.n_envs)
        dones = self.locals["dones"]
        self.policy_state = self.model.policy.reset_states(self.model.n_envs, dones, self.policy_state)
        return super()._on_step()