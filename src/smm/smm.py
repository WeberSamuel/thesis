from typing import Any, Dict, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn
import torch as th
from gymnasium import Space, spaces
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.buffers import DictReplayBuffer

from src.cemrl.buffers import CEMRLReplayBuffer
from src.core.state_aware_algorithm import StateAwareOffPolicyAlgorithm, StateStorage
from src.smm.policies import SMMPolicy, SMMAgent
from stable_baselines3.common.vec_env import VecEnvWrapper


class SMM(StateAwareOffPolicyAlgorithm):
    def __init__(
        self,
        policy: str | type[SMMPolicy],
        env: GymEnv,
        learning_rate: float | Schedule = 1e-3,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        device: th.device | str = "auto",
        gamma: float = 0.99,
        learning_starts: int = 1000,
        log_prefix: str = "smm",
        monitor_wrapper: bool = True,
        optimize_memory_usage: bool = False,
        policy_kwargs: Dict[str, Any] | None = None,
        replay_buffer_class: type[DictReplayBuffer] | None = DictReplayBuffer,
        replay_buffer_kwargs: Dict[str, Any] | None = None,
        seed: int | None = None,
        stats_window_size: int = 100,
        supported_action_spaces: Tuple[type[Space], ...] | None = None,
        tau: float = 0.005,
        tensorboard_log: str | None = None,
        train_freq: int | Tuple[int, str] = (1, "step"),
        verbose: int = 0,
        z_dim: int = 4,
        action_noise: ActionNoise | None = None,
    ):
        policy_kwargs = policy_kwargs or {}
        policy_kwargs["z_dim"] = z_dim

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
            gradient_steps=-1,
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
            sde_support=False,
            supported_action_spaces=supported_action_spaces,
        )
        if log_prefix is not None:
            self.log_prefix = log_prefix + "/"
        else:
            self.log_prefix = ""

        self.env: SMMWrapper = SMMWrapper(self.env, z_dim)  # type: ignore
        self.observation_space = self.env.observation_space
        self.gradient_steps = self.n_envs // 8

        self._setup_model()
        self.policy: SMMPolicy
        self.state_storage = SMMStateStorage()

    def _get_replay_buffer_iterator(self, batch_size, num_batches):
        if self.replay_buffer is None:
            raise ValueError("The replay buffer is not initialized.")
        for _ in range(num_batches):
            batch = self.replay_buffer.sample(batch_size, self.get_vec_normalize_env())
            obs = batch.observations["observation"]  # type: ignore
            next_obs = batch.next_observations["observation"]  # type: ignore
            z = batch.observations["meta"]  # type: ignore
            actions = batch.actions
            dones = batch.dones
            extr_reward = batch.rewards

            yield obs, actions, next_obs, z, dones, extr_reward

    def train(self, gradient_steps: int, batch_size: int) -> None:
        smm: SMMAgent
        for i, smm in enumerate(self.policy.smm_ensemble):  # type: ignore
            metrics = smm.update(self._get_replay_buffer_iterator(batch_size, gradient_steps), self.num_timesteps)
            for k, v in metrics.items():
                self.logger.record(f"{self.log_prefix}{k}_{i}", v)


class SMMWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, z_dim: int):
        if isinstance(venv.observation_space, spaces.Dict):
            obs_space = venv.observation_space.spaces
        else:
            obs_space = {"observation": venv.observation_space}
        obs_space = spaces.Dict(
            {
                **obs_space,
                "meta": spaces.Box(low=0, high=1, shape=(z_dim,), dtype=np.float32),
            }
        )
        super().__init__(venv, observation_space=obs_space)
        self.z_dim = z_dim

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        if not isinstance(obs, dict):
            obs = {"observation": obs}

        return obs  # type: ignore

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, dones, infos = self.venv.step_wait()
        if not isinstance(obs, dict):
            obs = {"observation": obs}
        return obs, reward, dones, infos  # type: ignore


class SMMStateStorage(StateStorage):
    def _change_obs(self, obs: Dict[str, np.ndarray]):
        obs["meta"] = self.policy_state[1] # type: ignore

    def _change_terminal_obs(self, terminals: list[Dict[str, np.ndarray] | None]):
        for i, terminal in enumerate(terminals):
            if terminal is not None:
                terminal["meta"] = self.policy_state[1][i] # type: ignore
