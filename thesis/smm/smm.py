from typing import Any, Dict, Tuple

import numpy as np
import torch as th
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from thesis.smm.config import SmmConfig

from ..core import BaseAlgorithm, BasePolicy, ReplayBuffer
from ..core.algorithm import StateStorage
from .policy import SMMAgent, SmmPolicy
from .wrapper import AddSmmMetaToObservationWrapper


class Smm(BaseAlgorithm):
    policy: SmmPolicy

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
        train_freq: int | Tuple[int, str] = 1,
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
        policy_kwargs = policy_kwargs or {}
        config: SmmConfig = policy_kwargs.setdefault("config", SmmConfig())
        self.meta_size = config.meta_size

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            policy_kwargs,
            stats_window_size,
            tensorboard_log,
            verbose,
            device,
            monitor_wrapper,
            seed,
        )
        self.state_storage = SmmStateStorageAndAddMetaObs()

        if _init_setup_model:
            self._setup_model()

    def _wrap_env(self, env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
        vec_env = super()._wrap_env(env, verbose, monitor_wrapper)
        vec_env = AddSmmMetaToObservationWrapper(vec_env, self.meta_size)
        return vec_env

    def train(self, gradient_steps: int, batch_size: int):
        def replay_buffer_iterator(batch_size):
            for _ in range(gradient_steps):
                batch = self.replay_buffer.sample(batch_size, self.get_vec_normalize_env())
                obs = batch.observations["observation"]  # type: ignore
                next_obs = batch.next_observations["observation"]  # type: ignore
                meta = batch.observations["smm_meta"]  # type: ignore
                actions = batch.actions
                dones = batch.dones
                extr_reward = batch.rewards
                yield obs, actions, next_obs, meta, dones, extr_reward

        smm: SMMAgent
        for i, smm in enumerate(self.policy.smm_ensemble):  # type: ignore
            metrics = smm.update(replay_buffer_iterator(batch_size), self.num_timesteps)
            for k, v in metrics.items():
                self.logger.record(f"smm/{k}_{i}", v)

        super().train(gradient_steps, batch_size)


class SmmStateStorageAndAddMetaObs(StateStorage):
    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        terminal_obs = [info.get("terminal_observation", None) for info in infos]
        self._change_terminal_obs(terminal_obs)

        super()._on_step()

        new_obs = self.locals["new_obs"]
        self._change_obs(new_obs)

        return True
    
    def _init_callback(self):
        super()._init_callback()
        self._change_obs(self.model._last_obs)  # type: ignore

    def _change_obs(self, obs: Dict[str, np.ndarray]):
        obs["smm_meta"] = self.policy_state[1]  # type: ignore

    def _change_terminal_obs(self, terminals: list[Dict[str, np.ndarray] | None]):
        for i, terminal in enumerate(terminals):
            if terminal is not None:
                terminal["smm_meta"] = self.policy_state[1][i]  # type: ignore
