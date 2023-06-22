import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, ConvertCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.vec_env import VecEnv
from src.envs.meta_env import MetaMixin

from src.cemrl.buffers import CEMRLReplayBuffer


class ExplorationCallback(BaseCallback):
    """Callback that uses an exploration policy to collect additional samples."""

    def __init__(
        self,
        exploration_algorithm: BaseAlgorithm,
        steps_per_rollout=2,
        pre_train_steps=200,
        verbose: int = 0,
    ):
        """Initialize the callback.
        Args:
            exploration_algorithm (BaseAlgorithm): Algorithm used for exploration.
            steps_per_rollout (int, optional): Amount of steps to perform at each original model's rollout. Defaults to 1000.
            pre_train_steps (int, optional): Amount of steps to perform before the original model's training start. Defaults to 10000.
            verbose (int, optional): Verbosity of the controller. Defaults to 0.
        """
        super().__init__(verbose)
        self.exploration_algorithm = exploration_algorithm
        self.steps_per_rollout = steps_per_rollout
        self.pre_train_steps = pre_train_steps
        self._dummy_callback = ConvertCallback(None)  # type: ignore

    def _init_callback(self) -> None:
        assert isinstance(self.model, OffPolicyAlgorithm)
        assert self.model.env is not None and self.exploration_algorithm.env is not None
        explore_envs = [env for env in self.exploration_algorithm.env.get_attr("unwrapped") if isinstance(env, MetaMixin)]
        envs = [env for env in self.model.env.get_attr("unwrapped") if isinstance(env, MetaMixin)]
        for explore_env, env in zip(explore_envs, envs):
            explore_env.goal_sampler = env.goal_sampler
            

        if self.logger is not None:
            self.exploration_algorithm.set_logger(self.logger)
        self._dummy_callback.init_callback(self.exploration_algorithm)
        if isinstance(self.exploration_algorithm, OffPolicyAlgorithm):
            self.exploration_algorithm.replay_buffer = self.model.replay_buffer

        model_base_env = self.model.env.unwrapped
        exploration_base_env = self.exploration_algorithm.env.unwrapped
        self._same_base_envs = exploration_base_env == model_base_env

        if not self._same_base_envs:
            assert isinstance(self.exploration_algorithm.env, VecEnv)
            self.exploration_algorithm._last_obs = self.exploration_algorithm.env.reset()  # type: ignore
            self.exploration_algorithm._last_episode_starts = np.ones((self.exploration_algorithm.env.num_envs,), dtype=bool)
            if self.exploration_algorithm._vec_normalize_env is not None:
                self.exploration_algorithm._last_original_obs = (
                    self.exploration_algorithm._vec_normalize_env.get_original_obs()
                )

    def _on_training_start(self) -> None:
        assert self.model is not None
        assert isinstance(self.exploration_algorithm.env, VecEnv)
        if self.pre_train_steps <= 0:
            return

        if isinstance(self.exploration_algorithm, OffPolicyAlgorithm):
            assert self.exploration_algorithm.replay_buffer is not None

            self._setup_rollout_collection(self.pre_train_steps)
            log_prefix = getattr(self.exploration_algorithm, "log_prefix", None)
            if log_prefix is not None:
                setattr(self.exploration_algorithm, "log_prefix", f"pretrain_{log_prefix}")
            self.exploration_algorithm.learn(
                self.pre_train_steps * self.exploration_algorithm.n_envs, tb_log_name="exploration", progress_bar=True
            )
            if log_prefix is not None:
                setattr(self.exploration_algorithm, "log_prefix", log_prefix)
            self._cleanup_rollout_collection()

    def _on_rollout_start(self) -> None:
        assert self.model is not None
        assert isinstance(self.exploration_algorithm.env, VecEnv)
        
        if isinstance(self.exploration_algorithm, OffPolicyAlgorithm):
            assert self.exploration_algorithm.replay_buffer is not None
            steps = self.steps_per_rollout
            if steps <= 0:
                return

            self._setup_rollout_collection(self.steps_per_rollout)
            self.exploration_algorithm.collect_rollouts(
                self.exploration_algorithm.env,
                self._dummy_callback,
                TrainFreq(steps, TrainFrequencyUnit.STEP),
                self.exploration_algorithm.replay_buffer,
            )
            self._cleanup_rollout_collection()

    def _on_step(self) -> bool:
        return super()._on_step()

    def _setup_rollout_collection(self, num_steps: int):
        assert isinstance(self.exploration_algorithm.env, VecEnv)
        assert self.model is not None

        self.exploration_algorithm._total_timesteps = num_steps
        self.exploration_algorithm.num_timesteps = 0

        if isinstance(self.model, OffPolicyAlgorithm) and isinstance(self.model.replay_buffer, CEMRLReplayBuffer):
            self.model.replay_buffer.is_exploring = True

        if self._same_base_envs:
            self.exploration_algorithm._last_obs = self.model._last_obs
            self.exploration_algorithm._last_episode_starts = self.model._last_episode_starts
            self.exploration_algorithm._last_original_obs = self.model._last_original_obs
            self.exploration_algorithm.ep_success_buffer = self.model.ep_info_buffer
            self.exploration_algorithm.ep_info_buffer = self.model.ep_info_buffer

        self._dummy_callback.locals["total_timesteps"] = num_steps
        self._dummy_callback._on_training_start()

    def _cleanup_rollout_collection(self):
        if self._same_base_envs:
            assert self.model is not None
            self.model._last_obs = self.exploration_algorithm._last_obs
            self.model._last_episode_starts = self.exploration_algorithm._last_episode_starts
            self.model._last_original_obs = self.exploration_algorithm._last_original_obs
            self.exploration_algorithm.ep_success_buffer = self.model.ep_info_buffer
            self.exploration_algorithm.ep_info_buffer = self.model.ep_info_buffer

        if isinstance(self.model, OffPolicyAlgorithm) and isinstance(self.model.replay_buffer, CEMRLReplayBuffer):
            self.model.replay_buffer.is_exploring = False