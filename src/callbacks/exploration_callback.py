import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, ConvertCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.vec_env import VecEnv
from src.envs.meta_env import MetaMixin

from src.cemrl.buffers import CEMRLReplayBuffer


class MaybeNoTraining:
    """
    A context manager that disables training for an OffPolicyAlgorithm if active is True.

    This is useful for exploration algorithms that don't require training, but still need to interact with the environment
    to generate data for the replay buffer.
    """

    def __init__(self, active: bool, model: BaseAlgorithm) -> None:
        """
        Initializes a new instance of the NoTraining class.

        :param active: Whether to disable training for the model.
        :param model: The model to disable training for.
        """
        if active and not isinstance(model, OffPolicyAlgorithm):
            raise ValueError("NoTraining is only supported for OffPolicyAlgorithm")
        self.active = active
        self.model = model

    def __enter__(self):
        if self.active and isinstance(self.model, (OffPolicyAlgorithm)):
            self.original_gradient_steps = self.model.gradient_steps
            self.model.gradient_steps = 0

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.active and isinstance(self.model, (OffPolicyAlgorithm)):
            self.model.gradient_steps = self.original_gradient_steps


class StateCopy:
    def __init__(self, model: BaseAlgorithm, exploration_model: BaseAlgorithm) -> None:
        """
        Initializes a new instance of the StateCopy class.

        :param model: The original algorithm to copy the state to.
        :param exploration_model: The exploration algorithm to copy the state from.
        """
        if (not isinstance(model, OnPolicyAlgorithm) or not isinstance(exploration_model, OnPolicyAlgorithm)) and (
            not isinstance(model, OffPolicyAlgorithm) or not isinstance(exploration_model, OffPolicyAlgorithm)
        ):
            raise ValueError("StateCopy is only supported for similar algorithms")

        self.model = model
        self.exploration_model = exploration_model

    def __enter__(self):
        if isinstance(self.model, OffPolicyAlgorithm):
            self.original_last_obs = self.model._last_obs
            self.original_last_episode_starts = self.model._last_episode_starts
            self.original_last_original_obs = self.model._last_original_obs
            self.model._last_obs = self.exploration_model._last_obs
            self.model._last_episode_starts = self.exploration_model._last_episode_starts
            self.model._last_original_obs = self.exploration_model._last_original_obs

    def __exit__(self, exc_type, exc_value, exc_tb):
        if isinstance(self.model, OffPolicyAlgorithm):
            self.model._last_obs = self.original_last_obs
            self.model._last_episode_starts = self.original_last_episode_starts
            self.model._last_original_obs = self.original_last_original_obs


class StoreExplorationToBufferCallback(BaseCallback):
    """Callback that stores exploration data to the replay buffer of an off-policy algorithm."""

    def __init__(self, model: OffPolicyAlgorithm):
        """
        Initializes a new instance of the StoreExplorationToBufferCallback class.

        :param model: The original algorithm to store the exploration data to.
        :param converter: A converter to convert the exploration data to the original algorithm's format.
        """
        super().__init__()
        self.original_model = model

    def _on_step(self) -> bool:
        buffer_actions = self.locals["buffer_actions"]
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        new_obs = self.locals["new_obs"]
        assert new_obs is not None

        for info in infos:
            info["is_exploration"] = True

        with StateCopy(self.original_model, self.model): # type: ignore
            assert self.original_model.replay_buffer is not None
            self.original_model._store_transition(
                self.original_model.replay_buffer, buffer_actions, new_obs, rewards, dones, infos
            )

        return super()._on_step()


class NewExplorationCallback(BaseCallback):
    """
    A callback that performs exploration using a separate exploration algorithm.

    This callback is designed to be used with an off-policy algorithm that requires a replay buffer. It performs
    exploration using a separate exploration algorithm, and stores the exploration data to the replay buffer of the
    original algorithm. The exploration algorithm can be any algorithm that implements the BaseAlgorithm interface.
    """

    def __init__(
        self,
        exploration_algorithm: BaseAlgorithm,
        steps_per_rollout=2,
        pre_train_steps=200,
        train_on_rollout: bool = False,
        exploration_log_interval = 1,
        verbose: int = 0,
    ):
        """
        Initializes a new instance of the ExplorationCallback class.

        :param exploration_algorithm: The algorithm used for exploration.
        :param steps_per_rollout: The amount of steps to perform at each original model's rollout. Defaults to 2.
        :param pre_train_steps: The amount of steps to perform before the original model's training start. Defaults to 200.
        :param train_on_rollout: Whether to train on the rollout data. Defaults to False.
        :param verbose: The verbosity of the controller. Defaults to 0.
        """
        super().__init__(verbose)
        self.exploration_algorithm = exploration_algorithm
        self.steps_per_rollout = steps_per_rollout
        self.pre_train_steps = pre_train_steps
        self.train_on_rollout = train_on_rollout
        self.exploration_log_interval = exploration_log_interval

    def _init_callback(self) -> None:
        assert isinstance(self.model, OffPolicyAlgorithm)
        self._store_callback = StoreExplorationToBufferCallback(self.model)

        # copy goal sampler
        assert self.model.env is not None and self.exploration_algorithm.env is not None
        explore_envs = [env for env in self.exploration_algorithm.env.get_attr("unwrapped") if isinstance(env, MetaMixin)]
        envs = [env for env in self.model.env.get_attr("unwrapped") if isinstance(env, MetaMixin)]
        for explore_env, env in zip(explore_envs, envs):
            explore_env.goal_sampler = env.goal_sampler

        # setup log_dir
        if self.model.tensorboard_log:
            self.exploration_algorithm.tensorboard_log = self.model.logger.get_dir()

        return super()._init_callback()

    def _on_training_start(self) -> None:
        self.exploration_algorithm.learn(
            self.pre_train_steps,
            self._store_callback,
            log_interval=self.exploration_log_interval,
            tb_log_name="exploration",
            reset_num_timesteps=False,
            progress_bar=True,
        )
        if isinstance(self.exploration_algorithm, OffPolicyAlgorithm):
            self.exploration_algorithm._dump_logs()
        return super()._on_training_start()

    def _on_rollout_start(self) -> None:
        with MaybeNoTraining(not self.train_on_rollout, self.exploration_algorithm):
            self.exploration_algorithm.learn(
                self.steps_per_rollout,
                self._store_callback,
                log_interval=self.exploration_log_interval,
                tb_log_name="exploration",
                reset_num_timesteps=False,
            )
        if isinstance(self.exploration_algorithm, OffPolicyAlgorithm):
            self.exploration_algorithm._dump_logs()
        return super()._on_rollout_start()

    def _on_step(self) -> bool:
        return super()._on_step()


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
