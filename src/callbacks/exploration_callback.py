import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.vec_env import VecEnv
from src.envs.meta_env import MetaMixin

from src.core.buffers import ReplayBuffer


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

        with StateCopy(self.original_model, self.model):  # type: ignore
            assert self.original_model.replay_buffer is not None
            self.original_model._store_transition(
                self.original_model.replay_buffer, buffer_actions, new_obs, rewards, dones, infos
            )

        return super()._on_step()


class TagExplorationDataCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            info["is_exploration"] = True
        return super()._on_step()


class ExplorationCallback(BaseCallback):
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
        exploration_log_interval=8,
        use_model_buffer: bool = True,
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
        self.use_model_buffer = use_model_buffer

    def _init_callback(self) -> None:
        assert isinstance(self.model, OffPolicyAlgorithm)

        if (
            self.use_model_buffer
            and self.model.replay_buffer is not None
            and isinstance(self.exploration_algorithm, OffPolicyAlgorithm)
        ):
            self._exploration_callback = TagExplorationDataCallback()
            explore_buffer = self.exploration_algorithm.replay_buffer
            if isinstance(explore_buffer, ReplayBuffer) and isinstance(self.model.replay_buffer, ReplayBuffer):
                explore_buffer.storage = self.model.replay_buffer.storage
                explore_buffer.storage_idxs = explore_buffer.storage.start_new_episode(len(explore_buffer.storage_idxs))
                if hasattr(self.model.replay_buffer, "encoder"):
                    setattr(explore_buffer, "encoder", getattr(self.model.replay_buffer, "encoder"))
            else:
                self.exploration_algorithm.replay_buffer = self.model.replay_buffer
        else:
            self._exploration_callback = CallbackList(
                [TagExplorationDataCallback(), StoreExplorationToBufferCallback(self.model)]
            )

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
            self._exploration_callback,
            log_interval=self.exploration_log_interval,
            tb_log_name="exploration",
            reset_num_timesteps=False,
            progress_bar=True,
        )
        self.exploration_algorithm.set_logger(self.exploration_algorithm.logger)
        if isinstance(self.exploration_algorithm, OffPolicyAlgorithm):
            self.exploration_algorithm._dump_logs()
        return super()._on_training_start()

    def _on_rollout_start(self) -> None:
        if isinstance(self.exploration_algorithm, OffPolicyAlgorithm):
            self.exploration_algorithm.gradient_steps = 1
        with MaybeNoTraining(not self.train_on_rollout, self.exploration_algorithm):
            self.exploration_algorithm.learn(
                self.steps_per_rollout,
                self._exploration_callback,
                log_interval=self.exploration_log_interval,
                tb_log_name="exploration",
                reset_num_timesteps=False,
            )
        if self.train_on_rollout and isinstance(self.exploration_algorithm, OffPolicyAlgorithm):
            self.exploration_algorithm._dump_logs()
        return super()._on_rollout_start()

    def _on_step(self) -> bool:
        return super()._on_step()
