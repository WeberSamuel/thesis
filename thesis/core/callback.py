from stable_baselines3.common.callbacks import BaseCallback

from .algorithm import BaseAlgorithm
from .buffer import ReplayBuffer
from .env import MetaMixin

class StateCopy:
    def __init__(self, model: BaseAlgorithm, exploration_model: BaseAlgorithm) -> None:
        self.model = model
        self.exploration_model = exploration_model

    def __enter__(self):
        self.original_last_obs = self.model._last_obs
        self.original_last_episode_starts = self.model._last_episode_starts
        self.original_last_original_obs = self.model._last_original_obs
        self.model._last_obs = self.exploration_model._last_obs
        self.model._last_episode_starts = self.exploration_model._last_episode_starts
        self.model._last_original_obs = self.exploration_model._last_original_obs

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.model._last_obs = self.original_last_obs
        self.model._last_episode_starts = self.original_last_episode_starts
        self.model._last_original_obs = self.original_last_original_obs


class StoreExplorationToBufferCallback(BaseCallback):
    """Callback that stores exploration data to the replay buffer of an off-policy algorithm."""
    model: BaseAlgorithm

    def __init__(self, model: BaseAlgorithm):
        """
        Initializes a new instance of the StoreExplorationToBufferCallback class.

        :param model: The original algorithm to store the exploration data to.
        :param converter: A converter to convert the exploration data to the original algorithm's format.
        """
        super().__init__()
        self.original_model = model

    def _on_step(self) -> bool:
        # we don't want to store the data twice
        if self.original_model.replay_buffer == self.model.replay_buffer:
            return super()._on_step()
        
        buffer_actions = self.locals["buffer_actions"]
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        new_obs = self.locals["new_obs"]

        for info in infos:
            info["is_exploration"] = True


        with StateCopy(self.original_model, self.model):  # type: ignore
            self.original_model._store_transition(
                self.original_model.replay_buffer, buffer_actions, new_obs, rewards, dones, infos
            )

        return super()._on_step()


class ExplorationCallback(BaseCallback):
    """
    A callback that performs exploration using a separate exploration algorithm.

    This callback is designed to be used with an off-policy algorithm that requires a replay buffer. It performs
    exploration using a separate exploration algorithm, and stores the exploration data to the replay buffer of the
    original algorithm. The exploration algorithm can be any algorithm that implements the BaseAlgorithm interface.
    """
    model: BaseAlgorithm

    def __init__(
        self,
        exploration_algorithm: BaseAlgorithm,
        steps_per_rollout=2,
        pre_train_steps=200,
        exploration_log_interval=1,
        exploration_rollout_gradient_steps=0,
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
        self.exploration_rollout_gradient_steps = exploration_rollout_gradient_steps
        self.exploration_log_interval = exploration_log_interval
        self.use_model_buffer = use_model_buffer

    def _init_callback(self) -> None:
        if self.use_model_buffer:
            self._store_callback = None
            self._link_replay_buffers()
        else:
            self._store_callback = StoreExplorationToBufferCallback(self.model)

        self._copy_goal_sampler()

        if self.model.tensorboard_log:
            self.exploration_algorithm.tensorboard_log = self.model.logger.get_dir()

        return super()._init_callback()

    def _copy_goal_sampler(self):
        assert self.model.env is not None and self.exploration_algorithm.env is not None
        explore_envs = [env for env in self.exploration_algorithm.env.get_attr("unwrapped") if isinstance(env, MetaMixin)]
        envs = [env for env in self.model.env.get_attr("unwrapped") if isinstance(env, MetaMixin)]
        for explore_env, env in zip(explore_envs, envs):
            explore_env.goal_sampler = env.goal_sampler

    def _link_replay_buffers(self):
        explore_buffer = self.exploration_algorithm.replay_buffer
        explore_buffer.storage = self.model.replay_buffer.storage
        explore_buffer.storage_idxs = explore_buffer.storage.start_new_episode(len(explore_buffer.storage_idxs))
        
    def _on_training_start(self) -> None:
        self.exploration_algorithm.learn(
            self.pre_train_steps,
            self._store_callback,
            log_interval=self.exploration_log_interval,
            tb_log_name="exploration",
            reset_num_timesteps=False,
            progress_bar=True,
        )

        self.exploration_algorithm.set_logger(self.exploration_algorithm.logger)  # enables reuse of logger at rollout
        self.exploration_algorithm._dump_logs()

        return super()._on_training_start()

    def _on_rollout_start(self) -> None:
        original_gradient_steps = self.model.gradient_steps
        self.exploration_algorithm.gradient_steps = self.exploration_rollout_gradient_steps

        self.exploration_algorithm.learn(
            self.steps_per_rollout,
            self._store_callback,
            log_interval=self.exploration_log_interval,
            tb_log_name="exploration",
            reset_num_timesteps=False,
        )
        self.exploration_algorithm._dump_logs()

        self.model.gradient_steps = original_gradient_steps
        return super()._on_rollout_start()

    def _on_step(self) -> bool:
        return super()._on_step()
