"""Callbacks used by stable-baselines3 that are useful for this project."""
import os
from typing import Any, List, Tuple, cast
from jsonargparse import ArgumentParser, Namespace
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import (
    EvalCallback,
    BaseCallback,
    CheckpointCallback,
    ConvertCallback,
)
from stable_baselines3.common.base_class import BaseAlgorithm
import torch as th
from src.cemrl.buffers import CombinedBuffer
from src.cemrl.policies import CEMRLPolicy
from src.cemrl.types import CEMRLObsTensorDict
from src.envs.wrappers.heatmap import HeatmapWrapper
from src.plan2explore.policies import Plan2ExplorePolicy
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from git.repo import Repo
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.vec_env import (
    VecEnvWrapper,
    unwrap_vec_wrapper,
    VecEnv,
    sync_envs_normalization,
    VecVideoRecorder,
)
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from typing import Union, Optional
from gym.core import Env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Figure


class EvalInLogFolderCallback(EvalCallback):
    def _init_callback(self):
        assert self.logger is not None
        if self.log_path is not None:
            self.log_path = os.path.join(self.logger.get_dir(), self.log_path)
        super()._init_callback()


class Plan2ExploreEvalCallback(EvalInLogFolderCallback):
    def __init__(
        self,
        eval_env: Union[Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        eval_model: Optional[BaseAlgorithm] = None,
    ):
        super().__init__(
            eval_env,
            callback_on_new_best,
            callback_after_eval,
            n_eval_episodes,
            eval_freq,
            log_path,
            best_model_save_path,
            deterministic,
            render,
            verbose,
            warn,
        )
        self.eval_model = eval_model

    def _init_callback(self) -> None:
        if self.eval_model is None:
            self.eval_model = self.model
        return super()._init_callback()

    """Modified EvalCallback that supports the Plan2ExplorePolicy.

    Since the Plan2ExplorePolicy is normally set to maximize the disagreement,
    the evaluation has to set a flag in the policy,
    such that it tries to maximize the future reward instead.
    """

    def _on_step(self) -> bool:
        assert self.eval_model is not None and isinstance(self.eval_model.policy, Plan2ExplorePolicy)
        self.eval_model.policy._is_collecting = False
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.eval_model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.eval_model.env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.eval_model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("p2e_eval/mean_reward", float(mean_reward))
            self.logger.record("p2e_eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("p2e_eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.eval_model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        self.eval_model.policy._is_collecting = True
        return continue_training


class ExplorationCallback(BaseCallback):
    """Callback that uses an exploration policy to collect additional samples."""

    def __init__(
        self,
        exploration_algorithm: BaseAlgorithm,
        steps_per_rollout=2,
        pre_train_steps=200,
        link_buffers=False,
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
        self.link_buffers = link_buffers
        self._dummy_callback = ConvertCallback(None)  # type: ignore
        self._dummy_callback.init_callback(self.exploration_algorithm)

    def _init_callback(self) -> None:
        assert isinstance(self.model, OffPolicyAlgorithm)
        assert self.model.env is not None and self.exploration_algorithm.env is not None

        if self.logger is not None:
            self.exploration_algorithm.set_logger(self.logger)
        if isinstance(self.exploration_algorithm, OffPolicyAlgorithm):
            if self.link_buffers:
                self.exploration_algorithm.replay_buffer = self.model.replay_buffer
            else:
                assert self.model.replay_buffer is not None
                assert self.exploration_algorithm.replay_buffer is not None
                self.model.replay_buffer = CombinedBuffer([self.model.replay_buffer, self.exploration_algorithm.replay_buffer])

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
            steps = self.pre_train_steps
            self.exploration_algorithm._setup_learn(steps, None)
            self._setup_rollout_collection(self.pre_train_steps)
            self.exploration_algorithm.collect_rollouts(
                self.exploration_algorithm.env,
                self._dummy_callback,
                TrainFreq(steps, TrainFrequencyUnit.STEP),
                self.exploration_algorithm.replay_buffer,
            )
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
        self.exploration_algorithm._total_timesteps = num_steps
        self.exploration_algorithm.num_timesteps = 0

        if self._same_base_envs:
            assert self.model is not None
            self.exploration_algorithm._last_obs = self.model._last_obs
            self.exploration_algorithm._last_episode_starts = self.model._last_episode_starts
            self.exploration_algorithm._last_original_obs = self.model._last_original_obs
            self.exploration_algorithm.ep_success_buffer = self.model.ep_info_buffer
            self.exploration_algorithm.ep_info_buffer = self.model.ep_info_buffer
        else:
            assert not self.link_buffers

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


class CheckpointInLogFolderCallback(CheckpointCallback):
    def _init_callback(self):
        assert self.logger is not None
        self.save_path = os.path.join(self.logger.get_dir(), self.save_path)
        super()._init_callback()


class SaveHeatmapCallback(BaseCallback):
    def __init__(
        self,
        envs: List[Tuple[str, VecEnv]],
        save_freq: int,
        save_path: str = "heatmaps",
        name_prefix: str = "heatmap",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.heatmap_wrappers = [(name, cast(HeatmapWrapper, unwrap_vec_wrapper(env, HeatmapWrapper))) for name, env in envs]
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        assert self.logger is not None
        self.save_path = os.path.join(self.logger.get_dir(), self.save_path)
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _get_heatmap_path(self, name: str) -> str:
        return os.path.join(self.save_path, f"{self.name_prefix}_{name}_{self.num_timesteps}_steps")

    def _on_step(self) -> bool:
        for name, heatmap_wrapper in self.heatmap_wrappers:
            if heatmap_wrapper is not None:
                if self.n_calls % self.save_freq == 0:
                    heatmap_path = self._get_heatmap_path(name)
                    heatmap_wrapper.save_heatmap(heatmap_path)

                    # for img in heatmap_wrapper.heatmaps_2d:
                    #     fig = plt.figure()
                    #     plt.imshow(img)
                    #     self.logger.record(
                    #         f"heatmap_{name}", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv")
                    #     )
                    #     plt.close()
                    #     plt.plot()

        return True


class LogLatentMedian(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.goals = []
        self.latents = []
        self.episode_goals = []
        self.episode_latents = []

    def _init_callback(self) -> None:
        assert isinstance(self.model.policy, CEMRLPolicy)
        self.model.policy.encoder.register_forward_hook(self.network_hook)

    def network_hook(self, module, args, output: Tuple[th.Tensor, th.Tensor]):
        input: CEMRLObsTensorDict = args[0]
        self.episode_goals.append(input["goal"][:, -1])
        self.episode_latents.append(output[1].detach())

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        if np.any(dones):
            latents = th.stack(self.episode_latents)
            goals = th.stack(self.episode_goals)
            self.goals.append(goals)
            self.latents.append(latents)
        return True


class RecordVideo(BaseCallback):
    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

    def _init_callback(self) -> None:
        self.save_path = os.path.join(self.logger.get_dir(), self.save_path)
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        assert self.parent is not None
        parent = self.parent
        while parent is not None:
            if isinstance(parent, EvalCallback):
                break
        video_env = VecVideoRecorder(parent.eval_env, self.save_path, lambda x: x == 0, name_prefix=f"{self.num_timesteps}")
        episode_rewards, episode_lengths = evaluate_policy(
            parent.model,
            video_env,
            n_eval_episodes=1,
            render=False,
            deterministic=parent.deterministic,
            return_episode_rewards=True,
            warn=parent.warn,
        )
        video_env.close()
        return True


class SaveConfigCallback(BaseCallback):
    def __init__(self, parser: ArgumentParser, cfg: Namespace) -> None:
        super().__init__()
        self.parser = parser
        self.cfg = cfg

    def _on_training_start(self) -> None:
        path = os.path.join(self.logger.get_dir(), "config.yaml")
        self.parser.save(self.cfg, path, format="yaml", skip_none=False, overwrite=True, multifile=False)

        cfg = {k: v for k,v in vars(self.cfg.as_flat()).items() if isinstance(v, (int, float, str, bool, th.Tensor))}
        metric_dict = {
            "eval/success_rate": 0.0,
            "eval/mean_reward": 0.0,
            "p2e-eval/success_rate": 0.0,
            "p2e-eval/mean_reward": 0.0,
        }
        self.logger.record("hparams", HParam(cfg, metric_dict), exclude=("stdout", "log", "json", "csv"))
        self.logger.dump()

        tag_name = self.logger.get_dir().replace("\\", "/")
        repo = Repo(".")
        tag_with_same_name = [t for t in repo.tags if t.name == tag_name]
        if len(tag_with_same_name) != 0:
            repo.delete_tag(tag_with_same_name[0])    
        repo.create_tag(tag_name)

        path = os.path.join(self.logger.get_dir(), "git_info")
        branch = repo.active_branch
        sha = repo.head.object.hexsha
        with open(path, "w") as f:
            f.write(f"{branch}+{sha}")

    def _on_step(self) -> bool:
        return super()._on_step()
