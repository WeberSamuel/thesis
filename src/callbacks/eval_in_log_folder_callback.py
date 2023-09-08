import os
from typing import Any, Dict

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
import wandb

class EvalInLogFolderCallback(EvalCallback):
    def __init__(self, *args, log_prefix="eval", **kwargs):
        super().__init__(*args, **kwargs)
        self.log_prefix = log_prefix

    def _init_callback(self):
        assert self.logger is not None
        logger_dir = self.logger.get_dir()
        if logger_dir is None:
            raise ValueError("No logger directory")
        if self.log_path is not None:
            self.log_path = os.path.join(logger_dir, self.log_path)
            self.video_dir = os.path.join(*self.log_path.split(os.sep)[:-1])
        else:
            self.video_dir = os.path.join(logger_dir, self.log_prefix)
        super()._init_callback()

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        super()._log_success_callback(locals_, globals_)
        info = locals_["info"]
        if isinstance(info, dict):
            self.success_reward += info.get("is_success", False)
            self.total_count += 1
        if self.callback_counter % self.eval_env.num_envs == 0:
            self.video_env.capture_frame()
        self.callback_counter += 1

    def step_wrapper(self, step_function):
        is_going_to_evaluate = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if not is_going_to_evaluate:
            return step_function()
        self.success_reward = 0
        self.total_count = 0
        self.callback_counter = 0
        self.video_env = VideoRecorder(self.eval_env, os.path.join(self.video_dir, str(self.num_timesteps) + ".mp4"), enabled=is_going_to_evaluate, disable_logger=True)
        
        result = step_function()
        
        if is_going_to_evaluate:
            frames = np.array(self.video_env.recorded_frames).transpose(0, -1, 1, 2)
            wandb.log({self.log_prefix: wandb.Video(frames, fps=30)})
            self.video_env.close()
            self.logger.record(f"{self.log_prefix}/success_reward", self.success_reward / self.total_count)
        return result

    def _on_step(self) -> bool:
        if hasattr(self.model.policy, "is_evaluating"):
            setattr(self.model.policy, "is_evaluating", True)

        result = self.step_wrapper(super()._on_step)

        if hasattr(self.model.policy, "is_evaluating"):
            setattr(self.model.policy, "is_evaluating", False)

        return result