import os
from typing import Any, Dict, Optional, Union
import gymnasium as gym

import torch as th
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import Video
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3.common.vec_env import VecEnv

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
            self.success_reward = +info.get("is_success", False)
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
            frames = th.tensor(np.array(self.video_env.recorded_frames))[None].permute(0, 1, -1, 2, 3)[-600::3]
            self.logger.record(self.log_prefix, Video(frames, fps=30))
            self.video_env.close()
            self.logger.record(f"{self.log_prefix}/success_reward", self.success_reward / self.total_count)
        return result

    def _on_step(self) -> bool:
        return self.step_wrapper(super()._on_step)