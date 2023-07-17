import os
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.logger import Video

class RecordVideo(BaseCallback):
    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.local_save_path = save_path
        self.save_path = save_path

    def _init_callback(self) -> None:
        logger_dir = self.logger.get_dir()
        if logger_dir is None:
            raise ValueError("No logger directory")
        self.save_path = os.path.join(logger_dir, self.save_path)
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        assert self.parent is not None
        parent = self.parent
        while parent is not None:
            if isinstance(parent, EvalCallback):
                break
        video_env = VecVideoRecorder(parent.eval_env, self.save_path, lambda x: x == 0, video_length=2000, name_prefix=f"{self.num_timesteps}")
        episode_rewards, episode_lengths = evaluate_policy(
            parent.model,
            video_env,
            n_eval_episodes=1,
            render=False,
            deterministic=parent.deterministic,
            return_episode_rewards=False,
            warn=parent.warn,
        )
        assert video_env.video_recorder is not None
        video_env.video_recorder.disable_logger = True
        frames = th.tensor(np.array(video_env.video_recorder.recorded_frames))[None].permute(0, 1, -1, 2, 3)[:, -200::2]
        self.logger.record(self.local_save_path, Video(frames, fps=30))
        video_env.close()
        
        return True