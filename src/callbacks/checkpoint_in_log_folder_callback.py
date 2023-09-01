import os
from stable_baselines3.common.callbacks import CheckpointCallback


class CheckpointInLogFolderCallback(CheckpointCallback):
    def _init_callback(self):
        assert self.logger is not None
        logger_dir = self.logger.get_dir()
        if logger_dir is None:
            raise ValueError("No logger directory")
        self.save_path = os.path.join(logger_dir, self.save_path)
        super()._init_callback()
