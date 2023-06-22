import os

from stable_baselines3.common.callbacks import EvalCallback


class EvalInLogFolderCallback(EvalCallback):
    def _init_callback(self):
        assert self.logger is not None
        logger_dir = self.logger.get_dir()
        if logger_dir is None:
            raise ValueError("No logger directory")
        if self.log_path is not None:
            self.log_path = os.path.join(logger_dir, self.log_path)
        super()._init_callback()