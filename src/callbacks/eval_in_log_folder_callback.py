import os
from typing import Any, Dict

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
        self.log_prefix = "eval"

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        super()._log_success_callback(locals_, globals_)
        info = locals_["info"]
        if isinstance(info, dict):
            self.success_reward = +info.get("is_success", False)
            self.total_count += 1

    def step_wrapper(self, step_function):
        self.success_reward = 0
        self.total_count = 0
        result = step_function()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.logger.record(f"{self.log_prefix}/success_reward", self.success_reward / self.total_count)
        return result

    def _on_step(self) -> bool:
        return self.step_wrapper(super()._on_step)