import os
import numpy as np

import torch as th
from git.repo import Repo
from jsonargparse import ArgumentParser, Namespace
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class SaveConfigCallback(BaseCallback):
    def __init__(self, parser: ArgumentParser, cfg: Namespace) -> None:
        super().__init__()
        self.parser = parser
        self.cfg = cfg
        self.cfg.pop("subcommand")

    def _on_training_start(self) -> None:
        logger_dir = self.logger.get_dir()
        if logger_dir is None:
            raise ValueError("No logger directory")
        print(f"Experiment is logged at {logger_dir}")

        path = os.path.join(logger_dir, "config.yaml")
        self.parser.save(self.cfg, path, format="yaml", skip_check=True, overwrite=True, multifile=False)

        cfg = {k: v for k, v in vars(self.cfg.as_flat()).items() if isinstance(v, (int, float, str, bool))}
        metric_dict = {
            "eval/success_rate": 0.0,
            "eval/mean_reward": 0.0,
            "p2e-eval/success_rate": 0.0,
            "p2e-eval/mean_reward": 0.0,
        }
        self.logger.record("hparams", HParam(cfg, metric_dict), exclude=("stdout", "log", "json", "csv"))
        self.logger.dump()

        log_dir = self.logger.get_dir()
        if log_dir is not None:
            tag_name = log_dir.replace("\\", "/")
            repo = Repo(".")
            tag_with_same_name = [t for t in repo.tags if t.name == tag_name]
            if len(tag_with_same_name) != 0:
                repo.delete_tag(tag_with_same_name[0])
            repo.create_tag(tag_name)

            path = os.path.join(log_dir, "git_info")
            branch = repo.active_branch
            sha = repo.head.object.hexsha
            with open(path, "w") as f:
                f.write(f"{branch}+{sha}")

    def _on_step(self) -> bool:
        return super()._on_step()
