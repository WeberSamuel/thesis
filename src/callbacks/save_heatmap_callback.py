import os
from typing import Dict, cast

from gymnasium import Env
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from stable_baselines3.common.vec_env import VecEnv

from ..envs.wrappers.heatmap import HeatmapWrapper


class SaveHeatmapCallback(BaseCallback):
    def __init__(
        self,
        envs: Dict[str, Env|VecEnv],
        save_freq: int,
        save_path: str = "heatmaps",
        name_prefix: str = "heatmap",
        verbose: int = 0,
    ):
        super().__init__(verbose)        
        self.envs = envs
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        assert self.logger is not None
        logger_dir = self.logger.get_dir()
        if logger_dir is None:
            raise ValueError("No logger directory")

        self.save_path = os.path.join(logger_dir, self.save_path)
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _get_heatmap_path(self, name: str) -> str:
        return os.path.join(self.save_path, f"{self.name_prefix}_{name}_{self.num_timesteps}_steps")

    def _on_step(self) -> bool:
        for name, env in self.envs.items():
            if self.n_calls % self.save_freq == 0:
                heatmap_path = self._get_heatmap_path(name)
                if isinstance(env, VecEnv):
                    heatmaps = np.sum(env.get_attr("heatmaps"), axis=0)
                    heatmaps_2d = np.sum(env.get_attr("heatmaps_2d"), axis=0)
                else:
                    heatmap_wrapper = cast(HeatmapWrapper, env)
                    heatmaps = heatmap_wrapper.heatmaps
                    heatmaps_2d = heatmap_wrapper.heatmaps_2d

                self.save_heatmap(heatmap_path, heatmaps, heatmaps_2d)
                for img in heatmaps_2d:
                    self.logger.record(
                        f"{self.name_prefix}_{name}", Image(img>1e-5, "HW"), exclude=("stdout", "log", "json", "csv")
                    )
        return True

    def save_heatmap(self, path: str, heatmaps:np.ndarray, heatmaps_2d:np.ndarray) -> None:
        np.save(path, heatmaps)
        np.save(path + "_2d", heatmaps_2d)