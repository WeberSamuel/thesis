from typing import Dict
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv, VecMonitor
from gym.spaces.utils import flatten_space, flatten
from stable_baselines3.common.vec_env.util import dict_to_obs

from src.envs.toy_goal_env import ToyGoalEnv


class HeatmapWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, bin_size_per_dimension=512, idxs_2d=[(0, 1)], cooling_factor=0.97):
        super().__init__(venv)
        self.idxs2d = idxs_2d
        self.cooling_factor = cooling_factor
        space = flatten_space(self.observation_space)
        self.heatmap_idxs = np.where(~(np.isinf(space.low) | np.isinf(space.high)))[0]
        self.heatmap_linspaces = np.stack(
            [np.linspace(space.low[idx], space.high[idx], bin_size_per_dimension) for idx in self.heatmap_idxs]
        )
        self.heatmaps = np.zeros((len(self.heatmap_idxs), bin_size_per_dimension))
        self.heatmaps_2d = np.zeros((len(idxs_2d), bin_size_per_dimension, bin_size_per_dimension))

    def add_to_heatmap(self, obs):
        if isinstance(obs, Dict):
            obs = [dict(zip(obs, t)) for t in zip(*obs.values())]
        flatten_obs = np.stack([flatten(self.observation_space, single_obs) for single_obs in obs])
        for idx, linspace, heatmap in zip(self.heatmap_idxs, self.heatmap_linspaces, self.heatmaps):
            insert_at = np.searchsorted(linspace, flatten_obs[:, idx])
            heatmap[insert_at] += 1

        for (idx0, idx1), heatmap2d in zip(self.idxs2d, self.heatmaps_2d):
            x = np.searchsorted(self.heatmap_linspaces[idx0], flatten_obs[:, self.heatmap_idxs[idx0]])
            y = np.searchsorted(self.heatmap_linspaces[idx1], flatten_obs[:, self.heatmap_idxs[idx1]])
            heatmap2d[x, y] += 1

        self.heatmaps *= self.cooling_factor
        self.heatmaps_2d *= self.cooling_factor

    def save_heatmap(self, path):
        np.save(path, self.heatmaps)
        np.save(path + "_2d", self.heatmaps_2d)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.add_to_heatmap(obs)
        return obs, rewards, dones, infos

    def reset(self):
        obs = self.venv.reset()
        self.add_to_heatmap(obs)
        return obs
