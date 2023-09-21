from typing import List, Tuple

import numpy as np
from gymnasium import Env, ObservationWrapper, spaces
from gymnasium.spaces.utils import flatten, flatten_space


class HeatmapWrapper(ObservationWrapper):
    def __init__(self, env: Env, bin_size_per_dimension=512, idxs_2d: List[Tuple[int, int]] = [(0, 1)], cooling_factor=0.999):
        super().__init__(env)
        self.idxs2d = idxs_2d
        self.cooling_factor = cooling_factor

        space = flatten_space(self.observation_space)
        assert isinstance(space, spaces.Box)

        self.heatmap_idxs = np.where(~(np.isinf(space.low) | np.isinf(space.high)))[0]
        if len(self.heatmap_idxs) == 0:
            self.heatmap_linspaces = np.zeros((0, 0))
        else:
            self.heatmap_linspaces = np.stack(
                [np.linspace(space.low[idx], space.high[idx], bin_size_per_dimension) for idx in self.heatmap_idxs]
            )
        self.heatmaps = np.zeros((len(self.heatmap_idxs), bin_size_per_dimension))
        self.heatmaps_2d = np.zeros((len(idxs_2d), bin_size_per_dimension, bin_size_per_dimension))

    def add_to_heatmap(self, obs: np.ndarray):
        flatten_obs = flatten(self.observation_space, obs)
        for idx, linspace, heatmap in zip(self.heatmap_idxs, self.heatmap_linspaces, self.heatmaps):
            insert_at = np.searchsorted(linspace, flatten_obs[idx])
            heatmap[insert_at] += 1

        for (idx0, idx1), heatmap2d in zip(self.idxs2d, self.heatmaps_2d):
            x = np.searchsorted(self.heatmap_linspaces[idx0], flatten_obs[self.heatmap_idxs[idx0]])
            y = np.searchsorted(self.heatmap_linspaces[idx1], flatten_obs[self.heatmap_idxs[idx1]])
            heatmap2d[x, y] += 1

        self.heatmaps *= self.cooling_factor
        self.heatmaps_2d *= self.cooling_factor

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.add_to_heatmap(observation)
        return observation
