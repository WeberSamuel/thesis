import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
from src.cemrl.buffers import NoLinkingCemrlReplayBuffer
import torch as th


class P2EBuffer(NoLinkingCemrlReplayBuffer):
    def __init__(self, *args, chunk_length=50, **kwargs):
        self.chunk_length = chunk_length
        super().__init__(*args, **kwargs)

    def sample(self, batch_size: int, env: VecNormalize | None = None):
        self._build_decoder_index()
        indices = np.random.choice(self.valid_indices(), batch_size)

        sac_obs = super().sample(batch_size, env)

        samples = self.get_decoder_targets(indices, env, num_decoder_targets=self.chunk_length)
        chunk_length = samples.observations.shape[1]
        z = sac_obs.observations["task_indicator"][:, None].expand(-1, chunk_length, -1)
        next_z = sac_obs.next_observations["task_indicator"][:, None].expand(-1, chunk_length, -1)

        obs = th.cat([samples.observations, z], dim=-1)
        next_obs = th.cat([samples.next_observations, next_z], dim=-1)

        return samples._replace(dones=samples.dones[..., None].float(), observations=obs, next_observations=next_obs)

    def _select_decoder_target_indices(self, indices: np.ndarray, num_decoder_targets: int) -> np.ndarray:
        ends = self.episode_ends[indices]
        lengths = self.episode_lengths[indices]

        min_length = min(lengths.min(), num_decoder_targets)
        indices = ends + np.random.randint(-lengths + 1, -min_length + 2, size=len(indices)) * self.n_envs
        ep_indices = indices[:, None] - np.flip(np.arange(min_length))[None] * self.n_envs

        is_exploration = indices >= self.buffer_size
        ep_indices = self.episode_ends[indices, None] - np.flip(np.arange(min_length))[None] * self.n_envs
        ep_indices[is_exploration] = (
            ep_indices[is_exploration] - self.buffer_size
        ) % self.explore_buffer_size + self.buffer_size
        ep_indices[~is_exploration] = ep_indices[~is_exploration] % self.buffer_size

        assert np.all(self.goal_idxs[ep_indices] == self.goal_idxs[ep_indices[:, 0]][:, None])
        return ep_indices
