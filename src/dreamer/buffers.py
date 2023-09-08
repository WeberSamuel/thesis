from typing import TypedDict
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
import torch as th

from src.core.buffers import ReplayBuffer, Storage

class DreamerReplayBufferSamples(TypedDict):
    observation: th.Tensor
    goal_idx: th.Tensor
    goal: th.Tensor
    is_first: th.Tensor
    is_terminal: th.Tensor
    reward: th.Tensor
    action: th.Tensor
    

class DreamerReplayBuffer(ReplayBuffer):
    def dreamer_sample(self, batch_size: int, env: VecNormalize | None = None, batch_lenght: int = 64):
        self.batch_length = batch_lenght
        samples = self.sample_context(batch_size, env, batch_lenght)

        return DreamerReplayBufferSamples(
            observation = samples.observations["observation"],
            goal_idx = samples.observations["goal_idx"],
            goal = samples.observations["goal"],
            is_first = th.zeros_like(samples.observations["is_first"]).squeeze(-1),
            is_terminal = samples.dones.squeeze(-1),
            reward = samples.rewards,
            action = samples.actions,
        )

    def sample_context(
        self, batch_size: int, env: VecNormalize | None = None, context_length: int = 64
    ) -> DictReplayBufferSamples:
        self.prepare_sampling_if_necessary()
        data = self.get_empty_data_template(batch_size, context_length)

        for i in range(batch_size):
            size = 0
            while size < context_length:
                episode_idx = np.random.choice(self.valid_episodes, p=self.episode_probabilities)
                ep_length = min(context_length - size, int(self.storage.episode_lengths[episode_idx]))
                self.add_episode_to_data_template(i, size, episode_idx, slice(ep_length), data)
                size += ep_length
        return self.post_process_samples(data, env)

    def prepare_sampling(self, storage: Storage):
        super().prepare_sampling(storage)
        valid_ep_length = storage.episode_lengths[self.valid_episodes]
        self.episode_probabilities = valid_ep_length / np.sum(valid_ep_length)
