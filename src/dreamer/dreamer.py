import numpy as np

from src.dreamer.buffers import DreamerReplayBuffer
from src.dreamer.policies import DreamerPolicy
from src.core.state_aware_algorithm import StateAwareOffPolicyAlgorithm


class Dreamer(StateAwareOffPolicyAlgorithm):
    policy: DreamerPolicy
    replay_buffer: DreamerReplayBuffer

    def __init__(
        self,
        *args,
        batch_size=16,
        gradient_steps: int = 1,
        learning_starts: int = 10_000,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
            batch_size=batch_size,
            support_multi_env=True,
            sde_support=False,
            learning_rate=1e-3,
            gradient_steps=gradient_steps,
            learning_starts=learning_starts,
        )
        self._setup_model()

    def train(self, gradient_steps: int, batch_size: int) -> None:
        for i in range(gradient_steps):
            self.policy.dreamer._train(
                self.replay_buffer.dreamer_sample(
                    batch_size, self.get_vec_normalize_env(), self.policy.dreamer._config.batch_length
                )
            )
        for name, values in self.policy.dreamer._metrics.items():
            self.logger.record(name, float(np.mean(values)))
            self.policy.dreamer._metrics[name] = []
        if self.num_timesteps // self.n_envs % self.log_interval == 0:
            self._dump_logs()