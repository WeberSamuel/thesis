from typing import Any

import torch as th

from .config import MPCConfig


class MPC(th.nn.Module):
    def __init__(self, world_model, config: MPCConfig) -> None:
        super().__init__()
        self.config = config
        self.world_model = world_model

    def _loss(self, state: th.Tensor, action: th.Tensor) -> th.Tensor:
        raise NotImplementedError()

    def forward(self, state: Any, initial_action: th.Tensor) -> th.Tensor:
        batch_size = initial_action.shape[0]
        action = th.zeros((self.config.horizon, *initial_action.shape), dtype=initial_action.dtype, device=initial_action.device)
        action[0] = initial_action
        optim = th.optim.Adam([action], lr=1e-3)
        
        old_loss = th.tensor([th.inf] * batch_size, dtype=th.float32, device=initial_action.device)
        for i in range(self.config.max_num_iterations):
            optim.zero_grad()
            loss = self._loss(state, action)
            loss.backward()
            optim.step()

            if th.all(th.abs(loss - old_loss) < self.config.delta_loss_threshold):
                break

        return action[0]