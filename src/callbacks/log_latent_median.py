from typing import Tuple
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

from src.cemrl.policies import CEMRLPolicy
from src.cemrl.types import CEMRLObsTensorDict

class LogLatentMedian(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.goals = []
        self.latents = []
        self.episode_goals = []
        self.episode_latents = []

    def _init_callback(self) -> None:
        assert isinstance(self.model.policy, CEMRLPolicy)
        self.model.policy.encoder.register_forward_hook(self.network_hook)

    def network_hook(self, module, args, output: Tuple[th.Tensor, th.Tensor]):
        self.episode_goals.append(args[0]["goal"][:, -1])
        self.episode_latents.append(output[1].detach())

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        if np.any(dones):
            latents = th.stack(self.episode_latents)
            goals = th.stack(self.episode_goals)
            self.goals.append(goals)
            self.latents.append(latents)
        return True
