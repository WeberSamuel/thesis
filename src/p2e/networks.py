import torch as th

from src.plan2explore.networks import Ensemble

class LatentDisagreementEnsemble(th.nn.Module):
    def __init__(self, ensemble: Ensemble) -> None:
        super().__init__()
        self.ensemble = ensemble

    def forward(self, *args, **kwargs):
        (state_var, state_mean), (reward_var, reward_mean) = self.ensemble(*args, **kwargs)
        return state_mean, th.cat([state_var, reward_var], dim=-1).mean(dim=-1)