"""Networks used by the Plan2Explore Algorithm."""
from typing import Optional
import torch as th
from torch.utils.data import default_collate
from stable_baselines3.common.torch_layers import create_mlp
from src.utils import apply_function_to_type
from thesis.core.utils import build_network

class Ensemble(th.nn.Module):
    """Ensemble of networks.

    This class is a wrapper for multiple networks of the same kind (same input- and output-shape).
    It can return the individual network predictions or the mean and variances.
    This class can be used to compute an uncertainty estimate of the networks predictions.
    """

    def __init__(self, ensemble: th.nn.ModuleList) -> None:
        """Initialize the ensemble.

        Args:
            ensemble (th.nn.ModuleList): List of networks in the ensemble.
        """
        super().__init__()
        self.ensemble = ensemble

    def __len__(self):
        """Return the number of networks in the ensemble.

        Returns:
            int: Number of networs in the ensemble
        """
        return len(self.ensemble)

    def forward(self, *args, return_raw=False, **kwargs):
        """Forward computation of the ensemble.

        Args:
            return_raw_predictions (bool, optional):
                Whether to return the individual predictions or just the mean and variance of the ensemble's prediction.
                Defaults to False.

        Returns:
            Any: Collated results of the individual network if return_raw_predictions is set.
                 Otherwise the mean and variance of the collated network predictions is returned.
        """
        predictions = []
        for network in self.ensemble:
            predictions.append(network(*args, **kwargs))
        raw_predictions = default_collate(predictions)

        if return_raw:
            return raw_predictions

        return apply_function_to_type(raw_predictions, th.Tensor, lambda x: th.var_mean(x, dim=0))


class WorldModel(th.nn.Module):
    """Module used to predict the world behaivor."""

    def __init__(self, obs_dim: int, action_dim: int, z_dim: int, complexity: float) -> None:
        """Initialize the model.

        Args:
            obs_dim (int): Dimensions of the flatten observation
            action_dim (int): Dimensions of the flatten action
            z_dim (int): Dimensions of the task encoding
            complexity (float): Used to tune the amount of parameters in the model
        """
        super().__init__()
        input_size = obs_dim + action_dim + z_dim
        self.state_predictor = th.nn.Sequential(*create_mlp(input_size, obs_dim, net_arch=[int(complexity * input_size)] * 2))
        self.reward_predictor = th.nn.Sequential(
            *create_mlp(input_size + obs_dim, 1, net_arch=[int(complexity * input_size)] * 2)
        )

    def forward(self, obs: th.Tensor, action: th.Tensor, next_obs: Optional[th.Tensor] = None, z: Optional[th.Tensor] = None):
        """Compute the observation and reward prediction given an observation and action.

        Args:
            obs (th.Tensor): Current observation from which an action is performed
            action (th.Tensor): Action to apply in the observation
            next_obs (th.Tensor, optional): Tensor to pass in a ground truth next state passed to the reward predictor. Defaults to None
            z (th.Tensor, optional): Additonal latent tensor to include. Defaults to None

        Returns:
            Tuple(th.Tensor): Tuple of the next observation and reward achieved by performing the action
        """
        z = z if z is not None else obs.new_empty(*obs.shape[:-1], 0)
        obs_pred = self.state_predictor(th.cat([obs, action, z], dim=-1))
        next_obs = obs_pred if next_obs is None else next_obs
        reward_pred = self.reward_predictor(th.cat([obs, action, next_obs, z], dim=-1))
        return obs_pred, reward_pred
    
class OneStepModel(th.nn.Module):
    def __init__(self, input_dim: int, obs_dim: int, reward_dim: int = 1, complexity = 20.) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.reward_dim = reward_dim
        self.network = build_network(input_dim, [int(complexity * input_dim)] * 2, th.nn.ELU, obs_dim + reward_dim)

    def forward(self, x: th.Tensor):
        x = self.network(x)
        return x[..., :self.obs_dim], x[..., -self.reward_dim:]