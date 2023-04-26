from typing import List, Optional
import torch as th
from torch.distributions import Categorical, Normal
from stable_baselines3.common.torch_layers import create_mlp

from src.cemrl.types import CEMRLObsTensorDict


class Encoder(th.nn.Module):
    def __init__(
        self, num_classes: int, latent_dim: int, obs_dim: int, action_dim: int, complexity: float, reward_dim=1
    ) -> None:
        super().__init__()
        self.input_dim = 2 * obs_dim + action_dim + reward_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        encoder_out_dim = int(self.input_dim * complexity)
        self.encoder = th.nn.GRU(
            input_size=self.input_dim,
            hidden_size=int(self.input_dim * complexity),
            batch_first=True,
        )
        self.class_encoder = th.nn.Sequential(th.nn.Linear(encoder_out_dim, num_classes), th.nn.Softmax(dim=-1))
        self.gauss_encoder_list = th.nn.ModuleList([th.nn.Linear(encoder_out_dim, latent_dim * 2) for _ in range(num_classes)])

    @th.autocast("cuda")
    def forward(self, x: CEMRLObsTensorDict):
        # used in evaluation. Next_obs is not available, therefore simulate by removing last item
        y_distribution, z_distributions = self.encode(
            x["observation"][:, :-1], x["action"][:, 1:], x["reward"][:, 1:], x["observation"][:, 1:]
        )
        y, z = self.sample(y_distribution, z_distributions)
        return y, z

    @th.autocast("cuda")
    def encode(self, obs, action, reward, next_obs):
        encoder_input = th.cat([obs, action, reward, next_obs], dim=-1)
        _, m = self.encoder(encoder_input)
        m = m.squeeze(dim=0)
        y = self.class_encoder(m)
        class_mu_sigma = th.stack([class_net(m) for class_net in self.gauss_encoder_list])
        mus, sigmas = th.split(class_mu_sigma, self.latent_dim, dim=-1)
        sigmas = th.nn.functional.softplus(sigmas)

        y_distribution = Categorical(probs=y)
        z_distributions = [Normal(mu, sigma) for mu, sigma in zip(mus, sigmas)]
        return y_distribution, z_distributions

    @th.autocast("cuda")
    def sample(
        self,
        y_distribution: Categorical,
        z_distributions: List[Normal],
        y: Optional[int] = None,
    ):
        self.training = y is not None
        if self.training:
            y = th.ones(len(y_distribution.probs), dtype=th.long, device=y_distribution.probs.device) * y
        else:
            y = th.argmax(y_distribution.probs, dim=1)

        if self.training:
            sampled = th.cat(
                [th.unsqueeze(z_distribution.rsample(), 0) for z_distribution in z_distributions],
                dim=0,
            )
        else:
            sampled = th.cat(
                [th.unsqueeze(z_distribution.mean, 0) for z_distribution in z_distributions],
                dim=0,
            )

        # tensor with shape [batch, class, latent]
        permute = sampled.permute(1, 0, 2)
        mask = y[:, None, None].repeat(1, 1, self.latent_dim)
        z = th.squeeze(th.gather(permute, 1, mask), 1)
        return y, z
