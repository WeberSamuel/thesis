from typing import List, Optional
import torch as th
from torch.distributions import Categorical, Normal, Distribution

from stable_baselines3.common.type_aliases import DictReplayBufferSamples

from src.cemrl.types import CEMRLObsTensorDict


class Decoder(th.nn.Module):
    def __init__(self, world_model: th.nn.Module) -> None:
        super().__init__()
        self.world_model = world_model

    def forward(self, x, z):
        return self.world_model(**x, z=z)

    def from_samples_to_decoder_target(self, decoder_samples: DictReplayBufferSamples) -> tuple[th.Tensor, th.Tensor]:
        return decoder_samples.rewards, decoder_samples.next_observations["observation"]

    def from_samples_to_decoder_input(self, decoder_samples: DictReplayBufferSamples):
        return dict(
            obs=decoder_samples.observations["observation"],
            action=decoder_samples.actions,
            next_obs=decoder_samples.next_observations["observation"],
            return_raw=True,
        )


class Encoder(th.nn.Module):
    def __init__(
        self,
        num_classes: int,
        latent_dim: int,
        obs_dim: int,
        action_dim: int,
        complexity: float,
        reward_dim=1,
        preprocess_dim=64,
    ) -> None:
        super().__init__()
        self.input_dim = 2 * obs_dim + action_dim + reward_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.encoder_state_dim = int(self.input_dim * complexity)
        self.pre_process_input = th.nn.Linear(self.input_dim, preprocess_dim)
        self.encoder = th.nn.GRU(
            input_size=preprocess_dim,
            hidden_size=self.encoder_state_dim,
            batch_first=True,
        )
        self.class_encoder = th.nn.Sequential(th.nn.Linear(self.encoder_state_dim, num_classes), th.nn.Softmax(dim=-1))
        self.gauss_encoder_list = th.nn.ModuleList(
            [th.nn.Linear(self.encoder_state_dim, latent_dim * 2) for _ in range(num_classes)]
        )

    def forward(self, encoder_input: th.Tensor, encoder_state: th.Tensor | None = None):
        # used in evaluation. Next_obs is not available, therefore simulate by removing last item
        y_distribution, z_distributions, encoder_state = self.encode(encoder_input, encoder_state)
        y, z = self.sample(y_distribution, z_distributions)
        return y, z, encoder_state

    def encode(self, encoder_input: th.Tensor, encoder_state: th.Tensor | None = None):
        encoder_input = self.pre_process_input(encoder_input)
        _, m = self.encoder(encoder_input, encoder_state)
        m = m.squeeze(dim=0)
        y = self.class_encoder(m)
        class_mu_sigma = th.stack([class_net(m) for class_net in self.gauss_encoder_list])
        mus, sigmas = th.split(class_mu_sigma, self.latent_dim, dim=-1)
        sigmas = th.nn.functional.softplus(sigmas)

        y_distribution = Categorical(probs=y)
        z_distributions = [Normal(mu, sigma) for mu, sigma in zip(mus, sigmas)]
        return y_distribution, z_distributions, m[None]

    def sample(
        self,
        y_distribution: Distribution,
        z_distributions: List[Distribution],
        y_int: Optional[int] = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        self.training = y_int is not None
        if self.training:
            y = th.ones(len(y_distribution.probs), dtype=th.long, device=y_distribution.probs.device) * y_int  # type: ignore
        else:
            y = th.argmax(y_distribution.probs, dim=1)  # type: ignore

        if self.training:
            sampled = th.stack([z_distribution.rsample() for z_distribution in z_distributions])
        else:
            sampled = th.stack([z_distribution.mean for z_distribution in z_distributions])

        # tensor with shape [batch, class, latent]
        permute = sampled.permute(1, 0, 2)
        mask = y[:, None, None].repeat(1, 1, self.latent_dim)
        z = th.squeeze(th.gather(permute, 1, mask), 1)
        return y, z

    def from_samples_to_encoder_input(self, encoder_context: DictReplayBufferSamples):
        return th.cat(
            [
                encoder_context.observations["observation"],
                encoder_context.next_observations["action"],
                encoder_context.next_observations["reward"],
                encoder_context.next_observations["observation"],
            ],
            dim=-1,
        )

    def from_obs_to_encoder_input(self, previous_obs: CEMRLObsTensorDict, current_obs: CEMRLObsTensorDict):
        return th.cat(
            [
                previous_obs["observation"],
                current_obs["action"],
                current_obs["reward"],
                current_obs["observation"],
            ],
            dim=-1,
        )[
            :, None
        ]  # add time dimension
