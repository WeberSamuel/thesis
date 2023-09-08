from typing import Any
import torch as th
from stable_baselines3.common.type_aliases import DictReplayBufferSamples

from src.cemrl.networks import Decoder, Encoder
    
def train_encoder(
    encoder: Encoder,
    decoder: Decoder|Any,
    encoder_samples: DictReplayBufferSamples,
    decoder_samples: DictReplayBufferSamples,
    optimizer: th.optim.Optimizer,
):
    encoder.train(True)
    encoder_input = encoder.from_samples_to_encoder_input(encoder_samples)
    decoder_input = decoder.from_samples_to_decoder_input(decoder_samples)
    reward_target, observation_target = decoder.from_samples_to_decoder_target(decoder_samples)
    
    y_distribution, z_distributions, _ = encoder.encode(encoder_input)

    # helper variables
    batch_size, decoder_batch_lenght = decoder_samples.actions.shape[:2]
    device = encoder_samples.actions.device
    num_classes = y_distribution.batch_shape[-1]

    # calculate component losses
    kl_qz_pz = th.empty(batch_size, num_classes, device=device)
    state_losses = th.empty(batch_size, num_classes, device=device)
    reward_losses = th.empty(batch_size, num_classes, device=device)
    nll_px = th.empty(batch_size, num_classes, device=device)

    for y in range(y_distribution.probs.shape[-1]): # type: ignore
        _, z = encoder.sample(y_distribution, z_distributions, y_int=y) # type: ignore
        # resize z to match decoder batch lenght
        z = z.unsqueeze(1).repeat(1, decoder_batch_lenght, 1)
    
        # get reconstruction
        state_estimate, reward_estimate = decoder(decoder_input, z)

        reward_loss = th.sum((reward_estimate - reward_target[None].expand(reward_estimate.shape)) ** 2, dim=-1)
        reward_loss = th.mean(reward_loss, dim=-1)
        reward_loss = th.mean(reward_loss, dim=0)
        reward_losses[:, y] = reward_loss

        state_loss = th.sum((state_estimate - observation_target[None].expand(state_estimate.shape)) ** 2, dim=-1)
        state_loss = th.mean(state_loss, dim=-1)
        state_loss = th.mean(state_loss, dim=0)
        state_losses[:, y] = state_loss

        # p(x|z_k)
        nll_px[:, y] = 0.3333 * state_loss + 0.6666 * reward_loss

        # KL ( q(z | x,y=k) || p(z|y=k))
        ones = th.ones(batch_size, encoder.latent_dim, device=device)
        prior_pz = th.distributions.normal.Normal(ones * y, ones * 0.5)
        kl_qz_pz[:, y] = th.sum(th.distributions.kl.kl_divergence(z_distributions[y], prior_pz), dim=-1)

    # KL ( q(y | x) || p(y) )
    ones = th.ones(batch_size, num_classes, device=device)
    prior_py = th.distributions.categorical.Categorical(probs=ones * (1.0 / num_classes))
    kl_qy_py = th.distributions.kl.kl_divergence(y_distribution, prior_py)

    alpha_kl_z = 1e-3  # weighting factor KL loss of z distribution vs prior
    beta_kl_y = 1e-3  # weighting factor KL loss of y distribution vs prior
    elbo = th.sum(th.sum(th.mul(y_distribution.probs, -nll_px - alpha_kl_z * kl_qz_pz), dim=-1) - beta_kl_y * kl_qy_py) # type: ignore
    loss = -elbo

    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()

    loss = loss / batch_size
    state_loss = th.mean(state_losses)
    reward_loss = th.mean(reward_losses)

    encoder.train(False)

    return loss, state_loss, reward_loss