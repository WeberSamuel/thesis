from typing import Any
import torch as th
from dreamer.buffers import DreamerReplayBufferSamples
from dreamer.config import DreamerConfig
from gymnasium.spaces import flatdim
from submodules.dreamer import tools
from submodules.dreamer.models import WorldModel
from submodules.dreamer.networks import RSSM


class MetaEncoder(th.nn.Module):
    def __init__(
        self, num_classes: int, latent_dim: int, obs_dim: int, action_dim: int, complexity: float, reward_dim=1
    ) -> None:
        super().__init__()
        self.input_dim = obs_dim + action_dim + reward_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        encoder_out_dim = int(self.input_dim * complexity)
        self.encoder = th.nn.GRUCell(
            input_size=self.input_dim,
            hidden_size=int(self.input_dim * complexity),
        )

        self.class_encoder = th.nn.Sequential(th.nn.Linear(encoder_out_dim, num_classes), th.nn.Softmax(dim=-1))
        self.gauss_encoder_list = th.nn.ModuleList([th.nn.Linear(encoder_out_dim, latent_dim * 2) for _ in range(num_classes)])

    def forward(self, meta_state, features, action, reward):
        y_distribution, z_distributions, meta_state = self.encode(meta_state, features, action, reward)
        y, z = self.sample(y_distribution, z_distributions)
        return y, z, meta_state

    def encode(self, meta_state, features, action, reward):
        encoder_input = th.cat([features, action, reward], dim=-1)
        meta_state = self.encoder.forward(encoder_input, meta_state)
        meta_state = meta_state.squeeze(dim=0)
        y = self.class_encoder(meta_state)
        class_mu_sigma = th.stack([class_net(meta_state) for class_net in self.gauss_encoder_list])
        mus, sigmas = th.split(class_mu_sigma, self.latent_dim, dim=-1)
        sigmas = th.nn.functional.softplus(sigmas)

        y_distribution = th.distributions.Categorical(probs=y)
        z_distributions = [th.distributions.Normal(mu, sigma) for mu, sigma in zip(mus, sigmas)]
        return y_distribution, z_distributions, meta_state

    def sample(
        self,
        y_distribution: th.distributions.Categorical,
        z_distributions: list[th.distributions.Normal],
        y: int | None = None,
    ):
        self.training = y is not None
        if self.training:
            y = th.ones(len(y_distribution.probs), dtype=th.long, device=y_distribution.probs.device) * y  # type: ignore
        else:
            y = th.argmax(y_distribution.probs, dim=1)  # type: ignore

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
        mask = y[:, None, None].repeat(1, 1, self.latent_dim)  # type: ignore
        z = th.squeeze(th.gather(permute, 1, mask), 1)
        return y, z


class MetaRSSM(RSSM):
    def __init__(self, *args, meta_latent_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_latent_dim = meta_latent_dim
        if self._initial == "learned":
            self.meta_W = th.nn.Parameter(
                th.zeros((1, meta_latent_dim), device=th.device(self._device)), requires_grad=True  # type: ignore
            )

    def get_feat(self, state):
        result = super().get_feat(state)
        return th.cat([result, state["meta"]])

    def img_step(self, prev_state, prev_action, embed=None, sample=True):
        meta = prev_state["meta"]
        return super().img_step(prev_state, prev_action, embed, sample)

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        meta = prev_state["meta"]
        return super().obs_step(prev_state, prev_action, embed, is_first, sample)

    def initial(self, batch_size):
        result = super().initial(batch_size)
        if self._initial == "zeros":
            result["meta"] = th.zeros(batch_size, self.meta_latent_dim)
            return result
        elif self._initial == "learned":
            result["meta"] = th.tanh(self.meta_W).repeat(batch_size, 1)
            return result
        else:
            raise NotImplementedError(self._initial)


class MetaWorldModel(WorldModel):
    def __init__(self, obs_space, act_space, step, config: DreamerConfig):
        super().__init__(obs_space, act_space, step, config)
        stoch_size = config.dyn_stoch * (config.dyn_discrete if config.dyn_discrete else 1)
        self.dynamics = MetaRSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_input_layers,
            config.dyn_output_layers,
            config.dyn_rec_depth,
            config.dyn_shared,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_temp_post,
            config.dyn_min_std,
            config.dyn_cell,
            config.unimix_ratio,
            config.initial,
            flatdim(act_space),
            self.embed_size,
            config.device,
        )
        self.meta_encoder = MetaEncoder(
            config.meta_num_classes,
            config.meta_latent_dim,
            config.dyn_deter + stoch_size,
            flatdim(act_space),
            config.meta_encoder_complexity,
        )
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )

    def _train(self, data: DreamerReplayBufferSamples):
        batch_size, batch_length = data["action"].shape[:2]
        post, context, metrics = super()._train(data)
        
        init_meta = self.dynamics.initial(batch_size)["meta"]
        meta = tools.static_scan(
            lambda state, embed, action, reward: self.meta_encoder.forward(state, embed, action, reward),
            (context["embed"], data["action"], data["reward"]),
            init_meta
        )

        
        return post, context, metrics