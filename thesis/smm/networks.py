import torch as th

from ..core.module import BaseModule
from .utils import TruncatedNormal, weight_init


class Encoder(th.nn.Module):
    def __init__(self, obs_shape: tuple[int, ...]):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = th.nn.Sequential(
            th.nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 32, 3, stride=1),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 32, 3, stride=1),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 32, 3, stride=1),
            th.nn.ReLU(),
        )

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(th.nn.Module):
    def __init__(self, obs_type, obs_dim: int, action_dim: int, feature_dim: int, hidden_dim: int):
        super().__init__()

        feature_dim = feature_dim if obs_type == "pixels" else hidden_dim

        self.trunk = th.nn.Sequential(th.nn.Linear(obs_dim, feature_dim), th.nn.LayerNorm(feature_dim), th.nn.Tanh())

        policy_layers = []
        policy_layers += [th.nn.Linear(feature_dim, hidden_dim), th.nn.ReLU(inplace=True)]
        # add additional hidden layer for pixels
        if obs_type == "pixels":
            policy_layers += [th.nn.Linear(hidden_dim, hidden_dim), th.nn.ReLU(inplace=True)]
        policy_layers += [th.nn.Linear(hidden_dim, action_dim)]

        self.policy = th.nn.Sequential(*policy_layers)

        self.apply(weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = th.tanh(mu)
        std = th.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)
        return dist


class Critic(th.nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == "pixels":
            # for pixels actions will be added after trunk
            self.trunk = th.nn.Sequential(th.nn.Linear(obs_dim, feature_dim), th.nn.LayerNorm(feature_dim), th.nn.Tanh())
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = th.nn.Sequential(
                th.nn.Linear(obs_dim + action_dim, hidden_dim), th.nn.LayerNorm(hidden_dim), th.nn.Tanh()
            )
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [th.nn.Linear(trunk_dim, hidden_dim), th.nn.ReLU(inplace=True)]
            if obs_type == "pixels":
                q_layers += [th.nn.Linear(hidden_dim, hidden_dim), th.nn.ReLU(inplace=True)]
            q_layers += [th.nn.Linear(hidden_dim, 1)]
            return th.nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(weight_init)

    def forward(self, obs, action):
        inpt = obs if self.obs_type == "pixels" else th.cat([obs, action], dim=-1)
        h = self.trunk(inpt)
        h = th.cat([h, action], dim=-1) if self.obs_type == "pixels" else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2


class VAE(BaseModule):
    def __init__(self, obs_dim: int, z_dim: int, code_dim: int, vae_beta: float):
        super().__init__()
        self.z_dim = z_dim
        self.code_dim = code_dim

        self.make_networks(obs_dim, z_dim, code_dim)
        self.beta = vae_beta

        self.apply(weight_init)

    def make_networks(self, obs_dim, z_dim, code_dim):
        self.enc = th.nn.Sequential(th.nn.Linear(obs_dim + z_dim, 150), th.nn.ReLU(), th.nn.Linear(150, 150), th.nn.ReLU())
        self.enc_mu = th.nn.Linear(150, code_dim)
        self.enc_logvar = th.nn.Linear(150, code_dim)
        self.dec = th.nn.Sequential(
            th.nn.Linear(code_dim, 150), th.nn.ReLU(), th.nn.Linear(150, 150), th.nn.ReLU(), th.nn.Linear(150, obs_dim + z_dim)
        )

    def encode(self, obs_z):
        enc_features = self.enc(obs_z)
        mu = self.enc_mu(enc_features)
        logvar = self.enc_logvar(enc_features)
        stds = (0.5 * logvar).exp()
        return mu, logvar, stds

    def forward(self, obs_z, epsilon):
        mu, logvar, stds = self.encode(obs_z)
        code = epsilon * stds + mu
        obs_distr_params = self.dec(code)
        return obs_distr_params, (mu, logvar, stds)

    def loss(self, obs_z):
        epsilon = th.randn([obs_z.shape[0], self.code_dim]).to(self.device)
        obs_distr_params, (mu, logvar, stds) = self(obs_z, epsilon)
        kle = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        log_prob = th.nn.functional.mse_loss(obs_z, obs_distr_params, reduction="none")

        loss = self.beta * kle + log_prob.mean()
        return loss, log_prob.sum(list(range(1, len(log_prob.shape)))).view(log_prob.shape[0], 1)


class PVae(VAE):
    def make_networks(self, obs_shape: tuple[int], z_dim: int, code_dim: int):
        self.enc = th.nn.Sequential(
            th.nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 32, 3, stride=1),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 32, 3, stride=1),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 32, 3, stride=1),
            th.nn.ReLU(),
            th.nn.Flatten(),
            th.nn.Linear(32 * 35 * 35, 150),
            th.nn.ReLU(),
        )
        self.enc_mu = th.nn.Linear(150, code_dim)
        self.enc_logvar = th.nn.Linear(150, code_dim)
        self.dec = th.nn.Sequential(
            th.nn.Linear(code_dim, 32 * 35 * 35),
            th.nn.ReLU(),
            th.nn.Unflatten(dim=1, unflattened_size=(32, 35, 35)),
            th.nn.ConvTranspose2d(32, 32, 3, stride=1),
            th.nn.ReLU(),
            th.nn.ConvTranspose2d(32, 32, 3, stride=1),
            th.nn.ReLU(),
            th.nn.ConvTranspose2d(32, 32, 3, stride=1),
            th.nn.ReLU(),
            th.nn.ConvTranspose2d(32, 32, 3, stride=1),
            th.nn.ReLU(),
            th.nn.ConvTranspose2d(32, 32, 3, stride=2),
            th.nn.ReLU(),
            th.nn.Conv2d(32, obs_shape[0], 4, stride=1),
        )


class SMM(th.nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim, vae_beta):
        super().__init__()
        self.z_dim = z_dim
        self.z_pred_net = th.nn.Sequential(
            th.nn.Linear(obs_dim, hidden_dim),
            th.nn.ReLU(),
            th.nn.Linear(hidden_dim, hidden_dim),
            th.nn.ReLU(),
            th.nn.Linear(hidden_dim, z_dim),
        )
        self.vae = VAE(obs_dim=obs_dim, z_dim=z_dim, code_dim=128, vae_beta=vae_beta)
        self.apply(weight_init)

    def predict_logits(self, obs):
        z_pred_logits = self.z_pred_net(obs)
        return z_pred_logits

    def loss(self, logits, z):
        z_labels = th.argmax(z, 1)
        return th.nn.CrossEntropyLoss(reduction="none")(logits, z_labels)
