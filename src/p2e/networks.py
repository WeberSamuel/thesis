from dataclasses import dataclass
from typing import Tuple

import torch as th
import torch.nn as nn
from torch.distributions import Independent, Normal, TanhTransform

from src.p2e.utils import build_network, create_normal_dist, horizontal_forward, asdict
from src.utils import DeviceAwareModuleMixin


@dataclass(frozen=True)
class EncoderKwargs:
    activation: str = "ReLU"
    depth: int = 32
    kernel_size: int = 4
    stride: int = 2


@dataclass(frozen=True)
class DecoderKwargs:
    activation: str = "ReLU"
    depth: int = 32
    kernel_size: int = 5
    stride: int = 2


@dataclass(frozen=True)
class RSSMRecurrentKwargs:
    activation: str = "ELU"
    hidden_size: int = 100


@dataclass(frozen=True)
class RSSMTransitionKwargs:
    activation: str = "ELU"
    hidden_size: int = 100
    num_layers: int = 2
    min_std: float = 0.1


@dataclass(frozen=True)
class RSSMRepresentationKwargs:
    activation: str = "ELU"
    hidden_size: int = 100
    num_layers: int = 2
    min_std: float = 0.1


@dataclass(frozen=True)
class RSSMKwargs:
    recurrent_kwargs: RSSMRecurrentKwargs = RSSMRecurrentKwargs()
    transition_kwargs: RSSMTransitionKwargs = RSSMTransitionKwargs()
    representation_kwargs: RSSMRepresentationKwargs = RSSMRepresentationKwargs()


@dataclass(frozen=True)
class RewardModelKwargs:
    hidden_size: int = 200
    num_layers: int = 2
    activation: str = "ELU"


@dataclass(frozen=True)
class ContinueModelKwargs:
    hidden_size: int = 200
    num_layers: int = 3
    activation: str = "ELU"


@dataclass(frozen=True)
class ActorKwargs:
    hidden_size: int = 200
    min_std: float = 1e-4
    init_std: float = 5.0
    mean_scale: float = 5.0
    activation: str = "ELU"
    num_layers: int = 4


@dataclass(frozen=True)
class CriticKwargs:
    hidden_size: int = 200
    num_layers: int = 3
    activation: str = "ELU"


@dataclass(frozen=True)
class OneStepModelKwargs:
    hidden_size: int = 200
    num_layers: int = 4
    activation: str = "ELU"


class OneStepModel(nn.Module):
    def __init__(
        self,
        action_size: int,
        embedded_state_size: int,
        stochastic_size: int,
        deterministic_size: int,
        hidden_size: int,
        num_layers: int,
        activation: str,
    ):
        """
        For plan2explore
        There are several variations, but in our implementation,
        we use stochastic and deterministic actions as input and embedded observations as output
        """
        super().__init__()
        self.embedded_state_size = embedded_state_size
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size

        self.action_size = action_size

        self.network = build_network(
            self.deterministic_size + self.stochastic_size + action_size,
            hidden_size,
            num_layers,
            activation,
            self.embedded_state_size,
        )

    def forward(self, action, stochastic: th.Tensor, deterministic: th.Tensor):
        stoch_deter = th.concat([stochastic, deterministic], dim=-1)
        x = horizontal_forward(
            self.network,
            action,
            stoch_deter,
            output_shape=(self.embedded_state_size,),
        )
        dist = create_normal_dist(x, std=1, event_shape=1)
        return dist


class Critic(nn.Module):
    def __init__(
        self,
        stochastic_size: int,
        deterministic_size: int,
        hidden_size: int = 400,
        num_layers: int = 3,
        activation: str = "ELU",
    ):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            hidden_size,
            num_layers,
            activation,
            1,
        )

    def forward(self, posterior: th.Tensor, deterministic: th.Tensor):
        x = horizontal_forward(self.network, posterior, deterministic, output_shape=(1,))
        dist = create_normal_dist(x, std=1, event_shape=1)
        return dist


class Actor(nn.Module):
    def __init__(
        self,
        discrete_action_bool: bool,
        action_size: int,
        stochastic_size: int,
        deterministic_size: int,
        hidden_size: int = 400,
        num_layers: int = 2,
        activation: str = "ELU",
        mean_scale: float = 5.0,
        init_std: float = 5.0,
        min_std: float = 0.0001,
    ):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.discrete_action_bool = discrete_action_bool
        self.mean_scale = mean_scale
        self.init_std = init_std
        self.min_std = min_std

        action_size = action_size if discrete_action_bool else 2 * action_size

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            hidden_size,
            num_layers,
            activation,
            action_size,
        )
        self.intrinsic = False

    def forward(self, posterior: th.Tensor, deterministic: th.Tensor) -> th.Tensor:
        x = th.cat((posterior, deterministic), -1)
        x = self.network(x)
        if self.discrete_action_bool:
            dist: th.distributions.Distribution = th.distributions.OneHotCategorical(logits=x)
            action = dist.sample() + dist.probs - dist.probs.detach() # type: ignore
        else:
            dist = create_normal_dist(
                x,
                mean_scale=self.mean_scale,
                init_std=self.init_std,
                min_std=self.min_std,
                activation=th.tanh,
            )
            dist = th.distributions.TransformedDistribution(dist, TanhTransform())
            action = th.distributions.Independent(dist, 1).rsample()
        return action


class RewardModel(nn.Module):
    """
    A neural network module that predicts the reward for a given state.

    Args:
        state_size (int): The size of the state input.
        action_size (int): The size of the action input.
        hidden_size (int, optional): The number of units in each hidden layer. Defaults to 400.
        num_layers (int, optional): The number of hidden layers. Defaults to 2.
        activation (str, optional): The activation function to use in the hidden layers. Defaults to "ELU".
    """
    def __init__(
        self,
        stochastic_size: int,
        deterministic_size: int,
        hidden_size: int = 400,
        num_layers: int = 2,
        activation: str = "ELU",
    ):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            hidden_size,
            num_layers,
            activation,
            1,
        )

    def forward(self, posterior, deterministic):
        x = horizontal_forward(self.network, posterior, deterministic, output_shape=(1,))
        dist = create_normal_dist(x, std=1, event_shape=1)
        return dist


class ContinueModel(nn.Module):
    """
    A PyTorch module that represents the continuation model used in the P2E algorithm.

    This model takes as input the posterior and deterministic states of the agent and outputs a Bernoulli distribution
    that represents the probability of continuing the current episode.

    Args:
        stochastic_size (int): The size of the stochastic state space.
        deterministic_size (int): The size of the deterministic state space.
        hidden_size (int, optional): The number of units in each hidden layer of the network. Defaults to 400.
        num_layers (int, optional): The number of hidden layers in the network. Defaults to 3.
        activation (str, optional): The activation function to use in the hidden layers. Defaults to "ELU".
    """

    def __init__(
        self,
        stochastic_size: int,
        deterministic_size: int,
        hidden_size: int = 400,
        num_layers: int = 3,
        activation: str = "ELU",
    ):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            hidden_size,
            num_layers,
            activation,
            1,
        )

    def forward(self, posterior, deterministic):
        x = horizontal_forward(self.network, posterior, deterministic, output_shape=(1,))
        dist = th.distributions.Bernoulli(logits=x)
        return dist


class RecurrentModel(DeviceAwareModuleMixin, nn.Module):
    """
    Recurrent module that models the dynamics of the stochastic latent variable and the deterministic latent variable over time.

    Args:
        stochastic_size (int): The size of the stochastic latent variable.
        deterministic_size (int): The size of the deterministic latent variable.
        action_size (int): The size of the action tensor.
        activation (str): The activation function to use. Defaults to "ELU".
        hidden_size (int): The size of the hidden layer. Defaults to 200.
    """
    def __init__(
        self,
        action_size: int,
        stochastic_size: int,
        deterministic_size: int,
        activation: str = "ELU",
        hidden_size: int = 200,
    ):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size

        self.activation = getattr(nn, activation)()

        self.linear = nn.Linear(self.stochastic_size + action_size, hidden_size)
        self.recurrent = nn.GRUCell(hidden_size, self.deterministic_size)

    def forward(self, embedded_state: th.Tensor, action: th.Tensor, deterministic: th.Tensor) -> th.Tensor:
        """
        Forward pass of the recurrent model.

        Args:
            embedded_state (th.Tensor): The embedded state.
            action (th.Tensor): The action tensor.
            deterministic (th.Tensor): The deterministic tensor.

        Returns:
            th.Tensor: The output tensor.
        """
        x = th.cat((embedded_state, action), 1)
        x = self.activation(self.linear(x))
        x = self.recurrent(x, deterministic)
        return x

    def input_init(self, batch_size):
        """
        Initializes the input tensor.

        Args:
            batch_size (int): The batch size.

        Returns:
            th.Tensor: The input tensor.
        """
        return th.zeros(batch_size, self.deterministic_size).to(self.device)


class TransitionModel(DeviceAwareModuleMixin, nn.Module):
    """
    A PyTorch module that represents the transition model of a P2E agent.

    This module takes as input the current state of the environment
    and outputs a distribution of next states.

    Args:
        stochastic_size (int): The size of the stochastic latent variable (next states).
        deterministic_size (int): The size of the deterministic latent variable (current state).
        activation (str, optional): The activation function to use in the neural network layers.
            Defaults to "ELU".
        hidden_size (int, optional): The number of neurons in the hidden layers of the neural network.
            Defaults to 200.
        num_layers (int, optional): The number of hidden layers in the neural network. Defaults to 2.
        min_std (float, optional): The minimum standard deviation of the Gaussian distribution used to
            sample the stochastic latent variable. Defaults to 0.1.
    """

    def __init__(
        self,
        stochastic_size: int,
        deterministic_size: int,
        activation: str = "ELU",
        hidden_size: int = 200,
        num_layers: int = 2,
        min_std: float = 0.1,
    ):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.min_std = min_std

        self.network = build_network(
            self.deterministic_size,
            hidden_size,
            num_layers,
            activation,
            self.stochastic_size * 2,
        )

    def forward(self, x: th.Tensor) -> Tuple[th.distributions.Independent | th.distributions.Normal, th.Tensor]:
        x = self.network(x)
        prior_dist = create_normal_dist(x, min_std=self.min_std)
        prior = prior_dist.rsample()
        return prior_dist, prior

    def input_init(self, batch_size):
        """
        Initializes the starting input tensor for the model's forward method.

        Args:
            batch_size (int): The size of the batch.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, self.stochastic_size) filled with zeros.
        """
        return th.zeros(batch_size, self.stochastic_size).to(self.device)


class RepresentationModel(nn.Module):
    """
    A PyTorch module that represents the representation model of a P2E agent.

    This module takes as input the current state of the environment and the current observation
    and outputs a latent representation of the state as well as the distribution from which the latents are sampled.

    Args:
        embedded_state_size (int): The size of the embedded state (current state).
        stochastic_size (int): The size of the stochastic latent variable (latent representation).
        deterministic_size (int): The size of the deterministic latent variable (current state).
        activation (str, optional): The activation function to use in the neural network layers.
            Defaults to "ELU".
        hidden_size (int, optional): The number of neurons in the hidden layers of the neural network.
            Defaults to 200.
        num_layers (int, optional): The number of hidden layers in the neural network. Defaults to 2.
    """

    def __init__(
        self,
        embedded_state_size: int,
        stochastic_size: int,
        deterministic_size: int,
        activation: str = "ELU",
        hidden_size: int = 200,
        num_layers: int = 2,
        min_std: float = 1,
    ):
        super().__init__()
        self.embedded_state_size = embedded_state_size
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.min_std = min_std

        self.network = build_network(
            self.embedded_state_size + self.deterministic_size,
            hidden_size,
            num_layers,
            activation,
            self.stochastic_size * 2,
        )

    def forward(self, embedded_observation: th.Tensor, deterministic: th.Tensor) -> Tuple[Independent | Normal, th.Tensor]:
        x = self.network(th.cat((embedded_observation, deterministic), 1))
        posterior_dist = create_normal_dist(x, min_std=self.min_std)
        posterior = posterior_dist.rsample()
        return posterior_dist, posterior


class RSSM(nn.Module):
    """
    The Recurrent State-Space Model (RSSM) of a P2E agent.

    Args:
        action_size (int): The size of the action space.
        stochastic_size (int): The size of the stochastic latent variable.
        deterministic_size (int): The size of the deterministic latent variable.
        embedded_state_size (int): The size of the embedded state (current state).
        recurrent_kwargs (RSSMRecurrentKwargs, optional): The keyword arguments for the recurrent model.
        transition_kwargs (RSSMTransitionKwargs, optional): The keyword arguments for the transition model.
        representation_kwargs (RSSMRepresentationKwargs, optional): The keyword arguments for the representation model.
    """
    def __init__(
        self,
        action_size: int,
        stochastic_size: int,
        deterministic_size: int,
        embedded_state_size: int,
        recurrent_kwargs: RSSMRecurrentKwargs = RSSMRecurrentKwargs(),
        transition_kwargs: RSSMTransitionKwargs = RSSMTransitionKwargs(),
        representation_kwargs: RSSMRepresentationKwargs = RSSMRepresentationKwargs(),
    ):
        super().__init__()

        self.recurrent_model = RecurrentModel(action_size, stochastic_size, deterministic_size, **asdict(recurrent_kwargs))
        self.transition_model = TransitionModel(stochastic_size, deterministic_size, **asdict(transition_kwargs))
        self.representation_model = RepresentationModel(
            embedded_state_size, stochastic_size, deterministic_size, **asdict(representation_kwargs)
        )

    def recurrent_model_input_init(self, batch_size):
        """
        Initializes the input for the RSSM submodules.

        Args:
            batch_size (int): The size of the batch.

        Returns:
            Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
            - The initial input for the transition model. Shape: (batch_size, stochastic_size).
            - The initial hidden state of the recurrent model. Shape: (batch_size, deterministic_size).
        """
        return self.transition_model.input_init(batch_size), self.recurrent_model.input_init(batch_size)
