import math
import re
from typing import TypedDict, cast

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import distributions as torchd
from torch import nn

import src.dreamer.distributions as distributions
from src.dreamer import tools
from src.utils import DeviceAwareModuleMixin


class Context(TypedDict):
    """A dictionary containing tensors representing the context of a Dreamer agent."""

    embed: th.Tensor
    """A tensor representing the embedding of the context."""
    feat: th.Tensor
    """A tensor representing the features of the context."""
    kl: th.Tensor
    """A tensor representing the KL divergence of the context."""
    postent: th.Tensor
    """A tensor representing the posterior entropy of the context."""


class State(TypedDict):
    """A dictionary that contains the state of the dreamer model."""

    stoch: th.Tensor
    """A tensor representing the stochastic part of the state."""
    deter: th.Tensor
    """A tensor representing the deterministic part of the state."""


class ContinousState(State):
    """A dictionary representing a continuous state of the dreamer model."""

    mean: th.Tensor
    """The mean value of the state."""
    std: th.Tensor
    """The standard deviation of the state."""


class DiscreteState(State):
    """A dictionary representing a discrete state of the dreamer model."""

    logit: th.Tensor
    """The logit of the state."""


class RSSM(DeviceAwareModuleMixin, nn.Module):
    """
    Recurrent State-Space Model (RSSM) module that models the dynamics of a system given observations and actions.

    This version support continuous or discrete latent variables.
    """

    def __init__(
        self,
        num_actions: int,
        embed: int,
        stoch=30,
        deter=200,
        hidden=200,
        layers_input=1,
        layers_output=1,
        rec_depth=1,
        shared=False,
        discrete=0,
        activation="SiLU",
        normalization="LayerNorm",
        mean_act="none",
        std_activation="softplus",
        temp_post=True,
        min_std=0.1,
        cell="gru",
        unimix_ratio=0.01,
        initial="learned",
    ):
        """
        Initialize a Dreamer neural network module.

        Args:
            num_actions (int): The number of possible actions.
            embed (int): The size of the embedding layer.
            stoch (int, optional): The number of stochastic variables. Defaults to 30.
            deter (int, optional): The number of deterministic variables. Defaults to 200.
            hidden (int, optional): The number of hidden units in the network. Defaults to 200.
            layers_input (int, optional): The number of input layers. Defaults to 1.
            layers_output (int, optional): The number of output layers. Defaults to 1.
            rec_depth (int, optional): The number of recurrent layers. Defaults to 1.
            shared (bool, optional): Whether to use a shared embedding layer. Defaults to False.
            discrete (int, optional): The number of discrete variables. Defaults to 0.
            act (str, optional): The activation function to use. Defaults to "SiLU".
            norm (str, optional): The normalization function to use. Defaults to "LayerNorm".
            mean_act (str, optional): The activation function for the mean output. Defaults to "none".
            std_act (str, optional): The activation function for the standard deviation output. Defaults to "softplus".
            temp_post (bool, optional): Whether to use a temporal posterior. Defaults to True.
            min_std (float, optional): The minimum standard deviation. Defaults to 0.1.
            cell (str, optional): The type of recurrent cell to use. Defaults to "gru".
            unimix_ratio (float, optional): The ratio of uniform to Gaussian mixture components. Defaults to 0.01.
            initial (str, optional): The type of initialization to use. Defaults to "learned".
        """
        super().__init__()
        self._stoch = stoch
        """Number of stochastic latent variables."""

        self._deter = deter
        """Number of deterministic latent variables."""

        self._hidden = hidden
        """Number of hidden units in the recurrent cell and output layers."""

        self._min_std = min_std
        """Minimum standard deviation for the output distribution."""

        self._layers_input = layers_input
        """Number of layers in the input processing network."""

        self._layers_output = layers_output
        """Number of layers in the output processing networks."""

        self._rec_depth = rec_depth
        """Number of recurrent iterations."""

        self._shared = shared
        """Whether to share the input processing network across time steps."""

        self._discrete = discrete
        """Number of categories for each discrete latent variable (0 for continuous)."""

        activation = getattr(th.nn, activation)
        normalization = getattr(th.nn, normalization)
        self._mean_act = mean_act
        """Activation function to use for the mean of the output distribution."""

        self._std_act = std_activation
        """Activation function to use for the standard deviation of the output distribution."""

        self._temp_post = temp_post
        """Whether to use the temporal posterior for the output distribution."""

        self._unimix_ratio = unimix_ratio
        """Ratio of uniform mixture to Gaussian mixture in the output distribution."""

        self._initial = initial
        """Type of initial state to use (zeros or learned)."""

        self._embed = embed
        """Embedding size for the input observations."""

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        if self._shared:
            inp_dim += self._embed
        for i in range(self._layers_input):
            inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            inp_layers.append(normalization(self._hidden, eps=1e-03))
            inp_layers.append(activation())
            if i == 0:
                inp_dim = self._hidden
        self._inp_layers = nn.Sequential(*inp_layers)
        """Input processing network."""
        self._inp_layers.apply(tools.weight_init)

        if cell == "gru":
            cell_network = GRUCell(self._hidden, self._deter)
            cell_network.apply(tools.weight_init)
        elif cell == "gru_layer_norm":
            cell_network = GRUCell(self._hidden, self._deter, norm=True)
            cell_network.apply(tools.weight_init)
        else:
            raise NotImplementedError(cell)

        self.cell = cell_network
        """Recurrent cell"""

        img_out_layers = []
        inp_dim = self._deter
        for i in range(self._layers_output):
            img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            img_out_layers.append(normalization(self._hidden, eps=1e-03))
            img_out_layers.append(activation())
            if i == 0:
                inp_dim = self._hidden
        self._img_out_layers = nn.Sequential(*img_out_layers)
        """Output processing network for image observations."""
        self._img_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        if self._temp_post:
            inp_dim = self._deter + self._embed
        else:
            inp_dim = self._embed
        for i in range(self._layers_output):
            obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            obs_out_layers.append(normalization(self._hidden, eps=1e-03))
            obs_out_layers.append(activation())
            if i == 0:
                inp_dim = self._hidden
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        """Output processing network for other observations."""
        self._obs_out_layers.apply(tools.weight_init)

        if self._discrete:
            ims_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            ims_stat_layer.apply(tools.weight_init)
            obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            obs_stat_layer.apply(tools.weight_init)
        else:
            ims_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            ims_stat_layer.apply(tools.weight_init)
            obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            obs_stat_layer.apply(tools.weight_init)
        self._ims_stat_layer = ims_stat_layer
        """Linear layer for computing the state from the image observation."""
        self._obs_stat_layer = obs_stat_layer
        """Linear layer for computing the state from the other observation."""

        if self._initial == "learned":
            self.W = th.nn.Parameter(
                th.zeros((1, self._deter), device=th.device(self.device)),
                requires_grad=True,
            )
            """Learnable parameter for the initial deterministic state."""

    def initial(self, batch_size: int) -> State:
        """
        Return an initial state for the agent.

        Args:
            batch_size (int): The number of states to generate.

        Returns:
            State: An initial state for the agent.
        """
        state: State

        deter = th.zeros(batch_size, self._deter).to(self.device)
        if self._discrete:
            state = DiscreteState(
                logit=th.zeros([batch_size, self._stoch, self._discrete]).to(self.device),
                stoch=th.zeros([batch_size, self._stoch, self._discrete]).to(self.device),
                deter=deter,
            )
        else:
            state = ContinousState(
                mean=th.zeros([batch_size, self._stoch]).to(self.device),
                std=th.zeros([batch_size, self._stoch]).to(self.device),
                stoch=th.zeros([batch_size, self._stoch]).to(self.device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = th.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(
        self, embed: th.Tensor, action: th.Tensor, is_first: th.Tensor, state: State | None = None
    ) -> tuple[State, State]:
        """
        Compute the posterior and prior states of the world model given the current observation.

        Args:
            embed (torch.Tensor): The embedding of the current observation, with shape (batch_size, time, embedding_size).
            action (torch.Tensor): The action taken after the current observation, with shape (batch_size, time, action_size).
            is_first (torch.Tensor): A binary tensor indicating whether each element in the batch is the first observation, with shape (batch_size, time).
            state (State, optional): The previous state of the world model. If None, the initial state is used. Defaults to None.

        Returns:
            A tuple of two States: the posterior state and the prior state of the world model, both with the same structure as the initial state.
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(prev_state[0], prev_act, embed, is_first),
            (action, embed, is_first),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return cast(State, post), cast(State, prior)

    def imagine(self, action: th.Tensor, state: State | None = None) -> State:
        """
        Imagine the next state given the current action and state.

        Args:
            action (th.Tensor): The action tensor of shape (batch_size, action_dim).
            state (State, optional): The current state. Defaults to None.

        Returns:
            State: The next state.
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = tools.static_scan(self.img_step, (action,), state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return cast(State, prior)

    def get_features(self, state: State):
        """
        Return the features of the state.

        Args:
            state (State): The input state tensor.

        Returns:
            Tensor: The concatenated tensor of stochastic and deterministic components.
        """
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return th.cat([stoch, state["deter"]], -1)

    def get_distribution(self, state: State) -> th.distributions.Distribution:
        """
        Return a probability distribution over the actions, given the current state.

        Args:
            state (State): The current state of the environment.

        Returns:
            torch.distributions.Distribution: A probability distribution over the actions.
        """
        dist: th.distributions.Distribution
        if self._discrete:
            state = cast(DiscreteState, state)
            logit = state["logit"]
            dist = torchd.independent.Independent(distributions.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)
        else:
            state = cast(ContinousState, state)
            mean, std = state["mean"], state["std"]
            dist = distributions.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), 1))
        return dist

    def obs_step(self, prev_state: State, prev_action: th.Tensor, embed: th.Tensor, is_first: th.Tensor, sample=True):
        """
        Take in the previous state, previous action, embedding, and a boolean flag indicating whether it is the first step.

        Returns the posterior and prior states.

        Args:
            prev_state (State): A dictionary containing the previous state information.
            prev_action (th.Tensor): A tensor containing the previous action.
            embed (th.Tensor): A tensor containing the embedding.
            is_first (th.Tensor): A boolean tensor indicating whether it is the first step.
            sample (bool): A boolean flag indicating whether to sample from the distribution.

        Returns:
            post (State): A dictionary containing the posterior state information.
            prior (State): A dictionary containing the prior state information.
        """
        # if shared is True, prior and post both use same networks(inp_layers, _img_out_layers, _ims_stat_layer)
        # otherwise, post use different network(_obs_out_layers) with prior[deter] and embed as inputs
        prev_action *= (1.0 / th.clip(th.abs(prev_action), min=1.0)).detach()

        if th.sum(is_first) > 0:
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                val = cast(th.Tensor, val)
                is_first_r = th.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = val * (1.0 - is_first_r) + init_state[key] * is_first_r

        prior = self.img_step(prev_state, prev_action, None, sample)
        if self._shared:
            post = self.img_step(prev_state, prev_action, embed, sample)
        else:
            if self._temp_post:
                x = th.cat([prior["deter"], embed], -1)
            else:
                x = embed
            # (batch_size, prior_deter + embed) -> (batch_size, hidden)
            x = self._obs_out_layers(x)
            # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
            stats = self._suff_stats_layer("obs", x)
            if sample:
                stoch = self.get_distribution(stats).sample()
            else:
                stoch = self.get_distribution(stats).mode
            post = cast(State, {"stoch": stoch, "deter": prior["deter"], **stats})
        return post, prior

    # this is used for making future image
    def img_step(self, prev_state: State, prev_action: th.Tensor, embed: th.Tensor | None = None, sample=True) -> State:
        """
        Compute a forward step of the image prediction model.

        Args:
            prev_state (State): The previous state of the model.
            prev_action (th.Tensor): The previous action taken by the agent.
            embed (th.Tensor, optional): An optional tensor to use as input to the embedding layer. Defaults to None.
            sample (bool, optional): Whether to sample from the predicted distribution or use its mode. Defaults to True.

        Returns:
            State: The updated state of the model after the forward step.
        """
        # (batch, stoch, discrete_num)
        prev_action *= (1.0 / th.clip(th.abs(prev_action), min=1.0)).detach()
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        if self._shared:
            if embed is None:
                shape = list(prev_action.shape[:-1]) + [self._embed]
                embed = th.zeros(shape)
            # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action, embed)
            x = th.cat([prev_stoch, prev_action, embed], -1)
        else:
            x = th.cat([prev_stoch, prev_action], -1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self._inp_layers(x)
        deter = th.empty(0)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self.cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_distribution(stats).sample()
        else:
            stoch = self.get_distribution(stats).mode
        prior = {"stoch": stoch, "deter": deter, **stats}
        return cast(State, prior)

    def get_stoch(self, deter: th.Tensor):
        """
        Return the stochastic output of the network given a deterministic input.

        Args:
            deter (th.Tensor): The deterministic input tensor.

        Returns:
            th.Tensor: The stochastic output tensor.
        """
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_distribution(stats)
        return dist.mode

    def _suff_stats_layer(self, name: str, x: th.Tensor) -> State:
        """
        Compute the state of a given input tensor `x` for a specific module `name`.

        Args:
            name (str): The name of the module for which to compute the state.
            x (th.Tensor): The input tensor for which to compute the state.

        Returns:
            State: A `State` object.
        """
        if self._discrete:
            if name == "ims":
                x = self._ims_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return DiscreteState(logit=logit)  # type: ignore
        else:
            if name == "ims":
                x = self._ims_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = th.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * th.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: th.nn.functional.softplus(std),
                "abs": lambda: th.abs(std + 1),
                "sigmoid": lambda: th.sigmoid(std),
                "sigmoid2": lambda: 2 * th.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return ContinousState(mean=mean, std=std)  # type: ignore

    def kl_loss(self, post: State, prior: State, free: float, dyn_scale: float, rep_scale: float):
        """
        Compute the KL divergence loss between the posterior and prior distributions.

        Args:
            post (State): The posterior state distribution.
            prior (State): The prior state distribution.
            free (float): The minimum value for the loss.
            dyn_scale (float): The scaling factor for the dynamic loss.
            rep_scale (float): The scaling factor for the representation loss.

        Returns:
            tuple: A tuple containing the total loss, the value loss, the dynamic loss, and the representation loss.
        """
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_distribution(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,  # type: ignore
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,  # type: ignore
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,  # type: ignore
            dist(prior) if self._discrete else dist(prior)._dist,  # type: ignore
        )
        rep_loss = th.mean(th.clip(rep_loss, min=free))
        dyn_loss = th.mean(th.clip(dyn_loss, min=free))
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class MultiEncoder(nn.Module):
    """A PyTorch module that encodes observations using a combination of convolutional and multi-layer perceptron (MLP) encoders."""

    def __init__(
        self,
        shapes: dict[str, tuple[int, ...]],
        mlp_keys: str,
        cnn_keys: str,
        activation: str,
        normalization: str,
        cnn_depth: int,
        kernel_size: int,
        min_res: int,
        mlp_layers: int,
        mlp_units: int,
        symlog_inputs: bool,
    ):
        """
        Initialize a MultiEncoder object that encodes observation tensors using both convolutional and MLP encoders.

        Args:
            shapes (dict): A dictionary containing the shapes of the observation tensors.
            mlp_keys (str): A regular expression pattern that matches the keys of the observation tensors that should be encoded using an MLP encoder.
            cnn_keys (str): A regular expression pattern that matches the keys of the observation tensors that should be encoded using a convolutional encoder.
            act (nn.Module): The activation function to use in the encoders.
            norm (nn.Module): The normalization function to use in the encoders.
            cnn_depth (int): The depth of the convolutional encoder.
            kernel_size (int): The kernel size of the convolutional encoder.
            minres (bool): Whether to use residual connections in the convolutional encoder.
            mlp_layers (int): The number of layers in the MLP encoder.
            mlp_units (int): The number of units in each layer of the MLP encoder.
            symlog_inputs (bool): Whether to use symmetrical logarithmic scaling for the input values.
        """
        super(MultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {k: v for k, v in shapes.items() if k not in excluded and not k.startswith("log_")}
        self.cnn_shapes = {k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)}
        """A dictionary containing the shapes of the observation tensors that should be encoded using a convolutional encoder."""

        self.mlp_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2) and re.match(mlp_keys, k)}
        """A dictionary containing the shapes of the observation tensors that should be encoded using an MLP encoder."""

        self.outdim = 0
        """The dimensionality of the encoded observations."""

        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(input_shape, cnn_depth, activation, normalization, kernel_size, min_res)
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = np.sum([np.prod(v) for v in self.mlp_shapes.values()]).item()
            self._mlp = MLP(input_size, None, mlp_layers, mlp_units, activation, normalization, symlog_inputs=symlog_inputs)
            self.outdim += mlp_units

    def forward(self, obs: dict[str, th.Tensor]):
        """
        Forward pass of the neural network.

        Args:
            obs (dict[str, th.Tensor]): Dictionary containing the input tensors.

        Returns:
            th.Tensor: Output tensor of the neural network.
        """
        outputs = []
        if self.cnn_shapes:
            inputs = th.cat([obs[k] for k in self.cnn_shapes], -1)
            outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            inputs = th.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        return th.cat(outputs, -1)


class MultiDecoder(nn.Module):
    """Decoder of the Dreamer observations. Can handle image and non-image data."""

    def __init__(
        self,
        feat_size: int,
        shapes: dict[str, tuple[int, ...]],
        mlp_keys: str,
        cnn_keys: str,
        activation: str | type[nn.Module],
        normalization: str | type[nn.Module],
        cnn_depth: int,
        kernel_size: int,
        min_res: int,
        mlp_layers: int,
        mlp_units: int,
        cnn_sigmoid: bool,
        image_dist: str,
        vector_dist: str,
    ):
        """
        Initialize a MultiDecoder module.

        Args:
            feat_size (int): The size of the input feature tensor.
            shapes (dict[str, tuple[int, ...]]): A dictionary of shapes for each input tensor.
            mlp_keys (str): A regular expression pattern for selecting MLP keys from the shapes dictionary.
            cnn_keys (str): A regular expression pattern for selecting CNN keys from the shapes dictionary.
            activation (str | type[nn.Module]): The activation function to use in the decoder.
            normalization (str | type[nn.Module]): The normalization function to use in the decoder.
            cnn_depth (int): The depth of the CNN decoder.
            kernel_size (int): The kernel size of the CNN decoder.
            min_res (int): The minimum resolution of the CNN decoder.
            mlp_layers (int): The number of layers in the MLP decoder.
            mlp_units (int): The number of units in each layer of the MLP decoder.
            cnn_sigmoid (bool): Whether to apply a sigmoid activation function to the output of the CNN decoder.
            image_dist (str): The distribution to use for image outputs.
            vector_dist (str): The distribution to use for vector outputs.
        """
        super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)}
        """A dictionary mapping CNN output keys to their expected shapes."""

        self.mlp_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2) and re.match(mlp_keys, k)}
        """A dictionary mapping MLP output keys to their expected shapes."""

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                feat_size, shape, cnn_depth, activation, normalization, kernel_size, min_res, cnn_sigmoid=cnn_sigmoid
            )
            """The convolutional decoder, if any."""

        if self.mlp_shapes:
            self._mlp = MLP(feat_size, self.mlp_shapes, mlp_layers, mlp_units, activation, normalization, vector_dist)
            """The MLP decoder, if any."""

        self._image_dist = image_dist
        """The type of distribution to use for image outputs."""

    def forward(self, features: th.Tensor) -> dict[str, th.distributions.Distribution]:
        """
        Forward pass of the Decoder network.

        Args:
            features (th.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            dict[str, th.distributions.Distribution]: A dictionary of distributions, where the keys are the names of the
            corresponding layers and the values are instances of `th.distributions.Distribution`.
        """
        dists = {}
        if self.cnn_shapes:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = th.split(outputs, split_sizes, -1)
            dists.update({key: self._make_image_dist(output) for key, output in zip(self.cnn_shapes.keys(), outputs)})
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean: th.Tensor):
        """
        Create an image distribution based on the specified type of distribution.

        Args:
            mean (torch.Tensor): The mean tensor used to create the distribution.

        Returns:
            tools.ContDist or tools.MSEDist: The image distribution object.
        """
        if self._image_dist == "normal":
            return distributions.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3))
        if self._image_dist == "mse":
            return distributions.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    """Convolutional encoder module that encodes an input tensor into a latent representation."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        depth: int = 32,
        activation: str | type[th.nn.Module] = "SiLU",
        normalization: str | type[th.nn.Module] = "LayerNorm",
        kernel_size: int = 4,
        min_res: int = 4,
    ):
        """
        Initialize a ConvEncoder instance.

        Args:
            input_shape (tuple[int, ...]): The shape of the input tensor (height, width, channels).
            depth (int, optional): The number of channels in the first layer. Defaults to 32.
            activation (str, optional): The name of the activation function to use. Defaults to "SiLU".
            normalization (str, optional): The name of the normalization layer to use. Defaults to "LayerNorm".
            kernel_size (int, optional): The size of the convolutional kernel. Defaults to 4.
            min_res (int, optional): The minimum resolution of the output feature map. Defaults to 4.
        """
        super(ConvEncoder, self).__init__()
        if isinstance(activation, str):
            activation = cast(type[th.nn.Module], getattr(th.nn, activation))
        if isinstance(normalization, str):
            normalization = cast(type[th.nn.Module], getattr(th.nn, normalization))
        h, w, input_ch = input_shape
        layers: list[th.nn.Module] = []
        out_dim: int = 0
        for i in range(int(np.log2(h) - np.log2(min_res))):
            if i == 0:
                in_dim = input_ch
            else:
                in_dim = 2 ** (i - 1) * depth
            out_dim = 2**i * depth
            layers.append(Conv2dSame(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, stride=2, bias=False))
            layers.append(normalization(out_dim))
            layers.append(activation())
            h, w = h // 2, w // 2

        self.outdim = out_dim * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        """
        Forward pass of the network.

        Args:
            obs (torch.Tensor): Input tensor of shape (batch, time, h, w, ch).

        Returns:
            torch.Tensor: Output tensor of shape (batch, time, -1).
        """
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1, *obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    """Convolutional decoder module that maps a feature vector to an image tensor."""

    def __init__(
        self,
        feat_size: int,
        shape: tuple[int, ...] = (3, 64, 64),
        depth: int = 32,
        activation: str | type[th.nn.Module] | None = "ELU",
        normalization: str | type[th.nn.Module] | None = "LayerNorm",
        kernel_size: int = 4,
        min_res: int = 4,
        outscale: float = 1.0,
        cnn_sigmoid: bool = False,
    ):
        """
        Initialize a ConvDecoder instance.

        Args:
            feat_size (int): The size of the input feature vector.
            shape (tuple[int, ...], optional): The shape of the output tensor. Defaults to (3, 64, 64).
            depth (int, optional): The number of channels in the first layer. Defaults to 32.
            activation (str or type[th.nn.Module] or None, optional): The activation function to use. Can be a string
                representing the name of a torch.nn module, a torch.nn module class, or None. Defaults to "ELU".
            normalization (str or type[th.nn.Module] or None, optional): The normalization layer to use. Can be a string
                representing the name of a torch.nn module, a torch.nn module class, or None. Defaults to "LayerNorm".
            kernel_size (int, optional): The size of the convolutional kernel. Defaults to 4.
            min_res (int, optional): The minimum resolution of the output tensor. Defaults to 4.
            outscale (float, optional): The scale of the output tensor. Defaults to 1.0.
            cnn_sigmoid (bool, optional): Whether to apply a sigmoid activation to the output tensor. Defaults to False.
        """
        super(ConvDecoder, self).__init__()
        if isinstance(activation, str):
            activation = cast(type[th.nn.Module], getattr(th.nn, activation))
        if isinstance(normalization, str):
            normalization = cast(type[th.nn.Module], getattr(th.nn, normalization))
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(min_res))
        self._min_res = min_res
        self._embed_size = min_res**2 * depth * 2 ** (layer_num - 1)

        self._linear_layer = nn.Linear(feat_size, self._embed_size)
        self._linear_layer.apply(tools.weight_init)
        in_dim = self._embed_size // (min_res**2)

        layers: list[th.nn.Module] = []
        h, w = min_res, min_res
        for i in range(layer_num):
            out_dim = self._embed_size // (min_res**2) // (2 ** (i + 1))
            bias = False
            initializer = tools.weight_init
            if i == layer_num - 1:
                out_dim = self._shape[0]
                activation = None
                bias = True
                normalization = None
                initializer = tools.uniform_weight_init(outscale)

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim, out_dim, kernel_size, 2, padding=(pad_h, pad_w), output_padding=(outpad_h, outpad_w), bias=bias
                )
            )
            if normalization is not None:
                layers.append(normalization(out_dim))
            if activation is not None:
                layers.append(activation())
            [m.apply(initializer) for m in layers[-3:]]
            h, w = h * 2, w * 2

        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        """
        Calculate the padding required for a convolutional layer with 'same' padding.

        Args:
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.

        Returns:
            tuple: A tuple containing the padding required on both sides of the input tensor.
        """
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features: th.Tensor) -> th.Tensor:
        """
        Forward pass of the network.

        Args:
            features (torch.Tensor): Input tensor of shape (batch_size, sequence_length, feature_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, height, width, channels).
        """
        x: th.Tensor = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape([-1, self._min_res, self._min_res, self._embed_size // self._min_res**2])
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch, time, -1) -> (batch * time, ch, h, w) necessary???
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch * time, ch, h, w) -> (batch * time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean) - 0.5
        return mean


class MLP(nn.Module):
    """A multi-layer perceptron (MLP) network."""

    mean_layer: th.nn.Module
    std_layer: th.nn.Module

    def __init__(
        self,
        inp_dim: int,
        shape: dict[str, tuple[int, ...]] | tuple[int, ...] | int | None,
        num_layers: int,
        hidden_units: int,
        activation: str | type[th.nn.Module] = "SiLU",
        normalization: str | type[th.nn.Module] = "LayerNorm",
        distribution: str = "normal",
        std: float = 1.0,
        outscale: float = 1.0,
        symlog_inputs: bool = False,
    ):
        """
        Initialize a multi-layer perceptron (MLP) network.

        Args:
            inp_dim (int): The input dimension of the network.
            shape (dict[str, tuple[int, ...]] | tuple[int, ...] | int | None):
                The output shape of the network. If a dictionary is provided, it should
                map output names to output shapes. If a tuple is provided, it should
                specify a single output shape. If an integer is provided, it should
                specify the number of output units (i.e., a single output shape of that
                size).
            num_layers (int): The number of hidden layers in the network.
            hidden_units (int): The number of units in each hidden layer.
            activation (str | type[th.nn.Module], optional): The activation function
                to use in the hidden layers. Can be a string (e.g., "ReLU", "Tanh",
                "SiLU", etc.) or a PyTorch module. Defaults to "SiLU".
            normalization (str | type[th.nn.Module], optional): The normalization
                function to use in the hidden layers. Can be a string (e.g., "LayerNorm",
                "BatchNorm1d", etc.) or a PyTorch module. Defaults to "LayerNorm".
            distribution (str, optional): The distribution to use for weight initialization.
                Can be "normal" or "uniform". Defaults to "normal".
            std (float, optional): The standard deviation of the weight initialization.
                If "learned", the standard deviation will be learned during training.
                Defaults to 1.0.
            outscale (float, optional): The scaling factor for the output weights.
                Defaults to 1.0.
            symlog_inputs (bool, optional): Whether to use a symmetrical logarithmic
                transformation for the input. Defaults to False.
            device (th.device | str, optional): The device to use for the network.
                Defaults to "cuda".
        """
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        self._layers = num_layers
        if isinstance(activation, str):
            activation = cast(type[th.nn.Module], getattr(th.nn, activation))
        if isinstance(normalization, str):
            normalization = cast(type[th.nn.Module], getattr(th.nn, normalization))
        self._dist = distribution
        self._std = std
        self._symlog_inputs = symlog_inputs

        layers: list[th.nn.Module] = []
        for index in range(self._layers):
            layers.append(nn.Linear(inp_dim, hidden_units, bias=False))
            layers.append(normalization(hidden_units, eps=1e-03))
            layers.append(activation())
            if index == 0:
                inp_dim = hidden_units
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, sub_shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(sub_shape).item())
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.ModuleDict()
                for name, sub_shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(sub_shape).item())
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape).item())
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.Linear(hidden_units, np.prod(self._shape).item())
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features: th.Tensor) -> dict[str, torchd.Distribution] | torchd.Distribution:
        """
        Compute the forward pass of the network, given a batch of input features.

        Args:
            features (torch.Tensor): A tensor of shape (batch_size, input_size) containing the input features.

        Returns:
            torch.distributions.Distribution: A distribution object representing the output of the network.
        """
        x: th.Tensor = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            assert isinstance(self.mean_layer, nn.ModuleDict)

            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    assert isinstance(self.std_layer, nn.ModuleDict)
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist: str, mean: th.Tensor, std: float, shape: tuple[int, ...]) -> th.distributions.Distribution:
        """
        Return a PyTorch distribution object based on the specified distribution type.

        Args:
            dist (str): The type of distribution to use. Can be one of "normal", "huber", "binary", "symlog_disc", or "symlog_mse".
            mean (th.Tensor): The mean of the distribution.
            std (th.Tensor): The standard deviation of the distribution.
            shape (tuple[int, ...]): The shape of the distribution.

        Returns:
            th.distributions.Distribution: A PyTorch distribution object based on the specified distribution type.
        """
        if dist == "normal":
            return distributions.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), len(shape)))
        if dist == "huber":
            return distributions.ContDist(
                torchd.independent.Independent(distributions.UnnormalizedHuber(mean, std, 1.0), len(shape))
            )
        if dist == "binary":
            return distributions.Bernoulli(torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=mean), len(shape)))
        if dist == "symlog_disc":
            return distributions.DiscDist(logits=mean)
        if dist == "symlog_mse":
            return distributions.SymlogDist(mean)
        raise NotImplementedError(dist)


class ActionHead(nn.Module):
    """A PyTorch module that represents the action head of the Dreamer neural network."""

    def __init__(
        self,
        input_dim: int,
        size: int,
        num_layers: int,
        hidden_units: int,
        activation: str | type[th.nn.Module] = "ELU",
        normalization: str | type[th.nn.Module] = "LayerNorm",
        distribution: str = "trunc_normal",
        init_std: float = 0.0,
        min_std: float = 0.1,
        max_std: float = 1.0,
        temp: float = 0.1,
        outscale: float = 1.0,
        unimix_ratio: float = 0.01,
    ):
        """
        Initialize an instance of the ActionHead class.

        Args:
            input_dim (int): The dimensionality of the input tensor.
            size (int): The size of the output tensor.
            num_layers (int): The number of layers in the network.
            hidden_units (int): The number of hidden units in each layer.
            activation (str or type[th.nn.Module], optional): The activation function to use. Defaults to "ELU".
            normalization (str or type[th.nn.Module], optional): The normalization layer to use. Defaults to "LayerNorm".
            distribution (str, optional): The distribution to use for the output tensor. Defaults to "trunc_normal".
            init_std (float, optional): The standard deviation of the initialization distribution. Defaults to 0.0.
            min_std (float, optional): The minimum standard deviation for the output tensor. Defaults to 0.1.
            max_std (float, optional): The maximum standard deviation for the output tensor. Defaults to 1.0.
            temp (float, optional): The temperature parameter for the Gumbel-Softmax distribution. Defaults to 0.1.
            outscale (float, optional): The scaling factor for the output tensor. Defaults to 1.0.
            unimix_ratio (float, optional): The ratio of the uniform mixture component for the output tensor. Defaults to 0.01.
        """
        super(ActionHead, self).__init__()
        self._size = size
        self._layers = num_layers
        self._units = hidden_units
        self._dist = distribution
        if isinstance(activation, str):
            activation = cast(type[th.nn.Module], getattr(th.nn, activation))
        if isinstance(normalization, str):
            normalization = cast(type[th.nn.Module], getattr(th.nn, normalization))
        self._min_std = min_std
        self._max_std = max_std
        self._init_std = init_std
        self._unimix_ratio = unimix_ratio
        self._temp = temp() if callable(temp) else temp

        pre_layers: list[th.nn.Module] = []
        for index in range(self._layers):
            pre_layers.append(nn.Linear(input_dim, self._units, bias=False))
            pre_layers.append(normalization(self._units, eps=1e-03))
            pre_layers.append(activation())
            if index == 0:
                input_dim = self._units
        self._pre_layers = nn.Sequential(*pre_layers)
        self._pre_layers.apply(tools.weight_init)

        if self._dist in ["tanh_normal", "tanh_normal_5", "normal", "trunc_normal"]:
            self._dist_layer = nn.Linear(self._units, 2 * self._size)
            self._dist_layer.apply(tools.uniform_weight_init(outscale))

        elif self._dist in ["normal_1", "onehot", "onehot_gumbel"]:
            self._dist_layer = nn.Linear(self._units, self._size)
            self._dist_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features: th.Tensor) -> torchd.Distribution:
        """
        Computes the forward pass of the network, given a batch of input features.

        Args:
            features (torch.Tensor): A tensor of shape (batch_size, input_size) containing the input features.

        Returns:
            torch.distributions.Distribution: A distribution object representing the output distribution of the network.
        """
        dist: torchd.Distribution
        x = features
        x = self._pre_layers(x)
        if self._dist == "tanh_normal":
            x = self._dist_layer(x)
            mean, std = th.split(x, 2, -1)
            mean = th.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(dist, distributions.TanhBijector())
            dist = torchd.independent.Independent(dist, 1)
            dist = distributions.SampleDist(dist)
        elif self._dist == "tanh_normal_5":
            x = self._dist_layer(x)
            mean, std = th.split(x, 2, -1)
            mean = 5 * th.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(dist, distributions.TanhBijector())
            dist = torchd.independent.Independent(dist, 1)
            dist = distributions.SampleDist(dist)
        elif self._dist == "normal":
            x = self._dist_layer(x)
            mean, std = th.split(x, [self._size] * 2, -1)
            std = (self._max_std - self._min_std) * th.sigmoid(std + 2.0) + self._min_std
            dist = torchd.normal.Normal(th.tanh(mean), std)
            dist = distributions.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "normal_1":
            mean = self._dist_layer(x)
            dist = torchd.normal.Normal(mean, 1)
            dist = distributions.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "trunc_normal":
            x = self._dist_layer(x)
            mean, std = th.split(x, [self._size] * 2, -1)
            mean = th.tanh(mean)
            std = 2 * th.sigmoid(std / 2) + self._min_std
            dist = distributions.SafeTruncatedNormal(mean, std, -1, 1)
            dist = distributions.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "onehot":
            x = self._dist_layer(x)
            dist = distributions.OneHotDist(x, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            x = self._dist_layer(x)
            temp = self._temp
            dist = distributions.ContDist(torchd.gumbel.Gumbel(x, 1 / temp))
        else:
            raise NotImplementedError(self._dist)
        return dist


class GRUCell(nn.Module):
    """Gated Recurrent Unit (GRU) cell implementation."""

    def __init__(self, inp_size: int, size: int, norm: bool = False, activation=th.tanh, update_bias=-1):
        """
        Initializes a GRUCell.

        Args:
            inp_size (int): The size of the input tensor.
            size (int): The size of the hidden state tensor.
            norm (bool, optional): Whether to apply layer normalization. Defaults to False.
            activation (function, optional): The activation function to use. Defaults to th.tanh.
            update_bias (int, optional): The bias to use for the update gate. Defaults to -1.
        """
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = activation
        self._update_bias = update_bias
        self._layer = nn.Linear(inp_size + size, 3 * size, bias=False)
        self._norm = nn.LayerNorm(3 * size, eps=1e-03) if norm else None

    @property
    def state_size(self):
        """
        Returns the size of the state space of the network.

        Returns:
            int: The size of the state space.
        """
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(th.cat([inputs, state], -1))
        if self._norm is not None:
            parts = self._norm(parts)
        reset, cand, update = th.split(parts, [self._size] * 3, -1)
        reset = th.sigmoid(reset)
        cand = self._act(reset * cand)
        update = th.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSame(th.nn.Conv2d):
    """
    2D convolutional layer with 'SAME' padding, which pads the input tensor
    with zeros so that the output tensor has the same spatial dimensions as
    the input tensor. This implementation extends the PyTorch's Conv2d layer
    and overrides its forward method to perform the padding.
    """

    def calc_same_pad(self, i, k, s, d):
        """
        Calculate the amount of padding needed to maintain the same spatial dimensions
        when using a convolutional layer with the given kernel size, stride, and dilation.

        Args:
            i (int): Input size.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.

        Returns:
            int: The amount of padding needed to maintain the same spatial dimensions.
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: th.Tensor):
        """
        Apply a 2D convolution over an input signal composed of several input planes.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        ret = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return ret
