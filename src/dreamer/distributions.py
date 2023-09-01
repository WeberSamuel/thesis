import math
from typing import Callable
import torch as th
from torch import distributions as torchd
from torch.nn import functional as F

from src.dreamer.tools import symexp, symlog


class SampleDist(torchd.Distribution):
    def __init__(self, dist: torchd.Distribution, samples: int = 100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    @property
    def mean(self):
        samples = self._dist.sample(th.Size((self._samples,)))
        return th.mean(samples, 0)

    @property
    def mode(self):
        sample = self._dist.sample(th.Size((self._samples,)))
        logprob = self._dist.log_prob(sample)
        return sample[th.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(th.Size((self._samples,)))
        logprob = self.log_prob(sample)
        return -th.mean(logprob, 0)

    def log_prob(self, value: th.Tensor) -> th.Tensor:
        return self._dist.log_prob(value)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits: th.Tensor | None = None, probs=None, unimix_ratio: float = 0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = th.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    @property
    def mode(self) -> th.Tensor:
        logits: th.Tensor = super().logits  # type: ignore
        _mode = F.one_hot(th.argmax(logits, dim=-1), logits.shape[-1])
        return _mode.detach() + logits - logits.detach()

    def sample(self, sample_shape=(), seed=None) -> th.Tensor:
        if seed is not None:
            raise NotImplementedError("seed is not implemented")
        sample: th.Tensor = super().sample(sample_shape)
        probs: th.Tensor = super().probs  # type: ignore
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class DiscDist(th.distributions.Distribution):
    def __init__(
        self,
        logits: th.Tensor,
        low: float = -20.0,
        high: float = 20.0,
        transfwd: Callable[[th.Tensor], th.Tensor] = symlog,
        transbwd: Callable[[th.Tensor], th.Tensor] = symexp,
    ):
        super().__init__(validate_args=False)
        self.logits = logits
        self.probs = th.softmax(logits, -1)
        self.buckets = th.linspace(low, high, steps=255).to(logits.device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    @property
    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(th.sum(_mean, dim=-1, keepdim=True))

    @property
    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(th.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x: th.Tensor):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = th.sum((self.buckets <= x[..., None]).to(th.int32), dim=-1) - 1
        above = len(self.buckets) - th.sum((self.buckets > x[..., None]).to(th.int32), dim=-1)
        below = th.clip(below, 0, len(self.buckets) - 1)
        above = th.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = th.where(equal, 1, th.abs(self.buckets[below] - x))
        dist_to_above = th.where(equal, 1, th.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - th.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target: th.Tensor):
        log_pred = self.logits - th.logsumexp(self.logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)


class MSEDist(torchd.Distribution):
    def __init__(self, mode: th.Tensor, agg="sum"):
        self._mode = mode
        self._agg = agg

    @property
    def mode(self):
        return self._mode

    @property
    def mean(self):
        return self._mode

    def log_prob(self, value: th.Tensor):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist(th.distributions.Distribution):
    def __init__(self, mode: th.Tensor, dist: str = "mse", agg: str = "sum", tol: float = 1e-8):
        super().__init__(validate_args=False)
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    @property
    def mode(self):
        return symexp(self._mode)

    @property
    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value:th.Tensor):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = th.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = th.abs(self._mode - symlog(value))
            distance = th.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class ContDist(th.distributions.Distribution):
    def __init__(self, dist: th.distributions.Distribution):
        super().__init__(validate_args=False)
        self._dist = dist

    @property
    def mean(self) -> th.Tensor:
        return self._dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    @property
    def mode(self):
        return self._dist.mean

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli(th.distributions.Distribution):
    def __init__(self, dist: th.distributions.Independent):
        super().__init__(validate_args=False)
        self._dist = dist

    @property
    def mean(self) -> th.Tensor:
        return self._dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    @property
    def mode(self):
        _mode = th.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return log_probs0 * (1 - x) + log_probs1 * x


class UnnormalizedHuber(torchd.normal.Normal):
    def __init__(self, loc, scale, threshold: float = 1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(th.sqrt((event - self.mean) ** 2 + self._threshold**2) - self._threshold)

    @property
    def mode(self):
        return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = th.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(torchd.Transform):
    def __init__(self, validate_args=False, name="tanh"):
        super().__init__()

    def _forward(self, x):
        return th.tanh(x)

    def _inverse(self, y):
        y = th.where((th.abs(y) <= 1.0), th.clamp(y, -0.99999997, 0.99999997), y)
        y = th.atanh(y)
        return y

    def _forward_log_det_jacobian(self, x: th.Tensor) -> th.Tensor:
        log2 = math.log(2.0)
        return 2.0 * (log2 - x - th.nn.functional.softplus(-2.0 * x))