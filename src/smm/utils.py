import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def hard_update_params(net, target_net):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(param.data)


def to_torch(xs, device):
    if isinstance(xs, dict):
        return {k: to_torch(x, device=device) for k, x in xs.items()}
    elif isinstance(xs, list):
        return [to_torch(x, device=device) for x in xs]
    elif isinstance(xs, tuple):
        return tuple(to_torch(x, device=device) for x in xs)
    else:
        return torch.as_tensor(xs, device=device)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)  # type: ignore
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)  # type: ignore


def grad_norm(params, norm_type=2.0):
    params = [p for p in params if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]), norm_type)
    return total_norm.item()


def param_norm(params, norm_type=2.0):
    total_norm = torch.norm(torch.stack([torch.norm(p.detach(), norm_type) for p in params]), norm_type)
    return total_norm.item()


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl: str|float, step: int):
    """
    The `schedule` function is a utility function that is used to implement learning rate schedules in PyTorch.

    Args:
        schdl (str): A string that specifies the schedule.
        step (int): An integer that represents the current step in the training process.

    Returns:
        float: The learning rate for the current step.

    The function first tries to convert `schdl` to a float. If this succeeds, it simply returns the float value. This allows the user to specify a constant learning rate by passing a float value as `schdl`.

    If `schdl` is not a float, the function tries to match it against two regular expressions. The first regular expression matches strings of the form `linear(init, final, duration)`. This specifies a linear schedule that starts at `init`, ends at `final`, and lasts for `duration` steps. The function calculates the current mix value as `step / duration`, clips it to the range [0, 1], and returns the linearly interpolated value between `init` and `final`.

    The second regular expression matches strings of the form `step_linear(init, final1, duration1, final2, duration2)`. This specifies a schedule that starts at `init`, transitions to `final1` after `duration1` steps, and then transitions to `final2` after `duration2` additional steps. The function checks whether `step` is less than or equal to `duration1`. If it is, it calculates the current mix value as `step / duration1`, clips it to the range [0, 1], and returns the linearly interpolated value between `init` and `final1`. If `step` is greater than `duration1`, it calculates the current mix value as `(step - duration1) / duration2`, clips it to the range [0, 1], and returns the linearly interpolated value between `final1` and `final2`.
    """
    if isinstance(schdl, float):
        return schdl
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [float(g) for g in match.groups()]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
