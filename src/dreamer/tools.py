from typing import Any, Callable

import numpy as np
import torch as th
from torch import nn

to_np = lambda x: x.detach().cpu().numpy()


def symlog(x: th.Tensor):
    """
    Applies the symmetric logarithmic transformation to the input tensor x.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The transformed tensor.
    """
    return th.sign(x) * th.log(th.abs(x) + 1.0)


def symexp(x: th.Tensor):
    """
    Applies a symmetric exponential function to the input tensor.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The output tensor after applying the symmetric exponential function.
    """
    return th.sign(x) * (th.exp(th.abs(x)) - 1.0)


class RequiresGrad:
    """Context Manager that sets the `requires_grad` attribute to `True` within the context and `False` outside of it."""

    def __init__(self, model: th.nn.Module):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)


from typing import Callable, Tuple
import torch as th

def static_scan_for_lambda_return(fn: Callable, inputs: Tuple[th.Tensor, ...], start: th.Tensor) -> th.Tensor:
    """
    Applies a function `fn` to a sequence of inputs `inputs` in reverse order, starting from `start`.
    The function `fn` should take two arguments: the previous output and the current input.
    The function returns a tensor of outputs, where each output is the result of applying `fn` to the corresponding input.

    Args:
        fn (Callable): A function that takes two arguments: the previous output and the current input.
        inputs (Tuple[th.Tensor, ...]): A tuple of tensors representing the inputs to the function.
        start (th.Tensor): A tensor representing the starting point for the scan.

    Returns:
        th.Tensor: A tensor of outputs, where each output is the result of applying `fn` to the corresponding input.
    """
    last = start
    indices = reversed(range(inputs[0].shape[0]))
    outputs_list = []
    inp = lambda x: (_input[x] for _input in inputs)
    for index in indices:
        last = fn(last, *inp(index))
        outputs_list.append(last)
    outputs = th.cat(outputs_list[::-1], dim=-1)
    outputs = th.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    return outputs


def lambda_return(
    reward: th.Tensor, value: th.Tensor, pcont: int | float | th.Tensor, bootstrap: th.Tensor | None, lambda_: float, axis: int
):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * th.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = th.zeros_like(value[-1])
    next_values = th.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    # returns = static_scan(
    #    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    #    (inputs, pcont), bootstrap, reverse=True)
    # reimplement to optimize performance
    returns = static_scan_for_lambda_return(lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap)
    if axis != 0:
        returns = returns.permute(dims)
    return returns


class Optimizer(th.nn.Module):
    def __init__(
        self,
        name: str,
        parameters: Callable,
        lr: float,
        eps=1e-4,
        grad_clip: float | None = None,
        weight_decay: float | None = None,
        weight_decay_pattern:str=r".*",
        optimizer:str="adam",
        use_amp:bool=False,
    ):
        super().__init__()
        
        assert weight_decay is None or 0 <= weight_decay < 1
        assert not grad_clip or 1 <= grad_clip
        self._name = name
        self._grad_clip = grad_clip
        self._weight_decay = weight_decay
        self._weight_decay_pattern = weight_decay_pattern
        self._opt = {
            "adam": lambda: th.optim.Adam(parameters(), lr=lr, eps=eps),
            "adamax": lambda: th.optim.Adamax(parameters(), lr=lr, eps=eps),
            "sgd": lambda: th.optim.SGD(parameters(), lr=lr),
            "momentum": lambda: th.optim.SGD(parameters(), lr=lr, momentum=0.9),
        }[optimizer]()
        self._scaler = th.cuda.amp.GradScaler(enabled=use_amp)  # type: ignore

    def __call__(self, loss: th.Tensor, params, retain_graph=False):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
        self._scaler.scale(loss).backward()  # type: ignore
        self._scaler.unscale_(self._opt)
        # loss.backward(retain_graph=retain_graph)
        if self._grad_clip:
            norm = th.nn.utils.clip_grad_norm_(params, self._grad_clip)  # type: ignore
            metrics[f"{self._name}_grad_norm"] = norm.item()
        if self._weight_decay:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
        self._opt.zero_grad()
        return metrics

    def _apply_weight_decay(self, variabs):
        assert self._weight_decay is not None
        nontrivial = self._weight_decay_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in variabs:
            var.data = (1 - self._weight_decay) * var.data


def static_scan(fn:Callable, inputs, start):
    inp = lambda x: (_input[x] for _input in inputs)
    
    last = fn(start, *inp(0))
    indices = range(inputs[0].shape[0])
    
    index = indices[0]
    if isinstance(last, dict):
        outputs = {
            key: value.clone().unsqueeze(0) for key, value in last.items()
        }
    else:
        outputs = []
        for _last in last:
            if isinstance(_last, dict):
                outputs.append(
                    {
                        key: value.clone().unsqueeze(0)
                        for key, value in _last.items()
                    }
                )
            else:
                outputs.append(_last.clone().unsqueeze(0))

    for index in indices[1:]:
        last = fn(last, *inp(index))
        if isinstance(last, dict):
            for key in last.keys():
                outputs[key] = th.cat(
                    [outputs[key], last[key].unsqueeze(0)], dim=0
                )
        else:
            for j in range(len(outputs)):
                if type(last[j]) == type({}):
                    for key in last[j].keys():
                        outputs[j][key] = th.cat(
                            [outputs[j][key], last[j][key].unsqueeze(0)], dim=0
                        )
                else:
                    outputs[j] = th.cat(
                        [outputs[j], last[j].unsqueeze(0)], dim=0
                    )
    if isinstance(last, dict):
        outputs = [outputs]
    return outputs

def weight_init(m: th.nn.Module):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale: float):
    def f(m: th.nn.Module):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    return f


def tensorstats(tensor:th.Tensor, prefix:str|None=None):
    """
    Compute various statistics (mean, std, min, max) for a PyTorch tensor.

    Args:
        tensor (th.Tensor): The input tensor.
        prefix (str, optional): A prefix to add to the metric names. Defaults to None.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    metrics = {
        "mean": to_np(th.mean(tensor)),
        "std": to_np(th.std(tensor)),
        "min": to_np(th.min(tensor)),
        "max": to_np(th.max(tensor)),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics
