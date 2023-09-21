from typing import NamedTuple

import torch as th

class EncoderInput(NamedTuple):
    obs: th.Tensor
    action: th.Tensor
    next_obs: th.Tensor
    reward: th.Tensor
