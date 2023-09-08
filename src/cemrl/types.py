from typing import TypedDict
import torch as th
import numpy as np


class CEMRLObsDict(TypedDict):
    observation: np.ndarray
    reward: np.ndarray
    action: np.ndarray


class CEMRLObsTensorDict(TypedDict):
    observation: th.Tensor
    reward: th.Tensor
    action: th.Tensor


class CEMRLPolicyInput(TypedDict):
    observation: th.Tensor
    task_indicator: th.Tensor
