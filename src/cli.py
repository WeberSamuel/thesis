from dataclasses import dataclass
from typing import Optional

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.policies import BasePolicy

from src.callbacks import EvalCallback, ExplorationCallback, Plan2ExploreEvalCallback, SaveHeatmapCallback


class DummyPolicy(BasePolicy):
    def __init__(self, *args, **kwargs) -> None:
        del kwargs["use_sde"]
        super().__init__(*args, **kwargs)

    def _predict(self, *args, **kwargs):
        raise NotImplementedError()


class DummyReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        pass


@dataclass
class Callbacks:
    callbacks: Optional[CallbackList] = None
    eval_callback: Optional[EvalCallback] = None
    save_heatmap_callback: Optional[SaveHeatmapCallback] = None
    eval_exploration_callback: Optional[Plan2ExploreEvalCallback] = None
    exploration_callback: Optional[ExplorationCallback] = None
