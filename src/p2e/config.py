from jsonargparse import lazy_instance
from dataclasses import dataclass

@dataclass
class OneStepModelConfig:
    ensemble_size: int = 5
    layers: int = 2
    complexity: float = 20.0

@dataclass
class P2EConfig:
    imagination_horizon: int = 10
    use_world_model_as_ensemble: bool = False
    use_ground_truth_as_one_step_target: bool = False
    one_step_model: OneStepModelConfig = lazy_instance(OneStepModelConfig)
    use_online_disagreement: bool = False