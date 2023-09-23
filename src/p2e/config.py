from dataclasses import dataclass

@dataclass
class P2EConfig:
    imagination_horizon: int = 10
    ensemble_size: int = 5
    complexity: float = 20.0
    layers: int = 2
    use_world_model_as_ensemble: bool = False
    use_ground_truth_as_one_step_target: bool = False
