from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from typing import Any

class ExplorationAlgorithmMixin:
    policy_kwargs: dict[str, Any]
    def _setup_model(self, *args, parent_algorithm: OffPolicyAlgorithm, **kwargs):
        algorithm:OffPolicyAlgorithm = self # type: ignore
        self.parent_algorithm = parent_algorithm
        self.policy_kwargs.setdefault("main_policy", parent_algorithm.policy)
        super()._setup_model(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:	
        return super()._excluded_save_params() + ["parent_algorithm"]