from core.state_aware_algorithm import StateAwareOffPolicyAlgorithm
from src.meta_dreamer.policy import MetaDreamerPolicy

class MetaDreamer(StateAwareOffPolicyAlgorithm):
    policy: MetaDreamerPolicy
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        self.policy