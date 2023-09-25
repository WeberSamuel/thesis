from typing import Literal
import torch as th
from dataclasses import field, dataclass
from jsonargparse import lazy_instance


@dataclass
class EncoderConfig:
    num_classes: int = 1
    latent_dim: int = 2
    complexity: float = 40.0
    preprocessed_state_size: int = 0
    use_simplified_state_preprocessor: bool = False


@dataclass
class DecoderConfig:
    complexity: float = 40.0
    use_state_decoder: bool = True
    use_next_state_for_reward: bool = True
    num_layers: int = 2
    activation: type[th.nn.Module] = th.nn.ReLU
    ensemble_size: int = 1


class TaskInferenceTrainingConfig:
    use_state_diff: bool = False
    reconstruct_all_steps: bool = True
    prior_mode: Literal["fixedOnY", "network"] = "fixedOnY"
    alpha_kl_z_query: float | None = None
    beta_kl_y_query: float | None = None
    alpha_kl_z: float = 1e-3
    beta_kl_y: float = 1e-3
    optimizer_class: type[th.optim.Optimizer] = th.optim.AdamW
    prior_sigma: float = 0.5
    lr: float = 1e-3


@dataclass
class TaskInferenceConfig:
    encoder: EncoderConfig = lazy_instance(EncoderConfig)
    decoder: DecoderConfig = lazy_instance(DecoderConfig)
    training: TaskInferenceTrainingConfig = lazy_instance(TaskInferenceTrainingConfig)


@dataclass
class CemrlTrainingConfig:
    task_inference_gradient_steps: int = 50
    policy_gradient_steps: int = 40
    num_decoder_targets: int = 400
    encoder_context_length: int = 30
    validation_frequency: int = 10
    loss_weight_adjustment_frequency: int = 10
    imagination_horizon: int = 0
    early_stopping_threshold: int = 20



@dataclass
class CemrlConfig:
    task_inference: TaskInferenceConfig = lazy_instance(TaskInferenceConfig)
    training: CemrlTrainingConfig = lazy_instance(CemrlTrainingConfig)
