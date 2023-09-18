import torch as th
from dataclasses import dataclass, field


@dataclass
class SequenceModelConfig:
    hidden_size: int = 200
    activation: type[th.nn.Module] = th.nn.ELU


@dataclass
class TransitionModelConfig:
    layers: tuple[int, ...] = (200,)
    activation: type[th.nn.Module] = th.nn.ELU
    min_std: float = 0.1


@dataclass
class RepresentationModelConfig:
    layers: tuple[int, ...] = (200,)
    activation: type[th.nn.Module] = th.nn.ELU
    min_std: float = 0.1


@dataclass
class RecurrentStateSpaceModelConfig:
    transition_model: TransitionModelConfig = field(default_factory=TransitionModelConfig)
    representation_model: RepresentationModelConfig = field(default_factory=RepresentationModelConfig)
    sequence_model: SequenceModelConfig = field(default_factory=SequenceModelConfig)


@dataclass
class ContinueModelConfig:
    layers: tuple[int, ...] = (200, 200,)
    activation: type[th.nn.Module] = th.nn.ELU


@dataclass
class RewardModelConfig:
    layers: tuple[int, ...] = (200, 200)
    activation: type[th.nn.Module] = th.nn.ELU


@dataclass
class CnnConfig:
    activation: type[th.nn.Module] = th.nn.ReLU
    depth: int = 32
    kernel_size: int = 4
    stride: int = 2
    extracted_size: int = 256

@dataclass
class MlpConfig:
    activation: type[th.nn.Module] = th.nn.ReLU
    layers: tuple[int, ...] = (200, )

@dataclass
class EncoderConfig:
    mlp_filter: str = "^(observation|goal)$"
    cnn_filter: str = "^image$"
    cnn_config: CnnConfig = field(default_factory=CnnConfig)
    mlp_config: MlpConfig = field(default_factory=MlpConfig)


@dataclass
class DecoderConfig:
    cnn_config: CnnConfig = field(default_factory=lambda: CnnConfig(kernel_size=5))
    mlp_config: MlpConfig = field(default_factory=lambda: MlpConfig())
    mlp_filter: str = "^(observation|goal)$"
    cnn_filter: str = "^image$"


@dataclass
class DreamerWorldModelTrainingConfig:
    use_continue_flag: bool = False
    kl_divergence_scale: float = 1.0
    free_nats: int = 3
    lr: float = 0.0006
    optimizer_class: type[th.optim.Optimizer] = th.optim.AdamW
    clip_grad: float = 100
    grad_norm_type: int = 2


@dataclass
class DreamerWorldModelConfig:
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    decoder_config: DecoderConfig = field(default_factory=DecoderConfig)
    continue_config: ContinueModelConfig = field(default_factory=ContinueModelConfig)
    reward_config: RewardModelConfig = field(default_factory=RewardModelConfig)
    recurrent_state_space_model_config: RecurrentStateSpaceModelConfig = field(default_factory=RecurrentStateSpaceModelConfig)
    world_model_training_config: DreamerWorldModelTrainingConfig = field(default_factory=DreamerWorldModelTrainingConfig)


@dataclass
class ActorCriticTrainingConfig:
    optimizer_class: type[th.optim.Optimizer] = th.optim.AdamW
    discount: float = 0.99
    lambda_: float = 0.95
    critic_optimizer_lr: float = 0.00008
    actor_optimizer_lr: float = 0.00008
    clip_grad: float = 100
    horizon_length: int = 15
    grad_norm_type: int = 2
    use_continue_flag: bool = False


@dataclass
class ActorConfig:
    layers: tuple[int, ...] = (200, 200)
    activation: type[th.nn.Module] = th.nn.ELU
    mean_scale: float = 5
    init_std: float = 5.0
    min_std: float = 0.0001


@dataclass
class CriticConfig:
    layers: tuple[int, ...] = (200, 200, 200)
    activation: type[th.nn.Module] = th.nn.ELU


@dataclass
class ActorCriticConfig:
    actor_config: ActorConfig = field(default_factory=ActorConfig)
    critic_config: CriticConfig = field(default_factory=CriticConfig)
    actor_critic_training_config: ActorCriticTrainingConfig = field(default_factory=ActorCriticTrainingConfig)


@dataclass
class DreamerConfig:
    stochastic_size: int = 30
    deterministic_size: int = 200
    embedded_size: int = 1024
    task_size: int = 0
    batch_length = 50

    actor_critic_config: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    world_model_config: DreamerWorldModelConfig = field(default_factory=DreamerWorldModelConfig)
