from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DdpgAgentConfig:
    reward_free: bool = True
    obs_type: Literal["symbolic", "pixels"] = "symbolic"
    lr: float = 1e-4
    feature_dim: int = 50
    hidden_dim: int = 1024
    critic_target_tau = 0.01
    stddev_schedule: float = 0.2
    stddev_clip: float = 0.3
    init_critic: bool = True
    use_tb: bool = False
    use_wandb: bool = False


@dataclass
class SmmAgentConfig(DdpgAgentConfig):
    sp_lr: float = 1e-3
    vae_lr: float = 1e-2
    vae_beta: float = 0.5
    state_ent_coef: float = 1.0
    latent_ent_coef: float = 1.0
    latent_cond_ent_coef: float = 1.0
    update_encoder: bool = True
    lr: float = 1e-4
    critic_target_tau: float = 0.01
    gamma = 0.99


@dataclass
class SmmConfig:
    meta_size: int = 4
    ensemble_size: int = 1
    smm_agent_config: SmmAgentConfig = field(default_factory=SmmAgentConfig)
