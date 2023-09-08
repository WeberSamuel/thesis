from dataclasses import dataclass

import torch as th


@dataclass
class EncoderConfig:
    mlp_keys: str = "observation|reward$"
    cnn_keys: str = "image"
    act: str = "SiLU"
    norm: str = "LayerNorm"
    cnn_depth: int = 32
    kernel_size: int = 4
    minres: int = 4
    mlp_layers: int = 2
    mlp_units: int = 512
    symlog_inputs: bool = True


@dataclass
class DecoderConfig:
    mlp_keys: str = "observation|reward$"
    cnn_keys: str = "image"
    act: str = "SiLU"
    norm: str = "LayerNorm"
    cnn_depth: int = 32
    kernel_size: int = 4
    minres: int = 4
    mlp_layers: int = 2
    mlp_units: int = 512
    cnn_sigmoid: bool = False
    image_dist: str = "mse"
    vector_dist: str = "symlog_mse"


@dataclass
class DreamerConfig:
    device: str | th.device = "cuda:0"
    offline_traindir: str = ""
    offline_evaldir: str = ""
    seed: int = 0
    deterministic_run: bool = False
    steps: int = int(1e6)
    parallel: bool = False
    eval_every: int = int(1e4)
    eval_episode_num: int = 10
    log_every: int = int(1e4)
    reset_every: int = 0
    compile: bool = True
    precision: int = 16
    debug: bool = False
    expl_gifs: bool = False
    video_pred_log: bool = True

    # Environment
    task: str = "dmc_walker_walk"
    envs: int = 1
    action_repeat: int = 2
    time_limit: int = 1000
    grayscale: bool = False
    prefill: int = 2500
    eval_noise: float = 0.0
    reward_EMA: bool = True

    # Model
    dyn_cell: str = "gru_layer_norm"
    dyn_hidden: int = 512
    dyn_deter: int = 512
    dyn_stoch: int = 32
    dyn_discrete: int = 32
    dyn_input_layers: int = 1
    dyn_output_layers: int = 1
    dyn_rec_depth: int = 1
    dyn_shared: bool = False
    dyn_mean_act: str = "none"
    dyn_std_act: str = "sigmoid2"
    dyn_min_std: float = 0.1
    dyn_temp_post: bool = True
    units: int = 512
    reward_layers: int = 2
    cont_layers: int = 2
    value_layers: int = 2
    actor_layers: int = 2
    act: str = "SiLU"
    norm: str = "LayerNorm"
    encoder = EncoderConfig()
    decoder = DecoderConfig()
    value_head: str = "symlog_disc"
    reward_head: str = "symlog_disc"
    dyn_scale: str = "0.5"
    rep_scale: str = "0.1"
    kl_free: str = "1.0"
    cont_scale: float = 1.0
    reward_scale: float = 1.0
    weight_decay: float = 0.0
    unimix_ratio: float = 0.01
    action_unimix_ratio: float = 0.01
    initial: str = "learned"

    # Training
    batch_size: int = 16
    batch_length: int = 64
    train_ratio: int = 512
    pretrain: int = 100
    model_lr: float = 1e-3
    opt_eps: float = 1e-8
    grad_clip: int = 1000
    value_lr: float = 3e-4
    actor_lr: float = 3e-4
    ac_opt_eps: float = 1e-5
    value_grad_clip: int = 100
    actor_grad_clip: int = 100
    dataset_size: int = 1000000
    slow_value_target: bool = True
    slow_target_update: int = 1
    slow_target_fraction: float = 0.02
    opt: str = "adam"

    # Behavior.
    discount: float = 0.997
    discount_lambda: float = 0.95
    imag_horizon: int = 15
    imag_gradient: str = "dynamics"
    imag_gradient_mix: str = "0.0"
    imag_sample: bool = True
    actor_dist: str = "normal"
    actor_entropy: str = "3e-4"
    actor_state_entropy: float = 0.0
    actor_init_std: float = 1.0
    actor_min_std: float = 0.1
    actor_max_std: float = 1.0
    actor_temp: float = 0.1
    expl_amount: float = 0.0
    eval_state_mean: bool = False
    collect_dyn_sample: bool = True
    behavior_stop_grad: bool = True
    value_decay: float = 0.0
    future_entropy: bool = False

    # Exploration
    expl_behavior: str = "greedy"
    expl_until: int = 0
    expl_extr_scale: float = 0.0
    expl_intr_scale: float = 1.0
    disag_target: str = "embed"
    disag_log: bool = True
    disag_models: int = 5
    disag_offset: int = 1
    disag_layers: int = 2
    disag_units: int = 400
    disag_action_cond: bool = True

    # Encoder
    encoder_mlp_keys: str = "observation|reward$"
    encoder_cnn_keys: str = "image"
    encoder_act: str = "SiLU"
    encoder_norm: str = "LayerNorm"
    encoder_cnn_depth: int = 32
    encoder_kernel_size: int = 4
    encoder_minres: int = 4
    encoder_mlp_layers: int = 2
    encoder_mlp_units: int = 512
    encoder_symlog_inputs: bool = True

    # Decoder
    decoder_mlp_keys: str = "observation|reward$"
    decoder_cnn_keys: str = "image"
    decoder_act: str = "SiLU"
    decoder_norm: str = "LayerNorm"
    decoder_cnn_depth: int = 32
    decoder_kernel_size: int = 4
    decoder_minres: int = 4
    decoder_mlp_layers: int = 2
    decoder_mlp_units: int = 512
    decoder_cnn_sigmoid: bool = False
    decoder_image_dist: str = "mse"
    decoder_vector_dist: str = "symlog_mse"

    # Meta
    meta: bool = False
    meta_num_classes: int = 1
    meta_latent_dim: int = 32
    meta_encoder_complexity: float = 40.0