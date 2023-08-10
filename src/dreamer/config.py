from dataclasses import dataclass


@dataclass
class EncoderConfig:
    mlp_keys = "$^"
    cnn_keys = "image"
    act = "SiLU"
    norm = "LayerNorm"
    cnn_depth = 32
    kernel_size = 4
    minres = 4
    mlp_layers = 2
    mlp_units = 512
    symlog_inputs = True


@dataclass
class DecoderConfig:
    mlp_keys = "$^"
    cnn_keys = "image"
    act = "SiLU"
    norm = "LayerNorm"
    cnn_depth = 32
    kernel_size = 4
    minres = 4
    mlp_layers = 2
    mlp_units = 512
    cnn_sigmoid = False
    image_dist = "mse"
    vector_dist = "symlog_mse"


@dataclass
class Config:
    device = "cuda=0"
    compile = True
    precision = 32

    # Environment
    reward_EMA = True

    # Model
    dyn_cell = "gru_layer_norm"
    dyn_hidden = 512
    dyn_deter = 512
    dyn_stoch = 32
    dyn_discrete = 32
    dyn_input_layers = 1
    dyn_output_layers = 1
    dyn_rec_depth = 1
    dyn_shared = False
    dyn_mean_act = "none"
    dyn_std_act = "sigmoid2"
    dyn_min_std = 0.1
    dyn_temp_post = True
    grad_heads = ["decoder", "reward", "cont"]
    units = 512
    reward_layers = 2
    cont_layers = 2
    critic_layers = 2
    actor_layers = 2
    act = "SiLU"
    norm = "LayerNorm"
    encoder = EncoderConfig()
    decoder = DecoderConfig()
    critic_head = "symlog_disc"
    reward_head = "symlog_disc"
    dyn_scale = 0.5
    rep_scale = 0.1
    kl_free = 1.0
    cont_scale = 1.0
    reward_scale = 1.0
    weight_decay = 0.0
    unimix_ratio = 0.01
    action_unimix_ratio = 0.01
    initial = "learned"

    # Training
    batch_size = 16
    batch_length = 64
    train_ratio = 512
    pretrain = 100
    model_lr = 1e-4
    opt_eps = 1e-8
    grad_clip = 1000
    value_lr = 3e-5
    actor_lr = 3e-5
    ac_opt_eps = 1e-5
    value_grad_clip = 100
    actor_grad_clip = 100
    dataset_size = 1000000
    slow_value_target = True
    slow_target_update = 1
    slow_target_fraction = 0.02
    opt = "adam"

    # Behavior.
    discount = 0.997
    discount_lambda = 0.95
    imag_horizon = 15
    imag_gradient = "dynamics"
    imag_gradient_mix = 0.0
    imag_sample = True
    actor_dist = "normal"
    actor_entropy = 3e-4
    actor_state_entropy = 0.0
    actor_init_std = 1.0
    actor_min_std = 0.1
    actor_max_std = 1.0
    actor_temp = 0.1
    expl_amount = 0.0
    eval_state_mean = False
    collect_dyn_sample = True
    behavior_stop_grad = True
    value_decay = 0.0
    future_entropy = False

    # Exploration
    expl_behavior = "greedy"
    expl_until = 0
    expl_extr_scale = 0.0
    expl_intr_scale = 1.0
    disag_target = "stoch"
    disag_log = True
    disag_models = 10
    disag_offset = 1
    disag_layers = 4
    disag_units = 400
    disag_action_cond = False


class DMCProPrioConfig(Config):
    steps = 5e5
    action_repeat = 2
    envs = 4
    train_ratio = 512
    video_pred_log = False
    # encoder = EncoderConfig(mlp_keys=".*", cnn_keys="$^")
    # decoder = DecoderConfig(mlp_keys=".*", cnn_keys="$^")


class DMCVisionConfig(Config):
    steps = 1e6
    action_repeat = 2
    envs = 4
    train_ratio = 512
    video_pred_log = True
    # encoder = EncoderConfig(mlp_keys="$^", cnn_keys="image")
    # decoder = DecoderConfig(mlp_keys="$^", cnn_keys="image")


class CrafterConfig(Config):
    task = "crafter_reward"
    step = 1e6
    action_repeat = 1
    envs = 1
    train_ratio = 512
    video_pred_log = True
    dyn_hidden = 1024
    dyn_deter = 4096
    units = 1024
    reward_layers = 5
    cont_layers = 5
    value_layers = 5
    actor_layers = 5
    actor_dist = "onehot"
    imag_gradient = "reinforce"
    # encoder = EncoderConfig(mlp_keys="$^", cnn_keys="image", cnn_depth=96, mlp_layers=5, mlp_units=1024)
    # decoder = DecoderConfig(mlp_keys="$^", cnn_keys="image", cnn_depth=96, mlp_layers=5, mlp_units=1024)


class Atari100kConfig(Config):
    steps = 4e5
    envs = 1
    action_repeat = 4
    train_ratio = 1024
    video_pred_log = True
    eval_episode_num = 100
    actor_dist = "onehot"
    imag_gradient = "reinforce"
    stickey = False
    lives = "unused"
    noops = 30
    resize = "opencv"
    actions = "needed"
    time_limit = 108000


class MinecraftConfig(Config):
    task = "minecraft_diamond"
    step = 1e8
    envs = 16
    action_repeat = 1
    train_ratio = 16
    video_pred_log = True
    dyn_hidden = 1024
    dyn_deter = 4096
    units = 1024
    reward_layers = 5
    cont_layers = 5
    value_layers = 5
    actor_layers = 5
    # encoder = EncoderConfig(
    #     mlp_keys="inventory|inventory_max|equipped|health|hunger|breath|reward",
    #     cnn_keys="image",
    #     cnn_depth=96,
    #     mlp_layers=5,
    #     mlp_units=1024,
    # )
    # decoder = DecoderConfig(
    #     mlp_keys="inventory|inventory_max|equipped|health|hunger|breath",
    #     cnn_keys="image",
    #     cnn_depth=96,
    #     mlp_layers=5,
    #     mlp_units=1024,
    # )
    actor_dist = "onehot"
    imag_gradient = "reinforce"
    break_speed = 100.0
    time_limit = 36000


class DebugConfig(Config):
    debug = True
    pretrain = 1
    prefill = 1
    batch_size = 10
    batch_length = 20


class MemoryMazeConfig(Config):
    steps = 1e8
    action_repeat = 2
    actor_dist = "onehot"
    imag_gradient = "reinforce"
    task = "MemoryMaze_9x9"