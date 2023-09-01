from dataclasses import dataclass, asdict
from typing import Tuple
import numpy as np

import torch
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.type_aliases import MaybeCallback, Schedule
from stable_baselines3.common.vec_env import VecEnv

from src.cemrl.buffers import CEMRLReplayBuffer
from src.core.state_aware_algorithm import StateAwareOffPolicyAlgorithm, StateAwarePolicy
from src.envs.samplers.base_sampler import BaseSampler
from submodules.dreamer.dreamer import Dreamer as ExternalDreamer


@dataclass
class EncoderConfig:
    mlp_keys: str = "observation|goal$"
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
    mlp_keys: str = "observation|goal$"
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
    precision: int = 32
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
    disag_target: str = "stoch"
    disag_log: bool = True
    disag_models: int = 10
    disag_offset: int = 1
    disag_layers: int = 4
    disag_units: int = 400
    disag_action_cond: bool = False


class DreamerPolicy(StateAwarePolicy):
    state: dict[str, th.Tensor]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        dreamer_config=DreamerConfig(),
        **kwargs,
    ):
        super().__init__(observation_space, action_space)
        dreamer_config.grad_heads = ["decoder", "reward", "cont"]  # type: ignore
        dreamer_config.encoder = asdict(dreamer_config.encoder)  # type: ignore
        dreamer_config.decoder = asdict(dreamer_config.decoder)  # type: ignore
        dreamer_config.num_actions = spaces.flatdim(action_space)  # type: ignore
        self.dreamer = ExternalDreamer(observation_space, action_space, dreamer_config)
        self.use_intrinsic = True

    def _predict(self, observation: dict[str, th.Tensor], deterministic: bool = False) -> th.Tensor:
        if self.use_intrinsic:
            self.dreamer._should_expl._until = 1
        else:
            self.dreamer._should_expl._until = 0

        latent = self.state
        action = latent.pop("_action_state")
        observation["is_first"] = observation["is_first"].squeeze(-1)
        observation["is_first"] = observation["is_terminal"].squeeze(-1)
        action, (latent, _) = self.dreamer._policy(observation, (latent, action), self.use_intrinsic)
        latent["_action_state"] = action["action"]
        self.state = latent
        return action["action"]

    def _reset_states(self, size: int) -> dict[str, th.Tensor]:
        latent = self.dreamer._wm.dynamics.initial(size)
        action = torch.zeros(size, self.dreamer._config.num_actions, device=self.device)  # type: ignore
        latent["_action_state"] = action
        return latent


class Dreamer(StateAwareOffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        batch_size=16,
        gradient_steps: int = 1,
        learning_starts: int = 200_000,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
            batch_size=batch_size,
            support_multi_env=True,
            sde_support=False,
            learning_rate=1e-3,
            gradient_steps = gradient_steps,
            learning_starts = learning_starts,
        )
        self._setup_model()

        self.policy: DreamerPolicy
        self.replay_buffer: CEMRLReplayBuffer

    def train(self, gradient_steps: int, batch_size: int) -> None:
        for i in range(gradient_steps):
            self.policy.dreamer._train(
                self.replay_buffer.dreamer_sample(
                    batch_size, self.get_vec_normalize_env(), self.goal_sampler.goals, self.policy.dreamer._config.batch_length
                )
            )

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        result = super()._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar)
        try:
            self.goal_sampler: BaseSampler = (
                self.env.get_attr("goal_sampler", 0)[0] if isinstance(self.env, VecEnv) else getattr(self.env, "goal_sampler")
            )
        except:
            raise Exception("No goal sampler found")
        return result

    def _dump_logs(self) -> None:
        for name, values in self.policy.dreamer._metrics.items():
            self.logger.record(name, float(np.mean(values)))
            self.policy.dreamer._metrics[name] = []
        return super()._dump_logs()
