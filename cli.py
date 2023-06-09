"""Main entry point for running trainings and evaluations."""
from dataclasses import dataclass
from pydoc import locate
from typing import Optional

import gymnasium
import torch
from jsonargparse import ActionConfigFile, ArgumentParser
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv
import wandb

from src.callbacks import ExplorationCallback, Plan2ExploreEvalCallback, SaveConfigCallback, SaveHeatmapCallback


@dataclass
class Callbacks:
    custom_callback: Optional[BaseCallback] = None
    eval_callback: Optional[EvalCallback] = None
    save_heatmap_callback: Optional[SaveHeatmapCallback] = None
    eval_exploration_callback: Optional[Plan2ExploreEvalCallback] = None
    exploration_callback: Optional[ExplorationCallback] = None
    save_config_callback: Optional[SaveConfigCallback] = None
    checkpoint_callback: Optional[CheckpointCallback] = None


def add_base_algorithm(
    parser: ArgumentParser,
    key: str,
    skip_on_algorithm=[],
    skip_on_policy=[],
    skip_policy=False,
    skip_replay_buffer=False,
    skip_optimizer=False,
    skip_feature_extractor=False,
    directly_use_policy=False,
):
    key = key + "."
    parser.add_subclass_arguments(
        BaseAlgorithm, key + "algorithm", skip=set(skip_on_algorithm), instantiate=len(skip_on_algorithm) == 0
    )
    if not skip_policy:
        parser.add_subclass_arguments(
            BasePolicy,
            key + "policy",
            skip=set(["observation_space", "action_space", "lr_schedule"] + skip_on_policy),
            instantiate=False or directly_use_policy,
        )
        if not skip_optimizer:
            parser.add_subclass_arguments(torch.optim.Optimizer, key + "optimizer", skip=set(["params"]), instantiate=False)
            parser.link_arguments(key + "optimizer.class_path", key + "policy.init_args.optimizer_class")
            parser.link_arguments(
                key + "optimizer.init_args", key + "policy.init_args.optimizer_kwargs", compute_fn=lambda x: vars(x)
            )
            parser.link_arguments(key + "optimizer.init_args.lr", key + "algorithm.init_args.learning_rate")
        if not skip_feature_extractor:
            parser.add_subclass_arguments(
                BaseFeaturesExtractor, key + "features_extractor", skip=set(["observation_space"]), instantiate=False
            )
            parser.link_arguments(key + "features_extractor.class_path", key + "policy.init_args.features_extractor_class")
            parser.link_arguments(
                key + "features_extractor.init_args",
                key + "policy.init_args.features_extractor_kwargs",
                compute_fn=lambda x: vars(x),
            )
        if directly_use_policy:
            parser.link_arguments(key + "policy", key + "algorithm.init_args.policy", apply_on="instantiate")
        else:
            parser.link_arguments(
                key + "policy.class_path", key + "algorithm.init_args.policy", compute_fn=lambda x: locate(x)
            )
            parser.link_arguments(
                key + "policy.init_args",
                key + "algorithm.init_args.policy_kwargs",
                compute_fn=lambda x: vars(x),
                apply_on="instantiate",
            )
    if not skip_replay_buffer:
        parser.add_subclass_arguments(
            ReplayBuffer,
            key + "replay_buffer",
            skip=set(
                ["observation_space", "action_space", "buffer_size", "encoder", "device", "n_envs", "optimize_memory_usage"]
            ),
            instantiate=False,
        )
        parser.link_arguments(key + "replay_buffer.class_path", key + "algorithm.init_args.replay_buffer_class")
        parser.link_arguments(
            key + "replay_buffer.init_args", key + "algorithm.init_args.replay_buffer_kwargs", compute_fn=lambda x: vars(x)
        )


if __name__ == "__main__":
    parser = ArgumentParser(parser_mode="omegaconf")
    parser.add_argument("--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format.")
    parser.add_method_arguments(BaseAlgorithm, "learn", "learn")

    # Argument registration and linking for callbacks
    parser.add_dataclass_arguments(Callbacks, "callback")
    parser.link_arguments(
        "callback",
        "learn.callback",
        compute_fn=lambda x: CallbackList(
            [
                callback
                for callback in [
                    x.save_config_callback,
                    x.custom_callback,
                    x.eval_callback,
                    x.save_heatmap_callback,
                    x.eval_exploration_callback,
                    x.exploration_callback,
                    x.checkpoint_callback,
                ]
                if callback is not None
            ]
        ),
        apply_on="instantiate",
    )

    # Argument registration and linking for algorithms
    add_base_algorithm(parser, "main", directly_use_policy=True)

    add_base_algorithm(parser, "exploration")
    parser.link_arguments("main.policy.encoder", "exploration.policy.init_args.encoder", apply_on="instantiate")
    parser.link_arguments("main.policy.decoder", "exploration.policy.init_args.ensemble", apply_on="instantiate")
    parser.link_arguments(
        "exploration.algorithm", "callback.exploration_callback.init_args.exploration_algorithm", apply_on="instantiate"
    )
    parser.link_arguments(
        "exploration.algorithm", "callback.eval_exploration_callback.init_args.eval_model", apply_on="instantiate"
    )

    add_base_algorithm(parser, "sub_algorithm", skip_on_algorithm=["env", "policy", "buffer_size"], skip_replay_buffer=True)
    parser.link_arguments("sub_algorithm.algorithm.class_path", "main.policy.init_args.sub_policy_algorithm_class")
    parser.link_arguments(
        "sub_algorithm.algorithm.init_args",
        "main.policy.init_args.sub_policy_algorithm_kwargs",
        apply_on="instantiate",
        compute_fn=lambda x: vars(x),
    )

    # Argument registration and linking for environments
    parser.add_subclass_arguments((gymnasium.Env, VecEnv), "envs.env")
    parser.link_arguments("envs.env", "main.algorithm.init_args.env", apply_on="instantiate")
    parser.link_arguments("envs.env", "main.policy.init_args.env", apply_on="instantiate")
    parser.link_arguments("envs.env.init_args.env.init_args.n_stack", "main.algorithm.init_args.encoder_window")
    parser.link_arguments("envs.env", "exploration.algorithm.init_args.env", apply_on="instantiate")

    parser.add_subclass_arguments((gymnasium.Env, VecEnv), "envs.exploration_env")
    parser.link_arguments("envs.exploration_env", "exploration.algorithm.init_args.env", apply_on="instantiate")
    parser.link_arguments("envs.exploration_env", "exploration.policy.init_args.env", apply_on="instantiate")

    parser.add_subclass_arguments((gymnasium.Env, VecEnv), "envs.eval_env")
    parser.link_arguments("envs.env.init_args.env.init_args.n_stack", "envs.eval_env.init_args.env.init_args.n_stack")
    parser.link_arguments("envs.eval_env", "callback.eval_callback.init_args.eval_env", apply_on="instantiate")

    parser.add_subclass_arguments((gymnasium.Env, VecEnv), "envs.exploration_eval_env")
    parser.link_arguments("envs.env.init_args.env.init_args.n_stack", "envs.exploration_eval_env.init_args.env.init_args.n_stack")
    parser.link_arguments(
        "envs.exploration_eval_env", "callback.eval_exploration_callback.init_args.eval_env", apply_on="instantiate"
    )

    parser.link_arguments(
        (
            "envs.env",
            "envs.exploration_env",
            "envs.eval_env",
            "envs.exploration_eval_env",
        ),
        "callback.save_heatmap_callback.init_args.envs",
        compute_fn=lambda train, exploration, eval, exploration_eval: locals(),
        apply_on="instantiate",
    )

    parser.add_argument("subcommand", choices=["train", "eval"])
    parser.add_argument("--wandb", type=bool, default=True)
    cfg = parser.parse_args()
    init_cfg = parser.instantiate_classes(cfg)

    command = cfg.pop("subcommand")
    use_wandb = cfg.pop("wandb")
    init_cfg.learn.callback.callbacks.append(SaveConfigCallback(parser, cfg))

    if command == "train":
        if use_wandb:
            env, *log_path_parts = cfg.main.algorithm.init_args.tensorboard_log.replace("logs/", "").split("/")
            name = ' '.join(log_path_parts + [cfg.learn.tb_log_name])
            wandb.init(project="cemrl", sync_tensorboard=True, save_code=True, group=f"{env}-{name}", tags=[env, name], config=cfg.as_dict())
        init_cfg.main.algorithm.learn(**init_cfg.learn)
    elif command == "eval":
        raise NotImplementedError()
    elif command == "train_with_best_replay_buffer":
        init_cfg.main.algorithm.replay_buffer