"""Main entry point for running trainings and evaluations."""
import jsonargparse
import stable_baselines3 as sb3
import torch
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, ProgressBarCallback
from stable_baselines3.common.vec_env import VecExtractDictObs

from src.callbacks import (
    CheckpointInLogFolderCallback,
    ExplorationCallback,
    Plan2ExploreEvalCallback,
    SaveConfigCallback,
    SaveHeatmapCallback,
)
from src.cemrl.buffers import EpisodicBuffer
from src.cemrl.cemrl import CEMRL
from src.cemrl.exploration_cemrl import CEMRL
from src.cemrl.policies import CEMRLPolicy
from src.cemrl.wrappers import CEMRLHistoryWrapper
from src.cemrl.wrappers.reward_and_action_to_obs_wrapper import RewardAndActionToObsWrapper
from src.cli import Callbacks
from src.envs.samplers.random_box_sampler import RandomBoxSampler
from src.envs.samplers.uniform_circle_sampler import UniformCircleSampler
from src.envs.toy_goal_env import ToyGoalEnv
from src.envs.wrappers.heatmap import HeatmapWrapper
from src.envs.wrappers.include_goal import IncludeGoalWrapper
from src.plan2explore.plan2explore import Plan2Explore
from src.plan2explore.policies import CEMRLExplorationPolicy


def plan2explore_meta():
    env = ToyGoalEnv(256, RandomBoxSampler(), step_size=1.0)
    env = CEMRLHistoryWrapper(HeatmapWrapper(env), 1)
    algorithm = CEMRL(
        CEMRLPolicy,
        env,
        learning_rate=1e-2,
        tensorboard_log="logs/cemrl",
        policy_kwargs={"num_classes": 1, "latent_dim": 2},
        device="cuda",
        encoder_gradient_steps=3 / 5,
        policy_gradient_steps=lambda x: 2 / 5 if x > 0.3 else 10.0,
        gradient_steps=5,
    )
    algorithm = Plan2Explore(
        CEMRLExplorationPolicy,
        env,
        learning_rate=1e-3,
        tensorboard_log="logs/plan2explore",
        policy_kwargs={
            "ensemble": algorithm.policy.decoder,
            "encoder": algorithm.policy.encoder,
            "num_timesteps": 10,
            "best_selection_strategy": torch.max,
        },
        device="cuda",
        gradient_steps=25,
    )
    algorithm.learn(
        total_timesteps=100_000,
        callback=CallbackList(
            [
                ProgressBarCallback(),
                Plan2ExploreEvalCallback(
                    CEMRLHistoryWrapper(ToyGoalEnv(16, UniformCircleSampler(), step_size=1.0), 30), eval_freq=100, render=True
                ),
            ]
        ),
    )
    path = algorithm.logger.get_dir()
    algorithm.save(f"{path}/model.zip")

    algorithm = Plan2Explore.load(f"{path}/model.zip")
    algorithm.policy.is_collecting = False
    env = CEMRLHistoryWrapper(ToyGoalEnv(16, UniformCircleSampler(), step_size=0.25), 30, use_box=True)
    obs = env.reset()
    while True:
        action, _ = algorithm.predict(obs)
        obs, *_ = env.step(action)
        env.render()


def cemrl_meta():
    env = CEMRLHistoryWrapper(ToyGoalEnv(256, RandomBoxSampler(), step_size=0.25), 30, use_box=False)
    algorithm = CEMRL(
        CEMRLPolicy,
        env,
        learning_rate=1e-3,
        tensorboard_log="logs/cemrl",
        policy_kwargs={"num_classes": 1, "latent_dim": 2},
        device="cuda",
        encoder_gradient_steps=1,
        policy_gradient_steps=lambda x: 0 if x > 0.5 else 1,
        gradient_steps=25,
    )
    algorithm.learn(
        total_timesteps=1000,
        callback=CallbackList(
            [
                ProgressBarCallback(),
                EvalCallback(
                    CEMRLHistoryWrapper(
                        ToyGoalEnv(16, UniformCircleSampler(), step_size=0.25),
                        30,
                        use_box=False,
                    ),
                    eval_freq=500,
                    render=True,
                ),
            ]
        ),
    )
    path = algorithm.logger.get_dir()
    algorithm.save(f"{path}/model.zip")

    algorithm = CEMRL.load(f"{path}/model.zip")
    env = CEMRLHistoryWrapper(ToyGoalEnv(16, UniformCircleSampler(), step_size=0.25), 30, use_box=False)
    obs = env.reset()
    while True:
        action, _ = algorithm.predict(obs)
        obs, *_ = env.step(action)
        env.render()


def exploration_cemrl_meta():
    env = ToyGoalEnv(256, RandomBoxSampler(), step_size=0.25, random_position=True)
    env = CEMRLHistoryWrapper(HeatmapWrapper(env), 30)
    algorithm = CEMRL(
        CEMRLPolicy,
        env,
        learning_rate=1e-3,
        buffer_size=10_000_000,
        tensorboard_log="logs/cemrl",
        policy_kwargs={"num_classes": 1, "latent_dim": 2},
        replay_buffer_class=EpisodicBuffer,
        replay_buffer_kwargs={"original_obs_space": env.original_obs_space},
        device="cuda",
        encoder_gradient_steps=1,
        policy_gradient_steps=1,
        gradient_steps=2,
        batch_size=256,
    )
    # algorithm = CEMRL.load("logs\\cemrl\\run_114\\checkpoints\\rl_model_350000_steps.zip", env)

    assert isinstance(algorithm.policy, CEMRLPolicy)
    exploration_env = env.unwrapped
    exploration_env = CEMRLHistoryWrapper(HeatmapWrapper(exploration_env), 30)
    exploration_algorithm = Plan2Explore(
        CEMRLExplorationPolicy,
        exploration_env,
        learning_rate=1e-4,
        policy_kwargs={
            "ensemble": algorithm.policy.decoder,
            "encoder": algorithm.policy.encoder,
        },
        replay_buffer_class=EpisodicBuffer,
        replay_buffer_kwargs={"original_obs_space": env.original_obs_space},
        device="cuda",
        gradient_steps=0,
    )
    exploration_callback = ExplorationCallback(
        exploration_algorithm, steps_per_rollout=20, pre_train_steps=200, link_buffers=True
    )

    eval_env = HeatmapWrapper((ToyGoalEnv(256, UniformCircleSampler(), random_position=True)))
    eval_env = CEMRLHistoryWrapper(eval_env, 30)

    eval_exploration_env = ToyGoalEnv(256, UniformCircleSampler(), random_position=True, step_size=0.25)
    eval_exploration_env = CEMRLHistoryWrapper(HeatmapWrapper(eval_exploration_env), 30)

    algorithm.learn(
        total_timesteps=1_000_000,
        callback=CallbackList(
            [
                ProgressBarCallback(),
                # LatentToGoalCallback(gradient_steps=10),
                CheckpointInLogFolderCallback(100, save_path="checkpoints"),
                exploration_callback,
                SaveHeatmapCallback(exploration_env, 100, "heatmaps", "exploration"),
                SaveHeatmapCallback(env, 100, "heatmaps", "training"),
                EvalCallback(eval_env, eval_freq=100, render=True),
                Plan2ExploreEvalCallback(eval_exploration_env, eval_freq=100, render=True, eval_model=exploration_algorithm),
                SaveHeatmapCallback(eval_env, 100, "heatmaps", "eval"),
                SaveHeatmapCallback(eval_exploration_env, 100, "heatmaps", "eval_exploration"),
            ]
        ),
        log_interval=1,
    )
    path = algorithm.logger.get_dir()
    algorithm.save(f"{path}/model.zip")

    env = CEMRLHistoryWrapper((ToyGoalEnv(16, UniformCircleSampler(), step_size=0.25)), 30, use_box=False)
    algorithm = CEMRL.load(f"{path}/model.zip", env)
    obs = env.reset()
    while True:
        action, _ = algorithm.predict(obs)
        obs, *_ = env.step(action)
        env.render()


def sac():
    env = IncludeGoalWrapper(
        VecExtractDictObs(
            RewardAndActionToObsWrapper(ToyGoalEnv(256, RandomBoxSampler(), step_size=0.25), use_box=False), "observation"
        )
    )
    eval_env = IncludeGoalWrapper(
        VecExtractDictObs(
            RewardAndActionToObsWrapper(ToyGoalEnv(16, UniformCircleSampler(), step_size=0.25), use_box=False), "observation"
        )
    )
    sac = sb3.SAC("MultiInputPolicy", env, 1e-3, gradient_steps=2)
    sac.learn(1_000_000, EvalCallback(eval_env, render=True, eval_freq=100), progress_bar=True)

from jsonargparse import CLI, capture_parser

if __name__ == "__main__":
    parser = capture_parser(lambda: CLI(CEMRL, parser_mode="omegaconf"))
    parser.add_dataclass_arguments(Callbacks, "callback")

    parser.link_arguments("policy.encoder", "policy_algorithm.init_args.cemrl_policy_encoder", apply_on="instantiate")
    parser.link_arguments(
        "env", "callback.exploration_callback.init_args.exploration_algorithm.init_args.policy.init_args.env"
    )
    parser.link_arguments(
        "policy.encoder",
        "callback.exploration_callback.init_args.exploration_algorithm.init_args.policy.init_args.encoder",
        apply_on="instantiate",
    )
    parser.link_arguments(
        "policy.decoder",
        "callback.exploration_callback.init_args.exploration_algorithm.init_args.policy.init_args.ensemble",
        apply_on="instantiate",
    )
    parser.link_arguments("replay_buffer", "policy_algorithm.init_args.cemrl_replay_buffer", apply_on="instantiate")
    parser.link_arguments("env.goal_sampler", "policy.init_args.num_classes", lambda x: x.num_tasks, apply_on="instantiate")
    parser.link_arguments("env", "replay_buffer.init_args.env")
    parser.link_arguments("env", "policy.init_args.env")
    parser.link_arguments("env", "policy_algorithm.init_args.env")
    parser.link_arguments("env", "callback.exploration_callback.init_args.exploration_algorithm.init_args.env")
    parser.link_arguments("encoder_window", "env.init_args.n_stack")
    parser.link_arguments("encoder_window", "policy_algorithm.init_args.encoder_window")
    parser.link_arguments(
        "replay_buffer", "callback.exploration_callback.init_args.exploration_algorithm.init_args.replay_buffer"
    )

    parser.link_arguments("callback.eval_callback.init_args.eval_env", "callback.eval_exploration_callback.init_args.eval_env")
    parser.link_arguments(
        "callback.exploration_callback.exploration_algorithm",
        "callback.eval_exploration_callback.init_args.eval_model",
        apply_on="instantiate",
    )
    # parser.link_arguments(
    #     (
    #         "env",
    #         "callback.exploration_callback.init_args.exploration_algorithm.init_args.env",
    #         "callback.eval_callback.init_args.eval_env",
    #         "callback.eval_exploration_callback.init_args.eval_env",
    #     ),
    #     "callback.save_heatmap_callback.init_args.envs",
    #     lambda env, ex_env, eval, ex_eval: list(zip(["env", "ex_env", "eval", "ex_eval"], [env, ex_env, eval, ex_eval])),
    #     apply_on="instantiate",
    # )

    cfg = parser.parse_args()
    init_cfg = parser.instantiate_classes(cfg)

    if hasattr(init_cfg[cfg.subcommand], "callback"):
        callback: Callbacks = init_cfg.callback

        callbacks = init_cfg[cfg.subcommand].callback or []
        callbacks = (
            [SaveConfigCallback(parser, cfg)]
            + ([callback.exploration_callback] if callback.exploration_callback is not None else [])
            + callbacks
            + [
                callback.callbacks,
                callback.eval_callback,
                callback.eval_exploration_callback,
                SaveHeatmapCallback(
                    [("env", init_cfg.env)]
                    + (
                        [("ex_env", callback.exploration_callback.exploration_algorithm.env)]
                        if callback.exploration_callback is not None
                        else []
                    )
                    + ([("eval", callback.eval_callback.eval_env)] if callback.eval_callback is not None else [])
                    + (
                        [("ex_eval", callback.eval_exploration_callback.eval_env)]
                        if callback.eval_exploration_callback is not None
                        else []
                    ),
                    save_freq=100,
                ),
            ]
        )
        callbacks = list(filter(lambda x: x is not None, callbacks))
        init_cfg[cfg.subcommand].callback = callbacks or None
    init_cfg.pop("callback")
    jsonargparse.cli._run_component(CEMRL, init_cfg)
