import matplotlib.pyplot as plt
import numpy as np
from callbacks.exploration_callback import ExplorationCallback
from src.cemrl.buffers import EpisodicBuffer
from LogLatentMedian import LogLatentMedian
from src.plan2explore.policies import CEMRLExplorationPolicy
from src.cemrl.policies import CEMRLPolicy
from cemrl.wrappers.cemrl_wrapper import CEMRLHistoryWrapper
from src.cemrl.cemrl import CEMRL
from src.envs.toy_goal_env import ToyGoal1DEnv, ToyGoalEnv
from src.envs.samplers import RandomBoxSampler
from src.plan2explore.plan2explore import Plan2Explore
from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback
import torch as th


def test_1d_latents():
    env = ToyGoal1DEnv(ToyGoalEnv(256, RandomBoxSampler(), step_size=0.25, random_position=True))
    env = CEMRLHistoryWrapper(env, 20)
    algorithm = CEMRL(
        CEMRLPolicy,
        env,
        buffer_size=10_000_000,
        learning_rate=1e-3,
        policy_kwargs={"num_classes": 1, "latent_dim": 1},
        device="cuda",
        replay_buffer_class=EpisodicBuffer,
        replay_buffer_kwargs={"original_obs_space": env.original_obs_space},
        encoder_gradient_steps=1.,
        policy_gradient_steps=lambda x: 0.,
        gradient_steps=5,
        batch_size=512,
        learning_starts=256 * 200,
    )

    assert isinstance(algorithm.policy, CEMRLPolicy)
    exploration_env = ToyGoal1DEnv(ToyGoalEnv(256, RandomBoxSampler(), step_size=0.25, random_position=True))
    exploration_env = CEMRLHistoryWrapper(exploration_env, 20)
    exploration_algorithm = Plan2Explore(
        CEMRLExplorationPolicy,
        exploration_env,
        buffer_size=10_000_000,
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
    exploration_callback = ExplorationCallback(exploration_algorithm, steps_per_rollout=2, pre_train_steps=200)

    algorithm.learn(
        total_timesteps=200*256,#500_000,
        callback=CallbackList(
            [
                ProgressBarCallback(),
                exploration_callback,
            ]
        ),
        log_interval=1,
    )

    algorithm.train(100_000, 512)
    
    # algorithm.save_replay_buffer("test_buffer")
    # exploration_algorithm.save_replay_buffer("test_exploration_buffer")
    # algorithm.save("test.zip")

    callback = LogLatentMedian()
    callback.init_callback(algorithm)
    eval_env = ToyGoal1DEnv(ToyGoalEnv(256, RandomBoxSampler(), random_position=False))
    eval_env = CEMRLHistoryWrapper(eval_env, 20)

    obs = eval_env.reset()
    while True:
        action, _ = algorithm.predict(obs)
        obs, rewards, dones, infos = eval_env.step(action)
        callback.update_locals(locals())
        callback.on_step()
        if np.any(dones):
            break

    latents = th.cat(callback.latents)
    goals = th.cat(callback.goals)
    plt.scatter(goals[..., 0].view(-1).cpu(), latents.view(-1).cpu())
    plt.show()


test_1d_latents()
