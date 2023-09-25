from stable_baselines3.common.evaluation import evaluate_policy
import cv2
import cv2
import seaborn as sns
import torch as th
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.cemrl.cemrl import CEMRL
from stable_baselines3.common.vec_env import VecEnv
from src.cemrl.task_inference import EncoderInput
from src.core.state_aware_algorithm import StateAwareOffPolicyAlgorithm


def evaluate_performance(model, env):
    def render_callback(locals, globals):
        if locals["i"] == 0:
            img = env.render()
            cv2.imshow("img", img)
            cv2.waitKey(1)

    reward, reward_std = evaluate_policy(
        model, env, n_eval_episodes=100, callback=render_callback
    )
    print(f"{reward} +- {reward_std}")

def evaluate_world_model(model:CEMRL, env:VecEnv):
    model.policy.task_inference.eval()
    world_model = model.policy.task_inference.decoder
    init_obs = env.reset()

    states = [model.policy._reset_states(env.num_envs)]
    task_encodings = []
    observations = [init_obs]
    actions = []
    rewards = []
    for i in range(100):
        action, new_state = model.predict(observations[-1], states[-1]) # type:ignore
        task_encodings.append(
            model.policy.task_inference(
                EncoderInput(
                    obs=states[-1]["observation"],
                    action=new_state["action"],
                    reward=new_state["reward"],
                    next_obs=new_state["observation"],
                )
            )[1]
        )
        states.append(new_state)
        obs, reward, _, _ = init_cfg.envs.eval_env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)

        img = init_cfg.envs.eval_env.render()
        cv2.imshow("img", img)
        cv2.waitKey(1)

    predicted_obs = [th.tensor(init_obs["observation"], device=model.device)[None].expand(len(world_model.ensemble), -1, -1)]
    predicted_rewards = []
    for i in range(100):
        obs, reward = world_model(
            th.mean(predicted_obs[-1], dim=0),
            th.tensor(model.policy.scale_action(actions[i]), device=model.device),
            None,
            task_encodings[i],
            return_raw=True
        )
        predicted_obs.append(obs)
        predicted_rewards.append(reward)

    obs = np.stack([obs["observation"] for obs in observations])[1:]
    pred_obs = th.stack(predicted_obs).detach().cpu().numpy()[1:]
    pred_rewards = th.stack(predicted_rewards).detach().cpu().numpy()
    actions = np.stack(actions)
    rewards = np.stack(rewards)

    print(obs.shape)
    print(pred_obs.shape)
    print(actions.shape)
    print(rewards.shape)
    print(pred_rewards.shape)


    df_real = pd.DataFrame({"x": obs[:, 0, 0], "y": obs[:, 0, 1], "reward": rewards[:, 0], "mode": "real"})
    df_imagined = pd.DataFrame({"x": np.mean(pred_obs, axis=1)[:, 0, 0], "y": np.mean(pred_obs, axis=1)[:, 0, 1], "reward": np.mean(pred_rewards, axis=1)[:, 0, 0], "mode": "imagined"})
    df_single_model = pd.DataFrame({"x": pred_obs[:, 0, 0, 0], "y": pred_obs[:, 0, 0, 1], "reward": pred_rewards[:, 0, 0, 0], "mode": "single_model"})
    df = pd.concat([df_real, df_imagined, df_single_model])
    sns.scatterplot(data=df, x="x", y="y", hue="mode", style="mode", size="reward", sizes=(10, 100))
    plt.show()

def evaluate(model, env):
    evaluate_performance(model, env)
    evaluate_world_model(model, env)