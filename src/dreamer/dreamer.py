from typing import Callable, cast

import numpy as np
import torch as th
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.type_aliases import DictReplayBufferSamples


from src.core.state_aware_algorithm import StateAwareOffPolicyAlgorithm
from src.dreamer import tools
from src.dreamer.config import Config
from src.dreamer.exploration import Plan2Explore
from src.dreamer.models import DreamReplayBufferSamples, ImagitiveBehavior
from src.dreamer.networks import Context, State
from src.dreamer.policies import DreamerPolicy
from src.dreamer.tools import to_np


class Dreamer(StateAwareOffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        use_amp=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, support_multi_env=True, sde_support=False)
        self.policy: DreamerPolicy

        self._update_count = 0
        self._use_amp = use_amp

        self._setup_model()

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Trains the Dreamer agent for a given number of gradient steps using data from the replay buffer.

        Args:
            gradient_steps (int): The number of gradient steps to perform.
            batch_size (int): The size of the batches to sample from the replay buffer.

        Returns:
            None
        """
        assert self.replay_buffer is not None
        metrics = {}
        for step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self.get_vec_normalize_env())
            post, context, mets = self.train_world_model(data)
            metrics.update(mets)
            start = post
            metrics.update(self.train_behaivor(self.policy.task_behavior, start, self._training_reward)[-1])

            if self.policy.expl_behavior != "greedy":
                mets = self.train_exploration(start, context, data)
                metrics.update({"expl_" + key: value for key, value in mets.items()})

            self._update_count += 1
        for name, value in metrics.items():
            if isinstance(value, np.ndarray):
                self.logger.record(name, value.item())
            else:
                self.logger.record(name, value)
        self.logger.record("update_count", self._update_count)
        self.logger.dump(self.num_timesteps)

    def train_exploration(self, start: State, context: Context, data: DreamReplayBufferSamples):
        """
        Trains the exploration behavior of the Dreamer agent using the given start state, context, and data.

        Args:
            start (dict): The starting state of the agent.
            context (dict): The context of the agent.
            data (dict): The data used for training.

        Returns:
            dict: A dictionary containing the metrics of the training process.
        """
        metrics = {}
        if isinstance(self.policy.expl_behavior, Plan2Explore):
            with tools.RequiresGrad(self.policy.expl_behavior):
                metrics = {}
                stoch = start["stoch"]
                if self.policy.dynamics_discrete:
                    stoch = th.reshape(stoch, (stoch.shape[:-2] + ((stoch.shape[-2] * stoch.shape[-1]),)))
                target = {
                    "embed": context["embed"],
                    "stoch": stoch,
                    "deter": start["deter"],
                    "feat": context["feat"],
                }[self.policy.disagrement_target]
                inputs = context["feat"]
                if self.policy.disagrement_action_cond:
                    inputs = th.concat([inputs, th.Tensor(data.actions).to(self.device)], -1)

                # update one-step-models
                with th.cuda.amp.autocast(self._use_amp):  # type: ignore
                    if self.policy.disagrement_offset:
                        target = target[:, self.policy.disagrement_offset :]
                        inputs = inputs[:, : -self.policy.disagrement_offset]
                    target = target.detach()
                    inputs = inputs.detach()
                    preds = [head(inputs) for head in self.policy.expl_behavior._networks]
                    likes = th.cat([th.mean(pred.log_prob(target))[None] for pred in preds], 0)
                    loss = -th.mean(likes)
                    metrics.update(self.policy.expl_behavior._model_opt(loss, self.policy.expl_behavior.parameters()))
                metrics.update(self.train_behaivor(self.policy.expl_behavior.behavior, start, self._intrinsic_reward)[-1])
        return metrics

    def train_world_model(self, data: DictReplayBufferSamples):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)

        world_model = self.policy.world_model
        kl_free = self.policy.kl_free
        rep_scale = self.policy.representation_scale
        dyn_scale = self.policy.dynamics_scale

        with tools.RequiresGrad(world_model):
            with th.cuda.amp.autocast(self._use_amp):  # type:ignore
                embed = world_model.encoder.forward(data.observations)
                post, prior = world_model.dynamics.observe(embed, data.actions, data.observations["is_first"])
                kl_loss, kl_value, dyn_loss, rep_loss = world_model.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )

                features = world_model.dynamics.get_features(post)
                pred_reward = world_model.reward.forward(
                    features if "reward" in self.policy.dynamics_grad_heads else features.detach()
                )
                pred_continue = world_model.cont.forward(
                    features if "cont" in self.policy.dynamics_grad_heads else features.detach()
                )
                pred_observation = world_model.decoder.forward(
                    features if "observation" in self.policy.dynamics_grad_heads else features.detach()
                )

                reward_loss = -th.mean(pred_reward.log_prob(data.rewards) * world_model._scales.get("reward", 1.0))
                continue_loss = -th.mean(pred_continue.log_prob(1 - data.dones) * world_model._scales.get("cont", 1.0))
                observation_loss = sum(
                    -th.mean(v.log_prob(data.observations[k]) * world_model._scales.get(k, 1.0))
                    for k, v in pred_observation.items()
                )
                model_loss = kl_loss + reward_loss + continue_loss + observation_loss

            metrics = world_model._model_opt(model_loss, world_model.parameters())

        metrics["reward_loss"] = to_np(reward_loss)
        metrics["continue_loss"] = to_np(continue_loss)
        metrics["observation_loss"] = to_np(observation_loss)
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(th.mean(kl_value))
        with th.cuda.amp.autocast(self._use_amp):  # type:ignore
            metrics["prior_ent"] = to_np(th.mean(world_model.dynamics.get_distribution(prior).entropy()))
            metrics["post_ent"] = to_np(th.mean(world_model.dynamics.get_distribution(post).entropy()))
            context = Context(
                embed=embed,
                feat=world_model.dynamics.get_features(post),
                kl=kl_value,
                postent=world_model.dynamics.get_distribution(post).entropy(),
            )
        post = {k: cast(th.Tensor, v).detach() for k, v in post.items()}
        return cast(State, post), context, metrics

    def train_behaivor(
        self,
        behavior: ImagitiveBehavior,
        start: State,
        objective: Callable[[th.Tensor, State, th.Tensor], th.Tensor],
    ):
        if self.policy.slow_value_target and self._update_count % self.policy.slow_target_update == 0:
            polyak_update(
                behavior.critic.parameters(), behavior._slow_value.parameters(), tau=self.policy.slow_target_fraction
            )

        metrics = {}

        with tools.RequiresGrad(behavior.actor):
            with th.cuda.amp.autocast(self._use_amp):  # type: ignore
                imag_feat, imag_state, imag_action = behavior._imagine(start, behavior.actor, self.policy.imagine_horizon)
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = behavior.actor.forward(
                    imag_feat.detach() if self.policy.behavior_stop_grad else imag_feat
                ).entropy()
                state_ent: th.Tensor = behavior._world_model.dynamics.get_distribution(imag_state).entropy()
                # this target is not scaled
                # slow is flag to indicate whether slow_target is used for lambda-return
                target, weights, base = behavior._compute_target(imag_feat, imag_state, reward, actor_ent, state_ent)
                actor_loss, mets = behavior._compute_actor_loss(
                    imagine_features=imag_feat,
                    imagine_action=imag_action,
                    actor_entropy=actor_ent,
                    target=target,
                    state_entropy=state_ent,
                    weights=weights,
                    base=base,
                )
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(behavior.critic):
            with th.cuda.amp.autocast(self._use_amp):  # type: ignore
                value = behavior.critic(value_input[:-1].detach())
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach().transpose(0, 1))
                slow_target = behavior._slow_value(value_input[:-1].detach())
                if self.policy.slow_value_target:
                    value_loss = value_loss - value.log_prob(slow_target.mode.detach())
                if self.policy.value_decay:
                    value_loss += self.policy.value_decay * value.mode
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = th.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode, "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self.policy.actor_distribution in ["onehot"]:
            metrics.update(tools.tensorstats(th.argmax(imag_action, dim=-1).float(), "imag_action"))
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(th.mean(actor_ent))
        with tools.RequiresGrad(behavior):
            metrics.update(behavior._actor_opt(actor_loss, behavior.actor.parameters()))
            metrics.update(behavior._critic_opt(value_loss, behavior.critic.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _reward(self, feat: th.Tensor, state: State, action: th.Tensor):
        return self.policy.world_model.reward.forward(feat).mean

    def _intrinsic_reward(self, feat: th.Tensor, state: State, action: th.Tensor):
        assert isinstance(self.policy.expl_behavior, Plan2Explore)
        inputs = feat
        if self.policy.disagrement_action_cond:
            inputs = th.concat([inputs, action], -1)
        preds = th.cat([head(inputs).mode[None] for head in self.policy.expl_behavior._networks], 0)
        disag = th.mean(th.std(preds, 0), -1)[..., None]
        if self.policy.use_log_disagrement:
            disag = th.log(disag)
        reward = self.policy.exploration_intrinsic_scale * disag
        if self.policy.exploration_extrinsic_scale:
            reward += self.policy.exploration_extrinsic_scale * self._reward(feat, state, action)
        return reward

    def _training_reward(self, feat: th.Tensor, state: State, action: th.Tensor):
        return self.policy.world_model.reward.forward(self.policy.world_model.dynamics.get_features(state)).mode


if __name__ == "__main__":
    from src.envs.toy_goal_env import ToyGoalEnv
    from src.envs.samplers import RandomBoxSampler
    from src.dreamer.policies import DreamerPolicy
    from src.cemrl.wrappers.cemrl_wrapper_1 import CEMRLWrapper
    from src.cemrl.buffers1 import EpisodicReplayBuffer, BufferModes
    from gymnasium.wrappers.time_limit import TimeLimit
    from stable_baselines3.common.vec_env import DummyVecEnv

    dreamer = Dreamer(
        env=DummyVecEnv([lambda: CEMRLWrapper(TimeLimit(ToyGoalEnv(RandomBoxSampler(num_goals=1)), 200))] * 512),
        policy=DreamerPolicy,
        replay_buffer_class=EpisodicReplayBuffer,
        replay_buffer_kwargs=dict(max_episode_length=200, storage_path="tmp/buffer", mode=BufferModes.Episode),
        learning_rate=1e3,
        learning_starts=200 * 512,
        batch_size=5,
        verbose=1,
        tensorboard_log="tmp/tensorboard",   
    )
    dreamer.learn(200 * 512 * 5, progress_bar=True, log_interval=1)
