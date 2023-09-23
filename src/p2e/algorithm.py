import torch as th
from gymnasium import Env
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from src.cemrl.buffers import CEMRLReplayBuffer, ImagineBuffer
from src.cemrl.cemrl import CEMRL
from src.cemrl.wrappers.cemrl_policy_wrapper import CEMRLPolicyVecWrapper
from src.core.exploration_algorithm import ExplorationAlgorithmMixin
from src.core.has_sub_algorithm import HasSubAlgorithm
from src.core.state_aware_algorithm import StateAwareOffPolicyAlgorithm
from src.p2e.policies import P2EPolicy
from src.plan2explore.networks import Ensemble

from .networks import LatentDisagreementEnsemble


class P2E(ExplorationAlgorithmMixin, HasSubAlgorithm, StateAwareOffPolicyAlgorithm):
    replay_buffer: CEMRLReplayBuffer
    policy: P2EPolicy

    def __init__(
        self,
        env: Env | VecEnv,
        learning_rate=1e-3,
        learning_starts=1024,
        gradient_steps=1,
        train_freq=1,
        **kwargs,
    ):
        super().__init__(
            None,  # type: ignore
            env,
            learning_rate,
            learning_starts=learning_starts,
            gradient_steps=gradient_steps,
            train_freq=train_freq,
            support_multi_env=True,
            **kwargs,
            sde_support=False,
        )
        self.log_prefix = "p2e/"

    def _setup_model(self, parent_algorithm: OffPolicyAlgorithm):
        super()._setup_model(parent_algorithm=parent_algorithm)
        if self.policy.config.use_world_model_as_ensemble and isinstance(self.policy.task_inference.decoder, Ensemble):
            self.disagreement_ensemble = LatentDisagreementEnsemble(self.policy.task_inference.decoder)
        elif self.policy.one_step_models is not None:
            self.disagreement_ensemble = LatentDisagreementEnsemble(self.policy.one_step_models)
        else:
            raise ValueError("OneStepModel is not set up correctly.")

    def _setup_sub_algorithm(self):
        assert isinstance(self.parent_algorithm, CEMRL)
        latent_dim = self.parent_algorithm.config.task_inference.encoder.latent_dim
        wrapper = CEMRLPolicyVecWrapper(self.env, latent_dim)
        self.sub_algorithm = self.sub_algorithm_class("MultiInputPolicy", wrapper, buffer_size=0, **self.sub_algorithm_kwargs)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        state_loss, reward_loss = th.zeros(2)
        for _ in range(gradient_steps):
            enc_samples, dec_samples = self.replay_buffer.cemrl_sample(batch_size)
            metrics, y_distribution, z_distributions = self.policy.task_inference.training_step(enc_samples, dec_samples, True)
            task_encoding = self.policy.task_inference.encoder.sample_z(
                y_distribution, z_distributions, y_usage="most_likely"
            )[1]
            task_encoding = task_encoding[:, None].detach()
            if self.policy.one_step_models is not None and not self.policy.config.use_world_model_as_ensemble:
                if self.policy.config.use_ground_truth_as_one_step_target:
                    state_target = dec_samples.next_observations
                    reward_target = dec_samples.rewards
                else:
                    with th.no_grad():
                        state_estimate, reward_estimate = self.policy.task_inference.decoder.ensemble[0](
                            dec_samples.observations, dec_samples.actions, dec_samples.next_observations, task_encoding
                        )
                    state_target = state_estimate
                    reward_target = reward_estimate

                state_estimate, reward_estimate = self.policy.one_step_models(
                    th.cat([dec_samples.observations, dec_samples.actions, task_encoding], dim=-1)
                )

                state_loss = th.nn.functional.mse_loss(state_estimate, state_target)
                reward_loss = th.nn.functional.mse_loss(reward_estimate, reward_target)
                self.policy.optimizer.zero_grad()
                (state_loss + reward_loss).backward()
                self.policy.optimizer.step()

            for k, v in metrics.items():
                self.logger.record_mean("reconstruction/" + k, v)
            self.logger.record_mean("p2e/state_loss", state_loss.item())
            self.logger.record_mean("p2e/reward_loss", reward_loss.item())

        self.sub_algorithm.replay_buffer = ImagineBuffer(
            self.policy.config.imagination_horizon,
            self.policy,
            self.replay_buffer,
            self.disagreement_ensemble,
            self.gradient_steps,
            batch_size,
            self.action_space,  # type: ignore
            self.get_vec_normalize_env(),
        )  # type: ignore
        self.sub_algorithm.replay_buffer = self.replay_buffer
        self.sub_algorithm.train(self.gradient_steps, batch_size)
