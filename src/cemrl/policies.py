from typing import Any, Dict, Tuple, Type
from gymnasium import spaces
import numpy as np
import torch as th
from gymnasium import Env
from src.core.state_aware_algorithm import StateAwarePolicy
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from src.envs.samplers.base_sampler import BaseSampler
from src.cemrl.types import CEMRLObsTensorDict, CEMRLPolicyInput
from src.cemrl.wrappers.cemrl_policy_wrapper import CEMRLPolicyVecWrapper, CEMRLPolicyWrapper
from src.plan2explore.networks import Ensemble, WorldModel
from src.cemrl.networks import Decoder, Encoder
from src.utils import apply_function_to_type


class CEMRLPolicy(StateAwarePolicy):
    def __init__(
        self,
        *args,
        env: Env | VecEnv,
        sub_policy_algorithm_class: Type[OffPolicyAlgorithm],
        sub_policy_algorithm_kwargs: Dict[str, Any],
        latent_dim: int,
        num_classes: int | None = None,
        decoder_ensemble_size: int = 5,
        encoder_window: int = 30,
        net_complexity: float = 40.0,
        **kwargs,
    ):
        super().__init__(env.observation_space, env.action_space, *args, **kwargs)
        if num_classes is None:
            goal_sampler: BaseSampler = (
                env.get_attr("goal_sampler")[0] if isinstance(env, VecEnv) else getattr(env, "goal_sampler")
            )
            num_classes = len(goal_sampler.available_tasks)
        o_obs_space = env.get_attr("original_obs_space")[0] if isinstance(env, VecEnv) else getattr(env, "original_obs_space")
        self.encoder_window = encoder_window
        self.encoder = Encoder(
            num_classes,
            latent_dim,
            spaces.flatdim(o_obs_space),
            spaces.flatdim(self.action_space),
            net_complexity,
        )
        self.decoder = Decoder(
            Ensemble(
                th.nn.ModuleList(
                    [
                        WorldModel(
                            spaces.flatdim(o_obs_space),
                            spaces.flatdim(self.action_space),
                            latent_dim,
                            net_complexity,
                        )
                        for i in range(decoder_ensemble_size)
                    ]
                )
            )
        )
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)

        wrapper = CEMRLPolicyVecWrapper(env, latent_dim) if isinstance(env, VecEnv) else CEMRLPolicyWrapper(env, latent_dim)
        self.sub_policy_algorithm = sub_policy_algorithm_class(
            "MultiInputPolicy", wrapper, buffer_size=0, **sub_policy_algorithm_kwargs
        )

        assert isinstance(self.action_space, spaces.Box)

    @th.no_grad()
    def _predict(
        self,
        observation: CEMRLObsTensorDict,
        deterministic: bool = False,
    ) -> th.Tensor:
        prev_observation: CEMRLObsTensorDict = self.state  # type: ignore
        next_obs: CEMRLObsTensorDict = {}
        for k, v in prev_observation.items():  # type:ignore
            v: th.Tensor
            next_obs[k] = v.clone()
            next_obs[k][:, :-1] = v[:, 1:]
            next_obs[k][:, -1] = observation[k]

        with th.no_grad():
            y, z, encoder_state = self.encoder(
                self.encoder.from_obs_to_encoder_input(prev_observation, next_obs)
            )
        policy_obs = CEMRLPolicyInput(
            observation=observation["observation"].to(self.sub_policy_algorithm.device),
            task_indicator=z.to(self.sub_policy_algorithm.device),
        )

        action = self.sub_policy_algorithm.policy._predict(policy_obs, deterministic)  # type: ignore

        self.state = next_obs
        return action

    def _reset_states(self, size: int) -> Tuple[np.ndarray, ...]:
        return apply_function_to_type(
            self.observation_space.sample(),
            np.ndarray,
            lambda x: th.zeros((size, self.encoder_window, *x.shape), device=self.device),
        )
