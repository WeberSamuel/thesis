import os.path
import sys
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import dm_env
from gym import spaces
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq
from stable_baselines3.common.vec_env import VecEnv
import torch as th
from dm_env._environment import StepType
from dm_env.specs import BoundedArray

sys.path.append("submodules\\url_benchmark")
from submodules.url_benchmark.dmc import ExtendedTimeStep
from submodules.url_benchmark.pretrain import generate_model
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)


class URLBAgent(th.nn.Module):
    def __init__(
        self,
        env,
        eval_env,
        exploration_type,
        experiment_log_dir,
        max_timesteps,
        pretraining_steps,
        ensemble_id=None,
    ):
        super().__init__()
        # Expect the specific agent to be specified in the format urlb_rnd
        agent_type = exploration_type.split("_", 1)
        if len(agent_type) == 2:
            agent_type = agent_type[1]
        else:
            raise ValueError("Specific URLB agent unspecified or not understood.")
        self.agent_type = agent_type
        self.pretraining_steps = pretraining_steps
        self.agent_type = agent_type

        self.workdir = os.path.join(experiment_log_dir, "exploration")
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        self.train_env = URLBAgent.URLBEnvWrapper(env, max_timesteps)
        self.eval_env = URLBAgent.URLBEnvWrapper(eval_env, max_timesteps)

        cfg_override = [
            f"agent={self.agent_type}",
            f"save_video=false",
            f"num_train_frames={pretraining_steps+1}",
            f"snapshots={[pretraining_steps]}",
            f'snapshot_dir="."',
        ]

        self.workspace, self.pretrained = generate_model(
            self.train_env,
            self.eval_env,
            cfg_override,
            self.workdir,
            snapshot_prefix=f"agent_{ensemble_id}_" if ensemble_id is not None else "",
        )

    def train_agent(self, steps=0):
        self.workspace.train(additional_frames=steps, save_snapshots=False)

    def save_agent(self, epoch):
        self.workspace.save_snapshot(epoch)

    def get_action(self, obs, state):
        if state is None:
            state = {}

        if "meta" not in state:
            state["meta"] = meta = self.workspace.agent.init_meta()
        else:
            meta = state["meta"]

        with th.no_grad():
            action = self.workspace.agent.act(obs, meta, self.workspace.global_step, eval_mode=False)
        return action
    
    def forward(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)


    class URLBEnvWrapper(dm_env.Environment):
        def __init__(self, env, max_timesteps):
            self.env = env
            self.timestep = 0
            self.max_timesteps = max_timesteps
            self._observation_spec = BoundedArray(
                shape=tuple(env.observation_space.shape),
                dtype=env.observation_space.dtype,
                minimum=env.observation_space.low,
                maximum=env.observation_space.high,
                name="observation",
            )
            self._action_spec = BoundedArray(
                shape=tuple(env.action_space.shape),
                dtype=env.action_space.dtype,
                minimum=env.action_space.low,
                maximum=env.action_space.high,
                name="action",
            )

        def reset(self) -> ExtendedTimeStep:
            self.timestep = 0
            ob = self.env.reset()
            action = np.zeros(self._action_spec.shape, dtype=self._action_spec.dtype)
            return ExtendedTimeStep(StepType.FIRST, 0, 0.99, ob[0].astype(np.float32), action)  # Todo: 0.99 should be the discount factor

        def step(self, action) -> ExtendedTimeStep:
            self.timestep += 1
            assert self.timestep <= self.max_timesteps
            ob, reward, done, info = self.env.step(action[None])
            step_type = StepType.LAST if self.timestep == self.max_timesteps or done[0] else StepType.MID
            return ExtendedTimeStep(step_type, reward[0], 0.99, ob[0].astype(np.float32), action)

        def observation_spec(self):
            return self._observation_spec

        def action_spec(self):
            return self._action_spec


class EnsembleURLBAgent(th.nn.Module):
    def __init__(self, num_ensemble_agents, *args, **kwargs):
        super().__init__()
        self.num_ensemble_agents = num_ensemble_agents
        self.agents = th.nn.ModuleList([URLBAgent(*args, ensemble_id=i, **kwargs) for i in range(num_ensemble_agents)])

    def get_action(self, obs, state:dict|None):
        if state is None:
            state = {}
            state["id"] = id = np.random.randint(0, self.num_ensemble_agents)
        else:
            id = state["id"]
        agent = self.agents[id]
        assert isinstance(agent, URLBAgent)
        action = agent.get_action(obs, state)
        return action, state
    
    def forward(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

    def train_agent(self, steps=0):
        for idx, agent in enumerate(self.agents):
            assert isinstance(agent, URLBAgent)
            agent.train_agent(steps=steps)


class SMMPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        squash_output: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )


class SMMPolicy(BasePolicy):
    def __init__(self, *args, use_sde=False, **kwargs):
        super().__init__(*args, **kwargs)

    def _predict(self, observation, deterministic: bool = False):
        raise NotImplementedError()


class SMMExplorationAlgorithm(OffPolicyAlgorithm):
    def __init__(
        self,
        env: GymEnv | str,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 256,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: Type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: Dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Dict[str, Any] | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        verbose: int = 0,
        device: str = "auto",
        monitor_wrapper: bool = True,
        seed: int | None = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        gradient_steps=1
    ):
        super().__init__(
            SMMPolicy,
            env,
            0.0,
            buffer_size,
            learning_starts,
            batch_size,
            0.005,
            0.99,
            (100, "episode"),
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            policy_kwargs,
            stats_window_size,
            tensorboard_log,
            verbose,
            device,
            True,
            monitor_wrapper,
            seed,
            use_sde,
            sde_sample_freq,
            use_sde_at_warmup,
            sde_support,
        )
        self.smm = EnsembleURLBAgent(5, self.env, self.env, "urlb_smm", "exploration", 200, 20000)
        self._setup_model()
        self.pretrained = False
        self.state = None

    def predict(
        self,
        observation: np.ndarray | Dict[str, np.ndarray],
        state: Dict | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...] | None]:
        action, self.state = self.smm.get_action(observation[0].astype(np.float32), self.state)
        return action
    
    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: np.ndarray | None = None) -> None:
        if np.any(dones):
            self.state = None
        return super()._update_info_buffer(infos, dones)
    
    def train(self, gradient_steps: int, batch_size: int) -> None:
        if not self.pretrained:
            self.smm.train_agent(0) # Pretrain
            self.pretrained = True
        self.smm.train_agent(steps=gradient_steps)
