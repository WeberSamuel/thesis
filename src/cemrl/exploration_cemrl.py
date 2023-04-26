from typing import List, Optional, Tuple
from src.cemrl.buffers import CEMRLPolicyBuffer
from src.cemrl.policies import CEMRLPolicy
from src.cemrl.cemrl import CEMRL
from src.callbacks import ExplorationCallback
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.buffers import DictReplayBuffer


class ExplorationalCEMRL(CEMRL):
    def __init__(self, *args, exploration_callback: Optional[ExplorationCallback] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration_callback = exploration_callback

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        total_timesteps, callback = super()._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )
        callback = CallbackList([self._init_callback(self.exploration_callback), callback])
        callback.init_callback(self)

        # replay_buffer may have changed --> update policy buffer as well
        assert isinstance(self.policy, CEMRLPolicy)
        assert isinstance(self.policy.policy_algorithm.replay_buffer, CEMRLPolicyBuffer)
        assert isinstance(self.replay_buffer, DictReplayBuffer)
        self.policy.policy_algorithm.replay_buffer.cemrl_replay_buffer = self.replay_buffer
        return total_timesteps, callback

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["exploration_callback"]
