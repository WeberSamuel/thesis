from src.cemrl.buffers import CEMRLReplayBuffer
from src.envs.toy_goal_env import ToyGoalEnv
from src.envs.samplers import RandomBoxSampler
from src.cemrl.wrappers import CEMRLWrapper
from src.envs.vec_env import SequentialVecEnv
from gymnasium import spaces
import torch as th
import numpy as np

class Encoder(th.nn.Module):
    latent_dim = 1

    def forward(self, *args, **kwargs):
        return th.zeros((1,))

def setup_cemrl_replay_buffer(normal_size=1_000_000, exploration_size=1_000_000, n_envs=512, num_normal_steps=1000, num_exploration_steps=1000):
    env = SequentialVecEnv(CEMRLWrapper(ToyGoalEnv(RandomBoxSampler()), n_stack=30), 512)
    explore_env = SequentialVecEnv(CEMRLWrapper(ToyGoalEnv(RandomBoxSampler()), n_stack=30), 512)
    sut = CEMRLReplayBuffer(normal_size, env.observation_space, env.action_space, Encoder(), exploration_buffer_size=1000, n_envs=n_envs)

    obs = env.reset()
    for i in range(num_normal_steps):
        action = [env.action_space.sample() for i in range(n_envs)]
        next_obs, reward, done, info = env.step(action)
        sut.add(obs=obs, reward=reward, action=action, next_obs=next_obs, done=done, infos=info)
        obs = next_obs

    sut.is_exploring = True
    
    obs = explore_env.reset()
    for i in range(num_exploration_steps):
        action = [env.action_space.sample() for i in range(n_envs)]
        next_obs, reward, done, info = env.step(action)
        sut.add(obs=obs, reward=reward, action=action, next_obs=next_obs, done=done, infos=info)
        obs = next_obs

    return sut
    
def test_get_encoder_context():
    sut = setup_cemrl_replay_buffer()
    encoder_context = sut.get_encoder_context(np.array([40]), encoder_window=30)
    assert encoder_context.rewards.shape == (1, 30, 1)

def test_exploration_does_not_affect_normal_data():
    sut = setup_cemrl_replay_buffer()
