# %%
from copy import deepcopy
from thesis.cemrl.buffer import CemrlReplayBuffer as NewBuffer
from src.cemrl.buffers import CEMRLReplayBuffer as Buffer
from src.envs.toy_goal_env import ToyGoalEnv
from src.envs.samplers import RandomBoxSampler
from src.cemrl.wrappers import CEMRLWrapper
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.sac import SAC
from src.cemrl.networks import Encoder
from thesis.cemrl.task_inference import Encoder as NewEncoder
from thesis.cemrl.config import EncoderConfig
from tqdm import tqdm
import numpy as np

sampler = RandomBoxSampler()
env = DummyVecEnv([lambda: TimeLimit(CEMRLWrapper(ToyGoalEnv(sampler)), 200)] * 101)

encoder = Encoder(1, 2, 2, 2, 40., 1).cuda()
buffer = Buffer(1_000_000, env.observation_space, env.action_space, encoder, n_envs=101, use_bin_weighted_decoder_target_sampling=False)

new_encoder = NewEncoder(2, 2, EncoderConfig()).cuda()
new_buffer = NewBuffer(1_000_000, env.observation_space, env.action_space, 200, 100, n_envs=101)
new_buffer.task_inference = new_encoder # type: ignore

_last_obs = env.reset()
for _ in tqdm(range(1002)):
    action = np.stack([env.action_space.sample() for _ in range(env.num_envs)])
    new_obs, reward, dones, infos = env.step(action)
    

    _last_original_obs, new_obs_, reward_ = _last_obs, new_obs, reward

    # Avoid modification by reference
    next_obs = deepcopy(new_obs_)
    # As the VecEnv resets automatically, new_obs is already the
    # first observation of the next episode
    for i, done in enumerate(dones):
        if done and infos[i].get("terminal_observation") is not None:
            next_obs_ = infos[i]["terminal_observation"]
            # Replace next obs for the correct envs
            for key in next_obs.keys():
                next_obs[key][i] = next_obs_[key]

    buffer.add(_last_original_obs, next_obs, action, reward_, dones, infos)
    new_buffer.add(_last_original_obs, next_obs, action, reward_, dones, infos)

    _last_obs = new_obs




# %%
import matplotlib.pyplot as plt

# %%
buffer_samples = buffer.sample(10000)
new_buffer_samples = new_buffer.sample(10000)

fig, axis = plt.subplots(4, 3, figsize=(10, 10))
axis[0, 0].scatter(buffer_samples.observations["observation"][:, 0].cpu(), buffer_samples.observations["observation"][:, 1].cpu(), alpha=0.09)
axis[0, 0].scatter(new_buffer_samples.observations["obs"][:, 0].cpu(), new_buffer_samples.observations["obs"][:, 1].cpu(), alpha=0.09)
axis[0, 1].hist(buffer_samples.observations["observation"][:, 0].cpu(), alpha=0.4)
axis[0, 1].hist(new_buffer_samples.observations["obs"][:, 0].cpu(), alpha=0.4)
axis[0, 2].hist(buffer_samples.observations["observation"][:, 1].cpu(), alpha=0.4)
axis[0, 2].hist(new_buffer_samples.observations["obs"][:, 1].cpu(), alpha=0.4)

axis[1, 0].scatter(buffer_samples.next_observations["observation"][:, 0].cpu(), buffer_samples.next_observations["observation"][:, 1].cpu(), alpha=0.09)
axis[1, 0].scatter(new_buffer_samples.next_observations["obs"][:, 0].cpu(), new_buffer_samples.next_observations["obs"][:, 1].cpu(), alpha=0.09)
axis[1, 1].hist(buffer_samples.next_observations["observation"][:, 0].cpu(), alpha=0.4)
axis[1, 1].hist(new_buffer_samples.next_observations["obs"][:, 0].cpu(), alpha=0.4)
axis[1, 2].hist(buffer_samples.next_observations["observation"][:, 1].cpu(), alpha=0.4)
axis[1, 2].hist(new_buffer_samples.next_observations["obs"][:, 1].cpu(), alpha=0.4)

axis[2, 0].scatter(buffer_samples.actions[:, 0].cpu(), buffer_samples.actions[:, 1].cpu(), alpha=0.09)
axis[2, 0].scatter(new_buffer_samples.actions[:, 0].cpu(), new_buffer_samples.actions[:, 1].cpu(), alpha=0.09)
axis[2, 1].hist(buffer_samples.actions[:, 0].cpu(), alpha=0.4)
axis[2, 1].hist(new_buffer_samples.actions[:, 0].cpu(), alpha=0.4)
axis[2, 2].hist(buffer_samples.actions[:, 1].cpu(), alpha=0.4)
axis[2, 2].hist(new_buffer_samples.actions[:, 1].cpu(), alpha=0.4)

axis[3, 0].hist(buffer_samples.rewards[:, 0].cpu(), alpha=0.4)
axis[3, 0].hist(new_buffer_samples.rewards[:, 0].cpu(), alpha=0.4)

axis[3, 1].hist(buffer_samples.dones[:, 0].cpu(), alpha=0.4)
axis[3, 1].hist(new_buffer_samples.dones[:, 0].cpu(), alpha=0.4)
None

# %%
import numpy as np
new_samples, _ = new_buffer.cemrl_sample(1000, decoder_context_length=1)
samples, indices, (batch, time) = buffer.get_encoder_context(np.random.choice(buffer.valid_indices(), 1000), return_indices=True)
goals = buffer.goal_idxs[indices]
goals[(batch.cpu(), time.cpu())] = 0

fig, axis = plt.subplots(6, 4, figsize=(15, 20))
axis[0, 0].hist2d(samples.observations.view(-1, 2)[:, 0].cpu().numpy(), samples.observations.view(-1, 2)[:, 1].cpu().numpy())
axis[0, 1].hist2d(new_samples.observations["observation"].view(-1, 2)[:, 0].cpu().numpy(), new_samples.observations["observation"].view(-1, 2)[:, 1].cpu().numpy())
axis[0, 2].hist(samples.observations.view(-1, 2)[:, 0].cpu(), alpha=0.4)
axis[0, 2].hist(new_samples.observations["observation"].view(-1, 2)[:, 0].cpu(), alpha=0.4)
# axis[0, 2].hist(new_samples_seq.observations["observation"].view(-1, 2)[:, 0].cpu(), alpha=0.4)
axis[0, 3].hist(samples.observations.view(-1, 2)[:, 1].cpu(), alpha=0.4)
axis[0, 3].hist(new_samples.observations["observation"].view(-1, 2)[:, 1].cpu(), alpha=0.4)
# axis[0, 3].hist(new_samples_seq.observations["observation"].view(-1, 2)[:, 1].cpu(), alpha=0.4)
axis[1, 0].hist2d(samples.next_observations.view(-1, 2)[:, 0].cpu().numpy(), samples.next_observations.view(-1, 2)[:, 1].cpu().numpy())
axis[1, 1].hist2d(new_samples.next_observations["observation"].view(-1, 2)[:, 0].cpu().numpy(), new_samples.next_observations["observation"].view(-1, 2)[:, 1].cpu().numpy())
axis[1, 2].hist(samples.next_observations.view(-1, 2)[:, 0].cpu(), alpha=0.4)
axis[1, 2].hist(new_samples.next_observations["observation"].view(-1, 2)[:, 0].cpu(), alpha=0.4)
# axis[1, 2].hist(new_samples_seq.nexPt_observations["observation"].view(-1, 2)[:, 0].cpu(), alpha=0.4)
axis[1, 3].hist(samples.next_observations.view(-1, 2)[:, 1].cpu(), alpha=0.4)
axis[1, 3].hist(new_samples.next_observations["observation"].view(-1, 2)[:, 1].cpu(), alpha=0.4)
# axis[1, 3].hist(new_samples_seq.next_observations["observation"].view(-1, 2)[:, 1].cpu(), alpha=0.4)
axis[2, 0].hist2d(samples.actions.view(-1, 2)[:, 0].cpu().numpy(), samples.actions.view(-1, 2)[:, 1].cpu().numpy())
axis[2, 1].hist2d(new_samples.actions.view(-1, 2)[:, 0].cpu().numpy(), new_samples.actions.view(-1, 2)[:, 1].cpu().numpy())
axis[2, 2].hist(samples.actions.view(-1, 2)[:, 0].cpu(), alpha=0.4)
axis[2, 2].hist(new_samples.actions.view(-1, 2)[:, 0].cpu(), alpha=0.4)
# axis[2, 2].hist(new_samples_seq.actions.view(-1, 2)[:, 0].cpu(), alpha=0.4)
axis[2, 3].hist(samples.actions.view(-1, 2)[:, 1].cpu(), alpha=0.4)
axis[2, 3].hist(new_samples.actions.view(-1, 2)[:, 1].cpu(), alpha=0.4)
# axis[2, 3].hist(new_samples_seq.actions.view(-1, 2)[:, 1].cpu(), alpha=0.4)
axis[3, 0].hist(samples.rewards.view(-1).cpu())
axis[3, 1].hist(new_samples.rewards.view(-1).cpu())
axis[3, 2].hist(samples.rewards.view(-1).cpu(), alpha=0.4)
axis[3, 2].hist(new_samples.rewards.view(-1).cpu(), alpha=0.4)
# axis[3, 2].hist(new_samples_seq.rewards.view(-1).cpu(), alpha=0.4)
axis[4, 0].hist(samples.dones.view(-1).cpu())
axis[4, 1].hist(new_samples.dones.view(-1).cpu())
axis[4, 2].hist(samples.dones.view(-1).cpu(), alpha=0.4)
axis[4, 2].hist(new_samples.dones.view(-1).cpu(), alpha=0.4)
# axis[4, 2].hist(new_samples_seq.dones.view(-1).cpu(), alpha=0.4)
axis[5, 0].hist(goals.reshape(-1))
axis[5, 1].hist(new_samples.next_observations["goal_idx"].view(-1).cpu())
axis[5, 2].hist(goals.reshape(-1), alpha=0.4)
axis[5, 2].hist(new_samples.next_observations["goal_idx"].view(-1).cpu(), alpha=0.4)
# axis[5, 2].hist(new_samples_seq.next_observations["goal_idx"].view(-1).cpu(), alpha=0.4)

None

# %%
samples = buffer.get_decoder_targets(np.random.choice(buffer.valid_indices(), 1000))
ep_idx = np.random.choice(new_buffer.valid_episodes, 1000)
sample_idx = np.random.randint(0, new_buffer.storage.episode_lengths[ep_idx], 1000)
new_samples = new_buffer.get_decoder_targets(ep_idx, sample_idx, num_targets=300, env=None)
new_buffer.decoder_context_mode = "sequential"
new_ep_samples = new_buffer.get_decoder_targets(ep_idx, sample_idx, num_targets=300, env=None)
new_buffer.decoder_context_mode = "random"

fig, axis = plt.subplots(7, 3, figsize=(10, 15))
axis[0, 0].hist2d(samples.observations.view(-1, 2)[:, 0].cpu().numpy(), samples.observations.view(-1, 2)[:, 1].cpu().numpy())
axis[0, 1].hist2d(new_samples.observations["observation"].view(-1, 2)[:, 0].cpu().numpy(), new_samples.observations["observation"].view(-1, 2)[:, 1].cpu().numpy())
axis[0, 2].hist2d(new_ep_samples.observations["observation"].view(-1, 2)[:, 0].cpu().numpy(), new_ep_samples.observations["observation"].view(-1, 2)[:, 1].cpu().numpy())
axis[1, 0].hist2d(samples.next_observations.view(-1, 2)[:, 0].cpu().numpy(), samples.next_observations.view(-1, 2)[:, 1].cpu().numpy())
axis[1, 1].hist2d(new_samples.next_observations["observation"].view(-1, 2)[:, 0].cpu().numpy(), new_samples.next_observations["observation"].view(-1, 2)[:, 1].cpu().numpy())
axis[1, 2].hist2d(new_ep_samples.next_observations["observation"].view(-1, 2)[:, 0].cpu().numpy(), new_ep_samples.next_observations["observation"].view(-1, 2)[:, 1].cpu().numpy())
axis[2, 0].hist2d(samples.actions.view(-1, 2)[:, 0].cpu().numpy(), samples.actions.view(-1, 2)[:, 1].cpu().numpy())
axis[2, 1].hist2d(new_samples.actions.view(-1, 2)[:, 0].cpu().numpy(), new_samples.actions.view(-1, 2)[:, 1].cpu().numpy())
axis[2, 2].hist2d(new_ep_samples.actions.view(-1, 2)[:, 0].cpu().numpy(), new_ep_samples.actions.view(-1, 2)[:, 1].cpu().numpy())
axis[3, 0].hist(samples.rewards.view(-1).cpu())
axis[3, 1].hist(new_samples.rewards.view(-1).cpu())
axis[3, 2].hist(new_ep_samples.rewards.view(-1).cpu())
axis[4, 0].hist(samples.dones.view(-1).cpu())
axis[4, 1].hist(new_samples.dones.view(-1).cpu())
axis[4, 2].hist(new_ep_samples.dones.view(-1).cpu())
# axis[5, 0].hist(goals.reshape(-1))
axis[5, 1].hist(new_samples.observations["goal_idx"].view(-1).cpu())
axis[5, 2].hist(new_ep_samples.observations["goal_idx"].view(-1).cpu())
axis[6, 0].hist(samples.observations.view(-1, 2)[:, 0].cpu().numpy())
axis[6, 1].hist(new_samples.observations["observation"].view(-1, 2)[:, 0].cpu().numpy())
axis[6, 2].hist(new_ep_samples.observations["observation"].view(-1, 2)[:, 0].cpu().numpy())


None


# %%
import torch as th
goals = new_samples.next_observations["goal_idx"] == 15
plt.hist(new_samples.rewards[goals].view(-1).cpu())
goals_ep = new_ep_samples.next_observations["goal_idx"] == 15
plt.hist(new_ep_samples.rewards[goals_ep].view(-1).cpu(), alpha=0.4)
plt.show()


# %%
new_samples.observations["observation"][th.where(goals)[0], :, 0].shape

# %%
import torch as th
goals = new_samples.next_observations["goal_idx"] == 15
fig, ax = plt.subplots(1, 3)
ax[0].hist2d(new_samples.observations["observation"][th.where(goals)[0], :, 0].view(-1).cpu().numpy(), new_samples.observations["observation"][th.where(goals)[0], :, 1].view(-1).cpu().numpy())
goals_ep = new_ep_samples.next_observations["goal_idx"] == 15
ax[1].hist2d(new_ep_samples.observations["observation"][th.where(goals_ep)[0], :, 0].view(-1).cpu().numpy(), new_ep_samples.observations["observation"][th.where(goals_ep)[0], :, 1].view(-1).cpu().numpy())
plt.show()

# %%
import seaborn as sns

plt.scatter(new_samples.observations["observation"][th.where(goals)[0][0], :, 0].view(-1).cpu().numpy(), new_samples.observations["observation"][th.where(goals)[0][0], :, 1].view(-1).cpu().numpy())
plt.scatter(new_ep_samples.observations["observation"][th.where(goals_ep)[0][0], :, 0].view(-1).cpu().numpy(), new_ep_samples.observations["observation"][th.where(goals_ep)[0][0], :, 1].view(-1).cpu().numpy())
plt.scatter(samples.observations[3, :, 0].cpu().numpy(), samples.observations[3, :, 1].cpu().numpy())

# %%



