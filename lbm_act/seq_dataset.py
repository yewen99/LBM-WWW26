import torch
from torch.utils.data import Dataset
import numpy as np
import random
import os


def remove_a0_trajs(trajectories):
    filtered_trajs = []
    for traj in trajectories:
        if traj['actions'][0] > 2:
            filtered_trajs.append(traj)
    trajectories = filtered_trajs  

    return trajectories

class EpisodeReplayBuffer(Dataset):
    def __init__(self, state_dim, act_dim, data_path=None, max_ep_len=48, scale=2000, K=4, load_preprocessed_data=True, final_stage=False):
        self.device = "cpu"
        super(EpisodeReplayBuffer, self).__init__()
        self.max_ep_len = max_ep_len
        self.scale = scale
        print(f'scale for normalize r and rtg: {scale}')

        self.state_dim = state_dim
        self.act_dim = act_dim
        if load_preprocessed_data:
            if final_stage:
                print(f'loading sparse data')
                traj_0 = np.load(os.path.join(data_path, 'preprocessed_trajectory_data_final_1.npy'), allow_pickle=True).tolist()
                traj_1 = np.load(os.path.join(data_path, 'preprocessed_trajectory_data_final_2.npy'), allow_pickle=True).tolist()
                traj_2 = np.load(os.path.join(data_path, 'preprocessed_trajectory_data_final_3.npy'), allow_pickle=True).tolist()
            else:
                print(f'loading dense data')
                traj_0 = np.load(os.path.join(data_path, 'preprocessed_trajectory_data_0.npy'), allow_pickle=True).tolist()
                traj_1 = np.load(os.path.join(data_path, 'preprocessed_trajectory_data_1.npy'), allow_pickle=True).tolist()
                traj_2 = np.load(os.path.join(data_path, 'preprocessed_trajectory_data_2.npy'), allow_pickle=True).tolist()

            self.trajectories = traj_0 + traj_1 + traj_2
            filtered_trajs = []
            for traj in self.trajectories:
                if traj["observations"].shape[0] > 2:
                    filtered_trajs.append(traj)
            self.trajectories = filtered_trajs
            self.trajectories = remove_a0_trajs(self.trajectories)

            self.traj_lens, self.returns = [], []
            self.states, self.rewards = [], []
            for i, t_ in enumerate(self.trajectories):
                self.returns.append(sum(t_["rewards"]))
                self.traj_lens.append(len(t_["observations"]))
                self.states.append(t_["observations"])
                self.rewards.append(t_["rewards"])
            self.traj_lens, self.returns = np.array(self.traj_lens), np.array(self.returns).reshape(-1)
            
            
            tmp_states = np.concatenate(self.states, axis=0)
            self.state_mean, self.state_std = np.mean(tmp_states, axis=0), np.std(tmp_states, axis=0)

        self.K = K
        self.pct_traj = 1.

        num_timesteps = sum(self.traj_lens)
        num_timesteps = max(int(self.pct_traj * num_timesteps), 1)
        sorted_inds = np.argsort(self.returns)
        num_trajectories = 1
        timesteps = self.traj_lens[sorted_inds[-1]]
        ind = len(self.trajectories) - 2
        while ind >= 0 and timesteps + self.traj_lens[sorted_inds[ind]] <= num_timesteps:
            timesteps += self.traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        self.sorted_inds = sorted_inds[-num_trajectories:]

        self.p_sample = self.traj_lens[self.sorted_inds] / sum(self.traj_lens[self.sorted_inds])


    def __getitem__(self, index):
        traj = self.trajectories[int(self.sorted_inds[index])]
        start_t = random.randint(0, traj['rewards'].shape[0] - 1)

        s = traj['observations'][start_t: start_t + self.K]
        a = traj['actions'][start_t: start_t + self.K]
        r = traj['rewards'][start_t: start_t + self.K].reshape(-1, 1)
        c = traj["cost_ts"][start_t: start_t + self.K].reshape(-1, 1)
        if 'terminals' in traj:
            d = traj['terminals'][start_t: start_t + self.K]
        else:
            d = traj['dones'][start_t: start_t + self.K]
        timesteps = np.arange(start_t, start_t + s.shape[0])
        timesteps[timesteps >= self.max_ep_len] = self.max_ep_len - 1

        rtg = self.discount_cumsum(traj['rewards'][start_t:], gamma=1.)[:s.shape[0] + 1].reshape(-1, 1)
        if rtg.shape[0] <= s.shape[0]:
            rtg = np.concatenate([rtg, np.zeros((1, 1))], axis=0)

        tlen = s.shape[0]
        s = np.concatenate([np.zeros((self.K - tlen, self.state_dim)), s], axis=0)
        if self.scale != 1:
            s = (s - self.state_mean) / self.state_std
        a = np.concatenate([np.ones((self.K - tlen, self.act_dim)) * -10., a], axis=0)
        r = np.concatenate([np.zeros((self.K - tlen, 1)), r], axis=0)
        r = r / self.scale

        c = np.concatenate([np.zeros((self.K - tlen, 1)), c], axis=0)
        c = c / traj["budget"]

        d = np.concatenate([np.ones((self.K - tlen)) * 2, d], axis=0)
        rtg = np.concatenate([np.zeros((self.K - tlen, 1)), rtg], axis=0) / self.scale

        timesteps = np.concatenate([np.zeros((self.K - tlen)), timesteps], axis=0)
        mask = np.concatenate([np.zeros((self.K - tlen)), np.ones((tlen))], axis=0)

        s = torch.from_numpy(s).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(a).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(r).to(dtype=torch.float32, device=self.device)
        c = torch.from_numpy(c).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(d).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32, device=self.device)

        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(mask).to(device=self.device)
        budget = traj["budget"]
        cpa = traj["cpa_constrain"]
        
        return s, a, r, d, rtg, timesteps, mask, budget, cpa, c

    def discount_cumsum(self, x, gamma=1.):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum
