import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
import argparse
import numpy as np
import random

def make_prefix_bid(example, template_type):
    history_info = example["history_info"]
    budget = example["budget"]
    CPA = example["CPA"]
    expected_rtg = example["expected_rtg"]
    current_s = example["current_s"]
    current_won_conversions = example["current_won_conversions"]
    if template_type == 'qwen-instruct':
        """This works for bidding-r1 based on Qwen Instruct Models"""
        import bidding_template
        prefix = bidding_template.build_promt(history_info, current_s, budget, CPA, expected_rtg, current_won_conversions)
    
    return prefix

def discount_cumsum(x, gamma=1.):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

def remove_a0_trajs(trajectories):
    # 先过滤一遍初始动作为0或1的垃圾轨迹
    filtered_trajs = []
    for traj in trajectories:
        if traj['actions'][0] > 2:
            filtered_trajs.append(traj)
    trajectories = filtered_trajs  

    return trajectories


def filter_trajs(trajectories, budget_left_ratio=0.05, cpa_down_ratio=0.7, cpa_up_ratio=1.3, used_up_step=24, rtg_scale=2000):
    """
        筛选出符合以下条件的轨迹：
        优质轨迹占50% (调价的groundtruth方向为数据集方向), 其余轨迹占50%
        
        优质轨迹判断条件:
        1) 剩余预算比例 < budget_left_ratio  
        2) CPA_ratio < 1.0
        3) 花完钱的步数 > used_up_step
    """

    # 先过滤一遍太短的垃圾轨迹
    filtered_trajs = []
    for traj in trajectories:
        if traj["observations"].shape[0] > 2:
            filtered_trajs.append(traj)
    trajectories = filtered_trajs  

    cnt_good, cnt_over_cpa, cnt_low_cost, cnt_bad = 0, 0, 0, 0
    filtered_trajs = []
    for traj in trajectories:
        real_budget = traj['budget']
        real_budget_left_ratio = traj['observations'][-1][1]
        real_total_reward = sum(traj['rewards'])
        real_cpa_ratio =  ((1-real_budget_left_ratio)*real_budget / real_total_reward) / traj['cpa_constrain']
        real_used_up_steps = len(traj['dones'])

        # 优质轨迹
        if real_budget_left_ratio < budget_left_ratio and real_cpa_ratio < 1.0 and real_used_up_steps > used_up_step:
            if np.random.rand() > 0.85:
                traj["is_expert_data"] = True
                cnt_good += 1
                filtered_trajs.append(traj)

        # 其余随机
        else:
            if np.random.rand() > 0.98:
                traj["is_expert_data"] = False
                filtered_trajs.append(traj)
                cnt_bad += 1

    print(f'{cnt_good=} {cnt_bad=} ') 
    return filtered_trajs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/jiangnan/liyewen/r1/data/AuctionNet')
    parser.add_argument('--save_dir', default='/jiangnan/liyewen/r1/data/AuctionNet/Expert_prompt/')
    parser.add_argument('--K', type=int, default=1, help='the sequence length of the history information')
    parser.add_argument('--repeat_sample_time', type=int, default=1, help='randomly generate data for how many times')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_dataset_ratio', type=int, default=0.90)
    parser.add_argument('--test_size', type=int, default=1024)  # unused
    parser.add_argument('--template_type', type=str, default='qwen-instruct')

    args = parser.parse_args()

    data_source = 'bidding'
    
    final_stage = False
    if final_stage:
        traj_0 = np.load('data/AuctionNet/final/preprocessed_trajectory_data_final_1.npy', allow_pickle=True).tolist()
        traj_1 = np.load('data/AuctionNet/final/preprocessed_trajectory_data_final_2.npy', allow_pickle=True).tolist()
        traj_2 = np.load('data/AuctionNet/final/preprocessed_trajectory_data_final_3.npy', allow_pickle=True).tolist()
    else:
        traj_0 = np.load(os.path.join(args.local_dir, 'preprocessed_trajectory_data_0.npy'), allow_pickle=True).tolist()
        traj_1 = np.load(os.path.join(args.local_dir, 'preprocessed_trajectory_data_1.npy'), allow_pickle=True).tolist()
        traj_2 = np.load(os.path.join(args.local_dir, 'preprocessed_trajectory_data_2.npy'), allow_pickle=True).tolist()

    trajectories = traj_0 + traj_1 + traj_2
    trajectories = filter_trajs(trajectories)
    # trajectories = trajectories[:100]
    data_dict = {
        'history_info': [],
        'current_s': [],
        'budget': [],
        'expected_rtg': [],
        'CPA': [],
        'current_won_conversions': [],
        'target_a': [],
        'cpa_ratio': []
    }

    for _ in tqdm(range(args.repeat_sample_time), desc="Repeating sampling Progress"):
        for i in tqdm(range(len(trajectories)), desc="Inner Loop Progress for Iterating All Data", leave=False):
            seq_len = len(trajectories[i]["actions"])
            if seq_len <= 2:
                continue
            K = args.K
            start_t = random.randint(0, np.max((1, seq_len - 1 - K)))
            cur_step = np.min((seq_len-1, start_t + K))
            history_info = {"state":[], 
                            "action": [], 
                            "reward": []}
            history_info["state"].append(np.round(trajectories[i]["observations"][start_t:cur_step], 3))
            history_info["action"].append(np.round(trajectories[i]["actions"][start_t:cur_step], 3))
            history_info["reward"].append(np.round(trajectories[i]["rewards"][start_t:cur_step], 3))

            current_s = np.round(trajectories[i]["observations"][cur_step], 3)
            budget = trajectories[i]["budget"]
            expected_rtg = discount_cumsum(trajectories[i]['rewards'][cur_step:], gamma=1.)[0]
            cpa = trajectories[i]["cpa_constrain"]
            target_a = trajectories[i]["actions"][cur_step]

            data_dict["history_info"].append(history_info)
            data_dict["current_s"].append(current_s)
            data_dict["budget"].append(budget)
            data_dict["expected_rtg"].append(expected_rtg)
            data_dict["CPA"].append(cpa)
            data_dict["target_a"].append(target_a)

            current_spent_budget = budget * (1. - current_s[1])
            current_won_values = discount_cumsum(trajectories[i]['rewards'][:cur_step], gamma=1.)[0]
            current_cpa = current_spent_budget / (current_won_values + 1e-6)
            cpa_ratio = current_cpa.item() / cpa
            assert cpa_ratio >=0, f"error in calculating cpa_ratio: {current_cpa=} {cpa=} {current_spent_budget=} {current_won_values=}"
            data_dict["cpa_ratio"].append(cpa_ratio)
            data_dict['current_won_conversions'].append(current_won_values)



    raw_dataset = Dataset.from_dict(data_dict, split='train')
    TRAIN_SIZE = int(args.train_dataset_ratio * len(raw_dataset))
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, len(raw_dataset) - 1))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix_bid(example, template_type=args.template_type)
            solution = {
                "target": example['target_a'],
                "cpa_ratio": example['cpa_ratio'],
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    save_dir = args.save_dir

    train_dataset.to_parquet(os.path.join(save_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(save_dir, 'test.parquet'))

