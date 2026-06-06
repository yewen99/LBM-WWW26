"""Generate GQPO prompt data from preprocessed AuctionNet trajectories."""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List

import numpy as np
from datasets import Dataset
from tqdm import tqdm


# Allow running the script directly from the repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from bidding_template import build_promt  # noqa: E402  (intentional spelling kept for compat)


# Filtering thresholds — see :func:`filter_trajs`.
DEFAULT_BUDGET_LEFT_RATIO = 0.05
DEFAULT_USED_UP_STEPS = 24
EXPERT_KEEP_PROB = 0.15      # keep ~15% of "good" trajectories
NON_EXPERT_KEEP_PROB = 0.02  # keep ~2% of the rest


def make_prefix_bid(example: dict, template_type: str) -> str:
    """Build the user prompt prefix for a single example."""
    if template_type != "qwen-instruct":
        raise ValueError(f"Unsupported template_type: {template_type}")
    return build_promt(
        example["history_info"],
        example["current_s"],
        example["budget"],
        example["CPA"],
        example["expected_rtg"],
        example["current_won_conversions"],
    )


def discount_cumsum(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    out = np.zeros_like(x)
    out[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        out[t] = x[t] + gamma * out[t + 1]
    return out


def remove_a0_trajs(trajectories: List[dict], min_initial_action: float = 2.0) -> List[dict]:
    """Drop trajectories whose initial action is degenerate (e.g. 0 or 1)."""
    return [t for t in trajectories if t["actions"][0] > min_initial_action]


def filter_trajs(
    trajectories: List[dict],
    *,
    budget_left_ratio: float = DEFAULT_BUDGET_LEFT_RATIO,
    used_up_step: int = DEFAULT_USED_UP_STEPS,
    expert_keep_prob: float = EXPERT_KEEP_PROB,
    non_expert_keep_prob: float = NON_EXPERT_KEEP_PROB,
) -> List[dict]:
    """Return a balanced subset of trajectories.

    A trajectory is considered *expert* when:
    1. its terminal budget-remaining ratio is below ``budget_left_ratio``,
    2. its realised CPA ratio is below 1.0, and
    3. it spent the budget over more than ``used_up_step`` steps.
    """
    trajectories = [t for t in trajectories if t["observations"].shape[0] > 2]

    expert_count, other_count = 0, 0
    kept: List[dict] = []

    for traj in trajectories:
        real_budget = traj["budget"]
        budget_left = traj["observations"][-1][1]
        total_reward = float(np.sum(traj["rewards"]))
        cpa_ratio = ((1 - budget_left) * real_budget / max(total_reward, 1e-10)) / traj["cpa_constrain"]
        used_up_steps = len(traj["dones"])

        is_expert = (
            budget_left < budget_left_ratio
            and cpa_ratio < 1.0
            and used_up_steps > used_up_step
        )

        keep_prob = expert_keep_prob if is_expert else non_expert_keep_prob
        if np.random.rand() < keep_prob:
            traj["is_expert_data"] = is_expert
            kept.append(traj)
            if is_expert:
                expert_count += 1
            else:
                other_count += 1

    print(f"[filter_trajs] expert={expert_count}  other={other_count}  total={len(kept)}")
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GQPO prompt parquet data.")
    parser.add_argument("--local_dir", required=True, help="directory containing the preprocessed AuctionNet shards")
    parser.add_argument("--save_dir", required=True, help="directory to write train.parquet / test.parquet to")
    parser.add_argument("--K", type=int, default=1, help="length of the historical window in each prompt")
    parser.add_argument("--repeat_sample_time", type=int, default=1, help="number of resampling passes over the data")
    parser.add_argument("--train_dataset_ratio", type=float, default=0.90, help="fraction of samples used for training")
    parser.add_argument("--template_type", default="qwen-instruct", help="prompt template variant")
    parser.add_argument("--sparse_data", action="store_true", help="use AuctionNet-sparse trajectories")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    return parser.parse_args()


def _load_trajectories(local_dir: str, sparse: bool) -> List[dict]:
    from lbm_act.seq_dataset import DENSE_TRAJ_FILES, SPARSE_TRAJ_FILES

    files = SPARSE_TRAJ_FILES if sparse else DENSE_TRAJ_FILES
    trajs: List[dict] = []
    for fname in files:
        path = os.path.join(local_dir, fname)
        trajs.extend(np.load(path, allow_pickle=True).tolist())
    return trajs


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    trajectories = _load_trajectories(args.local_dir, args.sparse_data)
    trajectories = filter_trajs(trajectories)

    data_dict: dict[str, list] = {
        "history_info": [], "current_s": [], "budget": [], "expected_rtg": [],
        "CPA": [], "current_won_conversions": [], "target_a": [], "cpa_ratio": [],
    }

    for _ in tqdm(range(args.repeat_sample_time), desc="resampling"):
        for traj in tqdm(trajectories, desc="trajectories", leave=False):
            seq_len = len(traj["actions"])
            if seq_len <= 2:
                continue
            K = args.K
            start_t = random.randint(0, max(1, seq_len - 1 - K))
            cur_step = min(seq_len - 1, start_t + K)

            history_info = {
                "state": [np.round(traj["observations"][start_t:cur_step], 3)],
                "action": [np.round(traj["actions"][start_t:cur_step], 3)],
                "reward": [np.round(traj["rewards"][start_t:cur_step], 3)],
            }

            current_s = np.round(traj["observations"][cur_step], 3)
            budget = traj["budget"]
            expected_rtg = discount_cumsum(traj["rewards"][cur_step:], gamma=1.0)[0]
            cpa = traj["cpa_constrain"]
            target_a = traj["actions"][cur_step]

            current_spent_budget = budget * (1.0 - current_s[1])
            current_won_values = discount_cumsum(traj["rewards"][:cur_step], gamma=1.0)[0]
            current_cpa = current_spent_budget / (current_won_values + 1e-6)
            cpa_ratio = float(current_cpa) / cpa
            assert cpa_ratio >= 0, f"negative CPA ratio: {current_cpa=} {cpa=}"

            data_dict["history_info"].append(history_info)
            data_dict["current_s"].append(current_s)
            data_dict["budget"].append(budget)
            data_dict["expected_rtg"].append(expected_rtg)
            data_dict["CPA"].append(cpa)
            data_dict["target_a"].append(target_a)
            data_dict["cpa_ratio"].append(cpa_ratio)
            data_dict["current_won_conversions"].append(current_won_values)

    raw = Dataset.from_dict(data_dict, split="train")
    train_size = int(args.train_dataset_ratio * len(raw))
    train = raw.select(range(train_size))
    test = raw.select(range(train_size, len(raw) - 1))

    def _process(split: str):
        def _fn(example, idx):
            return {
                "data_source": "bidding",
                "prompt": [{"role": "user", "content": make_prefix_bid(example, args.template_type)}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": example["target_a"], "cpa_ratio": example["cpa_ratio"]},
                },
                "extra_info": {"split": split, "index": idx},
            }
        return _fn

    train = train.map(function=_process("train"), with_indices=True)
    test = test.map(function=_process("test"), with_indices=True)

    train.to_parquet(os.path.join(args.save_dir, "train.parquet"))
    test.to_parquet(os.path.join(args.save_dir, "test.parquet"))
    print(f"[gen_prompt_data] wrote train ({len(train)}) and test ({len(test)}) to {args.save_dir}")


if __name__ == "__main__":
    main()
