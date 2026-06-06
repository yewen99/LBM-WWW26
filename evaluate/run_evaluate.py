"""Offline evaluation of LBM-Act (and optionally LBM-Think CoT) on AuctionNet."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer


# Make ``evaluate.*`` and project root importable when the script is launched
# directly (``python evaluate/run_evaluate.py``).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for path in (_THIS_DIR, _REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from bidding_train_env.baseline.dt_baselines.dt_critics import load_Q_net  # noqa: E402
from bidding_train_env.dataloader.test_dataloader import TestDataLoader  # noqa: E402
from bidding_train_env.environment.offline_env import OfflineEnv  # noqa: E402
from bidding_train_env.strategy import LbmBiddingStrategy  # noqa: E402
from lbm_act.algo import LbmActLearner  # noqa: E402
from lbm_act.llm_nets import BiddingModelConfig, LLMPolicy  # noqa: E402
from lbm_act.state_stats import STATE_DIM, get_state_stats  # noqa: E402
from lbm_act.utils import set_seed  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("run_evaluate")


# Score penalty exponent used by the official AuctionNet scoring rule.
SCORE_BETA = 2
EPS = 1e-10
MAX_CPA_RATIO = 100.0


def get_score_nips(reward: float, cpa: float, cpa_constraint: float) -> float:
    """Official AuctionNet scoring rule (lower CPA than the constraint preserves the score)."""
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + EPS)
        penalty = pow(coef, SCORE_BETA)
        return penalty * reward
    return reward


def _build_policy(
    sparse_data: bool,
    policy_load_dir: str,
    learning_rate: float,
    max_steps: int,
) -> tuple[LLMPolicy, AutoTokenizer]:
    state_mean, state_std, rtg_scale = get_state_stats(sparse_data)
    mconf = BiddingModelConfig(
        state_mean=state_mean,
        state_std=state_std,
        input_state_dim=STATE_DIM,
        rtg_scale=rtg_scale,
    )

    model = LLMPolicy(mconf)
    tokenizer = AutoTokenizer.from_pretrained(mconf.model_name)
    tokenizer.padding_side = "left"

    learner = LbmActLearner(
        policy=model,
        tokenizer=tokenizer,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=learning_rate),
        max_steps=max_steps,
    )

    checkpoint = torch.load(policy_load_dir, map_location="cpu")
    learner.load_state_dict(checkpoint, strict=False)
    return learner.policy.eval(), tokenizer


def run_test(
    *,
    policy_load_dir: str,
    file_path: str,
    sparse_data: bool = False,
    llm_gen_cot: bool = False,
    cot_llm_path: str = "",
    use_q_voting: bool = False,
    q_paths: Optional[Sequence[str]] = None,
    use_cold_start: bool = True,
    learning_rate: float = 5e-6,
    max_steps: int = 100_000,
) -> tuple[float, float, float]:
    """Run the offline evaluation loop and return ``(score, cpa_ratio, reward)`` averaged over advertisers."""
    data_loader = TestDataLoader(file_path=file_path)
    env = OfflineEnv()

    q_ensemble: List = []
    if use_q_voting:
        for path in q_paths or []:
            q_ensemble.append(load_Q_net(path))

    cot_llm_model = None
    sampling_params = None
    if llm_gen_cot:
        if not cot_llm_path:
            raise ValueError("--cot_llm_path must be provided when --llm_gen_cot is set")
        # Local import: vLLM is heavy and only required when generating CoT.
        from vllm import LLM, SamplingParams

        with torch.no_grad():
            cot_llm_model = LLM(model=cot_llm_path, gpu_memory_utilization=0.8)
        sampling_params = SamplingParams(temperature=0.1, top_p=1.0, max_tokens=1024)

    lbm_act_model, tokenizer = _build_policy(sparse_data, policy_load_dir, learning_rate, max_steps)

    keys = data_loader.keys

    overall_score = 0.0
    overall_reward = 0.0
    exceed_cpa_count = 0
    cpa_ratio_sum = 0.0
    budget_utilisation_sum = 0.0

    for key in keys:
        num_steps, p_values, p_value_sigmas, least_winning_costs, budget, cpa, _category = (
            data_loader.mock_data(key)
        )

        agent = LbmBiddingStrategy(
            budget=float(budget),
            cpa=float(cpa),
            cot_llm_model=cot_llm_model,
            llm_dt_model=lbm_act_model,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            llm_gen_cot=llm_gen_cot,
            use_cold_start=use_cold_start,
        )
        agent.llm_dt_model.init_eval()

        rewards = np.zeros(num_steps)
        history = {
            "historyBids": [], "historyAuctionResult": [], "historyImpressionResult": [],
            "historyLeastWinningCost": [], "historyPValueInfo": [], "historyWonValue": [],
        }

        for t in range(num_steps):
            if agent.remaining_budget < 1.0:
                break

            p_value = p_values[t]
            p_value_sigma = p_value_sigmas[t]
            least_winning_cost = least_winning_costs[t]

            bid, _alpha = agent.bidding(
                t, p_value, p_value_sigma,
                history["historyPValueInfo"], history["historyBids"],
                history["historyAuctionResult"], history["historyImpressionResult"],
                history["historyLeastWinningCost"],
                actual_excuted_action=None,
                historyWonValue=history["historyWonValue"],
            )

            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
                p_value, p_value_sigma, bid, least_winning_cost,
            )

            # Drop the fewest-impactful winning bids until the period budget is respected.
            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
            while over_cost_ratio > 0:
                pv_index = np.where(tick_status == 1)[0]
                dropped = np.random.choice(
                    pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)), replace=False,
                )
                bid[dropped] = 0
                tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
                    p_value, p_value_sigma, bid, least_winning_cost,
                )
                over_cost_ratio = max(
                    (np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0,
                )

            agent.remaining_budget -= float(np.sum(tick_cost))
            rewards[t] = float(np.sum(tick_conversion))

            history["historyPValueInfo"].append(np.stack([p_value, p_value_sigma], axis=1))
            history["historyBids"].append(bid)
            history["historyLeastWinningCost"].append(least_winning_cost)
            history["historyAuctionResult"].append(np.stack([tick_status, tick_status, tick_cost], axis=1))
            history["historyImpressionResult"].append(np.stack([tick_conversion, tick_conversion], axis=1))
            history["historyWonValue"].append(
                rewards[t] if t == 0 else history["historyWonValue"][-1] + rewards[t]
            )

        total_reward = float(np.sum(rewards))
        total_cost = agent.budget - agent.remaining_budget
        cpa_real = float(np.clip(total_cost / (total_reward + EPS), 0.0, MAX_CPA_RATIO))
        score = get_score_nips(total_reward, cpa_real, agent.cpa)

        overall_score += score
        overall_reward += total_reward
        if cpa_real > agent.cpa:
            exceed_cpa_count += 1
        cpa_ratio_sum += cpa_real / agent.cpa
        budget_utilisation_sum += total_cost / agent.budget

    n = len(keys)
    avg_reward = overall_reward / n
    avg_score = overall_score / n
    avg_cpa_ratio = cpa_ratio_sum / n
    exceed_rate = exceed_cpa_count / n
    avg_budget_util = budget_utilisation_sum / n

    logger.info(
        "avg_reward=%.4f  avg_score=%.4f  exceed_rate=%.4f  cpa_ratio=%.4f  budget_util=%.4f",
        avg_reward, avg_score, exceed_rate, avg_cpa_ratio, avg_budget_util,
    )
    return avg_score, avg_cpa_ratio, avg_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LBM on AuctionNet.")
    parser.add_argument("--policy_load_dir", required=True, help="path to LBM-Act model checkpoint")
    parser.add_argument("--test_file_path", required=True, help="path to the test traffic CSV file (e.g. period-7.csv)")
    parser.add_argument("--sparse_data", action="store_true", help="use AuctionNet-sparse dataset")
    parser.add_argument("--llm_gen_cot", action="store_true", help="use LBM-Think LLM to generate CoT guidance")
    parser.add_argument("--cot_llm_path", default="", help="path to the fine-tuned LBM-Think LLM")
    parser.add_argument("--use_Q_voting", action="store_true", dest="use_q_voting", help="use Q-net ensemble for voting")
    parser.add_argument("--Q_paths", nargs="*", default=[], dest="q_paths", help="paths to Q-net checkpoint directories")
    parser.add_argument("--use_cold_start", action="store_true", help="use CPA as the first-step bidding parameter")
    parser.add_argument("--seed", type=int, default=2, help="random seed")
    parser.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device)
    set_seed(args.seed)

    logger.info("Evaluating LBM checkpoint: %s", args.policy_load_dir)
    avg_score, avg_cpa_ratio, avg_reward = run_test(
        policy_load_dir=args.policy_load_dir,
        file_path=args.test_file_path,
        sparse_data=args.sparse_data,
        llm_gen_cot=args.llm_gen_cot,
        cot_llm_path=args.cot_llm_path,
        use_q_voting=args.use_q_voting,
        q_paths=args.q_paths,
        use_cold_start=args.use_cold_start,
    )
    logger.info(
        "Results: avg_score=%.4f, avg_cpa_ratio=%.4f, avg_reward=%.4f",
        avg_score, avg_cpa_ratio, avg_reward,
    )


if __name__ == "__main__":
    main()
