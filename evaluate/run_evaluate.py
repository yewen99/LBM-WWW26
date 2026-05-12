import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import math
import logging
from bidding_train_env.strategy import LBM_BiddingStrategy
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv
from transformers import AutoTokenizer
from lbm_act import algo
from lbm_act.llm_nets import BiddingModelConfig, LLMPolicy
import torch
import argparse
from bidding_train_env.baseline.dt_baselines.dt_critics import Q_voting, load_Q_net
import random

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(2)

def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward

def run_test(policy_load_dir, file_path, final_stage=False, llm_gen_cot=False, use_Q_voting=False, Q_paths=[], use_cold_start=True, cot_llm_path=''):
    data_loader = TestDataLoader(file_path=file_path)
    env = OfflineEnv()

    Q_ensemble = []
    if use_Q_voting:
        for path in Q_paths:
            Q_net = load_Q_net(path)
            Q_ensemble.append(Q_net)

    cot_llm_model = None
    sampling_params = None
    if llm_gen_cot:
        from vllm import LLM, SamplingParams
        with torch.no_grad():
            cot_llm_model = LLM(model=cot_llm_path, gpu_memory_utilization=0.8)
        sampling_params = SamplingParams(temperature=0.1, top_p=1., max_tokens=1024)

    mconf = BiddingModelConfig()
    if not final_stage:
        mconf.state_mean = np.array([5.48876391e-01, 6.91904804e-01, 4.80044229e-02, 4.47875045e-02,
                                    1.17763952e-01, 4.87555661e-03, 4.76420127e-04, 5.72794009e-02,
                                    9.93989091e-02, 4.84664169e-03, 5.83001837e-04, 7.04144008e-02,
                                    4.99805521e-03, 1.01522635e+04, 2.86396864e+04, 1.91412327e+05])
        mconf.state_std = np.array([2.84053382e-01, 3.53000441e-01, 3.01172049e-02, 3.21944272e-02,
                                    3.07672391e-02, 1.92189715e-03, 8.29556557e-04, 9.36906833e-02,
                                    3.75196803e-02, 2.45325444e-03, 1.18077056e-03, 1.27290708e-01,
                                    2.48126164e-03, 5.73180055e+03, 1.67849786e+04, 1.52535424e+05])
        mconf.input_state_dim = 16
        mconf.rtg_scale = 1500
    else:
        mconf.state_mean = np.array([5.41854588e-01, 7.19698607e-01, 4.17500439e-02, 4.35970703e-02,
                                    9.91188952e-02, 4.82405201e-04, 4.61863046e-05, 5.29802530e-02,
                                    9.24203256e-02, 4.84138679e-04, 5.76074165e-05, 6.75957800e-02,
                                    4.98045765e-04, 1.02017857e+04, 2.88687230e+04, 1.95333666e+05])
        mconf.state_std = np.array([2.84601949e-01, 3.27488061e-01, 2.76529743e-02, 3.31906076e-02,
                                    2.38985949e-02, 1.89047081e-04, 8.73831598e-05, 9.07318426e-02,
                                    2.65035680e-02, 2.45689550e-04, 1.26855462e-04, 1.23013225e-01,
                                    2.48356154e-04, 5.72176857e+03, 1.67729807e+04, 1.52914080e+05])
        mconf.input_state_dim = 16
        mconf.rtg_scale = 100

    model = LLMPolicy(mconf)
    tokenizer = AutoTokenizer.from_pretrained(mconf.model_name)
    tokenizer.padding_side = 'left'
    lbm_act_learner = LBM_ACT_LEARNER(
        policy=model,
        tokenizer=tokenizer,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.lr),
        max_steps=args.step_num,
        tau=0.9,
        beta=3.,
        alpha=0.005,
        discount=0.99
        )
    
    model_state_dict = lbm_act_learner.state_dict()
    checkpoint = torch.load(policy_load_dir)
    lbm_act_learner.load_state_dict(checkpoint, strict=False)
    lbm_act_model = lbm_act_learner.policy

    keys, test_dict = data_loader.keys, data_loader.test_dict

    overall_score = 0.0
    excced_rate = 0.0
    overall_reward = 0.0
    cpa_ratio = 0.0
    budget_utilzation = 0.0
    for key in keys:
        num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts, budget, cpa, category = data_loader.mock_data(key)

        budget = 1. * budget
        agent = LBM_BiddingStrategy(
            budget=budget, cpa=cpa,
            cot_llm_model=cot_llm_model, llm_dt_model=lbm_act_model,
            tokenizer=tokenizer, sampling_params=sampling_params,
            critic_model=Q_ensemble, llm_gen_cot=llm_gen_cot,
            use_Q_voting=use_Q_voting, use_cold_start=use_cold_start
        )
        agent.llm_dt_model.init_eval()

        rewards = np.zeros(num_timeStepIndex)
        history = {
            'historyBids': [],
            'historyAuctionResult': [],
            'historyImpressionResult': [],
            'historyLeastWinningCost': [],
            'historyPValueInfo': [],
            'historyWonValue': [],
        }

        for timeStep_index in range(num_timeStepIndex):
            pValue = pValues[timeStep_index]
            pValueSigma = pValueSigmas[timeStep_index]
            leastWinningCost = leastWinningCosts[timeStep_index]

            if agent.remaining_budget < 1.:
                break
            else:
                bid, alpha = agent.bidding(
                    timeStep_index, pValue, pValueSigma,
                    history["historyPValueInfo"], history["historyBids"],
                    history["historyAuctionResult"], history["historyImpressionResult"],
                    history["historyLeastWinningCost"],
                    actual_excuted_action=None,
                    historyWonValue=history['historyWonValue']
                )

            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid, leastWinningCost)

            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
            while over_cost_ratio > 0:
                pv_index = np.where(tick_status == 1)[0]
                dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)), replace=False)
                bid[dropped_pv_index] = 0
                tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid, leastWinningCost)
                over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

            agent.remaining_budget -= np.sum(tick_cost)
            rewards[timeStep_index] = np.sum(tick_conversion)
            temHistoryPValueInfo = [(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]
            history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
            history["historyBids"].append(bid)
            history["historyLeastWinningCost"].append(leastWinningCost)
            temAuctionResult = np.array([(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
            history["historyAuctionResult"].append(temAuctionResult)
            temImpressionResult = np.array([(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])])
            history["historyImpressionResult"].append(temImpressionResult)

            if timeStep_index == 0:
                history["historyWonValue"].append(rewards[timeStep_index])
            else:
                history["historyWonValue"].append(history["historyWonValue"][-1] + rewards[timeStep_index])

        all_reward = np.sum(rewards)
        all_cost = agent.budget - agent.remaining_budget
        cpa_real = np.clip(all_cost / (all_reward + 1e-10), a_min=0, a_max=100.)
        cpa_constraint = agent.cpa
        score = getScore_nips(all_reward, cpa_real, cpa_constraint)
        overall_score += score
        overall_reward += all_reward
        if cpa_real > cpa_constraint:
            excced_rate += 1
        cpa_ratio += cpa_real / cpa_constraint
        budget_utilzation += all_cost / budget

    logger.info(f'avg_reward: {overall_reward/len(keys):.4f}  avg_score: {overall_score/len(keys):.4f}  exceed_rate: {excced_rate/len(keys):.4f}  cpa_ratio: {cpa_ratio/len(keys):.4f}  budget_util: {budget_utilzation/len(keys):.4f}')
    return overall_score / len(keys), cpa_ratio / len(keys), overall_reward / len(keys)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate LBM on AuctionNet')
    parser.add_argument('--policy_load_dir', type=str, required=True, help='Path to LBM-Act model checkpoint')
    parser.add_argument('--test_file_path', type=str, required=True, help='Path to the test traffic csv file (e.g., period-7.csv)')
    parser.add_argument('--sparse_data', action='store_true', help='Use AuctionNet-sparse dataset')
    parser.add_argument('--llm_gen_cot', action='store_true', help='Use LBM-Think LLM to generate CoT guidance')
    parser.add_argument('--cot_llm_path', type=str, default='', help='Path to the fine-tuned LBM-Think LLM')
    parser.add_argument('--use_Q_voting', action='store_true', help='Use Q-net ensemble for direction voting')
    parser.add_argument('--Q_paths', type=str, nargs='*', default=[], help='Paths to Q-net checkpoint directories')
    parser.add_argument('--use_cold_start', action='store_true', help='Use CPA as first-step bid')
    args = parser.parse_args()

    print(f'Evaluating LBM checkpoint: {args.policy_load_dir}')
    avg_score, avg_cpa_ratio, avg_reward = run_test(
        policy_load_dir=args.policy_load_dir,
        file_path=args.test_file_path,
        final_stage=args.sparse_data,
        llm_gen_cot=args.llm_gen_cot,
        use_Q_voting=args.use_Q_voting,
        Q_paths=args.Q_paths,
        use_cold_start=args.use_cold_start,
        cot_llm_path=args.cot_llm_path
    )
    print(f'Results: avg_score={avg_score:.4f}, avg_cpa_ratio={avg_cpa_ratio:.4f}, avg_reward={avg_reward:.4f}')
