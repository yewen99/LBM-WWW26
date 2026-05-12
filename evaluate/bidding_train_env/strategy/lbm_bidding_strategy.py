import numpy as np
import torch
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from bidding_train_env.strategy.lbm_prompt_template import build_prompt_cot_llm_dt, extract_bid
from bidding_train_env.baseline.dt_baselines.dt_critics import Q_voting


class LBM_BiddingStrategy(BaseBiddingStrategy):
    def __init__(self, budget=100, cpa=2, category=1, cot_llm_model=None, llm_dt_model=None, tokenizer=None, sampling_params=None, critic_model=None, llm_gen_cot=False, use_Q_voting=False, use_cold_start=True):
        super().__init__(budget, "cot-rsa-llm-mlp", cpa, category)

        self.cot_llm_model = cot_llm_model
        self.llm_dt_model = llm_dt_model
        self.critic_model = critic_model
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.llm_gen_cot = llm_gen_cot
        self.use_Q_voting = use_Q_voting
        self.Q_ensemble = critic_model
        self.use_cold_start = use_cold_start

        self.cpa = cpa
        self.budget = budget
        self.category = category
        self.target_return = (budget / (0.3 * cpa)) / self.llm_dt_model.scale

        self.history_states = []
        self.history_actions = []

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost,
                actual_excuted_action=None, historyWonValue=None):

        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        history_xi = [result[:, 0] for result in historyAuctionResult]
        history_pValue = [result[:, 0] for result in historyPValueInfo]
        history_conversion = [result[:, 1] for result in historyImpressionResult]

        historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0
        historical_conversion_mean = np.mean([np.mean(reward) for reward in history_conversion]) if history_conversion else 0
        historical_LeastWinningCost_mean = np.mean([np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0
        historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0
        historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0

        def mean_of_last_n_elements(history, n):
            l = len(history)
            last_n_data = history[max(0, l - n):l]
            if len(last_n_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_n_data])

        last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
        last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
        last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
        last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
        last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

        current_pValues_mean = np.mean(pValues)
        current_pv_num = len(pValues)

        historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
        last_three_pv_num_total = sum(
            [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if historyBid else 0

        test_state = np.array([
            time_left, budget_left, historical_bid_mean, last_three_bid_mean,
            historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
            historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
            last_three_conversion_mean, last_three_xi_mean,
            current_pValues_mean, current_pv_num, last_three_pv_num_total,
            historical_pv_num_total
        ])

        self.history_states.append(np.round(test_state, 2).tolist())
        if timeStepIndex > 0:
            self.history_actions.append(self.last_alpha)

        states, actions, rtgs, timesteps, attention_mask = self.llm_dt_model.orgnize_RSA_inference(
            test_state,
            pre_reward=sum(history_conversion[-1]) if len(history_conversion) != 0 else None,
        )

        alpha = self._bidding_with_llm_cot(timeStepIndex, states, actions, rtgs, timesteps, attention_mask, budget_left, historyWonValue)

        if alpha is None:
            alpha = self.cpa

        bids = alpha * pValues
        self.last_alpha = alpha
        return bids, alpha


    def _bidding_with_llm_cot(self, timeStepIndex, states, actions, rtgs, timesteps, attention_mask, budget_left, historyWonValue):
        if timeStepIndex == 0:
            CoT = 'This is the first timestep of bidding.'
            cot_tokenized = self.tokenizer(CoT, return_tensors="pt", padding=True)
            cot_input_ids = cot_tokenized.input_ids.cuda()
            cot_prompt_embs = self.llm_dt_model.model.embed_tokens(cot_input_ids)
            pred_actions, _ = self.llm_dt_model.forward_Text_RSA_emb(states, actions, rtgs, timesteps, attention_mask, text_prompt_embs=cot_prompt_embs)
            self.llm_dt_model.eval_actions[-1] = pred_actions[0, -1]
            return pred_actions[0, -1].item()

        prompt = build_prompt_cot_llm_dt(
            history_states=self.history_states,
            history_actions=self.history_actions,
            historical_today_won_values=np.round(historyWonValue, 2).tolist(),
            current_state=self.history_states[-1],
            budget=self.budget,
            CPA=self.cpa,
            n_step_history=10,
            simple_prompt=False
        )
        generation = self.cot_llm_model.generate(prompt, self.sampling_params)
        response = generation[0].outputs[0].text
        extracted_cot = extract_bid(response)
        success_extract = 1

        try:
            if float(extracted_cot) == 1:
                CoT = "You should increase the bidding parameter."
            elif float(extracted_cot) == -1:
                CoT = "You should decrease the bidding parameter."
            else:
                success_extract = 0
        except:
            success_extract = 0

        if success_extract == 0:
            CoT = "Uncertain of optimal bid adjustment direction."

        cot_tokenized = self.tokenizer(CoT, return_tensors="pt", padding=True)
        cot_input_ids = cot_tokenized.input_ids.cuda()
        cot_prompt_embs = self.llm_dt_model.model.embed_tokens(cot_input_ids)
        pred_actions, _ = self.llm_dt_model.forward_Text_RSA_emb(states, actions, rtgs, timesteps, attention_mask, text_prompt_embs=cot_prompt_embs)
        self.llm_dt_model.eval_actions[-1] = pred_actions[0, -1]
        alpha = pred_actions[0, -1].item()
        return alpha
