"""Bidding strategy that wraps an LBM-Act policy with optional LBM-Think CoT."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from bidding_train_env.strategy.lbm_prompt_template import (
    build_prompt_cot_llm_dt,
    extract_bid,
)


# Number of decision steps in a delivery period (24 h, 30 min granularity).
NUM_TIMESTEPS_PER_DAY = 48
# Default high-level guides — must match the vocabulary used at training time.
GUIDE_INCREASE = "You should increase the bidding parameter."
GUIDE_DECREASE = "You should decrease the bidding parameter."
GUIDE_UNCERTAIN = "Uncertain of optimal bid adjustment direction."
GUIDE_FIRST_STEP = "This is the first timestep of bidding."


def _safe_mean(values) -> float:
    """Return the mean of a (possibly empty) sequence-of-arrays, falling back to 0."""
    if not values:
        return 0.0
    return float(np.mean([np.mean(v) for v in values]))


def _mean_of_last_n(values, n: int) -> float:
    if not values:
        return 0.0
    return _safe_mean(values[-n:])


class LbmBiddingStrategy(BaseBiddingStrategy):
    """Hierarchical LBM bidding strategy.

    Parameters
    ----------
    budget, cpa, category
        Standard advertiser-level parameters consumed by
        :class:`BaseBiddingStrategy`.
    cot_llm_model
        Optional vLLM ``LLM`` instance that produces the CoT high-level guide
        (LBM-Think). When ``None`` the strategy falls back to ``GUIDE_UNCERTAIN``.
    llm_dt_model
        The trained LBM-Act policy.
    tokenizer
        Tokenizer matching ``llm_dt_model``.
    sampling_params
        vLLM sampling parameters used together with ``cot_llm_model``.
    use_cold_start
        If True, the bidding parameter for the very first timestep is taken
        from the advertiser CPA target; this stabilises the first auctions.
    """

    def __init__(
        self,
        budget: float = 100.0,
        cpa: float = 2.0,
        category: int = 1,
        *,
        cot_llm_model=None,
        llm_dt_model=None,
        tokenizer=None,
        sampling_params=None,
        llm_gen_cot: bool = False,
        use_cold_start: bool = True,
        n_step_history: int = 10,
        simple_prompt: bool = False,
    ):
        super().__init__(budget, "lbm-cot-rsa", cpa, category)

        if llm_dt_model is None:
            raise ValueError("llm_dt_model (LBM-Act policy) is required")

        self.cot_llm_model = cot_llm_model
        self.llm_dt_model = llm_dt_model
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.llm_gen_cot = llm_gen_cot
        self.use_cold_start = use_cold_start
        self.n_step_history = n_step_history
        self.simple_prompt = simple_prompt

        # The DT-style RTG target is parameterised in terms of the budget
        # divided by a (CPA-conditioned) per-conversion cost; the 0.3 factor
        # is empirical and was kept from the original training recipe.
        self.target_return = (budget / (0.3 * cpa)) / self.llm_dt_model.scale

        self.history_states: list[list[float]] = []
        self.history_actions: list[float] = []
        self.last_alpha: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Bidding strategy API                                                #
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.remaining_budget = self.budget
        self.history_states.clear()
        self.history_actions.clear()
        self.last_alpha = None

    def bidding(
        self,
        timeStepIndex,
        pValues,
        pValueSigmas,
        historyPValueInfo,
        historyBid,
        historyAuctionResult,
        historyImpressionResult,
        historyLeastWinningCost,
        actual_excuted_action=None,
        historyWonValue: Optional[Sequence[float]] = None,
    ):
        state = self._compute_state(
            timeStepIndex,
            pValues,
            historyPValueInfo,
            historyBid,
            historyAuctionResult,
            historyImpressionResult,
            historyLeastWinningCost,
        )

        self.history_states.append(np.round(state, 2).tolist())
        if timeStepIndex > 0 and self.last_alpha is not None:
            self.history_actions.append(self.last_alpha)

        history_conversion = [r[:, 1] for r in historyImpressionResult]
        pre_reward = float(np.sum(history_conversion[-1])) if history_conversion else None

        states, actions, rtgs, timesteps, attention_mask = (
            self.llm_dt_model.organize_rsa_inference(
                state, pre_reward=pre_reward, target_return=self.target_return,
            )
        )

        if self.use_cold_start and timeStepIndex == 0:
            alpha = float(self.cpa)
        else:
            alpha = self._predict_alpha(
                timeStepIndex, states, actions, rtgs, timesteps, attention_mask, historyWonValue,
            )

        if alpha is None or not np.isfinite(alpha):
            alpha = float(self.cpa)

        bids = alpha * np.asarray(pValues)
        self.last_alpha = alpha
        return bids, alpha

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #
    def _compute_state(
        self,
        timeStepIndex,
        pValues,
        historyPValueInfo,
        historyBid,
        historyAuctionResult,
        historyImpressionResult,
        historyLeastWinningCost,
    ) -> np.ndarray:
        time_left = (NUM_TIMESTEPS_PER_DAY - timeStepIndex) / NUM_TIMESTEPS_PER_DAY
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0.0

        history_xi = [r[:, 0] for r in historyAuctionResult]
        history_pvalue = [r[:, 0] for r in historyPValueInfo]
        history_conversion = [r[:, 1] for r in historyImpressionResult]

        hist_xi_mean = _safe_mean(history_xi)
        hist_conv_mean = _safe_mean(history_conversion)
        hist_lwc_mean = _safe_mean(historyLeastWinningCost)
        hist_pv_mean = _safe_mean(history_pvalue)
        hist_bid_mean = _safe_mean(historyBid)

        last3_xi_mean = _mean_of_last_n(history_xi, 3)
        last3_conv_mean = _mean_of_last_n(history_conversion, 3)
        last3_lwc_mean = _mean_of_last_n(historyLeastWinningCost, 3)
        last3_pv_mean = _mean_of_last_n(history_pvalue, 3)
        last3_bid_mean = _mean_of_last_n(historyBid, 3)

        cur_pv_mean = float(np.mean(pValues))
        cur_pv_num = float(len(pValues))
        hist_pv_num_total = float(sum(len(b) for b in historyBid))
        last3_pv_num_total = float(
            sum(len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex))
        ) if historyBid else 0.0

        return np.array(
            [
                time_left, budget_left,
                hist_bid_mean, last3_bid_mean,
                hist_lwc_mean, hist_pv_mean, hist_conv_mean, hist_xi_mean,
                last3_lwc_mean, last3_pv_mean, last3_conv_mean, last3_xi_mean,
                cur_pv_mean, cur_pv_num, last3_pv_num_total, hist_pv_num_total,
            ]
        )

    def _predict_alpha(
        self,
        timeStepIndex: int,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
        historyWonValue: Optional[Sequence[float]],
    ) -> float:
        guide = self._select_guide(timeStepIndex, historyWonValue)
        tokenized = self.tokenizer(guide, return_tensors="pt", padding=True)
        guide_input_ids = tokenized.input_ids.to(self.llm_dt_model.device)
        guide_embs = self.llm_dt_model.model.embed_tokens(guide_input_ids)

        pred_actions, _ = self.llm_dt_model.forward_text_rsa(
            states, actions, rtgs, timesteps, attention_mask, text_prompt_embs=guide_embs,
        )
        # Sync the streaming buffer for subsequent steps.
        self.llm_dt_model.eval_actions[-1] = pred_actions[0, -1]
        return float(pred_actions[0, -1].item())

    def _select_guide(self, timeStepIndex: int, historyWonValue: Optional[Sequence[float]]) -> str:
        if timeStepIndex == 0:
            return GUIDE_FIRST_STEP

        if not (self.llm_gen_cot and self.cot_llm_model is not None):
            return GUIDE_UNCERTAIN

        won_values = list(historyWonValue) if historyWonValue is not None else []
        prompt = build_prompt_cot_llm_dt(
            history_states=self.history_states,
            history_actions=self.history_actions,
            historical_today_won_values=np.round(won_values, 2).tolist(),
            current_state=self.history_states[-1],
            budget=self.budget,
            CPA=self.cpa,
            n_step_history=self.n_step_history,
            simple_prompt=self.simple_prompt,
        )

        try:
            generation = self.cot_llm_model.generate(prompt, self.sampling_params)
            extracted = extract_bid(generation[0].outputs[0].text)
            direction = float(extracted) if extracted is not None else None
        except Exception:  # pragma: no cover - defensive
            direction = None

        if direction == 1:
            return GUIDE_INCREASE
        if direction == -1:
            return GUIDE_DECREASE
        return GUIDE_UNCERTAIN
