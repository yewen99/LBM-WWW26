"""GQPO helper rules: hallucination filter + Q-value based gain assessment."""

from __future__ import annotations

import copy
import re
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from evaluate.bidding_train_env.baseline.dt_baselines.dt_critics import Q_voting, load_Q_net
from lbm_act.algo import LbmActLearner
from lbm_act.llm_nets import BiddingModelConfig, LLMPolicy
from lbm_act.state_stats import STATE_DIM, get_state_stats


# Patterns used by the rule-based hallucination filter.
_RATIO_PATTERN = re.compile(r"<ratio>(.*?)</ratio>")
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>")
_VALID_ANSWERS = {-1, 0, 1}
_CPA_RATIO_TOLERANCE = 0.03  # Accept ratios within 3% of ground truth.

GUIDE_INCREASE = "You should increase the bidding parameter."
GUIDE_DECREASE = "You should decrease the bidding parameter."


def _to_batched(arr, dtype=torch.float32, device="cuda"):
    return torch.tensor(np.asarray(arr), dtype=dtype, device=device).unsqueeze(0)


def delta_Q(rsa_sequence: dict, q_ensemble: List, tokenizer, llm_dt_model) -> int:
    """Use the Q-value ensemble to decide if a CoT direction strictly improves the action.

    Returns one of ``{-1, 0, 1}``:
    * ``+1`` — the "increase" guide produces the highest-Q action,
    * ``-1`` — the "decrease" guide produces the highest-Q action,
    * ``0``  — neither guide improves over the dataset action.
    """
    states = np.vstack(rsa_sequence["s"])
    actions = np.vstack(rsa_sequence["a"])
    rtgs = np.stack(rsa_sequence["rtg"])[:-1, :]
    timesteps = np.stack(rsa_sequence["timesteps"])
    attention_mask = np.stack(rsa_sequence["mask"])

    device = next(llm_dt_model.parameters()).device
    states_t = _to_batched(states, torch.float32, device)
    actions_t = _to_batched(actions, torch.float32, device)
    rtgs_t = _to_batched(rtgs, torch.float32, device)
    timesteps_t = _to_batched(timesteps, torch.long, device)
    attn_t = _to_batched(attention_mask, torch.float32, device)

    actions_copy = copy.deepcopy(actions_t)

    up_ids = tokenizer(GUIDE_INCREASE, return_tensors="pt", padding=True).input_ids.to(device)
    down_ids = tokenizer(GUIDE_DECREASE, return_tensors="pt", padding=True).input_ids.to(device)

    with torch.no_grad():
        up_embs = llm_dt_model.model.embed_tokens(up_ids)
        down_embs = llm_dt_model.model.embed_tokens(down_ids)
        up_pred, _ = llm_dt_model.forward_text_rsa(
            states_t, actions_t, rtgs_t, timesteps_t, attn_t, text_prompt_embs=up_embs,
        )
        down_pred, _ = llm_dt_model.forward_text_rsa(
            states_t, actions_t, rtgs_t, timesteps_t, attn_t, text_prompt_embs=down_embs,
        )

        action_proposals = [up_pred[0, -1], down_pred[0, -1], actions_copy[0, -1]]
        rsa_for_Q = (states_t, actions_t, rtgs_t, timesteps_t, attn_t)
        _best, max_index, _values = Q_voting(action_proposals, rsa_for_Q, q_ensemble)

    return {0: 1, 1: -1, 2: 0}[max_index]


def load_llm_dt(
    sparse_data: bool,
    policy_load_dir: str,
    learning_rate: float = 5e-6,
    max_steps: int = 100_000,
):
    """Re-load a trained LBM-Act learner and return its underlying policy + tokenizer."""
    state_mean, state_std, rtg_scale = get_state_stats(sparse_data)
    mconf = BiddingModelConfig(
        state_mean=state_mean,
        state_std=state_std,
        input_state_dim=STATE_DIM,
        rtg_scale=rtg_scale,
    )
    policy = LLMPolicy(mconf)
    tokenizer = AutoTokenizer.from_pretrained(mconf.model_name)
    tokenizer.padding_side = "left"

    learner = LbmActLearner(
        policy=policy,
        tokenizer=tokenizer,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=learning_rate),
        max_steps=max_steps,
    )
    checkpoint = torch.load(policy_load_dir, map_location="cpu")
    learner.load_state_dict(checkpoint, strict=False)
    return learner.policy.eval(), tokenizer


def expert_rule(response: str, gt: dict, q_ensemble: List, tokenizer, llm_dt_model) -> bool:
    """Return True if ``response`` is non-hallucinated AND its direction is a strict gain.

    The response is required to contain exactly one ``<ratio>`` tag (matching
    the ground-truth realised CPA ratio within ``_CPA_RATIO_TOLERANCE``) and
    exactly one ``<answer>`` tag with value in ``{-1, 0, 1}`` that matches the
    direction recommended by the Q-ensemble.
    """
    gt_cpa_ratio = gt["cpa_ratio"]
    if gt_cpa_ratio == 0:
        return False

    ratios = _RATIO_PATTERN.findall(response)
    answers = _ANSWER_PATTERN.findall(response)
    if len(ratios) != 1 or len(answers) != 1:
        return False

    try:
        ratio_value = float(ratios[0].strip())
        answer_value = int(answers[0].strip())
    except ValueError:
        return False

    if answer_value not in _VALID_ANSWERS:
        return False

    lower = (1.0 - _CPA_RATIO_TOLERANCE) * gt_cpa_ratio
    upper = (1.0 + _CPA_RATIO_TOLERANCE) * gt_cpa_ratio
    if not (lower < ratio_value < upper):
        return False

    return answer_value == delta_Q(gt, q_ensemble, tokenizer, llm_dt_model)
