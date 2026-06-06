"""Prompt builder used by LBM-Think to produce a high-level CoT guide."""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np


NUM_TIMESTEPS_PER_DAY = 48

TASK_DESCRIPTION = (
    "You are an auto-bidding agent determining the optimal bidding parameter "
    "for the advertiser. There are 48 timesteps in a day; the aim is to "
    "maximise the total acquired number of conversions while keeping the "
    "realised CPA below the advertiser's CPA constraint."
)

ATTENTION_SIMPLE = (
    "You should summarise the history and reason about the best future "
    "adjustment direction. After your reasoning, you MUST output the direction "
    "in <answer></answer> tags with exactly one of two choices at the end of "
    "your response: <answer>1</answer> indicates increasing the bidding "
    "parameter, <answer>-1</answer> indicates decreasing it."
)

ATTENTION_COMPLEX = (
    "You should summarise the history and reason about the best future "
    "adjustment direction. Useful background:\n"
    "1. You should spend the budget evenly — do not exhaust it too early.\n"
    "2. The realised CPA is total_spent_budget / total_acquired_conversions, "
    "with total_spent_budget = budget * (1 - current_budget_remaining_ratio). "
    "If the projected realised CPA at full spend exceeds the CPA constraint, "
    "you should decrease the bidding parameter.\n"
    "After your reasoning, you MUST output the direction in <answer></answer> "
    "tags with exactly one of three choices at the end of your response: "
    "<answer>1</answer> for increase, <answer>-1</answer> for decrease, "
    "<answer>0</answer> for uncertain."
)


def format_states_for_cot(states: List[List[float]]) -> str:
    """Format a sequence of states for inclusion in the LLM prompt."""
    keys = (
        "timesteps remaining ratio",
        "budget remaining ratio",
        "predicted impression value",
    )
    result: dict[str, list[float]] = {k: [] for k in keys}

    for state in states:
        result[keys[0]].append(state[0])
        result[keys[1]].append(state[1])
        # current pValue mean (idx 12) * current pv num (idx 13)
        result[keys[2]].append(int(state[12] * state[13]))

    return "\n".join(f"{k}: {v}" for k, v in result.items())


def build_prompt_cot_llm_dt(
    history_states: List[List[float]],
    history_actions: List[float],
    historical_today_won_values: List[float],
    current_state: List[float],
    budget: float,
    CPA: float,
    n_step_history: int = 10,
    simple_prompt: bool = False,
) -> str:
    """Build the high-level reasoning prompt for LBM-Think."""
    cur = np.array(current_state).reshape(1, -1)
    if cur[0, 0] == 1.0:
        concat_states = cur.tolist()
    else:
        concat_states = (
            np.array(history_states[-n_step_history - 1 :])
            .reshape(-1, cur.shape[-1])
            .tolist()
        )

    timestep_start = NUM_TIMESTEPS_PER_DAY - int(concat_states[0][0] * NUM_TIMESTEPS_PER_DAY)
    timestep_end = NUM_TIMESTEPS_PER_DAY - int(concat_states[-1][0] * NUM_TIMESTEPS_PER_DAY)

    attention = ATTENTION_SIMPLE if simple_prompt else ATTENTION_COMPLEX

    return (
        "<|im_start|>system\n"
        f"{TASK_DESCRIPTION}\n"
        "<|im_end|>\n"
        "<|im_start|>Advertiser\n"
        f"The advertiser's budget is {budget} and the CPA constraint is {CPA}.\n"
        f"Its historical {len(concat_states) - 1} timesteps and current timestep's states "
        f"(from timestep {timestep_start} to timestep {timestep_end}) are:\n"
        f"{format_states_for_cot(concat_states)}.\n"
        f"For each historical timestep the bidding parameter is "
        f"{np.round(history_actions[-n_step_history:], 2).tolist()} and the achieved "
        f"number of conversions is {historical_today_won_values[-n_step_history:]}.\n"
        f"From timestep 0 to the current timestep, the total acquired number of "
        f"conversions is {sum(historical_today_won_values)}.\n"
        f"{attention}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Let me solve this step by step. Now, here is my chain of thought:"
    )


_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>")


def extract_bid(response: str) -> Optional[float]:
    """Extract the final answer (1 / -1 / 0) from the model's response.

    Returns ``None`` if no parseable answer is found.
    """
    matches = list(_ANSWER_PATTERN.finditer(response or ""))
    if not matches:
        return None
    try:
        return float(matches[-1].group(1).strip())
    except (TypeError, ValueError):
        return None
