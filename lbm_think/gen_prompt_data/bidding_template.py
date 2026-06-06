"""Prompt-template builders for the LBM-Think GQPO data generation."""

from __future__ import annotations

from typing import Sequence

import numpy as np


NUM_TIMESTEPS_PER_DAY = 48

TASK_DESCRIPTION = (
    "You are an auto-bidding agent determining the optimal bidding parameter "
    "for the advertiser. There are 48 timesteps in a day; the aim is to "
    "maximise the total acquired number of conversions while keeping the "
    "realised CPA below the advertiser's CPA constraint."
)

ATTENTION_COT_DT_FINAL = (
    "You should summarise the history and reason about the best future "
    "adjustment direction. Useful background:\n"
    "1. You should spend the budget evenly — do not exhaust it too early.\n"
    "2. The realised CPA is total_spent_budget / total_acquired_conversions, "
    "with total_spent_budget = budget * (1 - current_budget_remaining_ratio). "
    "If the projected realised CPA at full spend exceeds the CPA constraint, "
    "you should decrease the bidding parameter.\n"
    "As part of the summarisation, output the current cpa_ratio = "
    "realised_cpa / cpa_constraint inside <ratio></ratio> tags, e.g. "
    "<ratio>1.0</ratio>.\n"
    "After your reasoning, you MUST output the direction in <answer></answer> "
    "tags with exactly one of three choices: <answer>1</answer> for increase, "
    "<answer>-1</answer> for decrease, <answer>0</answer> for uncertain."
)


# State-vector indices used to build the textual representation.
_STATE_KEYS = (
    "remaining_timesteps/48",
    "remaining_budget/budget",
    "average bid over past timesteps",
    "average bid over last N timesteps",
    "average least cost required to win an impression over previous timesteps",
    "average pValues over past timesteps",
    "average numbers conversion over past timesteps",
    "average winning status over past timesteps",
    "average least cost over last N timesteps",
    "average pValues over last N timesteps",
    "average numbers conversion over last N timesteps",
    "average winning status over last N timesteps",
    "current pValues mean",
    "current number of impression opportunities",
    "last N timesteps' total impression opportunities",
    "previous timesteps' total impression opportunities",
)
_HISTORICAL_INDICES = (3, 8, 10, 11)
_CURRENT_INDICES = (0, 1, 4, 6, 7)


def format_states_for_cot(states: Sequence[Sequence[float]]) -> str:
    """Format states as a compact (timesteps_remaining, budget_remaining, predicted_value) view."""
    keys = ("timesteps remaining ratio", "budget remaining ratio", "predicted impression value")
    result: dict[str, list[float]] = {k: [] for k in keys}

    for state in states:
        result[keys[0]].append(state[0])
        result[keys[1]].append(state[1])
        result[keys[2]].append(int(state[12] * state[13]))

    return "\n".join(f"{k}: {v}" for k, v in result.items())


def format_states(states: Sequence[Sequence[float]]) -> str:
    """Format states with the historical / current split used by the CoT prompt."""
    n_steps = max(len(states) - 1, 1)
    keys = list(_STATE_KEYS)
    keys[3] = f"average bid over last {n_steps} timesteps"
    keys[8] = f"average least cost over last {n_steps} timesteps"
    keys[9] = f"average pValues over last {n_steps} timesteps"
    keys[10] = f"average numbers conversion over last {n_steps} timesteps"
    keys[11] = f"average winning status over last {n_steps} timesteps"
    keys[14] = f"last {n_steps} timesteps' total impression opportunities"

    columns: dict[str, list[float]] = {k: [] for k in keys}
    for state in states:
        for i, value in enumerate(state):
            columns[keys[i]].append(value)

    historical = "\n".join(f"{keys[i]}: {columns[keys[i]]}" for i in _HISTORICAL_INDICES)
    current = "\n".join(f"{keys[i]}: {columns[keys[i]]}" for i in _CURRENT_INDICES)
    return f"Historical state:\n{historical}\nCurrent state:\n{current}"


def build_promt(
    history_info: dict,
    current_state: Sequence[float],
    budget: float,
    CPA: float,
    expected_rtg: float,
    current_won_conversions: float,
) -> str:
    """Build the full prompt for one GQPO training example.

    The misspelled function name is kept for backwards compatibility with
    callers that imported ``build_promt`` from earlier versions of the code.
    """
    concat_states = np.concatenate(
        [
            np.array(history_info["state"]).reshape(-1, len(current_state)),
            np.array(current_state).reshape(1, -1),
        ]
    ).tolist()

    timestep_start = NUM_TIMESTEPS_PER_DAY - int(concat_states[0][0] * NUM_TIMESTEPS_PER_DAY)
    timestep_end = NUM_TIMESTEPS_PER_DAY - int(concat_states[-1][0] * NUM_TIMESTEPS_PER_DAY)

    return (
        "<|im_start|>system\n"
        f"{TASK_DESCRIPTION}\n"
        "<|im_end|>\n"
        "<|im_start|>Advertiser\n"
        f"The advertiser's budget is {budget} with a CPA constraint {CPA}.\n"
        f"Its historical {len(concat_states) - 1} timesteps and current timestep's states "
        f"(from timestep {timestep_start} to timestep {timestep_end}) are:\n"
        f"{format_states_for_cot(concat_states)}.\n"
        f"For each historical timestep, the bidding parameter is "
        f"{history_info['action']} and the achieved number of conversions is "
        f"{history_info['reward']}.\n"
        f"From timestep 0 to the current timestep, the total acquired conversions is "
        f"{current_won_conversions}.\n"
        f"{ATTENTION_COT_DT_FINAL}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Let me solve this step by step. Now, here is my chain of thought:\n"
    )


# Kept under both names for compatibility.
build_prompt = build_promt
