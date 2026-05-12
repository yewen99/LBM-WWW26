import numpy as np
import re


def format_states_for_cot(states):
    keys = [
        "timesteps remaining ratio",
        "budget remaining ratio",
        "predicted impression value"
    ]

    result = {key: [] for key in keys}

    for state in states:
        for i, value in enumerate(state[:2]):
            result[keys[i]].append(value)
        result["predicted impression value"].append(int(state[12] * state[13]))

    output = []
    for key in keys:
        output.append(f"{key}: {result[key]}")

    return "\n".join(output)


def build_prompt_cot_llm_dt(history_states, history_actions, historical_today_won_values, current_state, budget, CPA, n_step_history=10, simple_prompt=False):
    current_state = np.array(current_state).reshape(1, -1)
    if current_state[0, 0] == 1.:
        concat_states = current_state.tolist()
    else:
        concat_states = np.array(history_states[-n_step_history - 1:]).reshape(-1, current_state.shape[-1])

    timestep_start = 48 - int(concat_states[0][0] * 48)
    timestep_end = 48 - int(concat_states[-1][0] * 48)

    remain_budget = concat_states[-1][1] * budget
    rtg = remain_budget / CPA

    TASK_DESCRIPTION = "You are an auto-bidding agent determining the optimal bidding parameter for the advertiser. \
        There are 48 timesteps of a day, the aim is to maximize the total acquired number of conversions with a lower realized CPA."

    ATTENTION_simple = "You should summarize the history and reason for the best future adjustment direction.\
                After your summarization and reasoning, you MUST output the direction in <answer></answer> tags with only two choices at the end of your response: <answer>1</answer>indicates increasing the bidding parameter, <answer>-1</answer> indicates decreasing the parameter.\
                "

    ATTENTION_complex = "You should summarize the history and then reason for the best future adjustment direction. Here are some basic knowledge:\
                1. You should carefully and sufficiently spend your budget but do not spend all the budget too early; \
                2. The realized CPA is calculated as (total_spent_budget / total_acquired_number_of_conversions), where total_spent_budget = budget * (1 - current_budget_remaining_ratio).\
                If you think the realized CPA would be bigger than the CPA constraint when spent all the budget, you should decrease the bidding parameter. \
                After your summarization and reasoning, you MUST output the direction in <answer></answer> tags with only three choices at the end of your response: <answer>1</answer>indicates you are sure increasing the bidding parameter is better, <answer>-1</answer> indicates you are sure decreasing the parameter is better, and <answer>0</answer> indicates you are uncertain about the optimal adjustment direction.\
                "

    if simple_prompt:
        ATTENTION = ATTENTION_simple
    else:
        ATTENTION = ATTENTION_complex

    Prompt_Template = f"""\
        <|im_start|>\
            system\n\
                {TASK_DESCRIPTION}\
        <|im_end|>\n\
        <|im_start|>\
            Advertiser\n \
                The advertiser's budget is {budget} and the CPA constraint is {CPA}.\
                Its historical {len(concat_states) - 1} timesteps and current timestep's states (i.e., from timestep {timestep_start} to timestep {timestep_end}) are: \n{format_states_for_cot(concat_states)}.\
                For each historical timestep, their corresponding bidding parameter is: {np.round(history_actions[-n_step_history:], 2)}, and the corresponding achieved number of conversions of each step is: {historical_today_won_values[-n_step_history:]}.\
                From the 0-th timestep tp the current timestep, the total acquired number of conversions is {sum(historical_today_won_values)}.\
                {ATTENTION}\
        <|im_end|>\n\
        <|im_start|>assistant\n \
            Let me solve this step by step. Now, it is my chain of thought:"""

    return Prompt_Template


def extract_bid(response: str) -> float:
    answer_str = response
    pattern = r'<answer>(.*?)</answer>'

    match = re.finditer(pattern, answer_str)
    matches = list(match)

    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None

    try:
        alpha = float(final_answer)
    except:
        alpha = None

    return alpha
