import numpy as np

task = 'cot_dt' 

task_description_cot_dt =  "You are an auto-bidding agent determining the optimal bidding parameter for the advertiser. \
        There are 48 timesteps of a day, the aim is to maximize the total acquired number of conversions with a lower realized CPA."

TASK_DESCRIPTION = task_description_cot_dt

ATTENTION_cot_dt_final = "You should summarize the history and then reason for the best future adjustment direction. Here are some basic knowledge:\
                1. You should carefully and sufficiently spend your budget but do not spend all the budget too early; \
                2. The realized CPA is calculated as (total_spent_budget / total_acquired_number_of_conversions), where total_spent_budget = budget * (1 - current_budget_remaining_ratio).\
                If you think the realized CPA would be bigger than the CPA constraint when spent all the budget, you should decrease the bidding parameter. \
                As a part of summarization, you need to output the current timestep's cpa_tatio = realized_cpa/cpa_constraint in <ratio></ratio> tags, e.g., <ratio>1.0</ratio>.\
                After your summarization and reasoning, you MUST output the direction in <answer></answer> tags with only three choices at the end of your response: <answer>1</answer>indicates you are sure increasing the bidding parameter is better, <answer>-1</answer> indicates you are sure decreasing the parameter is better, and <answer>0</answer> indicates you are uncertain about the optimal adjustment direction.\
                "

def format_states_for_cot(states):
    # 定义每个维度对应的键
    keys = [
        "timesteps remaining ratio",
        "budget remaining ratio",
        "predicted impression value",
    ]

    # 初始化一个字典来存储结果
    result = {key: [] for key in keys}

    # 遍历每个state并填充到对应的键中
    for state in states:
        for i, value in enumerate(state[:2]):
            result[keys[i]].append(value)

        # 计算当前流量总价值：pValue * pValue_num
        result["predicted impression value"].append(int(state[12]*state[13]))


    # 格式化输出文本
    output = []
    for key in keys:
        output.append(f"{key}: {result[key]}")

    return "\n".join(output)




def format_states(states):
    # 定义每个维度对应的键
    keys = [
        "remaining_timesteps/48",
        "remaining_budget/budget",
        "average bid over past timesteps",
        f"average bid over last {len(states)-1} timesteps",
        "average least cost required to win an impression over previous timesteps",
        "average pValues over past timesteps",
        "average numbers conversion over past timesteps",
        "average winning status in impression opportunities, where 1 represents winning and 0 represents not winning",
        f"average least cost over last {len(states)-1} timesteps",
        f"average pValues over last {len(states)-1} timesteps",
        f"average numbers conversion over past {len(states)-1} timesteps",
        f"average winning status over last {len(states)-1} timesteps",
        "current pValues mean",
        "current number of impression oppotunities",
        f"last {len(states)-1} timesteps' total number of impression oppotunities",
        "previous timesteps' total number of impression oppotunities"
    ]

    # 初始化一个字典来存储结果
    result = {key: [] for key in keys}

    # 遍历每个state并填充到对应的键中
    for state in states:
        for i, value in enumerate(state):
            result[keys[i]].append(value)

    # 格式化输出文本
    if task=='cot_dt':
        current_output = []
        historical_output = []

        current_keys = [
        "remaining_timesteps/48",
        "remaining_budget/budget",
        "average least cost required to win an impression over previous timesteps",
        "average numbers conversion over past timesteps",
        "average winning status in impression opportunities, where 1 represents winning and 0 represents not winning",
        ]
        historical_keys = [
            f"average bid over last {len(states)-1} timesteps",
            f"average least cost over last {len(states)-1} timesteps",
            f"average numbers conversion over past {len(states)-1} timesteps",
            f"average winning status over last {len(states)-1} timesteps",
        ]

        for key in current_keys:
            current_output.append(f"{key}: {result[key]}")
        for key in historical_keys:
            historical_output.append(f"{key}: {result[key]}")
            
        return "Historical state:\n" + "\n".join(historical_output) + "Current state:\n" + "\n".join(current_output)

    else:
        output = []
        for key in keys:
            output.append(f"{key}: {result[key]}")

        return "\n".join(output)


def build_promt(history_info, current_state, budget, CPA, expected_rtg, current_won_conversions):
    concat_states = np.concatenate([np.array(history_info["state"]).reshape(-1, len(current_state)), np.array(current_state).reshape(1, -1)]).tolist()

    ATTENTION = ATTENTION_cot_dt_final
    Few_Shot_Example = None

    timestep_start = 48 - int(concat_states[0][0] * 48)
    timestep_end = 48 - int(concat_states[-1][0] * 48)

    Prompt_Template = f"""\
        <|im_start|>\
            system\n\
                {TASK_DESCRIPTION}\
        <|im_end|>\n\
        <|im_start|>\
            Advertiser\n \
                The advertiser's budget is {budget} with a CPA constraint {CPA}.\
                Its historical {len(concat_states)-1} timesteps and current timestep's states' change along with time (i.e., from timestep {timestep_start} to timestep {timestep_end}) are: \n{format_states_for_cot(concat_states)}.\
                For each historical timestep, their corresponding bidding parameter is: {history_info["action"]}, and the corresponding achieved number of conversions of each step is: {history_info["reward"]}.\
                From the 0-th timestep to the current timestep, the total acquired conversions is {current_won_conversions}.
                {ATTENTION}
                {Few_Shot_Example}
        <|im_end|>\n\
        <|im_start|>assistant \
                Let me solve this step by step. Now, it is my chain of thought:\n"""

    return Prompt_Template
