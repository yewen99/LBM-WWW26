import random
import numpy as np
import torch
import logging
import sys
import csv
from datetime import datetime
import json
from pathlib import Path
import string
import torch.nn as nn


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def config_logging(log_file="main.log"):
    date_format = '%Y-%m-%d %H:%M:%S'
    log_format = '%(asctime)s: [%(levelname)s]: %(message)s'
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler], force=True)

def convert_to_list(state):
    if isinstance(state, list):
        return state
    elif isinstance(state, np.ndarray):
        if state.ndim != 2:
            raise ValueError("Array must be 2-dimensional")
        return state.tolist()
    elif isinstance(state, torch.Tensor):
        if state.ndim != 2:
            raise ValueError("Tensor must be 2-dimensional")
        return state.tolist()
    else:
        raise ValueError("Unsupported state type")

def format_states(states):
    keys = [
        "timesteps left ratio",
        "budget left ratio",
        "average bid over past timesteps",
        "average bid over last three timesteps",
        "average least cost required to win an impression over previous timesteps",
        "average pValues over past timesteps",
        "average numbers conversion over past timesteps",
        "average winning status in impression opportunities, where 1 represents winning and 0 represents not winning",
        "average least cost over last three timesteps",
        "average pValues over last three timesteps",
        "average numbers conversion over past three timesteps",
        "average winning status over last three timesteps",
        "current pValues mean",
        "current number of impression oppotunities",
        "last three timesteps' total number of impression oppotunities",
        "previous timesteps' total number of impression oppotunities"
    ]

    result = {key: [] for key in keys}

    for state in states:
        for i, value in enumerate(state):
            result[keys[i]].append(float("{:.3f}".format(value.item())))

    output = []
    for key in keys:
        output.append(f"{key}: {result[key]}")

    return "\n".join(output)

def build_prompt(state, action, reward, rtg, timestep, budget, cpa):
    TASK_DESCRIPTION = "You are an auto-bidding agent determining the optimal bidding parameter for the advertiser, whic is a simplified target CPA task. \
    There are 48 timesteps of a day, the aim is to maximize the total number of conversions by spending the budget while satisfying the CPA constraint $C$ for the whole day.\
    If the realized CPA (calculated as spent_budget / total_number_of_conversions) exceeds the CPA constraint $C$, a penalty will be applied.\
    Thus, you need make a balance between the conversions and CPA to achieve a best score by adjusting the bidding parameter. \
    Note that, the bid for each impression oppotunity is calculated by bid = bidding_parameter * pValue, where pValue is the conversion probality of the impression."

    ATTENTION = f"You are expected to achieve {rtg[-2]} conversions within the remaining timesteps, while adhering to the CPA constraint."

    Prompt_Template = f"""\
        <|im_start|>\
            system\n\
                {TASK_DESCRIPTION}\
        <|im_end|>\n\
        <|im_start|>\
            Advertiser\n \
                The advertiser's budget is {budget} with a CPA constrain {cpa}.\
                Its historical {len(state)-1} timesteps and current timestep's states from timestep {timestep[0]} to timestep {timestep[-1]} are: \n{format_states(state)}.\
                For each historical timestep, their corresponding bidding parameter is: {action[:-1]}, and the corresponding achieved number of conversions of each step is: {reward[:-1]}.\
                {ATTENTION}\
        <|im_end|>\n\
        <|im_start|>assistant\n \
            The optimal bidding paramter is:"""
    
    return Prompt_Template

def get_prompts(states, actions, rewards, rtgs, timesteps, budgets, cpas):
    prompts = []

    for i in range(len(states)):
        state, action, reward, rtg, timestep, budget, cpa = states[i], actions[i], rewards[i], rtgs[i], timesteps[i], budgets[i], cpas[i]
        prompt = build_prompt(state, action, reward, rtg, timestep, budget, cpa)
        prompts.append(prompt)
    
    return prompts


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)



def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x
