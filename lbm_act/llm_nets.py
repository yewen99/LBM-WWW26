import torch
import torch.nn as nn
from lbm_act.utils import mlp
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PretrainedConfig, Qwen2Config, AutoModelForCausalLM
import logging
import os
import numpy as np
import random

logger = logging.getLogger(__name__)
torch.manual_seed(42)

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

def ks_mlp(input_size, hidden_size, output_size=1, output_activation=nn.Identity(), hidden_activation=nn.ELU()):
    sizes = [input_size] + list(hidden_size) + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = hidden_activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act]
    return nn.Sequential(*layers)


class BiddingModelConfig(Qwen2Config):

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    output_hidden_states = True

    output_head_size = 1
    output_head_hidden_activation = nn.ReLU()
    output_head_dropout_ratio = 0.0

    input_state_dim = 16
    rtg_scale = 2000
    state_mean = 0.
    state_std = 1.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k,v in kwargs.items():
            setattr(self, k, v)


class LLMPolicy(PreTrainedModel):
    config_class = BiddingModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(config.model_name, output_hidden_states=config.output_hidden_states) 
        hidden_size = self.model.config.hidden_size
        self.hidden_size = hidden_size
        print("Base model hidden size: ", hidden_size)

        self.output_heads = nn.ModuleDict()
        self.output_heads["output_head3"] = ks_mlp(input_size=hidden_size, 
                               hidden_size=[hidden_size, hidden_size, hidden_size], 
                               output_size=1, 
                               output_activation=nn.Identity(), 
                               hidden_activation=nn.ReLU())

        self.time_dim = 8
        self.max_ep_len = 48
        self.horizon = 10
        self.state_dim = config.input_state_dim
        self.action_dim = 1
        self.scale = config.rtg_scale
        self.target_return = 1.
        self.state_mean = config.state_mean
        self.state_std = config.state_std
        self.input_timestep_emb_layer = nn.Embedding(self.max_ep_len, self.time_dim)
        
        self.input_rtg_emb_layer = ks_mlp(input_size=1, 
                               hidden_size=[hidden_size, hidden_size], 
                               output_size=hidden_size, 
                               output_activation=nn.Identity(), 
                               hidden_activation=nn.ReLU())

        self.input_state_emb_layer = ks_mlp(input_size=config.input_state_dim, 
                               hidden_size=[hidden_size, hidden_size], 
                               output_size=hidden_size, 
                               output_activation=nn.Identity(), 
                               hidden_activation=nn.ReLU())
        
        self.input_action_emb_layer = ks_mlp(input_size=1, 
                               hidden_size=[hidden_size, hidden_size], 
                               output_size=hidden_size, 
                               output_activation=nn.Identity(), 
                               hidden_activation=nn.ReLU())
        
        self.trans_rtg_emb = torch.nn.Linear(self.time_dim+hidden_size, hidden_size)
        self.trans_state_emb = torch.nn.Linear(self.time_dim+hidden_size, hidden_size)
        self.trans_action_emb = torch.nn.Linear(self.time_dim+hidden_size, hidden_size)

        self.init_eval()

                        
    @torch.jit.ignore
    def log(self, msg):
        logger.warning_once(msg)
        logger.debug(msg)
    
    def forward_Text_RSA_emb(self, states, actions, rtgs, timesteps, attention_mask=None, text_prompt_embs=None):
        batch_size, seq_len = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        timestep_emb = self.input_timestep_emb_layer(timesteps)

        rtg_emb = self.input_rtg_emb_layer(rtgs)
        state_emb = self.input_state_emb_layer(states)
        action_emb = self.input_action_emb_layer(actions)
        
        rtg_emb = torch.cat((rtg_emb, timestep_emb), dim=-1)
        state_emb = torch.cat((state_emb, timestep_emb), dim=-1)
        action_emb = torch.cat((action_emb, timestep_emb), dim=-1)

        rtg_emb = self.trans_rtg_emb(rtg_emb)
        state_emb = self.trans_state_emb(state_emb)
        action_emb = self.trans_action_emb(action_emb)

        stacked_embs = torch.stack(
            (rtg_emb, state_emb, action_emb), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_len, self.hidden_size)

        stacked_attention_mask = torch.stack(
            ([attention_mask for _ in range(3)]), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_len).to(stacked_embs.dtype)

        text_rsa_emb = torch.concat((text_prompt_embs, stacked_embs), dim=1)
        
        outputs = self.model.forward(inputs_embeds=text_rsa_emb)

        hidden_states = outputs[0]

        output_embs = hidden_states[:, text_prompt_embs.shape[1]:].reshape(batch_size, seq_len, 3, self.hidden_size).permute(0, 2, 1, 3)
        output_state_embs = output_embs[:, -2]
        output = self.output_heads["output_head3"](output_state_embs)  

        return output, output_embs

    def get_action(self, states, actions, returns_to_go, timesteps, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.action_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.horizon is not None:
            states = states[:, -self.horizon:]
            actions = actions[:, -self.horizon:]
            returns_to_go = returns_to_go[:, -self.horizon:]
            timesteps = timesteps[:, -self.horizon:]

            attention_mask = torch.cat([torch.zeros(self.horizon - states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.horizon - states.shape[1], self.state_dim),
                             device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.ones((actions.shape[0], self.horizon - actions.shape[1], self.action_dim),
                             device=actions.device)*(-10.), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.horizon - returns_to_go.shape[1], 1),
                             device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.horizon - timesteps.shape[1]), device=timesteps.device),
                 timesteps],
                dim=1).to(device=states.device, dtype=torch.long)
        else:
            attention_mask = None

        action_preds, _ = self.forward_w_IO_emb(
            states=states, actions=actions, rtgs=returns_to_go, timesteps=timesteps, attention_mask=attention_mask)
        return action_preds[0, -1]

    def take_actions(self, state, target_return=None, pre_reward=None):
        self.eval()
        if self.eval_states is None:
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            ep_return = target_return.to(self.device) if target_return is not None else self.target_return
            self.eval_target_return = torch.tensor(ep_return, dtype=torch.float32).reshape(1, 1).to(self.device)

        else:
            assert pre_reward is not None
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0).to(self.device)
            
            self.eval_rewards[-1] = pre_reward

            pred_return = self.eval_target_return[0, -1] - (pre_reward / self.scale)
            self.eval_target_return = torch.cat([self.eval_target_return, pred_return.reshape(1, 1)], dim=1)

            self.eval_timesteps = self.eval_timesteps.to(self.device)
            self.eval_timesteps = torch.cat(
                [self.eval_timesteps, torch.ones((1, 1), dtype=torch.long).to(self.device) * self.eval_timesteps[:, -1] + 1], dim=1)

        self.eval_actions = torch.cat([self.eval_actions.to(self.device), torch.zeros(1, self.action_dim).to(self.device)], dim=0)        
        self.eval_rewards = torch.cat([self.eval_rewards.to(self.device), torch.zeros(1).to(self.device)])

        action = self.get_action(
            (self.eval_states.to(dtype=torch.float32) - torch.tensor(self.state_mean).to(self.device)) / torch.tensor(self.state_std).to(self.device),
            self.eval_actions.to(dtype=torch.float32),
            self.eval_target_return.to(dtype=torch.float32),
            self.eval_timesteps.to(dtype=torch.long),
        )
        self.eval_actions[-1] = action
        action = action.item()
        return action

    def init_eval(self):
        self.eval_states = None
        self.eval_actions = torch.zeros((0, self.action_dim), dtype=torch.float32).to(self.device)
        self.eval_rewards = torch.zeros(0, dtype=torch.float32).to(self.device)
        self.eval_costs = torch.zeros(0, dtype=torch.float32).to(self.device)

        self.eval_target_return = None
        self.eval_target_ctg = None

        self.eval_timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1).to(self.device)

        self.eval_episode_return, self.eval_episode_length = 0, 0