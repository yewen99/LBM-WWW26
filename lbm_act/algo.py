import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from lbm_act.utils import DEFAULT_DEVICE, compute_batched, update_exponential_moving_average

EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class LBM_ACT_LEARNER(nn.Module):
    def __init__(self, policy, tokenizer, optimizer_factory, max_steps,
                 tau, beta, discount=0.99, alpha=0.005):
        super().__init__()
        self.policy = policy.to(DEFAULT_DEVICE)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha

    def update_policy_Text_RSA_emb(self, states, actions, rtgs, timesteps, attention_mask):
        states, actions, rtgs, timesteps, attention_masks = states.cuda(), actions.cuda(), rtgs.cuda(), timesteps.cuda(), attention_mask.cuda()
        
        # ----------- Build the CoT High-level Guidance ------------ #
        directions = actions[:, -1] / actions[:, -2] 
        text_prompts = []
        for i in range(actions.shape[0]):
            if directions[i] > 1.:
                text_prompt = 'You should increase the bidding parameter.'
            elif directions[i] < 1.:
                if directions[i] < 0:  # padding action = -10
                    text_prompt = 'This is the first timestep of bidding.'
                else:
                    text_prompt = 'You should decrease the bidding parameter.'
            else: 
                text_prompt = 'Uncertain of optimal bid adjustment direction.'
            
            if np.random.rand() < 0.05:
                text_prompt = 'Uncertain of optimal bid adjustment direction.'
            
            text_prompts.append(text_prompt)
        
        # randomly drop some cot to be uncertain

            
        inputs = self.tokenizer(text_prompts, return_tensors="pt", padding=True)
        text_input_ids, text_attention_mask = inputs.input_ids.cuda(), inputs.attention_mask.cuda()

        # ----------- Token Embedding Layer is based on Qwen ------------- #
        with torch.no_grad():
            text_prompt_embs = self.policy.model.embed_tokens(text_input_ids)
        
        # ---------- Forward and Get pred actions --------- #
        pred_actions, embedding = self.policy.forward_Text_RSA_emb(states, actions, rtgs, timesteps, attention_mask, text_prompt_embs=text_prompt_embs)
        
        assert pred_actions.shape[1] != 1
        act_dim = pred_actions.shape[-1]
        pred_actions = pred_actions.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        actions = actions.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        positive_action_indices = actions >= 0
        pred_actions = pred_actions[positive_action_indices]
        actions = actions[positive_action_indices]
        
        action_loss = torch.mean((pred_actions - actions) ** 2)
        loss = action_loss

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()
        return loss.item()  
