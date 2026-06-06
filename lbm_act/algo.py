"""Training algorithm for LBM-Act.

The ``LbmActLearner`` wraps an :class:`lbm_act.llm_nets.LLMPolicy` and
performs supervised regression of the ground-truth bidding parameter on top
of a high-level CoT (Chain-of-Thought) guide that is derived from the
trajectory itself during training.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from lbm_act.utils import DEFAULT_DEVICE


# Padding value used for missing actions in the trajectory dataset.
ACTION_PADDING_VALUE = -10.0
# Probability of replacing a deterministic guide with the "uncertain" guide
# during training, to teach the policy to handle ambiguity at inference time.
UNCERTAIN_DROPOUT_PROB = 0.05

# Canonical high-level guide vocabulary used by both training and inference.
GUIDE_INCREASE = "You should increase the bidding parameter."
GUIDE_DECREASE = "You should decrease the bidding parameter."
GUIDE_UNCERTAIN = "Uncertain of optimal bid adjustment direction."
GUIDE_FIRST_STEP = "This is the first timestep of bidding."


def _build_guide(prev_action: float, curr_action: float) -> str:
    """Build a CoT high-level guide from two consecutive bidding parameters."""
    if prev_action < 0:  # padding placeholder => first valid timestep
        return GUIDE_FIRST_STEP
    if curr_action > prev_action:
        return GUIDE_INCREASE
    if curr_action < prev_action:
        return GUIDE_DECREASE
    return GUIDE_UNCERTAIN


class LbmActLearner(nn.Module):
    """Supervised learner for the LBM-Act policy."""

    def __init__(
        self,
        policy: nn.Module,
        tokenizer,
        optimizer_factory: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer],
        max_steps: int,
        device: torch.device = DEFAULT_DEVICE,
    ):
        super().__init__()
        self.device = device
        self.policy = policy.to(device)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)

    def _build_guide_prompts(self, actions: torch.Tensor) -> list[str]:
        """Construct per-sample guide prompts from the action sequence.

        ``actions`` has shape ``(B, K, 1)``; the guide compares the last two
        timesteps of each trajectory.
        """
        prev = actions[:, -2, 0].detach().cpu().numpy()
        curr = actions[:, -1, 0].detach().cpu().numpy()

        prompts: list[str] = []
        for prev_a, curr_a in zip(prev, curr):
            if np.random.rand() < UNCERTAIN_DROPOUT_PROB:
                prompts.append(GUIDE_UNCERTAIN)
            else:
                prompts.append(_build_guide(float(prev_a), float(curr_a)))
        return prompts

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> float:
        """Run one optimisation step on a batch and return the scalar loss."""
        states = states.to(self.device)
        actions = actions.to(self.device)
        rtgs = rtgs.to(self.device)
        timesteps = timesteps.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # ------------------- Build CoT high-level guide ------------------- #
        text_prompts = self._build_guide_prompts(actions)
        inputs = self.tokenizer(text_prompts, return_tensors="pt", padding=True)
        text_input_ids = inputs.input_ids.to(self.device)

        # Token embedding lookup; gradients are not propagated through the
        # text branch — only through the projection layers and the LLM body.
        with torch.no_grad():
            text_prompt_embs = self.policy.model.embed_tokens(text_input_ids)

        # ------------------------- Forward pass --------------------------- #
        pred_actions, _ = self.policy.forward_text_rsa(
            states, actions, rtgs, timesteps, attention_mask,
            text_prompt_embs=text_prompt_embs,
        )

        assert pred_actions.shape[1] != 1, "expected a sequence-length > 1"
        act_dim = pred_actions.shape[-1]
        flat_mask = attention_mask.reshape(-1) > 0
        pred_flat = pred_actions.reshape(-1, act_dim)[flat_mask]
        target_flat = actions.reshape(-1, act_dim)[flat_mask]

        # Drop padded (negative) actions from the loss.
        valid = target_flat[:, 0] >= 0
        loss = torch.mean((pred_flat[valid] - target_flat[valid]) ** 2)

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()
        return loss.item()
