"""LLM-based policy network for LBM-Act.

The model takes a Decision-Transformer–style ``(R, S, A)`` sequence together
with a textual high-level guide (CoT) and predicts the next bidding
parameter. It builds on top of a frozen Qwen2.5 backbone.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel, Qwen2Config

from lbm_act.utils import mlp


logger = logging.getLogger(__name__)


# Padding value for missing actions in the trajectory; consistent with the
# replay buffer in :mod:`lbm_act.seq_dataset`.
ACTION_PADDING_VALUE = -10.0


def _build_mlp(input_size: int, hidden_size: int, output_size: int = 1) -> nn.Sequential:
    """Convenience: 4-layer MLP used to project tokens into the LLM hidden space."""
    return mlp(
        [input_size, hidden_size, hidden_size, hidden_size, output_size],
        activation=nn.ReLU,
        output_activation=None,
    )


class BiddingModelConfig(Qwen2Config):
    """Configuration for :class:`LLMPolicy`."""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_hidden_states: bool = True

    input_state_dim: int = 16
    rtg_scale: float = 2000.0
    state_mean: float | np.ndarray = 0.0
    state_std: float | np.ndarray = 1.0

    # DT-style hyper-parameters
    max_ep_len: int = 48
    horizon: int = 10
    time_dim: int = 8
    action_dim: int = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)


class LLMPolicy(PreTrainedModel):
    """Policy network: ``(text guide, R, S, A) -> action``."""

    config_class = BiddingModelConfig

    def __init__(self, config: BiddingModelConfig):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(
            config.model_name,
            output_hidden_states=config.output_hidden_states,
        )
        hidden_size = self.model.config.hidden_size
        self.hidden_size = hidden_size

        # Output head: hidden -> action
        self.output_head = mlp(
            [hidden_size, hidden_size, hidden_size, hidden_size, 1],
            activation=nn.ReLU,
            output_activation=None,
        )

        # DT hyper-parameters
        self.time_dim = config.time_dim
        self.max_ep_len = config.max_ep_len
        self.horizon = config.horizon
        self.state_dim = config.input_state_dim
        self.action_dim = config.action_dim
        self.scale = config.rtg_scale
        self.state_mean = config.state_mean
        self.state_std = config.state_std

        # Token / DT embeddings
        self.input_timestep_emb_layer = nn.Embedding(self.max_ep_len, self.time_dim)
        self.input_rtg_emb_layer = _build_mlp(1, hidden_size, hidden_size)
        self.input_state_emb_layer = _build_mlp(self.state_dim, hidden_size, hidden_size)
        self.input_action_emb_layer = _build_mlp(1, hidden_size, hidden_size)

        # Project (token + time) embeddings into the LLM hidden space.
        self.trans_rtg_emb = nn.Linear(self.time_dim + hidden_size, hidden_size)
        self.trans_state_emb = nn.Linear(self.time_dim + hidden_size, hidden_size)
        self.trans_action_emb = nn.Linear(self.time_dim + hidden_size, hidden_size)

        self._init_eval_state()

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward_text_rsa(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        text_prompt_embs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the (text + R/S/A) sequence.

        Returns predicted actions of shape ``(B, K, action_dim)`` and the
        per-token output embeddings.
        """
        batch_size, seq_len = states.shape[:2]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=states.device)

        timestep_emb = self.input_timestep_emb_layer(timesteps)
        rtg_emb = self.input_rtg_emb_layer(rtgs)
        state_emb = self.input_state_emb_layer(states)
        action_emb = self.input_action_emb_layer(actions)

        rtg_emb = self.trans_rtg_emb(torch.cat((rtg_emb, timestep_emb), dim=-1))
        state_emb = self.trans_state_emb(torch.cat((state_emb, timestep_emb), dim=-1))
        action_emb = self.trans_action_emb(torch.cat((action_emb, timestep_emb), dim=-1))

        # Interleave (R, S, A) along the time axis.
        stacked_embs = (
            torch.stack((rtg_emb, state_emb, action_emb), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.hidden_size)
        )

        if text_prompt_embs is not None:
            inputs_embeds = torch.cat((text_prompt_embs, stacked_embs), dim=1)
        else:
            inputs_embeds = stacked_embs

        outputs = self.model.forward(inputs_embeds=inputs_embeds)
        hidden_states = outputs[0]

        prefix_len = text_prompt_embs.shape[1] if text_prompt_embs is not None else 0
        rsa_hidden = hidden_states[:, prefix_len:].reshape(batch_size, seq_len, 3, self.hidden_size).permute(0, 2, 1, 3)
        # index 0=R, 1=S, 2=A; the action prediction is conditioned on the state token (index 1).
        state_hidden = rsa_hidden[:, 1]
        pred_actions = self.output_head(state_hidden)
        return pred_actions, rsa_hidden

    # ------------------------------------------------------------------ #
    # Inference utilities                                                 #
    # ------------------------------------------------------------------ #
    def _init_eval_state(self) -> None:
        """Reset the streaming buffers used by :meth:`take_actions`."""
        self.eval_states = None
        self.eval_actions = torch.zeros((0, self.action_dim), dtype=torch.float32, device=self.device)
        self.eval_rewards = torch.zeros(0, dtype=torch.float32, device=self.device)
        self.eval_target_return: Optional[torch.Tensor] = None
        self.eval_timesteps = torch.tensor(0, dtype=torch.long, device=self.device).reshape(1, 1)

    def init_eval(self) -> None:
        """Public alias for resetting the inference state."""
        self._init_eval_state()

    def organize_rsa_inference(
        self,
        state: np.ndarray,
        pre_reward: Optional[float] = None,
        target_return: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append a new state to the eval buffers and pad to ``self.horizon``.

        Returns a tuple ``(states, actions, rtgs, timesteps, attention_mask)``
        ready to be consumed by :meth:`forward_text_rsa`.
        """
        device = self.device
        if self.eval_states is None:
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim).to(device)
            ep_return = float(target_return) if target_return is not None else 1.0
            self.eval_target_return = torch.tensor(ep_return, dtype=torch.float32, device=device).reshape(1, 1)
        else:
            assert pre_reward is not None, "pre_reward is required after the first step"
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim).to(device)
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0)

            pred_return = self.eval_target_return[0, -1] - (pre_reward / self.scale)
            self.eval_target_return = torch.cat([self.eval_target_return, pred_return.reshape(1, 1)], dim=1)

            self.eval_timesteps = torch.cat(
                [self.eval_timesteps, torch.ones((1, 1), dtype=torch.long, device=device) * self.eval_timesteps[:, -1] + 1],
                dim=1,
            )

        # Append placeholders for the action / reward of the current step.
        self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.action_dim, device=device)], dim=0)
        self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1, device=device)])

        states = self.eval_states.reshape(1, -1, self.state_dim).float()
        actions = self.eval_actions.reshape(1, -1, self.action_dim).float()
        rtgs = self.eval_target_return.reshape(1, -1, 1).float()
        timesteps = self.eval_timesteps.reshape(1, -1).long()

        # Normalise state.
        state_mean = torch.as_tensor(self.state_mean, dtype=torch.float32, device=device)
        state_std = torch.as_tensor(self.state_std, dtype=torch.float32, device=device)
        states = (states - state_mean) / state_std

        # Truncate / pad to the policy horizon.
        h = self.horizon
        states = states[:, -h:]
        actions = actions[:, -h:]
        rtgs = rtgs[:, -h:]
        timesteps = timesteps[:, -h:]

        cur_len = states.shape[1]
        pad = h - cur_len
        attention_mask = torch.cat(
            [torch.zeros(pad, dtype=torch.long, device=device), torch.ones(cur_len, dtype=torch.long, device=device)]
        ).reshape(1, -1)

        if pad > 0:
            states = torch.cat([torch.zeros((1, pad, self.state_dim), device=device), states], dim=1)
            actions = torch.cat(
                [torch.full((1, pad, self.action_dim), ACTION_PADDING_VALUE, device=device), actions], dim=1,
            )
            rtgs = torch.cat([torch.zeros((1, pad, 1), device=device), rtgs], dim=1)
            timesteps = torch.cat([torch.zeros((1, pad), dtype=torch.long, device=device), timesteps], dim=1)

        return states, actions, rtgs, timesteps, attention_mask
