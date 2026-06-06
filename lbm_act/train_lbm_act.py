"""Training entrypoint for LBM-Act.

Example
-------
::

    python -m lbm_act.train_lbm_act \\
        --data_path /path/to/preprocessed/data \\
        --outputs_path ./ckpt \\
        --sparse_data
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from lbm_act.algo import LbmActLearner
from lbm_act.llm_nets import BiddingModelConfig, LLMPolicy
from lbm_act.seq_dataset import EpisodeReplayBuffer
from lbm_act.utils import set_seed


# RTG normalisation scale; the dense and sparse AuctionNet datasets have very
# different reward magnitudes, so each ships with its own scale.
RTG_SCALE_DENSE = 1500.0
RTG_SCALE_SPARSE = 100.0
# Action clipping used as a soft outlier removal during training.
ACTION_CLIP_DENSE = 54.0
ACTION_CLIP_SPARSE = 500.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LBM-Act policy.")
    parser.add_argument("--outputs_path", default="./ckpt", help="checkpoint output directory")
    parser.add_argument("--data_path", required=True, help="path to the preprocessed trajectory data directory")
    parser.add_argument("--resume_from_pretrain", action="store_true", help="load a previously saved learner state dict")
    parser.add_argument("--pretrain_model_path", default="", help="path to the pretrained learner state dict")
    parser.add_argument("--sparse_data", action="store_true", help="use AuctionNet-sparse instead of AuctionNet-dense")
    parser.add_argument("--batch_size", type=int, default=64, help="training batch size")
    parser.add_argument("--step_num", type=int, default=100_000, help="total number of optimisation steps")
    parser.add_argument("--lr", type=float, default=5e-6, help="learning rate")
    parser.add_argument("--seq_len", type=int, default=10, help="sequence length K of the DT-style window")
    parser.add_argument("--state_dim", type=int, default=16, help="environment state dimension")
    parser.add_argument("--log_every_step", type=int, default=100, help="logging interval")
    parser.add_argument("--save_every_step", type=int, default=10_000, help="checkpoint saving interval")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--use_wandb", action="store_true", help="enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="lbm-act", help="W&B project name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    torch.set_default_dtype(torch.float32)

    if args.use_wandb:
        import wandb  # local import: optional dependency

        wandb.init(project=args.wandb_project, config=vars(args))

    # ------------------------- Model & learner ------------------------- #
    mconf = BiddingModelConfig(input_state_dim=args.state_dim)
    model = LLMPolicy(mconf)
    tokenizer = AutoTokenizer.from_pretrained(mconf.model_name)

    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    ckpt_path = os.path.join(args.outputs_path, timestamp, mconf.model_name)
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"[train] checkpoint path: {ckpt_path}", flush=True)

    writer = SummaryWriter(log_dir=ckpt_path)

    learner = LbmActLearner(
        policy=model,
        tokenizer=tokenizer,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.lr),
        max_steps=args.step_num,
    )

    if args.resume_from_pretrain:
        if not args.pretrain_model_path:
            raise ValueError("--pretrain_model_path must be set when --resume_from_pretrain is given")
        learner.load_state_dict(torch.load(args.pretrain_model_path))
        print(f"[train] loaded pretrained checkpoint from {args.pretrain_model_path}")

    # ----------------------------- Dataset ----------------------------- #
    rtg_scale = RTG_SCALE_SPARSE if args.sparse_data else RTG_SCALE_DENSE
    action_clip = ACTION_CLIP_SPARSE if args.sparse_data else ACTION_CLIP_DENSE

    replay_buffer = EpisodeReplayBuffer(
        state_dim=args.state_dim,
        act_dim=1,
        K=args.seq_len,
        data_path=args.data_path,
        load_preprocessed_data=True,
        scale=rtg_scale,
        sparse_data=args.sparse_data,
    )
    sampler = WeightedRandomSampler(
        replay_buffer.p_sample, num_samples=args.step_num * args.batch_size, replacement=True,
    )
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=args.batch_size)

    # ------------------------------ Train ------------------------------ #
    disable_tqdm = not sys.stdout.isatty()
    total_loss = 0.0
    with tqdm(dataloader, desc="Training", disable=disable_tqdm) as pbar:
        for step, batch in enumerate(pbar):
            states, actions, _rewards, _dones, rtgs, timesteps, attention_masks, *_ = batch
            actions = torch.clamp(actions, max=action_clip)

            loss = learner.update(states, actions, rtgs[:, :-1], timesteps, attention_masks)
            total_loss += loss

            if (step + 1) % args.log_every_step == 0:
                avg_loss = total_loss / args.log_every_step
                print(f"[train] step {step + 1}: avg loss = {avg_loss:.6f}", flush=True)
                writer.add_scalar("train/loss", avg_loss, step + 1)
                if args.use_wandb:
                    import wandb
                    wandb.log({"avg_loss": avg_loss}, step=step + 1)
                total_loss = 0.0

            if step % args.save_every_step == 0:
                torch.save(learner.state_dict(), os.path.join(ckpt_path, "final.pt"))

    torch.save(learner.state_dict(), os.path.join(ckpt_path, "final.pt"))
    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
