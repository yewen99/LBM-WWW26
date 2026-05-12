import os
import argparse
import torch
from transformers import AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_default_dtype(torch.float32)
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from lbm_act.utils import set_seed, get_prompts
from lbm_act.seq_dataset import EpisodeReplayBuffer
from torch.utils.data import DataLoader, WeightedRandomSampler
from lbm_act.llm_nets import BiddingModelConfig, LLMPolicy
from lbm_act.algo import LBM_ACT_LEARNER
from tqdm import tqdm
import wandb
import sys

set_seed(0)

argp = argparse.ArgumentParser()
argp.add_argument('--outputs_path', default="./ckpt", help='save model checkpoint directory')
argp.add_argument('--data_path', default="./data", help='path to the preprocessed trajectory data directory')
argp.add_argument('--resume_from_pretrain', default=False, help='whether loading a pretrained model')
argp.add_argument('--pretrain_model_path', default="", help='path to pretrained model checkpoint')
argp.add_argument('--sparse_data', action='store_true', help='whether use auctionNet-sparse')
argp.add_argument('--batch_size', type=int, default=64, help='training batch size')
argp.add_argument('--step_num', type=int, default=100000, help='total training steps')
argp.add_argument('--lr', type=float, default=5e-6, help='learning rate')
argp.add_argument('--log_every_step', type=int, default=100, help='logging interval')
argp.add_argument('--save_every_step', type=int, default=10000, help='checkpoint saving interval')
argp.add_argument('--debug', type=bool, default=True, help='disable wandb logging')
args = argp.parse_args()

if not args.debug:
    wandb.init(project='llm+mlp')

# ---------------------------- Setup the Model and Learner ---------------------------- #
mconf = BiddingModelConfig()

model = LLMPolicy(mconf)
tokenizer = AutoTokenizer.from_pretrained(mconf.model_name)

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d/%H-%M-%S")
ckpt_path = os.path.join(os.path.join(args.outputs_path, f"{timestamp}"), mconf.model_name)
os.makedirs(ckpt_path, exist_ok=True)
print(f'{ckpt_path=}', flush=True)
writer = SummaryWriter(log_dir=ckpt_path)

lbm_act_learner = LBM_ACT_LEARNER(
    policy=model,
    tokenizer=tokenizer,
    optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.lr),
    max_steps=args.step_num,
    tau=0.9,
    beta=3.,
    alpha=0.005,
    discount=0.99
)

if args.resume_from_pretrain:
    checkpoint = torch.load(args.pretrain_model_path)
    lbm_act_learner.load_state_dict(checkpoint)
    print(f'loaded pretrained model from {args.pretrain_model_path}')


# -------------------------------------- Dataset -------------------------------------- #
if args.sparse_data:
    rtg_scale = 100.
else:
    rtg_scale = 1500

replay_buffer = EpisodeReplayBuffer(state_dim=16, act_dim=1, K=10, data_path=args.data_path, load_preprocessed_data=True, scale=rtg_scale, final_stage=args.sparse_data)
sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=args.step_num * args.batch_size, replacement=True)
train_dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=args.batch_size)


# -------------------------------------- Train -------------------------------------- # 
disable_tqdm = not sys.stdout.isatty()
with tqdm(train_dataloader, desc=f"Training", disable=disable_tqdm) as pbar:
    total_loss = 0.0
    for step, batch in enumerate(pbar):
        states, actions, rewards, dones, rtgs, timesteps, attention_masks, budgets, cpas, costs = batch

        if args.sparse_data:
            actions = torch.clamp(actions, max=500.0)
        else:
            actions = torch.clamp(actions, max=54.0)
            
        policy_loss = lbm_act_learner.update_policy_Text_RSA_emb(states, actions, rtgs[:, :-1], timesteps, attention_masks)  # <----- LBM_ACT
        
        total_loss += policy_loss

        if (step + 1) % args.log_every_step == 0:
            average_loss = total_loss / args.log_every_step
            print(f"Average loss at step {step + 1}: {average_loss}", flush=True)
            if not args.debug:
                wandb.log({'Average Loss': average_loss}, step=step + 1)
            writer.add_scalar('train/loss', average_loss, step + 1)
            total_loss = 0.0

        if step % args.save_every_step == 0:
            torch.save(lbm_act_learner.state_dict(), os.path.join(ckpt_path, 'final.pt'))

if not args.debug:
    wandb.finish()
