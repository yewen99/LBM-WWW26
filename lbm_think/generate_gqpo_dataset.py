"""Generate the GQPO training dataset for LBM-Think.

Given a trained LBM-Act policy + Q-value critic, sample CoT generations from
the LBM-Think candidate LLM, filter hallucinated / non-improving ones with
:func:`lbm_think.rule.expert_rule`, and write the survivors to a JSONL SFT
file consumable by LLaMA-Factory.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import pandas as pd

# These imports are deliberately delayed to keep ``--help`` fast.


def _build_messages(prompt: str, response: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def generate_gqpo_data(
    *,
    train_data_path: str,
    cot_llm_path: str,
    q_path: str,
    save_data_path: str,
    lbm_act_path: str,
    sparse_data: bool,
    target_pair_num: int = 1000,
    batch_size: int = 512,
    gpu_memory_utilization: float = 0.6,
    temperature: float = 0.5,
    max_tokens: int = 512,
) -> None:
    from vllm import LLM, SamplingParams

    from lbm_think.rule import expert_rule, load_llm_dt
    from evaluate.bidding_train_env.baseline.dt_baselines.dt_critics import load_Q_net

    train_dataset = pd.read_parquet(train_data_path)
    print(f"[gqpo] loaded {len(train_dataset)} prompts from {train_data_path}")
    questions = [entry[0]["content"] for entry in train_dataset["prompt"]]

    print(f"[gqpo] loading candidate LLM from {cot_llm_path}")
    cot_model = LLM(model=cot_llm_path, gpu_memory_utilization=gpu_memory_utilization)
    sampling = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens)

    print(f"[gqpo] loading Q-net ensemble from {q_path}")
    q_ensemble = [load_Q_net(q_path)]

    print(f"[gqpo] loading LBM-Act from {lbm_act_path}")
    lbm_act, tokenizer = load_llm_dt(sparse_data=sparse_data, policy_load_dir=lbm_act_path)

    pair_count = 0
    os.makedirs(os.path.dirname(os.path.abspath(save_data_path)) or ".", exist_ok=True)
    with open(save_data_path, "w") as f:
        while pair_count < target_pair_num:
            for i in range(0, len(questions), batch_size):
                if pair_count >= target_pair_num:
                    break

                batch = questions[i : i + batch_size]
                generations = cot_model.generate(batch, sampling)

                for idx, gen in enumerate(generations):
                    if pair_count >= target_pair_num:
                        break
                    text = gen.outputs[0].text
                    gt = train_dataset["reward_model"][i + idx]["ground_truth"]
                    if expert_rule(text, gt, q_ensemble, tokenizer, lbm_act):
                        f.write(json.dumps(_build_messages(batch[idx], text)) + "\n")
                        pair_count += 1

                print(f"[gqpo] kept {pair_count} / target {target_pair_num} (batch end {i + batch_size})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GQPO SFT data for LBM-Think.")
    parser.add_argument("--train_data_path", required=True, help="path to the prompt parquet (output of bidding_data.py)")
    parser.add_argument("--cot_llm_path", required=True, help="HuggingFace name or local path of the candidate LBM-Think LLM")
    parser.add_argument("--Q_path", required=True, dest="q_path", help="directory containing the Q-net checkpoint")
    parser.add_argument("--llm_act_path", required=True, help="path to the trained LBM-Act state dict")
    parser.add_argument("--sparse_data", action="store_true", help="use AuctionNet-sparse normalisation")
    parser.add_argument("--save_data_path", default="./training_data.jsonl", help="output JSONL file")
    parser.add_argument("--target_pair_num", type=int, default=1000, help="number of (prompt, response) pairs to retain")
    parser.add_argument("--batch_size", type=int, default=512, help="generation batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_gqpo_data(
        train_data_path=args.train_data_path,
        cot_llm_path=args.cot_llm_path,
        q_path=args.q_path,
        save_data_path=args.save_data_path,
        lbm_act_path=args.llm_act_path,
        sparse_data=args.sparse_data,
        target_pair_num=args.target_pair_num,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
