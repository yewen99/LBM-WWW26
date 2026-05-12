import os
import json
import argparse
from vllm import LLM, SamplingParams
import pandas as pd
from rule import expert_rule, delta_Q, load_llm_dt
from dt_critics import Q_voting, load_Q_net


def generate_GQPO_data(train_data_path, LLM_path, Q_path, save_data_path, llm_dt, llm_dt_tokenizer):
    # load data
    train_dataset = pd.read_parquet(train_data_path)
    print(f'loaded data from {train_data_path}')

    # extract questions
    questions = [entry[0]['content'] for entry in train_dataset['prompt']]

    # load LLM and Q function
    model = LLM(model=LLM_path, gpu_memory_utilization=0.6)
    sampling_params = SamplingParams(temperature=0.5, top_p=1., max_tokens=512)
    print(f'load LLM from {LLM_path}')

    Q_ensemble = []
    Q_net = load_Q_net(Q_path) 
    Q_ensemble.append(Q_net)

    # generate SFT data
    batch_size = 512
    pair_num = 0
    data_num = 1000
    with open(save_data_path, 'w') as f:
        while pair_num <= data_num:
            for i in range(0, len(questions), batch_size):
                if pair_num > data_num:
                    break
                batch_questions = questions[i:i + batch_size]
                generations = model.generate(batch_questions, sampling_params)
                selected_generations = []
                RSA_sequences = []

                # filter our bad trajs according to delta-Q and Rules
                for idx, gen in enumerate(generations):
                    if expert_rule(gen.outputs[0].text, train_dataset["reward_model"][i+idx]['ground_truth'], Q_ensemble, llm_dt_tokenizer, llm_dt):
                        selected_generations.append({'question': batch_questions[idx], 'response': gen.outputs[0].text})
                        RSA_sequences.append(train_dataset["reward_model"][i+idx]['ground_truth'])
                
                for pair in selected_generations:
                    question = pair['question']
                    assistant_response = pair['response']
                    
                    messages = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant."
                            },
                            {
                                "role": "user",
                                "content": question
                            },
                            {
                                "role": "assistant",
                                "content": assistant_response
                            }
                        ]
                    }
                    
                    f.write(json.dumps(messages) + '\n')
                    pair_num  += 1

                print(f'num of selected pairs = {pair_num } / {i + batch_size} in total')

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Evaluating LLM for Bidding...')
    parser.add_argument('--train_data_path', type=str, default='/home/jiangnan07/liyewen/r1/data/AuctionNet/cot_dt_grpo_sparse/train.parquet', help='path to prompt parquet data.')
    parser.add_argument('--cot_llm_path', type=str, default='/home/jiangnan07/liyewen/r1/pretrained_LLMs/Qwen2.5/Qwen2.5-3B-Instruct', help='path to an LLM to generate the CoT.')
    parser.add_argument('--use_Q_voting', type=bool, default=False, help='whether use an ensemble of Q-nets to assess the decision performance by majority voting.')
    parser.add_argument('--Q_path', type=str, default='/jiangnan/liyewen/r1/biddingr1-paper/Bidding-R1-paper/AuctionNet_Evaluate/saved_model/dt_reweight_search_Q', help='Q model dir.')
    parser.add_argument('--llm_act_path', type=str, default="/home/jiangnan07/liyewen/r1/biddingr1-paper/Bidding-R1-paper/llm_mlp/ckpt/2025-09-20/13-32-17/Qwen/Qwen2.5-0.5B-Instruct/model.pt", help='LBM_ACT model path.')
    parser.add_argument('--sparse_data', type=bool, default=False, help='whether test on the auction_net_sparse.')
    parser.add_argument('--save_data_path', type=str, default='./training_data_sparse_3B_1000.jsonl', help='path to save sft data.')    
    args = parser.parse_args()
    
    lbm_act, tokenizer = load_llm_dt(sparse_data=args.sparse_data, policy_load_dir=args.llm_act_path)

    generate_GQPO_data(args.train_data_path, LLM_path=args.cot_llm_path, Q_path=args.Q_path, save_data_path=args.save_data_path, llm_dt=lbm_act, llm_dt_tokenizer=tokenizer)

