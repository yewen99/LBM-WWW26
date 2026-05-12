import numpy as np
import re
from lbm_act.llm_nets import BiddingModelConfig, LLMPolicy
from lbm_act.algo import LBM_ACT_LEARNER
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PretrainedConfig, Qwen2Config, AutoModelForCausalLM
import torch
from dt_critics import Q_voting, load_Q_net
import copy

# def find_pattern(pattern, sentence_str):
#     match = re.finditer(pattern, sentence_str)
#     matches = list(match)
#     if matches:
#         found_answer = matches[-1].group(1).strip()
#     else:
#         found_answer = None
    
#     return found_answer


def delta_Q(RSA_sequence, Q_ensemble, tokenizer, llm_dt_model):
    # use Q-value for judging the optimal adjustment direction
    cot_up = "You should increase the bidding parameter."
    cot_down = "You should decrease the bidding parameter."
    # ['a', 'budget', 'c', 'cpa', 'cpa_ratio', 'd', 'mask', 'r', 'rtg', 's', 'target', 'target_direction', 'timesteps']
    states, actions, rtgs, timesteps, attention_mask = RSA_sequence['s'], RSA_sequence['a'], RSA_sequence['rtg'], RSA_sequence['timesteps'], RSA_sequence['mask']
    
    states, actions, rtgs, timesteps, attention_mask = np.vstack(states), np.vstack(actions), np.stack(rtgs), np.stack(timesteps), np.stack(attention_mask)
    rtgs = rtgs[:-1, :]

    states, actions, rtgs, timesteps, attention_mask = torch.tensor(states, dtype=torch.float32).cuda(), torch.tensor(actions, dtype=torch.float32).cuda(), torch.tensor(rtgs, dtype=torch.float32).cuda(), torch.tensor(timesteps, dtype=torch.int).cuda(), torch.tensor(attention_mask).cuda()
    states, actions, rtgs, timesteps, attention_mask = states.unsqueeze(0), actions.unsqueeze(0), rtgs.unsqueeze(0), timesteps.unsqueeze(0), attention_mask.unsqueeze(0)

    actions_copy = copy.deepcopy(actions)

    cot_up_tokenized = tokenizer(cot_up, return_tensors="pt", padding=True)
    cot_down_tokenized = tokenizer(cot_down, return_tensors="pt", padding=True)

    cot_up_input_ids, cot_up_attention_mask = cot_up_tokenized.input_ids.cuda(), cot_up_tokenized.attention_mask.cuda()
    cot_down_input_ids, cot_down_attention_mask = cot_down_tokenized.input_ids.cuda(), cot_down_tokenized.attention_mask.cuda()
    with torch.no_grad():
        cot_up_prompt_embs = llm_dt_model.model.embed_tokens(cot_up_input_ids)
        cot_down_prompt_embs = llm_dt_model.model.embed_tokens(cot_down_input_ids)
        up_pred_actions, up_embedding = llm_dt_model.forward_Text_RSA_emb(states, actions, rtgs, timesteps, attention_mask, text_prompt_embs=cot_up_prompt_embs)
        down_pred_actions, down_embedding = llm_dt_model.forward_Text_RSA_emb(states, actions, rtgs, timesteps, attention_mask, text_prompt_embs=cot_down_prompt_embs)

        action_proposals = [up_pred_actions[0, -1], down_pred_actions[0, -1], actions_copy[0,-1]]
        RSA_sequence = (states, actions, rtgs, timesteps, attention_mask)
        best_action, max_index, action_values = Q_voting(action_proposals, RSA_sequence, Q_ensemble)

    if max_index == 2: # 意味着CoT不会给决策带来增益
        return 0  
    if max_index == 0:
        return 1
    if max_index == 1:
        return -1


def load_llm_dt(sparse_data=True, policy_load_dir=None):
    mconf = BiddingModelConfig()
    if not sparse_data:
        mconf.state_mean = np.array([5.48876391e-01, 6.91904804e-01, 4.80044229e-02, 4.47875045e-02,
                                    1.17763952e-01, 4.87555661e-03, 4.76420127e-04, 5.72794009e-02,
                                    9.93989091e-02, 4.84664169e-03, 5.83001837e-04, 7.04144008e-02,
                                    4.99805521e-03, 1.01522635e+04, 2.86396864e+04, 1.91412327e+05])
        mconf.state_std = np.array([2.84053382e-01, 3.53000441e-01, 3.01172049e-02, 3.21944272e-02,
                                    3.07672391e-02, 1.92189715e-03, 8.29556557e-04, 9.36906833e-02,
                                    3.75196803e-02, 2.45325444e-03, 1.18077056e-03, 1.27290708e-01,
                                    2.48126164e-03, 5.73180055e+03, 1.67849786e+04, 1.52535424e+05])
        mconf.input_state_dim = 16
        mconf.rtg_scale = 1500
    else:
        mconf.state_mean= np.array([5.41854588e-01, 7.19698607e-01, 4.17500439e-02, 4.35970703e-02,
                                    9.91188952e-02, 4.82405201e-04, 4.61863046e-05, 5.29802530e-02,
                                    9.24203256e-02, 4.84138679e-04, 5.76074165e-05, 6.75957800e-02,
                                    4.98045765e-04, 1.02017857e+04, 2.88687230e+04, 1.95333666e+05]) 
        mconf.state_std = np.array([2.84601949e-01, 3.27488061e-01, 2.76529743e-02, 3.31906076e-02,
                                    2.38985949e-02, 1.89047081e-04, 8.73831598e-05, 9.07318426e-02,
                                    2.65035680e-02, 2.45689550e-04, 1.26855462e-04, 1.23013225e-01,
                                    2.48356154e-04, 5.72176857e+03, 1.67729807e+04, 1.52914080e+05]) 
        mconf.input_state_dim = 16
        mconf.rtg_scale = 100

    model = LLMPolicy(mconf)
    llm_dt_tokenizer = AutoTokenizer.from_pretrained(mconf.model_name)
    llm_dt_tokenizer.padding_side = 'left'
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
    
    model_state_dict = lbm_act_learner.state_dict()
    checkpoint = torch.load(policy_load_dir)
    lbm_act_learner.load_state_dict(checkpoint, strict=False)
    lbm_act_model = lbm_act_learner.policy

    return lbm_act_model, llm_dt_tokenizer

def expert_rule(response, gt, Q_ensemble, tokenizer, llm_dt_model):
    # 看 cpa_ratio 是否出现幻觉
    matched_cpa_ratio = list(re.finditer(r'<ratio>(.*?)</ratio>', response))
    matched_answer = list(re.finditer(r'<answer>(.*?)</answer>', response))
    
    gt_cpa_ratio = gt['cpa_ratio']

    if gt_cpa_ratio == 0:
        return False

    if len(matched_cpa_ratio) != 1 or len(matched_answer) != 1:
        return False
    else:
        try:
            matched_cpa_ratio = float(matched_cpa_ratio[0].group(1).strip())
            matched_answer = int(matched_answer[0].group(1).strip())
            print(f'{matched_cpa_ratio=}  {gt_cpa_ratio=}   ||  {matched_answer=} ')
        except:
            return False

    if 0.97* gt['cpa_ratio'] < matched_cpa_ratio and matched_cpa_ratio < 1.03* gt['cpa_ratio'] and matched_answer in [0, -1, 1]:
        # 使用 delta-Q 判断是否有增益
        better_direct = delta_Q(gt, Q_ensemble, tokenizer, llm_dt_model)
        if matched_answer == better_direct: 
            return True
        else:
            return False
    else:
        return False


def infer_action(RSA_sequence, dt):
    states, actions, rtgs, timesteps, attention_mask = RSA_sequence['s'], RSA_sequence['a'], RSA_sequence['rtg'], RSA_sequence['timesteps'], RSA_sequence['mask']
    # 转换为 (10, 16) 的 numpy 数组
    states, actions, rtgs, timesteps, attention_mask = np.vstack(states), np.vstack(actions), np.stack(rtgs), np.stack(timesteps), np.stack(attention_mask)
    rtgs = rtgs[:-1, :]
    # 转换为 torch tensor
    states, actions, rtgs, timesteps, attention_mask = torch.tensor(states, dtype=torch.float32).cuda(), torch.tensor(actions, dtype=torch.float32).cuda(), torch.tensor(rtgs, dtype=torch.float32).cuda(), torch.tensor(timesteps, dtype=torch.int).cuda(), torch.tensor(attention_mask).cuda()
    states, actions, rtgs, timesteps, attention_mask = states.unsqueeze(0), actions.unsqueeze(0), rtgs.unsqueeze(0), timesteps.unsqueeze(0), attention_mask.unsqueeze(0)
    