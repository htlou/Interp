from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json
import torch

device = 'cuda:0'
base_path = '/mnt/hantao/models/gpt2_sft_2e5_3ep'
config = AutoConfig.from_pretrained(base_path)


path = "output_test.json"

with open(path, 'r') as f:
    data = json.load(f)

def compare_first_words(str1, str2):
    # 提取两个字符串中的第一个单词
    first_word_str1 = str1.split()[0]
    first_word_str2 = str2.split()[0]
    
    # 比较两个单词是否相同并返回结果
    return first_word_str1 == first_word_str2

for i in range(8,192,8):
    base_model = AutoModelForCausalLM.from_pretrained(base_path,
                                                       config=config,
    state_dict=torch.load(base_path + f"/pytorch_model_step_{i}.bin")).to(device)
    base_tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)
    success = 0
    for piece in data:
        prompt = piece['question']
        input_ids = base_tokenizer.encode(prompt, return_tensors='pt').to(device)
        outputs = base_model(input_ids=input_ids)
        logits = outputs.logits
        next_token = logits[0, -1].argmax(dim=-1)
        next_literal = base_tokenizer.decode(next_token, skip_special_tokens=True)
        if compare_first_words(next_literal, piece['completion']): 
            success += 1
        # print(next_literal)
        # print(piece['question'])
        # print(piece['completion'])
        # print("==" * 20)
    suc_rate = (success/50)*100
    print(f"ckpt{i}, win rate {suc_rate}%")

# base_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
# base_tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
# success = 0
# for piece in data:
#     prompt = piece['question']
#     input_ids = base_tokenizer.encode(prompt, return_tensors='pt').to(device)
#     outputs = base_model(input_ids=input_ids)
#     logits = outputs.logits
#     next_token = logits[0, -1].argmax(dim=-1)
#     next_literal = base_tokenizer.decode(next_token, skip_special_tokens=True)
#     if compare_first_words(next_literal, piece['completion']): 
#         success += 1
#     # print(next_literal)
#     # print(piece['question'])
#     # print(piece['completion'])
#     # print("==" * 20)
# suc_rate = (success/50)*100
# print(suc_rate)