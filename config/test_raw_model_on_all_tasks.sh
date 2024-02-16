#!/bin/bash

#for task in "nq_v1" "squad_v2" "pubmedqa_v2" "gsm8k_v1"; do
#  base_model='mistralai/Mistral-7B-v0.1'
#  data_p="/root/autodl-fs/${task}"
#  weight="none"
#  python kv_evaluate.py \
#    --model LLaMA-7B \
#    --dataset $data_p \
#    --base_model $base_model \
#    --lora_weights $weight \
#    --test_sample_num 501 \
#    --fix_wrong_special_token
#
#done

for task in "nq_v1" "squad_v2" "pubmedqa_v2"; do
  base_model='linhvu/decapoda-research-llama-7b-hf'
  data_p="/root/autodl-fs/${task}"
  weight="none"
  python kv_evaluate.py \
    --model LLaMA-7B \
    --dataset $data_p \
    --base_model $base_model \
    --lora_weights $weight \
    --test_sample_num 501 \
    --fix_wrong_special_token

done
#
#base_model='mistralai/Mistral-7B-v0.1'
#data_p="/root/autodl-fs/tool_v1"
#weight="none"
#python kv_evaluate.py \
#  --model LLaMA-7B \
#  --dataset $data_p \
#  --base_model $base_model \
#  --lora_weights $weight \
#  --test_sample_num 451 \
#  --fix_wrong_special_token
#
#base_model='linhvu/decapoda-research-llama-7b-hf'
#data_p="/root/autodl-fs/tool_v1"
#weight="none"
#python kv_evaluate.py \
#  --model LLaMA-7B \
#  --dataset $data_p \
#  --base_model $base_model \
#  --lora_weights $weight \
#  --test_sample_num 451 \
#  --fix_wrong_special_token
