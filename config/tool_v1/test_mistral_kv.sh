#!/bin/bash

base_model='mistralai/Mistral-7B-v0.1'
data_p='/root/autodl-fs/tool_v1'

#weight="/root/autodl-fs/trained_models/mistral7b-tool_v1-kv-topk0add768-ep1"
#python kv_evaluate.py \
#  --model LLaMA-7B \
#  --adapter "kv" \
#  --dataset $data_p \
#  --base_model $base_model \
#  --lora_weights "${weight}" \
#  --test_sample_num 451 \
#  --fix_wrong_special_token \
#  --bs 1
#
#base_model='mistralai/Mistral-7B-v0.1'
#data_p='/root/autodl-fs/nq_v1'
#
#weight="/root/autodl-fs/trained_models/mistral7b-nq_v1-kv-topk0add768-ep4"
#python kv_evaluate.py \
#  --model LLaMA-7B \
#  --adapter "kv" \
#  --dataset $data_p \
#  --base_model $base_model \
#  --lora_weights "${weight}" \
#  --test_sample_num 501 \
#  --fix_wrong_special_token \
#  --bs 1

base_model='mistralai/Mistral-7B-v0.1'
data_p='/root/autodl-fs/gsm8k_v1'

weight="/root/autodl-fs/trained_models/mistral7b-metamathqa_v1-kv-topk0add512-ep1"
python kv_evaluate.py \
  --model LLaMA-7B \
  --adapter "kv" \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights "${weight}" \
  --test_sample_num 501 \
  --fix_wrong_special_token \
  --bs 1

#base_model='mistralai/Mistral-7B-v0.1'
#data_p='/root/autodl-fs/squad_v2.1'
#
#weight="/root/autodl-fs/trained_models/mistral7b-squad_v2-kv-topk0add768-ep8"
#python kv_evaluate.py \
#  --model LLaMA-7B \
#  --adapter "kv" \
#  --dataset $data_p \
#  --base_model $base_model \
#  --lora_weights "${weight}" \
#  --test_sample_num 501 \
#  --fix_wrong_special_token \
#  --bs 1
