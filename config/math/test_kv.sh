#!/bin/bash

#base_model='linhvu/decapoda-research-llama-7b-hf'
#data_p='/root/autodl-fs/gsm8k_v1'
base_model='/data03/xxxlab_share/llama-7b-hf'
data_p="../datasets/gsm8k_v1"
export TRANSFORMERS_CACHE="../hub"
export HF_HOME="../hub"

#weight="/root/autodl-fs/trained_models/llama-metamathqa_v1-kv-topk64-add8464-MOE-add_progress-ep1"
weight="./trained_models/llama-metamathqa_v1-kv-topk64-add8464-MOE-add_progress-ep1"

python kv_evaluate.py \
  --model LLaMA-7B \
  --adapter "kv" \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights "${weight}" \
  --test_sample_num 501 \
  --bs 1

#  > test_nq_v1_bottleneck.log 2>&1
