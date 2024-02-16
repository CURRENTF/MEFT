#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p="/root/autodl-fs/nq_v1"

# 遍历ranks数组
#for rank in "${ranks[@]}"
#do
#
#done
weight="/root/autodl-fs/trained_models/llama7b-nq_v1-lora-rank256-ep1"
python raw_evaluate_my.py \
  --model LLaMA-7B \
  --adapter LORA \
  --dataset "$data_p" \
  --base_model "$base_model" \
  --lora_weights "$weight" \
  --test_sample_num 501