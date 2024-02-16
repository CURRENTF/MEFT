#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p="/root/autodl-fs/pubmedqa_v2"
weight="/root/autodl-fs/trained_models/llama-pubmedqa_v2-lora-rank256-ep4"
python raw_evaluate_my.py \
  --model LLaMA-7B \
  --adapter LORA \
  --dataset "$data_p" \
  --base_model "$base_model" \
  --lora_weights "$weight" \
  --test_sample_num 501

data_p="/root/autodl-fs/pubmedqa_v2"
weight="/root/autodl-fs/trained_models/llama-pubmedqa_v2-bottleneck-rank256-ep4"
python raw_evaluate_my.py \
  --model LLaMA-7B \
  --adapter Bottleneck \
  --dataset "$data_p" \
  --base_model "$base_model" \
  --lora_weights "$weight" \
  --test_sample_num 501