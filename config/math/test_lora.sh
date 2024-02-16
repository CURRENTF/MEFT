#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p="/root/autodl-fs/gsm8k_v1"

weight="/root/autodl-fs/trained_models/llama-lora-metamathqa_v1-256-ep1"
python raw_evaluate_my.py \
  --model LLaMA-7B \
  --adapter LORA \
  --dataset "$data_p" \
  --base_model "$base_model" \
  --lora_weights "$weight" \
  --test_sample_num 501