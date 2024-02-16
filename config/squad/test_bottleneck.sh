#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p='/root/autodl-fs/squad_v2.1'

weight="/root/autodl-fs/trained_models/llama-squad_v2-bottleneck-rank256-ep2"
python raw_evaluate_my.py \
  --model LLaMA-7B \
  --adapter AdapterH \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights "${weight}" \
  --test_sample_num 501 \
  --bs 1

#  > test_nq_v1_bottleneck.log 2>&1
