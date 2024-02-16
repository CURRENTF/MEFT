#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p='/root/autodl-fs/pubmedqa_v2'
weight='/root/autodl-fs/trained_models/llama7B-pubmedqa_v2-kv-add3072-topk0-ep2'

python kv_evaluate.py \
  --model LLaMA-7B \
  --adapter "kv" \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights "${weight}" \
  --test_sample_num 501 \
  --bs 1

#  --test_sample_num 500
#  --lora_weights "none" \
#  --lora_weights $weight \
#  > test_nq_v1_bottleneck.log 2>&1
