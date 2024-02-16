#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p='/root/autodl-fs/nq_v1'


for w in "llama7b-nq_v1-kv-topk32-add1024-LIM1024-ep4" "llama7b-nq_v1-kv-topk64-add1024-LIM1024-ep4"; do

weight="/root/autodl-fs/trained_models/${w}"
python kv_evaluate.py \
  --model LLaMA-7B \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights $weight \
  --test_sample_num 501 \
  --set_pll_0 \
  --bs 1

done