#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p='/root/autodl-fs/nq_v1'

#### ?????????
weight="/root/autodl-fs/trained_models/llama7b-nq_v1-kv-topk32add4096-MOE-F--BUG_FIX-ep4/checkpoint-1248"
#weight="/root/autodl-fs/trained_models/llama7b-nq_v1-kv-topk64-add6400-MOE-ep1"
python kv_evaluate.py \
  --model LLaMA-7B \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights $weight \
  --test_sample_num 501 \
  --set_pll_0 \
  --explosion