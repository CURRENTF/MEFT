#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p='/root/autodl-fs/nq_v1'

weight="/root/autodl-fs/trained_models/llama7b-nq_v1-kv-topk128add1024-ep1"
python kv_evaluate.py \
  --model LLaMA-7B \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights $weight \
  --test_sample_num 501 \
  --set_pll_0