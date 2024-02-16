#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p='/root/autodl-fs/nq_v1'

weight="/root/autodl-fs/trained_models/kv-no_cpu-lkd256-nq_v1-FKey-g_topk=32--ogs=16384--ep4/checkpoint-2496"
python kv_evaluate.py \
  --model LLaMA-7B \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights $weight \
  --test_sample_num 501 \
  --set_pll_0 \
#  --explosion