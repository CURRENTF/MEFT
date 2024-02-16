#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p='/root/autodl-fs/squad_v2.1'

weight="/root/autodl-fs/trained_models/kv-no_cpu-squad_v2-g_topk=0--ogs=6144--ep8"
python kv_evaluate.py \
  --model LLaMA-7B \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights $weight \
  --test_sample_num 101 \
  --set_pll_0 \
  --explosion
