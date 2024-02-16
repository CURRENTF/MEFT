#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p='/root/autodl-fs/squad_v2.1'

#for cps in 7720 5790 3860 1930; do
######
weight="/root/autodl-fs/trained_models/llama-squad_v2-kv-topk32-add6400-MOE-bbs256-ep8"
python kv_evaluate.py \
  --model LLaMA-7B \
  --adapter "kv" \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights "${weight}" \
  --test_sample_num 501 \
  --bs 1 \
  --explosion

#done ????/
#  > test_nq_v1_bottleneck.log 2>&1
