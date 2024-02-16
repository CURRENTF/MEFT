#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p='/root/autodl-fs/tool_v1'


weight="/root/autodl-fs/trained_models/llama7b-tool_v1-kv-topk32add1600-MOE-F-bbs256-ep1"
python kv_evaluate.py \
  --model LLaMA-7B \
  --adapter "kv" \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights "${weight}" \
  --test_sample_num 451 \
  --bs 1 \
  --explosion

#  > test_nq_v1_bottleneck.log 2>&1
