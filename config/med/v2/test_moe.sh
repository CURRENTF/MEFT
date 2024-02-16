#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p='/root/autodl-fs/pubmedqa_v2'


weight="/root/autodl-fs/trained_models/llama7b-pubmedqa_v2-kv-topk32-add1024-bbs256-MOE--bug_fixed-ep2"
python kv_evaluate.py \
  --model LLaMA-7B \
  --adapter "kv" \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights "${weight}" \
  --test_sample_num 501 \
  --bs 1 \
  --explosion

#  > test_nq_v1_bottleneck.log 2>&1
