#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p='/root/autodl-fs/gsm8k_v1'


weight="/root/autodl-fs/trained_models/llama-metamathqa_v1-kv-topk64-add6400-MOE-softmax-np8-bbs256-fwsp-ep1--Sun Feb 11 16:59:26 CST 2024"
python kv_evaluate.py \
  --model LLaMA-7B \
  --adapter "kv" \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights "${weight}" \
  --test_sample_num 501 \
  --bs 1

#  > test_nq_v1_bottleneck.log 2>&1
