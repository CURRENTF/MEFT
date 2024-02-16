#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p='/root/autodl-fs/triviaqa_v1'

#### ?????????
weight="/root/autodl-fs/trained_models/llama-nq_v1-kv-topk64-add6400-MOE-softmax-np8-bbs256-fwsp-ep4--Sat Feb 10 02:27:25 CST 2024"
#weight="/root/autodl-fs/trained_models/llama7b-nq_v1-kv-topk64-add6400-MOE-ep1"
python kv_evaluate.py \
  --model LLaMA-7B \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights "$weight" \
  --test_sample_num 501 \
  --set_pll_0 \
  --explosion