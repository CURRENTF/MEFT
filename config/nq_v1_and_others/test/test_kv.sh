#!/bin/bash

#base_model='linhvu/decapoda-research-llama-7b-hf'
#data_p='/root/autodl-fs/nq_v1'
base_model='/data03/xxxlab_share/llama-7b-hf'
data_p="../datasets/nq_v1"
export TRANSFORMERS_CACHE="../hub"
export HF_HOME="../hub"

weight="./trained_models/llama-nq_v1-kv-topk64-add8464-MOE-add_progress-ep4/checkpoint-6240"
python kv_evaluate.py \
  --model LLaMA-7B \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights $weight \
  --test_sample_num 501 \
  --set_pll_0