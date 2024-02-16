#!/bin/bash

base_model='mistralai/Mistral-7B-v0.1'
#data_p='/root/autodl-fs/tool_v1'
data_p="../datasets/squad_v2.1"
export TRANSFORMERS_CACHE="../hub"
export HF_HOME="../hub"

cp "./trained_models/mistral-squad_v2-kv-topk64-add3600-MOE-add_progress-ep8/config.json" \
 "./trained_models/mistral-squad_v2-kv-topk64-add3600-MOE-add_progress-ep8/checkpoint-2704/config.json"
#weight="/root/autodl-fs/trained_models/llama-tool_v1-kv-topk64-add8464-MOE-add_progress-ep1"
weight="./trained_models/mistral-squad_v2-kv-topk64-add3600-MOE-add_progress-ep8/checkpoint-1446"
python kv_evaluate.py \
  --model LLaMA-7B \
  --adapter "kv" \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights "${weight}" \
  --test_sample_num 501 \
  --fix_wrong_special_token \
  --bs 1


