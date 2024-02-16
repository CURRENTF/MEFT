#!/bin/bash
for task in "all_nq" "all_squad" "all_tool" "all_math"; do
  base_model='linhvu/decapoda-research-llama-7b-hf'
  data_p="/root/autodl-fs/${task}"
  weight="/root/autodl-fs/trained_models/llama7b-all_in_one_v1-kv-topk32-add4096-MOE-ep1"
  python kv_evaluate.py \
    --model LLaMA-7B \
    --dataset $data_p \
    --base_model $base_model \
    --lora_weights $weight \
    --test_sample_num 201 \
    --set_pll_0 \
    --explosion

done