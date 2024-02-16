#!/bin/bash
for task in "all_nq" "all_squad" "all_tool" "all_math"; do
  base_model='linhvu/decapoda-research-llama-7b-hf'
  data_p="/root/autodl-fs/${task}"
  weight="/root/autodl-fs/trained_models/llama7b-all_in_one_v1-lora-rank256-ep1"
  python raw_evaluate_my.py \
    --model LLaMA-7B \
    --adapter LORA \
    --dataset $data_p \
    --base_model $base_model \
    --lora_weights $weight \
    --test_sample_num 201 \
    --set_pll_0 \
    --explosion

done

for task in "all_nq" "all_squad" "all_tool" "all_math"; do
  base_model='linhvu/decapoda-research-llama-7b-hf'
  data_p="/root/autodl-fs/${task}"
  weight="/root/autodl-fs/trained_models/llama7b-all_in_one_v1-bottleneck-rank256-ep1"
  python raw_evaluate_my.py \
    --model LLaMA-7B \
    --adapter Bottleneck \
    --dataset $data_p \
    --base_model $base_model \
    --lora_weights $weight \
    --test_sample_num 201 \
    --set_pll_0 \
    --explosion

done
