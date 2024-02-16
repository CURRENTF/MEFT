#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'

echo "base model is $base_model"
model="llama7b"
task="nq_v1"
data_path="/root/autodl-fs/${task}"
adapter="lora"
rank=256
params="rank${rank}"
ep=1
python raw_finetune.py \
  --base_model "$base_model" \
  --data_path "$data_path" \
  --output_dir "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
  --batch_size 64 \
  --micro_batch_size 2 \
  --num_epochs $ep \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --adapter_name lora \
  --lora_r "${rank}" \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \



