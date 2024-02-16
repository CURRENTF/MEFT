#!/bin/bash
task="squad_v2"
base_model='linhvu/decapoda-research-llama-7b-hf'
data_path="/root/autodl-fs/${task}"

echo "base model is $base_model"
ep=2
rank=256
adapter="bottleneck"
bs=2
model="llama"
params="rank${rank}"

python raw_finetune.py \
  --base_model "$base_model" \
  --data_path "$data_path" \
  --output_dir "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
  --batch_size 64 \
  --micro_batch_size $bs \
  --num_epochs "${ep}" \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --adapter_name $adapter \
  --bottleneck_size ${rank} \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
  --save_step 0.2 \
  --eval_step 0.2 \
  --save_total_limit 1 \
  --fix_wrong_special_token

