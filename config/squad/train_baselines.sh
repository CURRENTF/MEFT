#!/bin/bash
task="squad_v2"
base_model='linhvu/decapoda-research-llama-7b-hf'
data_path="/root/autodl-fs/${task}"

echo "base model is $base_model"
ep=8
rank=256
adapter="lora"

python raw_finetune.py \
  --base_model "$base_model" \
  --data_path "$data_path" \
  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
  --batch_size 64 \
  --micro_batch_size 2 \
  --num_epochs "${ep}" \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --adapter_name $adapter \
  --lora_r ${rank} \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
  --save_step 0.1 \
  --eval_step 0.1 \
  --save_total_limit 10

adapter="bottleneck"
python raw_finetune.py \
  --base_model "$base_model" \
  --data_path "$data_path" \
  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
  --batch_size 64 \
  --micro_batch_size 2 \
  --num_epochs "${ep}" \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --adapter_name $adapter \
  --bottleneck_size ${rank} \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
  --save_step 0.1 \
  --eval_step 0.1 \
  --save_total_limit 10
