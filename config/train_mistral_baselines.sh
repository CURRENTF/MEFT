#!/bin/bash


#for task in "tool_v1" "metamathqa_v1"; do
#  adapter="lora"
#  params="rank${rank}"
#  rank=64
#  base_model='mistralai/Mistral-7B-v0.1'
#  model="mistral"
#  data_path="/root/autodl-fs/${task}"
#  ep=1
#  python raw_finetune.py \
#  --base_model "$base_model" \
#  --data_path "$data_path" \
#  --output_dir "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --batch_size 64 \
#  --micro_batch_size 4 \
#  --num_epochs "${ep}" \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --adapter_name $adapter \
#  --lora_r ${rank} \
#  --bottleneck_size ${rank} \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --save_step 0.2 \
#  --eval_step 0.2 \
#  --save_total_limit 1
#
#done

for task in "tool_v1" "metamathqa_v1"; do
  rank=256
  adapter="bottleneck"
  params="rank${rank}"
  base_model='mistralai/Mistral-7B-v0.1'
  model="mistral"
  data_path="/root/autodl-fs/${task}"
  ep=1
  python raw_finetune.py \
  --base_model "$base_model" \
  --data_path "$data_path" \
  --output_dir "./trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
  --batch_size 64 \
  --micro_batch_size 2 \
  --num_epochs "${ep}" \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --adapter_name $adapter \
  --lora_r ${rank} \
  --bottleneck_size ${rank} \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
  --save_step 0.2 \
  --eval_step 0.2 \
  --save_total_limit 1

done