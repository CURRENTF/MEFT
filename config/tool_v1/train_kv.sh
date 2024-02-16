#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'

echo "base model is $base_model"

task="tool_v1"
data_path="/root/autodl-fs/${task}"
bs=2
epoch=1
ep=$epoch
on_gpu_size=2048
gpu_topk=0
model="llama7b"
adapter="kv"
params="add${on_gpu_size}-topk${gpu_topk}"

python finetune.py \
  --data_path "$data_path" \
  --output_dir "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
  --batch_size 64 \
  --micro_batch_size $bs \
  --num_epochs "$epoch" \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
  --base_model "$base_model"  \
  --on_gpu_size "$on_gpu_size" \
  --eval_step 0.2 \
  --save_step 0.2 \
  --save_total_limit 1 \
  --gpu_topk "$gpu_topk" \
  --optimizer_kv "adamw_torch" \
  --pre_look_layers 0 \
  --use_list_load_data \
  --fix_wrong_special_token \
