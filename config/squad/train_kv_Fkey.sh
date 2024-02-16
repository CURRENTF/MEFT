#!/bin/bash
task="squad_v2"
base_model='linhvu/decapoda-research-llama-7b-hf'
data_path="/root/autodl-fs/${task}"

echo "base model is $base_model"

gpu_topk=128
bs=2
bbs=256
ep=8
on_gpu_size=8192
model="llama"
adapter="kv"
params="topk${gpu_topk}-add${on_gpu_size}-bbs${bbs}"

python finetune.py \
  --data_path "$data_path" \
  --output_dir "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
  --batch_size $bbs \
  --micro_batch_size $bs \
  --num_epochs "$ep" \
  --learning_rate 2e-4 \
  --cutoff_len 256 \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
  --base_model "$base_model"  \
  --on_gpu_size "$on_gpu_size" \
  --eval_step 0.1 \
  --save_step 0.1 \
  --gpu_topk "$gpu_topk" \
  --optimizer_kv "adamw_torch" \
  --pre_look_layers 0 \
  --location "all" \
  --save_total_limit 1 \
  --frozen_key \
  --fix_wrong_special_token \
  --init_weight_from_raw_model
