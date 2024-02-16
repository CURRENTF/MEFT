#!/bin/bash
task="nq_v1"
base_model='linhvu/decapoda-research-llama-7b-hf'
data_path="/root/autodl-fs/${task}"

echo "base model is $base_model"

gpu_topk=64
bbs=256
bs=1
ep=4
on_gpu_size=8192
low_key_dim=1024
model="llama"
adapter="kv"
params="topk${gpu_topk}-lkd${low_key_dim}-add${on_gpu_size}-bbs${bbs}-noFkey"

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
  --eval_step 0.2 \
  --save_step 0.2 \
  --gpu_topk "$gpu_topk" \
  --optimizer_kv "adamw_torch" \
  --pre_look_layers 0 \
  --location "all" \
  --save_total_limit 1 \
  --low_key_dim $low_key_dim \
#  --use_list_load_data
