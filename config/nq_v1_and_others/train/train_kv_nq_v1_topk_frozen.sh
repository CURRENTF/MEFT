#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'

echo "base model is $base_model"

task="nq_v1"
data_path="/root/autodl-fs/${task}"
bs=2
epoch=4

on_gpu_size=4096
# 4 16 32 64 128 256
for gpu_topk in 4 16 32 64 128 512 1024 2048 4096; do
  echo "Running with gpu_topk = $gpu_topk"
#  mv "./kv-no_cpu-${task}-g_topk=${gpu_topk}--ogs=${on_gpu_size}--ep${epoch}" \
#   "/root/autodl-fs/trained_models/kv-no_cpu-${task}-g_topk=${gpu_topk}--ogs=${on_gpu_size}--ep${epoch}"
  python finetune.py \
    --data_path "$data_path" \
    --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
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
    --location "all" \
    --frozen_key \
    --yingji_load_data
done