#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'

echo "base model is $base_model"

task="nq_v1"
data_path="/root/autodl-fs/${task}"
bs=2
epoch=4
model="llama7b"
adapter="kv"
ep=$epoch

on_gpu_size=1024
for gpu_topk in 64 32 128 256 4 16; do
  echo "Running with gpu_topk = $gpu_topk"
  params="topk${gpu_topk}-add${on_gpu_size}-LIM1024"
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
    --location "all"  \
    --limit_total_activated_neurons 1024

done

