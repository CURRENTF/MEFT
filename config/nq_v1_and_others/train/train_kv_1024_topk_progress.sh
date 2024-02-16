#!/bin/bash
task="nq_v1"
base_model='linhvu/decapoda-research-llama-7b-hf'
data_path="/root/autodl-fs/${task}"

echo "base model is $base_model"

for gpu_topk in 8 16 32 64 128 256 512 ; do
  bs=2
  ep=4
  on_gpu_size=1024
  model="llama7b"
  adapter="kv"
  params="topk${gpu_topk}add${on_gpu_size}"
  python finetune.py \
    --data_path "$data_path" \
    --output_dir "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
    --batch_size 64 \
    --micro_batch_size $bs \
    --num_epochs "$ep" \
    --learning_rate 1e-4 \
    --cutoff_len 256 \
    --wandb_project "meft_all_exps" \
    --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
    --base_model "$base_model"  \
    --on_gpu_size "$on_gpu_size" \
    --eval_step 0.1 \
    --save_step 0.1 \
    --save_total_limit 1 \
    --gpu_topk "$gpu_topk" \
    --optimizer_kv "adamw_torch" \
    --pre_look_layers 0 \
    --location "all"
done