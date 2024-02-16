#!/bin/bash
task="metamathqa_v1"
base_model='linhvu/decapoda-research-llama-7b-hf'
data_path="/root/autodl-fs/${task}"

echo "base model is $base_model"

for on_gpu_size in 3072 4096; do
  gpu_topk=0
  ep=1
  bs=2
  model="llama7b"
  adapter="kv"
  params="topk${gpu_topk}-add${on_gpu_size}"

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
    --eval_step 0.2 \
    --save_step 0.2 \
    --gpu_topk "$gpu_topk" \
    --optimizer_kv "adamw_torch" \
    --pre_look_layers 0 \
    --location "all" \
    --save_total_limit 1 \
    --fix_wrong_special_token

done