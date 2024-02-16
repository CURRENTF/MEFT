#!/bin/bash

# 获取0号GPU的显存信息
mem_info=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)

# 删除多余的空格
mem_info=$(echo $mem_info | tr -d '[:space:]')

base_model='linhvu/decapoda-research-llama-7b-hf'

echo "base model is $base_model"

task="tool_v1"
data_path="/root/autodl-fs/${task}"
ep=1

# 判断显存大小
if (( mem_info < 30000 )); then
    echo "显存小于30G"
    bs=2
    for on_gpu_size in 128 512 1024; do
      gpu_topk=0
      echo "Running with gpu_topk = $gpu_topk"
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
        --save_total_limit 1 \
        --gpu_topk "$gpu_topk" \
        --optimizer_kv "adamw_torch" \
        --pre_look_layers 0 \
        --fix_wrong_special_token \

    done
elif (( mem_info > 30000 )); then
    echo "显存大于30G"
    bs=2
    for on_gpu_size in 1536 2048 6144; do
      gpu_topk=0
      echo "Running with gpu_topk = $gpu_topk"
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
        --save_total_limit 1 \
        --gpu_topk "$gpu_topk" \
        --optimizer_kv "adamw_torch" \
        --pre_look_layers 0 \
        --fix_wrong_special_token \

    done
else
    echo "显存等于30G"
fi



