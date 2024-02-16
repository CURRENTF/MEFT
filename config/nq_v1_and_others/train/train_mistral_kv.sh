#!/bin/bash
#task="nq_v1"
#base_model='mistralai/Mistral-7B-v0.1'
#data_path="/root/autodl-fs/${task}"
#gpu_topk=0
#bs=2
#ep=4
#on_gpu_size=768
#model="mistral7b"
#adapter="kv"
#params="topk${gpu_topk}add${on_gpu_size}"
#
#python mistral_finetune.py \
#  --data_path "$data_path" \
#  --output_dir "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --batch_size 64 \
#  --micro_batch_size $bs \
#  --num_epochs "$ep" \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --base_model "$base_model"  \
#  --on_gpu_size "$on_gpu_size" \
#  --eval_step 0.1 \
#  --save_step 0.1 \
#  --save_total_limit 1 \
#  --gpu_topk "$gpu_topk" \
#
#
#task="squad_v2"
#base_model='mistralai/Mistral-7B-v0.1'
#data_path="/root/autodl-fs/${task}"
#gpu_topk=0
#bs=2
#ep=8
#on_gpu_size=768
#model="mistral7b"
#adapter="kv"
#params="topk${gpu_topk}add${on_gpu_size}"
#
#python mistral_finetune.py \
#  --data_path "$data_path" \
#  --output_dir "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --batch_size 64 \
#  --micro_batch_size $bs \
#  --num_epochs "$ep" \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --base_model "$base_model"  \
#  --on_gpu_size "$on_gpu_size" \
#  --eval_step 0.1 \
#  --save_step 0.1 \
#  --save_total_limit 1 \
#  --gpu_topk "$gpu_topk"
#
#
#task="tool_v1"
#base_model='mistralai/Mistral-7B-v0.1'
#data_path="/root/autodl-fs/${task}"
#gpu_topk=0
#bs=2
#ep=1
#on_gpu_size=768
#model="mistral7b"
#adapter="kv"
#params="topk${gpu_topk}add${on_gpu_size}"
#
#python mistral_finetune.py \
#  --data_path "$data_path" \
#  --output_dir "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --batch_size 64 \
#  --micro_batch_size $bs \
#  --num_epochs "$ep" \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --base_model "$base_model"  \
#  --on_gpu_size "$on_gpu_size" \
#  --eval_step 0.1 \
#  --save_step 0.1 \
#  --save_total_limit 1 \
#  --gpu_topk "$gpu_topk"


task="metamathqa_v1"
base_model='mistralai/Mistral-7B-v0.1'
data_path="/root/autodl-fs/${task}"
gpu_topk=0
bs=2
ep=1
on_gpu_size=768
model="mistral7b"
adapter="kv"
params="topk${gpu_topk}add${on_gpu_size}"

python mistral_finetune.py \
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
  --gpu_topk "$gpu_topk"

