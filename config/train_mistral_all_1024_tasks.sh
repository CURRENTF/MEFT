#!/bin/bash


task="nq_v1"
base_model='mistralai/Mistral-7B-v0.1'
data_path="/root/autodl-fs/${task}"

echo "base model is $base_model"
#
#done
gpu_topk=0
bs=2
ep=4
on_gpu_size=800
model="mistral7b"
adapter="kv"
params="topk${gpu_topk}add${on_gpu_size}"

python mistral_finetune.py \
  --data_path "$data_path" \
  --output_dir "./trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
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

weight="./trained_models/${model}-${task}-${adapter}-${params}-ep${ep}"
python kv_evaluate.py \
  --model LLaMA-7B \
  --adapter "kv" \
  --dataset $data_path \
  --base_model $base_model \
  --lora_weights "${weight}" \
  --test_sample_num 501 \
  --bs 1 \
  --explosion



#task="tool_v1"
#base_model='mistralai/Mistral-7B-v0.1'
#data_path="/root/autodl-fs/${task}"
#
#echo "base model is $base_model"
##
##done
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
#  --eval_step 0.2 \
#  --save_step 0.2 \
#  --save_total_limit 1 \
#  --gpu_topk "$gpu_topk" \


#task="squad_v2"
#base_model='mistralai/Mistral-7B-v0.1'
#data_path="/root/autodl-fs/${task}"
#
#echo "base model is $base_model"
##
##done
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
#  --batch_size 256 \
#  --micro_batch_size $bs \
#  --num_epochs "$ep" \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --base_model "$base_model"  \
#  --on_gpu_size "$on_gpu_size" \
#  --eval_step 0.2 \
#  --save_step 0.2 \
#  --save_total_limit 1 \
#  --gpu_topk "$gpu_topk" \
#
#
#weight="./trained_models/${model}-${task}-${adapter}-${params}-ep${ep}"
#python kv_evaluate.py \
#  --model LLaMA-7B \
#  --adapter "kv" \
#  --dataset $data_path \
#  --base_model $base_model \
#  --lora_weights "${weight}" \
#  --test_sample_num 501 \
#  --bs 1 \
#  --explosion
#
#
#task="metamathqa_v1"
#base_model='mistralai/Mistral-7B-v0.1'
#data_path="/root/autodl-fs/${task}"
#
#echo "base model is $base_model"
##
##done
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
#  --eval_step 0.2 \
#  --save_step 0.2 \
#  --save_total_limit 1 \
#  --gpu_topk "$gpu_topk" \
#  --fix_wrong_special_token \
#
#task="gsm8k_v1"
#data_path="/root/autodl-fs/${task}"
#weight="./trained_models/${model}-${task}-${adapter}-${params}-ep${ep}"
#python kv_evaluate.py \
#  --model LLaMA-7B \
#  --adapter "kv" \
#  --dataset $data_path \
#  --base_model $base_model \
#  --lora_weights "${weight}" \
#  --test_sample_num 501 \
#  --bs 1 \
#  --explosion

