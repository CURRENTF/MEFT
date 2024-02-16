#!/bin/bash

base_model=""
if [ -v SLURM_JOB_ID ]
then
#  export CUDA_VISIBLE_DEVICES="1"
  base_model='/data03/xxxlab_share/llama-7b-hf'
  data_path="../datasets/nq_v1"
else
  base_model='linhvu/decapoda-research-llama-7b-hf'
  data_path="/root/autodl-fs/nq_v1"
#  base_model='decapoda-research/llama-7b-hf'
fi

echo "base model is $base_model"

gpu_topk=0
bs=2
epoch=4

#on_gpu_size=16
#bs=4
#python finetune.py \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size $bs \
#  --num_epochs "$epoch" \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --base_model "$base_model"  \
#  --on_gpu_size "$on_gpu_size" \
#  --eval_step 0.02 \
#  --save_step 0.5 \
#  --gpu_topk "$gpu_topk" \
#  --optimizer_kv "adamw_torch" \
#  --pre_look_layers 0 \
#  --location "all" \
#
#on_gpu_size=128
#bs=2
#python finetune.py \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size $bs \
#  --num_epochs "$epoch" \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --base_model "$base_model"  \
#  --on_gpu_size "$on_gpu_size" \
#  --eval_step 0.02 \
#  --save_step 0.5 \
#  --gpu_topk "$gpu_topk" \
#  --optimizer_kv "adamw_torch" \
#  --pre_look_layers 0 \
#  --location "all" \
#
#on_gpu_size=512
#bs=2
#python finetune.py \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size $bs \
#  --num_epochs "$epoch" \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --base_model "$base_model"  \
#  --on_gpu_size "$on_gpu_size" \
#  --eval_step 0.02 \
#  --save_step 0.5 \
#  --gpu_topk "$gpu_topk" \
#  --optimizer_kv "adamw_torch" \
#  --pre_look_layers 0 \
#  --location "all" \

#on_gpu_size=1024
#bs=2
#python finetune.py \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size $bs \
#  --num_epochs "$epoch" \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --base_model "$base_model"  \
#  --on_gpu_size "$on_gpu_size" \
#  --eval_step 0.02 \
#  --save_step 0.5 \
#  --gpu_topk "$gpu_topk" \
#  --optimizer_kv "adamw_torch" \
#  --pre_look_layers 0 \
#  --location "all" \
#
#on_gpu_size=1536
#bs=1
#python finetune.py \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size $bs \
#  --num_epochs "$epoch" \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --base_model "$base_model"  \
#  --on_gpu_size "$on_gpu_size" \
#  --eval_step 0.02 \
#  --save_step 0.5 \
#  --gpu_topk "$gpu_topk" \
#  --optimizer_kv "adamw_torch" \
#  --pre_look_layers 0 \
#  --location "all" \

#on_gpu_size=2048
#bs=8
#python finetune.py \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size $bs \
#  --num_epochs "$epoch" \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --base_model "$base_model"  \
#  --on_gpu_size "$on_gpu_size" \
#  --eval_step 0.2 \
#  --save_step 0.2 \
#  --gpu_topk "$gpu_topk" \
#  --optimizer_kv "adamw_torch" \
#  --pre_look_layers 0 \
#  --location "all" \
#  --yingji_load_data

#on_gpu_size=3072
#bs=2
#python finetune.py \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size $bs \
#  --num_epochs "$epoch" \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --base_model "$base_model"  \
#  --on_gpu_size "$on_gpu_size" \
#  --eval_step 0.02 \
#  --save_step 0.08 \
#  --gpu_topk "$gpu_topk" \
#  --optimizer_kv "adamw_torch" \
#  --pre_look_layers 0 \
#  --location "all" \
#  --yingji_load_data


on_gpu_size=6144
bs=1
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
  --eval_step 0.02 \
  --save_step 0.08 \
  --gpu_topk "$gpu_topk" \
  --optimizer_kv "adamw_torch" \
  --pre_look_layers 0 \
  --location "all" \
  --yingji_load_data

on_gpu_size=4096
bs=1
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
  --eval_step 0.02 \
  --save_step 0.08 \
  --gpu_topk "$gpu_topk" \
  --optimizer_kv "adamw_torch" \
  --pre_look_layers 0 \
  --location "all" \
  --yingji_load_data



#--load_pkl \
#--added_on_cpu \
#  --base_model '/data03/xxxlab_share/llama-7b-hf'
#  'decapoda-research/llama-7b-hf'
#  --load_8bit true
#  "./converted_models/kv_llama"
#  /root/autodl-fs/kv_llama