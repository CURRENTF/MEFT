#!/bin/bash

task="nq_v1"
base_model='linhvu/decapoda-research-llama-7b-hf'
data_path="/root/autodl-fs/${task}"
echo "base model is $base_model"

#bs=1
#epoch=1
#on_gpu_size=8192
#add_num=2048
#sl=256
#gpu_topk=16
#echo "gpu_topk = ${gpu_topk}"
#
#python finetune.py\
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size "${bs}" \
#  --micro_batch_size "${bs}" \
#  --num_epochs "${epoch}" \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --base_model "$base_model"  \
#  --on_gpu_size "$on_gpu_size" \
#  --gpu_topk "$gpu_topk" \
#  --optimizer_kv "sgd" \
#  --add_num "${add_num}" \
#  --frozen_key \
#  --eval_step 0.2 \
#  --save_step 0.2 \
#  --save_total_limit 5 \
#  --warmup_ratio 0.001 \
#  --logging_steps 50 \
#  --low_key_dim 256 \
#  --location "back" \
#  --add_layer_num 30 \
#  --pre_look_layers 0 \
#  --yingji_load_data \
#
#bs=2
#epoch=1
#on_gpu_size=8192
#add_num=2048
#sl=256
#gpu_topk=16
#echo "gpu_topk = ${gpu_topk}"
#
#python finetune.py\
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size "${bs}" \
#  --micro_batch_size "${bs}" \
#  --num_epochs "${epoch}" \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --base_model "$base_model"  \
#  --on_gpu_size "$on_gpu_size" \
#  --gpu_topk "$gpu_topk" \
#  --optimizer_kv "sgd" \
#  --add_num "${add_num}" \
#  --frozen_key \
#  --eval_step 0.2 \
#  --save_step 0.2 \
#  --save_total_limit 5 \
#  --warmup_ratio 0.001 \
#  --logging_steps 50 \
#  --low_key_dim 256 \
#  --location "back" \
#  --add_layer_num 30 \
#  --pre_look_layers 0 \
#  --yingji_load_data \

bs=1
epoch=2
on_gpu_size=8192
add_num=2048
sl=256
gpu_topk=16
echo "gpu_topk = ${gpu_topk}"

python finetune.py\
  --data_path "$data_path" \
  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
  --batch_size "${bs}" \
  --micro_batch_size "${bs}" \
  --num_epochs "${epoch}" \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
  --base_model "$base_model"  \
  --on_gpu_size "$on_gpu_size" \
  --gpu_topk "$gpu_topk" \
  --optimizer_kv "adamw_torch" \
  --add_num "${add_num}" \
  --frozen_key \
  --eval_step 0.2 \
  --save_step 0.2 \
  --save_total_limit 5 \
  --warmup_ratio 0.001 \
  --logging_steps 50 \
  --low_key_dim 256 \
  --location "back" \
  --add_layer_num 30 \
  --pre_look_layers 0 \
  --yingji_load_data \
