#!/bin/bash

task="nq_v1"
base_model='linhvu/decapoda-research-llama-7b-hf'
data_path="/root/autodl-fs/${task}"
echo "base model is $base_model"

bbs=192
bs=2
epoch=4
on_gpu_size=2048
add_num=20000
gpu_topk=10
echo "gpu_topk = ${gpu_topk}"

python manual_train.py\
  --data_path "$data_path" \
  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
  --batch_size "${bbs}" \
  --micro_batch_size "${bs}" \
  --num_epochs "${epoch}" \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
  --base_model "$base_model"  \
  --on_gpu_size "$on_gpu_size" \
  --gpu_topk "$gpu_topk" \
  --optimizer_kv "adamw" \
  --pre_look_layers 2 \
  --location "back" \
  --add_layer_num 30 \
  --add_num "${add_num}" \
  --frozen_key \
  --added_on_cpu \
  --eval_step 0.2 \
  --save_step 0.2 \
  --save_total_limit 5 \
  --use_glass_vdb \
  --low_key_dim 784 \
  --async_compute \
  --yingji_load_data \
  --num_group 2 \

#  /root/autodl-fs/kv_llama        --use_torch_vecdb \    --cpp_mode \