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
bs=3
epoch=4
on_gpu_size=1024
model='llama7b'
task='nq_v1'
adapter='kv'
params="topk${gpu_topk}-add${on_gpu_size}"
ep=$epoch

python finetune.py \
  --data_path "$data_path" \
  --output_dir "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
  --batch_size 63 \
  --micro_batch_size $bs \
  --num_epochs "$epoch" \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
  --base_model "$base_model"  \
  --on_gpu_size "$on_gpu_size" \
  --eval_step 0.1 \
  --save_step 0.1 \
  --gpu_topk "$gpu_topk" \
  --optimizer_kv "adamw_torch" \
  --pre_look_layers 0 \
  --location "all" \




#--load_pkl \
#--added_on_cpu \
#  --base_model '/data03/xxxlab_share/llama-7b-hf'
#  'decapoda-research/llama-7b-hf'
#  --load_8bit true
#  "./converted_models/kv_llama"
#  /root/autodl-fs/kv_llama