#!/bin/bash

base_model=""
if [ -v SLURM_JOB_ID ]
then
#  export CUDA_VISIBLE_DEVICES="1"
  base_model='/data03/xxxlab_share/llama-7b-hf'
else
  base_model='linhvu/decapoda-research-llama-7b-hf'
#  base_model='decapoda-research/llama-7b-hf'
fi

echo "base model is $base_model"

# rojagtap/**natural_questions_clean**  './dataset/gsm8k/my_train.json'
#data_path='rojagtap/natural_questions_clean'
data_path="/root/autodl-fs/nq_v1"
gpu_topk=64
bs=4
epoch=4
task=nq_v1
on_gpu_size=4096

for low_key_dim in 128 256 512 784 1024 2048; do
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
    --eval_step 0.25 \
    --save_step 0.25 \
    --save_total_limit 5 \
    --gpu_topk "$gpu_topk" \
    --optimizer_kv "adamw_torch" \
    --frozen_key \
    --low_key_dim $low_key_dim \
    --yingji_load_data
done
#--load_pkl \ --simulate_recall 0.92 \
#--added_on_cpu \
#  --base_model '/data03/xxxlab_share/llama-7b-hf'
#  'decapoda-research/llama-7b-hf'
#  --load_8bit true
#  "./converted_models/kv_llama"
#  /root/autodl-fs/kv_llama