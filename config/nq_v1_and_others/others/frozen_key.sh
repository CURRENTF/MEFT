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
data_path="$1"
gpu_topk=0
bs=2
#on_gpu_size=$(expr $gpu_topk \* $bs)
on_gpu_size=1024

python finetune.py \
  --data_path "$data_path" \
  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
  --batch_size 64 \
  --micro_batch_size $bs \
  --num_epochs 8 \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --val_set_size 500 \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
  --base_model "$base_model"  \
  --add_layer_num 16 \
  --on_gpu_size "$on_gpu_size" \
  --eval_step 200 \
  --eval_on_train \
  --gpu_topk "$gpu_topk" \
  --add_num 2048 \
  --optimizer_kv "adamw_torch" \
  --use_torch_vecdb \
  --pre_look_layers 4 \
  --frozen_key \


#--load_pkl \
#--added_on_cpu \
#  --base_model '/data03/xxxlab_share/llama-7b-hf'
#  'decapoda-research/llama-7b-hf'
#  --load_8bit true
#  "./converted_models/kv_llama"
#  /root/autodl-fs/kv_llama