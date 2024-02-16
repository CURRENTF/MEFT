#!/bin/bash

base_model=""
if [ -v SLURM_JOB_ID ]
then
#  export CUDA_VISIBLE_DEVICES="1"
  echo "in slurm"
  base_model='/data03/xxxlab_share/yahma-llama-7b-hf'
  data_path="../datasets/nq_v1"
else
  base_model='linhvu/decapoda-research-llama-7b-hf'
  data_path="/root/autodl-fs/nq_v1"
fi

echo "base model is $base_model"

# rojagtap/**natural_questions_clean**  './dataset/gsm8k/my_train.json'

#rank=4
#python raw_finetune.py \
#  --base_model "$base_model" \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size 2 \
#  --num_epochs 4 \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --adapter_name lora \
#  --lora_r "${rank}" \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "lora-${rank}-ep4-nq_v1"
#
#rank=32
#python raw_finetune.py \
#  --base_model "$base_model" \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size 2 \
#  --num_epochs 4 \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --adapter_name lora \
#  --lora_r "${rank}" \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "lora-${rank}-ep4-nq_v1"
#
#rank=128
#python raw_finetune.py \
#  --base_model "$base_model" \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size 2 \
#  --num_epochs 4 \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --adapter_name lora \
#  --lora_r "${rank}" \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "lora-${rank}-ep4-nq_v1"
#
#rank=256
#python raw_finetune.py \
#  --base_model "$base_model" \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size 2 \
#  --num_epochs 4 \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --adapter_name lora \
#  --lora_r "${rank}" \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "lora-${rank}-ep4-nq_v1"

#rank=384
#python raw_finetune.py \
#  --base_model "$base_model" \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size 2 \
#  --num_epochs 4 \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --adapter_name lora \
#  --lora_r "${rank}" \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "lora-${rank}-ep4-nq_v1"

#rank=8
#python raw_finetune.py \
#  --base_model "$base_model" \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size 4 \
#  --num_epochs 4 \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --adapter_name lora \
#  --lora_r "${rank}" \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "lora-${rank}-ep4-nq_v1"
#
#rank=64
#python raw_finetune.py \
#  --base_model "$base_model" \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size 2 \
#  --num_epochs 4 \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --adapter_name lora \
#  --lora_r "${rank}" \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "lora-${rank}-ep4-nq_v1"

#rank=512
#python raw_finetune.py \
#  --base_model "$base_model" \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size 1 \
#  --num_epochs 4 \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --adapter_name lora \
#  --lora_r "${rank}" \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "lora-${rank}-ep4-nq_v1"



#rank=784
#python raw_finetune.py \
#  --base_model "$base_model" \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size 8 \
#  --num_epochs 4 \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --adapter_name lora \
#  --lora_r "${rank}" \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --yingji_load_data
##
#rank=1024
#python raw_finetune.py \
#  --base_model "$base_model" \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size 8 \
#  --num_epochs 4 \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --adapter_name lora \
#  --lora_r "${rank}" \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --yingji_load_data
#
#rank=2048
#python raw_finetune.py \
#  --base_model "$base_model" \
#  --data_path "$data_path" \
#  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
#  --batch_size 64 \
#  --micro_batch_size 4 \
#  --num_epochs 4 \
#  --learning_rate 1e-4 \
#  --cutoff_len 256 \
#  --adapter_name lora \
#  --lora_r "${rank}" \
#  --wandb_project "meft_all_exps" \
#  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#  --yingji_load_data

rank=4096
python raw_finetune.py \
  --base_model "$base_model" \
  --data_path "$data_path" \
  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
  --batch_size 64 \
  --micro_batch_size 1 \
  --num_epochs 4 \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --adapter_name lora \
  --lora_r "${rank}" \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
  --yingji_load_data


####
