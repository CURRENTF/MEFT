#!/bin/bash

base_model=""
if [ -v SLURM_JOB_ID ]
then
  export CUDA_VISIBLE_DEVICES="0"
  base_model='/data03/xxxlab_share/llama-7b-hf'
else
  base_model='decapoda-research/llama-7b-hf'
fi
data_path="$1"

python raw_finetune.py \
  --base_model "$base_model" \
  --data_path "$data_path" \
  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
  --batch_size 64 \
  --micro_batch_size 2 \
  --num_epochs 8 \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name bottleneck \
  --use_parallel_adapter \
  --bottleneck_size 256 \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
  --eval_on_train "true" \
