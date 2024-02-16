#!/bin/bash
task="squad_v2"
base_model='NousResearch/Llama-2-13b-hf'
data_path="/root/autodl-fs/${task}"

echo "base model is $base_model"

gpu_topk=0
bs=1
epoch=4
on_gpu_size=256

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
  --eval_step 0.2 \
  --save_step 0.2 \
  --gpu_topk "$gpu_topk" \
  --optimizer_kv "adamw_torch" \
  --pre_look_layers 0 \
  --location "all" \
  --save_total_limit 5 \
  --load_in_8bit

  #  --use_list_load_data \
