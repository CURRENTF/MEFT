#!/bin/bash
task="squad_v2"
base_model='linhvu/decapoda-research-llama-7b-hf'
data_path="/root/autodl-fs/${task}"

echo "base model is $base_model"

gpu_topk=32
bs=2
ep=8
on_gpu_size=6400
model="llama"
adapter="kv"
params="topk${gpu_topk}-add${on_gpu_size}-MOE-bbs256"

python finetune.py \
  --data_path "$data_path" \
  --output_dir "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
  --batch_size 256 \
  --micro_batch_size $bs \
  --num_epochs "$ep" \
  --learning_rate 4e-4 \
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
  --moe_style \
  --moe_expert_factor 0.25 \
  --n_probe 4 \
  --init_weight_from_raw_model \
  --fix_wrong_special_token \
  --use_unstable_feature \
#  --resume_from_checkpoint
#  --notrain_on_inputs
  --moe_softmax_score

