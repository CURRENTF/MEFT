#!/bin/bash
task="nq_v1"
base_model='linhvu/decapoda-research-llama-7b-hf'
data_path="/root/autodl-fs/${task}"

echo "base model is $base_model"



#for limit_total_activated_neurons in 1024 2048 3072; do
#for limit_total_activated_neurons in 1024 ; do
#  gpu_topk=16
#  bs=2
#  bbs=32
#  ep=1
#  on_gpu_size=3600
#  model="llama7b"
#  adapter="kv"
#  params="topk${gpu_topk}-add${on_gpu_size}-bbs${bbs}-MOE-progressR"
#  python finetune.py \
#    --data_path "$data_path" \
#    --output_dir "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
#    --batch_size $bbs \
#    --micro_batch_size $bs \
#    --num_epochs "$ep" \
#    --learning_rate 1e-4 \
#    --cutoff_len 256 \
#    --wandb_project "meft_all_exps" \
#    --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
#    --base_model "$base_model"  \
#    --on_gpu_size "$on_gpu_size" \
#    --eval_step 0.2 \
#    --save_step 0.2 \
#    --gpu_topk "$gpu_topk" \
#    --optimizer_kv "adamw_torch" \
#    --pre_look_layers 0 \
#    --location "all" \
#    --save_total_limit 5 \
#    --fix_wrong_special_token \
#    --moe_style \
#    --n_probe 2 \
#    --moe_expert_factor 2 \
#    --init_weight_from_raw_model \
#    --use_unstable_feature \
#    --limit_total_activated_neurons $limit_total_activated_neurons
#
#done

for limit_total_activated_neurons in 1024 2048 3072; do
#for limit_total_activated_neurons in 1024 ; do
  gpu_topk=32
  bs=2
  bbs=32
  ep=1
  on_gpu_size=3600
  model="llama7b"
  adapter="kv"
  params="topk${gpu_topk}-add${on_gpu_size}-bbs${bbs}-MOE-progressR"
  python finetune.py \
    --data_path "$data_path" \
    --output_dir "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
    --batch_size $bbs \
    --micro_batch_size $bs \
    --num_epochs "$ep" \
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
    --fix_wrong_special_token \
    --moe_style \
    --n_probe 2 \
    --moe_expert_factor 2 \
    --init_weight_from_raw_model \
    --use_unstable_feature \
    --limit_total_activated_neurons $limit_total_activated_neurons

done