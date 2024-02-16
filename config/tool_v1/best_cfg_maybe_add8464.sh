#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
task="tool_v1"

#data_path="/root/autodl-fs/${task}"
data_path="../datasets/${task}"
base_model='/data03/xxxlab_share/llama-7b-hf'

echo "base model is $base_model"
unset DEBUG_KV

gpu_topk=64
bs=1
ep=1
on_gpu_size=8464
model="llama"
adapter="kv"
params="topk${gpu_topk}-add${on_gpu_size}-MOE-add_progress"

mkdir "./trained_models"

python finetune.py \
  --data_path "$data_path" \
  --output_dir "./trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
  --batch_size 64 \
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
  --moe_style \
  --moe_expert_factor 0.25 \
  --n_probe 8 \
  --use_unstable_feature \
  --init_weight_from_raw_model \
  --simulate_moe \

#  --map_auto \
#  --fix_wrong_special_token \
#--lr_type "constant_with_warmup" \
#--moe_softmax_score \
# ????
#  --resume_from_checkpoint
#  --notrain_on_inputs
#  --moe_softmax_score

unset DEBUG_KV

weight="./trained_models/${model}-${task}-${adapter}-${params}-ep${ep}"
python kv_evaluate.py \
  --model LLaMA-7B \
  --adapter "kv" \
  --dataset $data_path \
  --base_model $base_model \
  --lora_weights "${weight}" \
  --test_sample_num 501 \
  --bs 1 \
  --explosion


d=$(date)
#mv "./trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
#   "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}--${d}"
