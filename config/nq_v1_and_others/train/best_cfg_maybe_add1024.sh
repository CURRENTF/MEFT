#!/bin/bash
task="nq_v1"
base_model='linhvu/decapoda-research-llama-7b-hf'
data_path="/root/autodl-fs/${task}"

echo "base model is $base_model"
export DEBUG_KV=1

gpu_topk=64
bs=2
ep=4
on_gpu_size=1024
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
  --eval_step 0.34 \
  --save_step 0.34 \
  --gpu_topk "$gpu_topk" \
  --optimizer_kv "adamw_torch" \
  --pre_look_layers 0 \
  --location "all" \
  --save_total_limit 2 \
  --moe_style \
  --moe_expert_factor 0.25 \
  --n_probe 8 \
  --use_unstable_feature \
  --init_weight_from_raw_model \
  --simulate_moe \

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
mv "./trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
   "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}--${d}"
