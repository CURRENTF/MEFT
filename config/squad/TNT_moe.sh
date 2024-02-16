#!/bin/bash
task="squad_v2"
base_model='linhvu/decapoda-research-llama-7b-hf'
data_path="/root/autodl-fs/${task}"

echo "base model is $base_model"

gpu_topk=64
bs=4
ep=8
on_gpu_size=2500
model="llama"
adapter="kv"
params="topk${gpu_topk}-add${on_gpu_size}-MOE-softmax-np8-bbs256-fwsp"

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
  --moe_expert_factor 0.1 \
  --n_probe 32 \
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
