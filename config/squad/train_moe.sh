#!/bin/bash
task="squad_v2"
base_model='linhvu/decapoda-research-llama-7b-hf'
data_path="/root/autodl-fs/${task}"

echo "base model is $base_model"

gpu_topk=64
bs=2
ep=8
on_gpu_size=5184
model="llama"
adapter="kv"
params="topk${gpu_topk}-add${on_gpu_size}-MOE-bbs256"

mkdir "./trained_models"

python finetune.py \
  --data_path "$data_path" \
  --output_dir "./trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
  --batch_size 256 \
  --micro_batch_size $bs \
  --num_epochs "$ep" \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
  --base_model "$base_model"  \
  --on_gpu_size "$on_gpu_size" \
  --eval_step 0.25 \
  --save_step 0.25 \
  --gpu_topk "$gpu_topk" \
  --optimizer_kv "adamw_torch" \
  --pre_look_layers 0 \
  --location "all" \
  --save_total_limit 4 \
  --moe_style \
  --moe_expert_factor 0.25 \
  --n_probe 8 \
  --init_weight_from_raw_model \
  --fix_wrong_special_token \
  --lr_type "constant_with_warmup" \

#  --use_unstable_feature \
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
