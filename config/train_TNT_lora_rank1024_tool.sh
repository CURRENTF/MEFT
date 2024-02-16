#!/bin/bash

task="tool_v1"
base_model='linhvu/decapoda-research-llama-7b-hf'
data_path="/root/autodl-fs/${task}"

echo "base model is $base_model"
ep=1
rank=1024
adapter="lora"
params="rank${rank}"
model="llama7b"

mkdir "./trained_models"

python raw_finetune.py \
  --base_model "$base_model" \
  --data_path "$data_path" \
  --output_dir "./trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
  --batch_size 60 \
  --micro_batch_size 12 \
  --num_epochs "${ep}" \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --adapter_name $adapter \
  --lora_r ${rank} \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
  --save_step 0.2 \
  --eval_step 0.2 \
  --save_total_limit 1 \
  --fix_wrong_special_token

#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p=$data_path

weight="./trained_models/${model}-${task}-${adapter}-${params}-ep${ep}"
python raw_evaluate_my.py \
  --model LLaMA-7B \
  --adapter LORA \
  --dataset "$data_p" \
  --base_model "$base_model" \
  --lora_weights "$weight" \
  --test_sample_num 451

d=$(date)
mv "./trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
   "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}--${d}"
