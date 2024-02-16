#!/bin/bash
#SBATCH --gres=gpu:1 \
#SBATCH --mem=30G \
#SBATCH -c4 \
#SBATCH --time=10-00:00:00 \
#SBATCH --nodelist=gpu09 \
#SBATCH --output=./examples/wtf38.8。C....out \

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p='/root/autodl-fs/nq_v1'
model="llama7b"
adapter="bottleneck"
rank=256
params="rank${rank}"
task="nq_v1"
ep=1
echo "base model is $base_model"

python raw_finetune.py \
  --base_model "$base_model" \
  --data_path "$data_p" \
  --output_dir "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
  --batch_size 64 \
  --micro_batch_size 2 \
  --num_epochs $ep \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
  --adapter_name bottleneck \
  --bottleneck_size $rank \
#  --eval_on_train "true" \
