#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH -c4
#SBATCH --time=10-00:00:00
#SBATCH --nodelist=gpu09
#SBATCH --output=./examples/kv_nq_v1.out

base_model='linhvu/decapoda-research-llama-7b-hf'

python kv_evaluate.py \
  --model LLaMA-7B \
  --dataset "/root/autodl-fs/nq_v1" \
  --base_model $base_model \
  --lora_weights './trained_models/kv-no_cpu-g_topk=64--ogs=1024--ep8--nq_v1/' \
  --test_sample_num 20

#  > test_nq_v1_bottleneck.log 2>&1
