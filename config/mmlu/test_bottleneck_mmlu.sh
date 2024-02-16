#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH -c4
#SBATCH --time=10-00:00:00
#SBATCH --nodelist=gpu09
#SBATCH --output=./examples/test_bottlenect_nq_v1.out

base_model=""
if [ -v SLURM_JOB_ID ]
then
#  export CUDA_VISIBLE_DEVICES="1"
  base_model='/data03/xxxlab_share/llama-7b-hf'
  data_p="../datasets/nq_v1"
  weight='./trained_models/llama-bottleneck_nq_v1_aligned'
else
  base_model='linhvu/decapoda-research-llama-7b-hf'
  data_p="/root/autodl-fs/mmlu_v1"
  weight='/root/autodl-fs/trained_models/bottleneck-mmlu_v1-256-ep4'
#  base_model='decapoda-research/llama-7b-hf'
fi

python raw_evaluate_my.py \
  --model LLaMA-7B \
  --adapter AdapterH \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights $weight \
  --test_sample_num 50 \
  --bs 1 \

#  > test_nq_v1_bottleneck.log 2>&1
