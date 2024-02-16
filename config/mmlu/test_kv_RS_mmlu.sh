#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH -c4
#SBATCH --time=10-00:00:00
#SBATCH --nodelist=gpu09
#SBATCH --output=./examples/kv_nq_v1.out

base_model=""
if [ -v SLURM_JOB_ID ]
then
#  export CUDA_VISIBLE_DEVICES="1"
#  base_model='/data03/xxxlab_share/llama-7b-hf'
#  weight='./trained_models/kv-no_cpu-g_topk=64--ogs=1024--ep8'
  data_p="../datasets/mmlu_v1"
else
  base_model='linhvu/decapoda-research-llama-7b-hf'
  weight='/root/autodl-fs/trained_models/kv-no_cpu-mmlu_v1-g_topk=64--ogs=1024--ep4'
  data_p='/root/autodl-fs/mmlu_v1'
#  base_model='decapoda-research/llama-7b-hf'
fi

python kv_evaluate.py \
  --model LLaMA-7B \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights $weight \
  --test_sample_num 50 \
  --bs 1 \

#  > test_nq_v1_bottleneck.log 2>&1
