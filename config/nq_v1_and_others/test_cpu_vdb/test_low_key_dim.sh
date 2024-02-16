#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH -c4
#SBATCH --time=10-00:00:00
#SBATCH --nodelist=gpu09
#SBATCH --output=./examples/kv_nq_v1.out

#base_model='linhvu/decapoda-research-llama-7b-hf'
#weight='/root/autodl-fs/trained_models/kv-simulate_cpu-nq_v1-g_topk=64--ogs=6144--ep4'
#data_p='/root/autodl-fs/nq_v1'
#
#python kv_evaluate.py \
#  --model LLaMA-7B \
#  --dataset $data_p \
#  --base_model $base_model \
#  --lora_weights $weight \
#  --test_sample_num 501

base_model='linhvu/decapoda-research-llama-7b-hf'
weight='/root/autodl-fs/trained_models/kv-simulate_cpu-nq_v1-g_topk=64--ogs=6144--ep4'
data_p='/root/autodl-fs/nq_v1'

python kv_evaluate.py \
  --model LLaMA-7B \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights $weight \
  --test_sample_num 501 \
  --set_pll_0

#base_model='linhvu/decapoda-research-llama-7b-hf'
#weight='/root/autodl-fs/trained_models/kv-simulate_cpu--g_topk=10--ogs=8192--ep4--pll=1'
#data_p='/root/autodl-fs/nq_v1'
#
#python kv_evaluate.py \
#  --model LLaMA-7B \
#  --dataset $data_p \
#  --base_model $base_model \
#  --lora_weights $weight \
#  --test_sample_num 501
#
#base_model='linhvu/decapoda-research-llama-7b-hf'
#weight='/root/autodl-fs/trained_models/kv-simulate_cpu--g_topk=10--ogs=8192--ep4'
#data_p='/root/autodl-fs/nq_v1'
#
#python kv_evaluate.py \
#  --model LLaMA-7B \
#  --dataset $data_p \
#  --base_model $base_model \
#  --lora_weights $weight \
#  --test_sample_num 501

#  > test_nq_v1_bottleneck.log 2>&1
