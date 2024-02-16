#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p='/root/autodl-fs/nq_v1'
#weight="/root/autodl-fs/trained_models/llama7b-nq_v1-kv-topk128-add4096-MOE-softmax-ep2"
#python kv_evaluate.py \
#  --model LLaMA-7B \
#  --dataset $data_p \
#  --base_model $base_model \
#  --lora_weights $weight \
#  --test_sample_num 501 \
#  --set_pll_0 \
#  --explosion

weight="/root/autodl-fs/trained_models/llama7b-nq_v1-kv-topk128-add4096-MOE-ep4"
python kv_evaluate.py \
  --model LLaMA-7B \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights $weight \
  --test_sample_num 501 \
  --set_pll_0 \
  --explosion

#base_model='linhvu/decapoda-research-llama-7b-hf'
#data_p='/root/autodl-fs/squad_v2'
#weight="/root/autodl-fs/trained_models/llama-squad_v2-kv-topk32-add1536-ep8"
#python kv_evaluate.py \
#  --model LLaMA-7B \
#  --dataset $data_p \
#  --base_model $base_model \
#  --lora_weights $weight \
#  --test_sample_num 501 \
#  --set_pll_0