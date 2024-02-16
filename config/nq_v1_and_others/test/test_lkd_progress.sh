#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p='/root/autodl-fs/nq_v1'


for lkd in 128 256 512 784 1024 2048; do
  weight="/root/autodl-fs/trained_models/kv-simulate_cpu_ablation-lkd=${lkd}-nq_v1-g_topk=64--ogs=4096--ep4"
  python kv_evaluate.py \
    --model LLaMA-7B \
    --dataset $data_p \
    --base_model $base_model \
    --lora_weights $weight \
    --test_sample_num 501 \
    --set_pll_0

done