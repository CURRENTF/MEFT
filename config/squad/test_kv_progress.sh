#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p='/root/autodl-fs/squad_v2.1'

for ogs in 6144 7777 ; do
  weight="/root/autodl-fs/trained_models/kv-no_cpu-squad_v2-g_topk=0--ogs=${ogs}--ep8"
  python kv_evaluate.py \
    --model LLaMA-7B \
    --adapter "kv-no_cpu" \
    --dataset $data_p \
    --base_model $base_model \
    --lora_weights "${weight}" \
    --test_sample_num 501 \
    --bs 1
done

#  > test_nq_v1_bottleneck.log 2>&1
