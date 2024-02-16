#!/bin/bash
task="squad_v2"
base_model='mistralai/Mistral-7B-v0.1'
data_path="/root/autodl-fs/${task}"

#cps=(965 1930 2895 3860)
cps=(1930 3860 5790 7720)
for cp in "${cps[@]}"
do
  weight="/root/autodl-fs/trained_models/mistral-kv-no_cpu-squad_v2-g_topk=0--ogs=512--ep8/checkpoint-${cp}"
  python kv_evaluate.py \
    --model mistral \
    --adapter "kv-mistral" \
    --dataset $data_path \
    --base_model $base_model \
    --lora_weights "${weight}" \
    --test_sample_num 501 \
    --bs 1
done

#  > test_nq_v1_bottleneck.log 2>&1
