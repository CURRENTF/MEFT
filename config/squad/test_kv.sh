#!/bin/bash

#base_model='linhvu/decapoda-research-llama-7b-hf'
#data_p='/root/autodl-fs/squad_v2.1'
base_model='/data03/xxxlab_share/llama-7b-hf'
data_p="../datasets/squad_v2.1"
export TRANSFORMERS_CACHE="../hub"
export HF_HOME="../hub"

#cps=(965 1930 2895 3860)
#cps=(1930 5790 9650 13510 17370)
#for cp in "${cps[@]}"
#do
#  weight="/root/autodl-fs/trained_models/kv-no_cpu-squad_v2-g_topk=0--ogs=1024--ep16/checkpoint-${cp}"
#  python kv_evaluate.py \
#    --model LLaMA-7B \
#    --adapter "kv-no_cpu" \
#    --dataset $data_p \
#    --base_model $base_model \
#    --lora_weights "${weight}" \
#    --test_sample_num 501 \
#    --bs 1
#done

#weight="/root/autodl-fs/trained_models/llama-squad_v2-kv-topk64-add8464-MOE-add_progress-ep8/checkpoint-1446"
weight="./trained_models/llama-squad_v2-kv-topk64-add8464-MOE-add_progress-ep8"

python kv_evaluate.py \
  --model LLaMA-7B \
  --adapter "KV" \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights "${weight}" \
  --test_sample_num 501 \
  --bs 1 \
  --set_pll_0 \
  --explosion

