#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p='/root/autodl-fs/squad_v2.1'

##cps=(965 1930 2895 3860)
#cps=(1930 3860 5790 7720)
#for cp in "${cps[@]}"
#do
#  weight="/root/autodl-fs/trained_models/llama-lora-squad_v2-128-ep8/checkpoint-${cp}"
#  mv "/root/autodl-fs/trained_models/llama-lora-squad_v2-128-ep8/checkpoint-${cp}/pytorch_model.bin" \
#  "/root/autodl-fs/trained_models/llama-lora-squad_v2-128-ep8/checkpoint-${cp}/adapter_model.bin"
#  cp "/root/autodl-fs/trained_models/llama-lora-squad_v2-128-ep8/adapter_config.json" \
#  "/root/autodl-fs/trained_models/llama-lora-squad_v2-128-ep8/checkpoint-${cp}/adapter_config.json"
#  python raw_evaluate_my.py \
#    --model LLaMA-7B \
#    --adapter LoRA \
#    --dataset $data_p \
#    --base_model $base_model \
#    --lora_weights "${weight}" \
#    --test_sample_num 501 \
#    --bs 1
#done

weight="/root/autodl-fs/trained_models/llama7b-squad_v2-lora-rank1024-ep8"
python raw_evaluate_my.py \
  --model LLaMA-7B \
  --adapter LoRA \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights "${weight}" \
  --test_sample_num 501 \
  --bs 1

#  > test_nq_v1_bottleneck.log 2>&1
