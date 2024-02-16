#!/bin/bash

base_model='linhvu/decapoda-research-llama-7b-hf'
data_p="/root/autodl-fs/nq_v1"

# 假设ranks是一个已经定义的数组，例如 ranks=(rank1 rank2 rank3)
#ranks=(4 8 32 64 128 256 384 512) # 这里需要您提供具体的rank值
#ranks=(4 32 128 256 384 512)
ranks=(512 1024 1536)
# 遍历ranks数组
for rank in "${ranks[@]}"
do
  weight="/root/autodl-fs/trained_models/llama-bottleneck-nq_v1-${rank}-progress"
  python raw_evaluate_my.py \
    --model LLaMA-7B \
    --adapter AdapterH \
    --dataset "$data_p" \
    --base_model "$base_model" \
    --lora_weights "$weight" \
    --test_sample_num 501
done
