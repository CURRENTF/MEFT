#!/bin/bash

task="$1"
data_p="/root/autodl-fs/${task}"
weight="$2"
sample_num="$3"
base_model="$4"
model_type="$5"

echo $task
echo $weight
echo $sample_num
echo $base_model
echo $model_type

python kv_evaluate.py \
  --model $model_type \
  --dataset $data_p \
  --base_model $base_model \
  --lora_weights $weight \
  --test_sample_num $sample_num \
  --set_pll_0 \
  --explosion
