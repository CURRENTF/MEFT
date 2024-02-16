#!/bin/bash

base_model='mistralai/Mistral-7B-v0.1'

#data_p="/root/autodl-fs/gsm8k_v1"
#weight="/root/autodl-fs/trained_models/mistral-metamathqa_v1-lora-rank64-ep1"
#python raw_evaluate_my.py \
#  --model LLaMA-7B \
#  --adapter LORA \
#  --dataset "$data_p" \
#  --base_model "$base_model" \
#  --lora_weights "$weight" \
#  --test_sample_num 501

data_p="/root/autodl-fs/gsm8k_v1"
#weight="/root/autodl-fs/trained_models/mistral-metamathqa_v1-bottleneck-rank256-ep1"
weight="./trained_models/mistral-metamathqa_v1-bottleneck-rank256-ep1"
python raw_evaluate_my.py \
  --model LLaMA-7B \
  --adapter Bottleneck \
  --dataset "$data_p" \
  --base_model "$base_model" \
  --lora_weights "$weight" \
  --test_sample_num 501

#data_p="/root/autodl-fs/squad_v2.1"
#weight="/root/autodl-fs/trained_models/mistral-lora-squad_v2-256-ep8"
#python raw_evaluate_my.py \
#  --model LLaMA-7B \
#  --adapter LORA \
#  --dataset "$data_p" \
#  --base_model "$base_model" \
#  --lora_weights "$weight" \
#  --test_sample_num 501
#
#data_p="/root/autodl-fs/squad_v2.1"
#weight="/root/autodl-fs/trained_models/mistral-bottleneck-squad_v2-256-ep8"
#python raw_evaluate_my.py \
#  --model LLaMA-7B \
#  --adapter Bottleneck \
#  --dataset "$data_p" \
#  --base_model "$base_model" \
#  --lora_weights "$weight" \
#  --test_sample_num 501
#
#data_p="/root/autodl-fs/nq_v1"
#weight="/root/autodl-fs/trained_models/mistral-lora-nq_v1-256-ep4"
#python raw_evaluate_my.py \
#  --model LLaMA-7B \
#  --adapter LORA \
#  --dataset "$data_p" \
#  --base_model "$base_model" \
#  --lora_weights "$weight" \
#  --test_sample_num 501
#
#data_p="/root/autodl-fs/nq_v1"
#weight="/root/autodl-fs/trained_models/mistral-bottleneck-nq_v1-256-ep4"
#python raw_evaluate_my.py \
#  --model LLaMA-7B \
#  --adapter Bottleneck \
#  --dataset "$data_p" \
#  --base_model "$base_model" \
#  --lora_weights "$weight" \
#  --test_sample_num 501
#
#data_p="/root/autodl-fs/pubmedqa_v2"
#weight="/root/autodl-fs/trained_models/mistral-pubmedqa_v2-lora-rank256-ep4"
#python raw_evaluate_my.py \
#  --model LLaMA-7B \
#  --adapter LORA \
#  --dataset "$data_p" \
#  --base_model "$base_model" \
#  --lora_weights "$weight" \
#  --test_sample_num 501
#
#data_p="/root/autodl-fs/pubmedqa_v2"
#weight="/root/autodl-fs/trained_models/mistral-pubmedqa_v2-bottleneck-rank256-ep4"
#python raw_evaluate_my.py \
#  --model LLaMA-7B \
#  --adapter Bottleneck \
#  --dataset "$data_p" \
#  --base_model "$base_model" \
#  --lora_weights "$weight" \
#  --test_sample_num 501
#
#
#data_p="/root/autodl-fs/tool_v1"
#weight="/root/autodl-fs/trained_models/mistral-tool_v1-bottleneck-rank64-ep1"
#python raw_evaluate_my.py \
#  --model LLaMA-7B \
#  --adapter Bottleneck \
#  --dataset "$data_p" \
#  --base_model "$base_model" \
#  --lora_weights "$weight" \
#  --test_sample_num 451
#
#data_p="/root/autodl-fs/tool_v1"
#weight="/root/autodl-fs/trained_models/mistral-tool_v1-lora-rank-ep1"
#python raw_evaluate_my.py \
#  --model LLaMA-7B \
#  --adapter LORA \
#  --dataset "$data_p" \
#  --base_model "$base_model" \
#  --lora_weights "$weight" \
#  --test_sample_num 451
