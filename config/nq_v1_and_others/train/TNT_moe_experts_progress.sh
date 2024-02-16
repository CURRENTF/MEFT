#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
task="nq_v1"
data_path="../datasets/${task}"
base_model='/data03/xxxlab_share/llama-7b-hf'
export TRANSFORMERS_CACHE="../hub"
export HF_HOME="../hub"
### ?
#base_model='linhvu/decapoda-research-llama-7b-hf'
#data_path="/root/autodl-fs/${task}"

echo "base model is $base_model"
#export DEBUG_KV=1
#0.25 0.1 0.5 1 0.1 1 4 2 10
for X in 0.25 0.5; do
  gpu_topk=64
  bs=4
  ep=1
  on_gpu_size=6400
  model="llama"
  adapter="kv"
  echo "2/${X}" | bc
  n_probe=$(echo "2/${X}" | bc)
  if [ "${n_probe}" -eq "0" ]
  then
      n_probe=1
  fi

  echo "n_probe=${n_probe}"
  bbs=128
  moe_expert_factor=$X
  params="topk${gpu_topk}-add${on_gpu_size}-MOE-softmax-np${n_probe}-bbs${bbs}-factor${X}"

  mkdir "./trained_models"

  python finetune.py \
    --data_path "$data_path" \
    --output_dir "./trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
    --batch_size $bbs \
    --micro_batch_size $bs \
    --num_epochs "$ep" \
    --learning_rate 1e-4 \
    --cutoff_len 256 \
    --wandb_project "meft_all_exps" \
    --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
    --base_model "$base_model"  \
    --on_gpu_size "$on_gpu_size" \
    --eval_step 0.25 \
    --save_step 0.25 \
    --gpu_topk "$gpu_topk" \
    --optimizer_kv "adamw_torch" \
    --pre_look_layers 0 \
    --location "all" \
    --save_total_limit 1 \
    --moe_style \
    --moe_expert_factor $moe_expert_factor \
    --n_probe "$n_probe" \
    --init_weight_from_raw_model \
    --fix_wrong_special_token \
    --use_unstable_feature \

  # ????
  #  --resume_from_checkpoint
  #  --notrain_on_inputs
  #  --moe_softmax_score

  weight="./trained_models/${model}-${task}-${adapter}-${params}-ep${ep}"
  python kv_evaluate.py \
    --model LLaMA-7B \
    --adapter "kv" \
    --dataset $data_path \
    --base_model $base_model \
    --lora_weights "${weight}" \
    --test_sample_num 501 \
    --bs 1 \
    --explosion


  d=$(date)
  mv "./trained_models/${model}-${task}-${adapter}-${params}-ep${ep}" \
     "/root/autodl-fs/trained_models/${model}-${task}-${adapter}-${params}-ep${ep}--${d}"


done



