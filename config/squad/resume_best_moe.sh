#!/bin/bash
python /root/autodl-fs/kvxxx/LLM-Adapters/finetune.py --data_path /root/autodl-fs/squad_v2 \
--output_dir /root/autodl-fs/trained_models/llama-squad_v2-kv-topk32-add6400-MOE-bbs256-ep4 \
 --batch_size 256 --micro_batch_size 4 --num_epochs 4 --learning_rate 2e-4 \
  --cutoff_len 256 --wandb_project meft_all_exps --wandb_run_name llama-squad_v2-kv-topk32-add6400-MOE-bbs256-ep4 \
   --base_model linhvu/decapoda-research-llama-7b-hf --on_gpu_size 6400 --eval_step 0.2 \
   --save_step 0.2 --gpu_topk 32 --optimizer_kv adamw_torch --pre_look_layers 0 --location all \
   --save_total_limit 5 --moe_style --init_weight_from_raw_model --notrain_on_inputs --resume_from_checkpoint \
   --frozen_key
