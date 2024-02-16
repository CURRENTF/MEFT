python convert_model.py \
  --data_path 'math_data.json' \
  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
  --batch_size 16 \
  --micro_batch_size 1 \
  --num_epochs 16 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
  --base_model 'decapoda-research/llama-7b-hf' \
  --add_num 32 \
  --location "mid" \
  --add_layer_num 30
#  --base_model '/data03/xxxlab_share/llama-7b-hf' --load_8bit true  'decapoda-research/llama-7b-hf'