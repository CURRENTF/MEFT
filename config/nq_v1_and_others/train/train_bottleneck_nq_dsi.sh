#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH -c4
#SBATCH --time=10-00:00:00
#SBATCH --nodelist=gpu09
#SBATCH --output=./examples/nq_index_bottleneck.out

base_model=""
if [ -v SLURM_JOB_ID ]
then
#  export CUDA_VISIBLE_DEVICES="1"
  echo "in slurm"
  base_model='/data03/xxxlab_share/yahma-llama-7b-hf'
else
  base_model='decapoda-research/llama-7b-hf'
fi

echo "base model is $base_model"

# rojagtap/**natural_questions_clean**  './dataset/gsm8k/my_train.json'
#data_path='rojagtap/natural_questions_clean'
#data_path="$1"
data_path="../datasets/nq_index/"

python raw_finetune.py \
  --base_model "$base_model" \
  --data_path "$data_path" \
  --output_dir !!!!!!!!!!!!!!!!!!!!!!!"/root/autodl-fs/trained_models/"
  --batch_size 64 \
  --micro_batch_size 1 \
  --num_epochs 8 \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --wandb_project "meft_all_exps" \
  --wandb_run_name "${model}-${task}-${adapter}-${params}-ep${ep}" \
  --adapter_name bottleneck \
  --bottleneck_size 384 \
#  --eval_on_train "true" \
