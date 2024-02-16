#export CUDA_VISIBLE_DEVICES="0"
python evaluate.py  \
    --model "LLaMA-7B" \
    --dataset "gsm8k" \
    --base_model './trained_models/llama-kv/pretrained'