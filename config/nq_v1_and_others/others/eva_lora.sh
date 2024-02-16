python raw_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset "gsm8k" \
    --base_model '/data03/xxxlab_share/yahma-llama-7b-hf' \
    --lora_weights './trained_models/llama-lora'