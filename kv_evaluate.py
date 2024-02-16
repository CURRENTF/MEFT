import os
import re
from transformers import LlamaForCausalLM
from raw_evaluate_my import *
from modeling.my_llama import KVLlamaForCausalLM, KVLlamaConfig

try:
    from modeling.my_mistral import KVMistralForCausalLM, KVMistralConfig
    from transformers import MistralForCausalLM
except ModuleNotFoundError:
    pass


def _load_model(config_cls, my_model_cls, raw_model_cls, args):
    base_model = args.base_model
    load_8bit = args.load_8bit
    if args.lora_weights != 'none':
        with open(os.path.join(args.lora_weights, "config.json"), "r", encoding='utf-8') as I:
            config = json.load(I)
            if args.explosion:
                config["use_unstable_feature"] = True
                if lim := config.get("limit_total_activated_neurons", 0):
                    config["gpu_topk"] *= 2

        cfg = config_cls.from_pretrained(base_model)
        cfg.set_kv_dict_params(config)
        if args.set_pll_0:
            cfg.pre_look_layers = 0
        model = my_model_cls.from_pretrained(
            base_model,
            config=cfg,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )  # fix zwq
        model.load_added(args.lora_weights)
        model.set_index()
    else:
        if args.lora_weights == 'none':
            if "checkpoint" in base_model:
                with open(os.path.join(base_model, "config.json"), "r", encoding='utf-8') as I:
                    config = json.load(I)
                    if args.explosion:
                        config["use_unstable_feature"] = True
                        if lim := config.get("limit_total_activated_neurons", 0):
                            config["gpu_topk"] *= 2

                cfg = config_cls.from_pretrained(base_model)
                cfg.set_kv_dict_params(config)
                if args.set_pll_0:
                    cfg.pre_look_layers = 0
                model = my_model_cls.from_pretrained(
                    base_model,
                    config=cfg,
                    load_in_8bit=load_8bit,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                model.set_index()
            else:
                model = raw_model_cls.from_pretrained(
                    base_model,
                    load_in_8bit=load_8bit,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
        else:
            raise NotImplementedError

    return model


def load_model(args) -> tuple:
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    lora_weights = args.lora_weights
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

    load_8bit = args.load_8bit
    if 'llama' in args.model.lower() or 'mistral' in args.model.lower():
        if "checkpoint" in base_model:
            if "SLURM_JOB_ID" in os.environ:
                tokenizer = LlamaTokenizer.from_pretrained("/data03/xxxlab_share/llama-7b-hf")
            else:
                tokenizer = LlamaTokenizer.from_pretrained("linhvu/decapoda-research-llama-7b-hf")
        else:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        raise NotImplementedError
    if device != 'cuda':
        raise NotImplementedError
    if 'llama' in args.model.lower():
        model = _load_model(KVLlamaConfig, KVLlamaForCausalLM, LlamaForCausalLM, args)
    elif 'mistral' in args.model.lower():
        model = _load_model(KVMistralConfig, KVMistralForCausalLM, MistralForCausalLM, args)
    else:
        raise ValueError("model not support yet")

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return tokenizer, model


if __name__ == "__main__":
    main(_load_model=load_model)
