import os
import sys
from typing import List
import random
import datasets
import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union
import wandb

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from their_peft.src.peft import (  # noqa: E402
    LoraConfig,
    BottleneckConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel  # noqa: F402
from modeling.my_mistral import KVMistralForCausalLM, KVMistralConfig


def my_prepare_model_for_int8_training(model, use_gradient_checkpointing=False):
    r"""
        This method wraps the entire protocol for preparing a model before running a training. This includes:
            1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
            head to fp32

        Args:
            model, (`transformers.PreTrainedModel`):
                The loaded model from `transformers`
        """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    # freeze base model's layers
    # model.freeze_other_params()
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad, param.dtype)

    # cast all non INT8 parameters to fp32
    # for param in model.parameters():
    #     if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
    #         param.data = param.data.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        print('code here')
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        adapter_name: str = "lora",
        load_8bit: bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        use_gradient_checkpointing: bool = False,
        eval_step: float = 0.25,
        save_step: float = 0.25,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        eval_on_train: bool = False,
        train_size=-1,
        val_size=-1,
        save_total_limit: int = 1,
        add_num=1234,
        location='all',
        add_layer_num=16,
        add_gate: bool = False,
        added_on_cpu: bool = False,  # 是否将一部分kv放在cpu上
        async_compute: bool = False,
        on_gpu_size=512,
        pre_look_layers=0,
        optimizer_kv="adamw_torch",
        gpu_topk=-1,  # when > 0, try topk gpu training
        use_torch_vecdb: bool = False,
        load_pkl: bool = False,
        check_similarity: bool = False,
        measure_1_decoder_layer_time: bool = False,
        frozen_key: bool = False,
        load_best_checkpoint_at_end: bool = False,
        low_key_dim=0,
        seq_split_train_len=0,
        yingji_load_data: bool = False,
        depend_on_update: bool = True,
        all_query_then_update: bool = False,
        simulate_recall: float = 1.0,
        use_glass_vdb: bool = False,
        simulate_ogs: int = 3072,
        cpp_mode: bool = False,
        num_group: int = 1,
        manual_seed: int = 42,
        use_list_load_data: bool = False,
        load_in_8bit: bool = False,
        fix_wrong_special_token: bool = False,
        use_bf: bool = False,
        grad_cut_value: float = 1.0,
        init_weight_from_raw_model: bool = False,
        use_peft_just_for_skip_bug: bool = False,
        moe_style: bool = False,
        n_probe: int = 2,
        use_unstable_feature: bool = False,
        moe_softmax_score: bool = False,
        moe_expert_factor: int = 1,
        moe_dropout: bool = False,
        limit_total_activated_neurons: int = 0,
        train_layernorm: bool = False,
        lr_type="linear",
        simulate_moe: bool = False,
        map_auto: bool = False,
):
    # todo = resume from checkpoint
    transformers.set_seed(42)
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    cfg = KVMistralConfig.from_pretrained(base_model)
    cfg.set_kv_dict_params(kv_config := {
        "add_num": add_num,
        "add_gate": add_gate,
        "location": location,
        "add_layer_num": add_layer_num,
        "added_on_cpu": added_on_cpu,
        "on_gpu_size": on_gpu_size,
        "pre_look_layers": pre_look_layers,
        "eval_on_train": eval_on_train,
        "train_size": train_size,
        "val_size": val_size,
        "gpu_topk": gpu_topk,
        "bbs": batch_size,
        "bs": micro_batch_size,
        "use_torch_vecdb": use_torch_vecdb,
        "check_similarity": check_similarity,
        "measure_1_decoder_layer_time": measure_1_decoder_layer_time,
        "frozen_key": frozen_key,
        "optimizer_kv": optimizer_kv,
        "low_key_dim": low_key_dim,
        "seq_split_train_len": seq_split_train_len,
        "async_compute": async_compute,
        "depend_on_update": depend_on_update,
        "all_query_then_update": all_query_then_update,
        "simulate_recall": simulate_recall,
        "use_glass_vdb": use_glass_vdb,
        "simulate_ogs": simulate_ogs,
        "cpp_mode": cpp_mode,
        "learning_rate": learning_rate,
        "num_group": num_group,
        "moe_style": moe_style,
        "n_probe": n_probe,
        "use_unstable_feature": use_unstable_feature,
        "moe_softmax_score": moe_softmax_score,
        "moe_expert_factor": moe_expert_factor,
        "moe_dropout": moe_dropout,
        "limit_total_activated_neurons": limit_total_activated_neurons,
        "run_name": wandb_run_name,
        "train_layernorm": train_layernorm,
        "lr_type": lr_type,
        "simulate_moe": simulate_moe,
        "map_auto": map_auto,
    })
    from transformers import MistralForCausalLM
    if load_8bit:
        model = KVMistralForCausalLM.from_pretrained(
            base_model,
            config=cfg,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = KVMistralForCausalLM.from_pretrained(
            base_model,
            config=cfg,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )

    if model.config.model_type == "llama":
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        tokenizer = LlamaTokenizer.from_pretrained(
            'linhvu/decapoda-research-llama-7b-hf',
            add_eos_token=True if fix_wrong_special_token else False
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    if fix_wrong_special_token:
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    model = my_prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    if adapter_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            modules_to_save=["add_up_proj_gpu", "add_down_proj_gpu", "add_gates", "add_experts"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "bottleneck":
        config = BottleneckConfig(
            bottleneck_size=bottleneck_size,
            non_linearity=non_linearity,
            adapter_dropout=adapter_dropout,
            use_parallel_adapter=use_parallel_adapter,
            use_adapterp=use_adapterp,
            target_modules=target_modules,
            scaling=scaling,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "prefix-tuning":
        config = PrefixTuningConfig(
            num_virtual_tokens=num_virtual_tokens,
            task_type="CAUSAL_LM",
        )
    model = get_peft_model(model, config)
    model.save_pretrained = model.base_model.save_pretrained
    model.freeze_other_params_and_set_kv_float32()
    if adapter_name == "prefix-tuning":
        model.to('cuda')

    try:
        data = datasets.load_from_disk(data_path)
    except Exception as e:
        # data = load_dataset(data_path)
        print("目前只支持自己造的数据")
        raise NotImplementedError

    if 'nq_v1' in data_path:
        if train_size > 0:
            data['train'] = data['train'].select(list(range(train_size)))
        if 'validation' in data and val_size > 0:
            data['validation'] = data['validation'].select(list(range(val_size)))
        # data = data.rename_columns({'question': 'instruction', 'answer': 'output'})
        data = data.map(lambda sample: {
            "input": "", "answer": "",
            "instruction": sample["question"], "output": ''.join(sample["long_answers"])
        })

    if eval_on_train:
        data['validation'] = data['train']
        if val_size > 0:
            data['validation'] = data['validation'].select(list(range(val_size)))

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    print(f"has valid and eval_on_train={eval_on_train}")
    if yingji_load_data:
        # x = data["train"].shuffle().to_list()
        # n = [generate_and_tokenize_prompt(_) for _ in tqdm(x)]
        # data['train'] = datasets.Dataset.from_list(n)
        #
        # try:
        #     x = data["validation"].shuffle().to_list()
        #     n = [generate_and_tokenize_prompt(_) for _ in tqdm(x)]
        #     data['validation'] = datasets.Dataset.from_list(n)
        # except KeyError:
        #     x = data["valid"].shuffle().to_list()
        #     n = [generate_and_tokenize_prompt(_) for _ in tqdm(x)]
        #     data['valid'] = datasets.Dataset.from_list(n)

        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        try:
            val_data = data["validation"].shuffle().map(generate_and_tokenize_prompt)
        except KeyError:
            val_data = data["valid"].shuffle().map(generate_and_tokenize_prompt)

    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=6)
        try:
            val_data = data["validation"].shuffle().map(generate_and_tokenize_prompt, num_proc=6)
        except KeyError:
            val_data = data["valid"].shuffle().map(generate_and_tokenize_prompt, num_proc=6)

    train_data = train_data.shuffle(seed=888)

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    wandb.init(
        project=wandb_project,
        name=wandb_run_name
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.02,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_step,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            logging_nan_inf_filter=False,
            max_grad_norm=0.4,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    with open(f'./examples/{os.path.split(base_model)[-1]}--structure.txt', 'w', encoding='utf-8') as O:
        for n, p in model.named_parameters():
            print(n, p.requires_grad, p.shape, p.dtype, file=O)

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    wandb.log(kv_config)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if "is_knowledge" in data_point:
        if data_point["is_knowledge"]:
            return (f'{data_point["instruction"]} \n'
                    f'{data_point["input"]} \n'
                    f'{data_point["output"]}')

    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}

                ### Input:
                {data_point["input"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501


if __name__ == "__main__":
    fire.Fire(train)
