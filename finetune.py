import os
import sys
import hashlib
import datasets
import fire
import torch
import transformers
import wandb

from raw_finetune import generate_prompt
from transformers import AutoTokenizer, LlamaTokenizer
from modeling import my_llama, my_trainer

# 获取当前Python的主版本号和次版本号
major_version, minor_version = sys.version_info[:2]

# 比较当前Python的版本号
if (major_version, minor_version) >= (3, 10):
    from modeling import my_mistral

transformers.logging.set_verbosity_info()
logger = my_llama.create_file_logger("running", "examples/run.log")


def hash_string(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()


class AnyObj:
    def __init__(self, d):
        self.__dict__ = d


def make_str_by_dict(d):
    # convert dict to string
    folder_name = str(d).strip("{}")
    # replace problematic characters
    folder_name = folder_name.replace(" ", "").replace(',', '，').replace(":", "：").replace("'", "")
    return hash_string(folder_name)


def get_my_kv_model(model_name, kv_cfg, load_in_8bit=False, map_auto=False):
    config_cls, model_cls = my_llama.KVLlamaConfig, my_llama.KVLlamaForCausalLM
    params_type = torch.float16
    if 'llama' in model_name.lower():
        config_cls, model_cls = my_llama.KVLlamaConfig, my_llama.KVLlamaForCausalLM
    elif 'mistral' in model_name.lower():
        config_cls, model_cls = my_mistral.KVMistralConfig, my_mistral.KVMistralForCausalLM
        # params_type = torch.bfloat16
    else:
        raise ValueError("not support yet")

    print("set new cfg ... ")
    cfg = config_cls.from_pretrained(model_name)
    cfg.set_kv_params(kv_cfg)
    print("init new model from config ... ")
    from transformers import BitsAndBytesConfig
    n_model = model_cls.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=cfg,
        torch_dtype=params_type,
        device_map=0 if not map_auto else 'auto',
        load_in_8bit=load_in_8bit,
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            llm_int8_skip_modules=["add_up_proj_gpu", "add_down_proj_gpu", "add_up_low_key"],
        ) if load_in_8bit else None,
    )
    print("Init Done.")
    if load_in_8bit:
        return n_model
    else:
        return n_model.cuda()


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "./dataset/gsm8k/real/train.jsonl",
        output_dir: str = "./lora-alpaca",
        adapter_name: str = "kv",
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        use_gradient_checkpointing: bool = False,
        eval_step: float = 0.25,
        save_step: float = 0.25,
        warmup_ratio: float = 0.02,
        logging_steps: int = 10,
        # # lora hyperparams
        # add_lora: bool = False,
        # lora_r: int = 8,
        # lora_alpha: int = 16,
        # lora_dropout: float = 0.05,
        # lora_target_modules: str = 'q_proj,k_proj',  # split with ","
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        # kvx params
        add_num=4096,
        location='all',
        add_layer_num=16,
        add_gate: bool = False,
        added_on_cpu: bool = False,  # 是否将一部分kv放在cpu上
        async_compute: bool = False,
        on_gpu_size=512,
        pre_look_layers=0,
        optimizer_kv="adamw_torch",
        eval_on_train: bool = False,
        train_size=-1,
        val_size=-1,
        gpu_topk=-1,  # when > 0, try topk gpu training
        use_torch_vecdb: bool = False,
        load_pkl: bool = False,
        check_similarity: bool = False,
        measure_1_decoder_layer_time: bool = False,
        frozen_key: bool = False,
        load_best_checkpoint_at_end: bool = False,
        save_total_limit=4,
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
        moe_expert_factor: float = 2.0,
        moe_dropout: bool = False,
        limit_total_activated_neurons: int = 0,
        train_layernorm: bool = False,
        lr_type="linear",
        simulate_moe: bool = False,
        map_auto: bool = False,
):
    # todo resume from checkpoint
    print("depend_on_update ==", depend_on_update)
    print("frozen key ==", frozen_key)
    if added_on_cpu:
        if 'adam' in optimizer_kv:
            raise NotImplementedError
    if async_compute:
        if not added_on_cpu:
            raise NotImplementedError
        if pre_look_layers <= 1:
            raise NotImplementedError

    if limit_total_activated_neurons:
        raise ValueError("will harm performance")

    transformers.set_seed(manual_seed)
    # if optimizer_kv != 'sgd' and added_on_cpu:
    #     raise NotImplementedError

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

    try:
        data = datasets.load_from_disk(data_path)
    except Exception as e:
        print("目前只支持自己造的数据")
        raise e

    if train_size > 0:
        data['train'] = data['train'].select(list(range(train_size)))
    if 'validation' in data and val_size > 0:
        data['validation'] = data['validation'].select(list(range(val_size)))
    if 'valid' in data and val_size > 0:
        data['valid'] = data['valid'].select(list(range(val_size)))

    print(data)
    print(f'data path: {data_path}, data size: {len(data)}')

    if 'nq_v1' in data_path:
        # data = data.rename_columns({'question': 'instruction', 'answer': 'output'})
        data = data.map(lambda sample: {
            "input": "", "answer": "",
            "instruction": sample["question"], "output": ''.join(sample["long_answers"])
        })
    elif "nq_index" in data_path:
        pass

    if eval_on_train:
        data['validation'] = data['train']
        if val_size > 0:
            data['validation'] = data['validation'].select(list(range(val_size)))

    kv_config = {
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
    }

    # abbs = set()
    # def get_abb(k):
    #     l = k.split('_')
    #     abb = ''.join([s[0] for s in l])
    #     _ = 2
    #     while abb in abbs:
    #         abb = ''.join([s[:_] for s in l])
    #         _ += 1
    #     abbs.add(abb)
    #     return abb
    #
    # wandb_run_name = '-'.join(
    #     [f"{get_abb(k)}{v}" for k, v in kv_config.items()]
    # )
    # kv_config["run_name"] = wandb_run_name
    # output_dir = f"./trained_models/{wandb_run_name}"

    kv_cfg = AnyObj(kv_config)
    model = get_my_kv_model(base_model, kv_cfg, load_in_8bit, map_auto=map_auto)
    if init_weight_from_raw_model:
        model.reinit_added_weight()

    if use_gradient_checkpointing:
        model.enable_input_require_grads()
        # from peft import prepare_model_for_kbit_training
        # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)

    if load_in_8bit:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

        # model.freeze_other_params_and_set_kv_float32()
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
        lora_config = LoraConfig(  # fake lora, will freeze`
            r=1,
            target_modules=["q_proj"],  # 最小化显存占用
            layers_to_transform=0,
            modules_to_save=["add_up_proj_gpu", "add_down_proj_gpu", "add_up_low_key"],
            bias='none',
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)

    if use_peft_just_for_skip_bug or model.config.model_type == "mistral":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

        lora_config = LoraConfig(  # fake lora, will freeze`
            r=1,
            target_modules=["q_proj"],  # 最小化显存占用
            layers_to_transform=30,
            modules_to_save=["add_up_proj_gpu", "add_down_proj_gpu", "add_up_low_key"],
            bias='none',
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)

    model.freeze_other_params_and_set_kv_float32()
    model.my_print_trainable_parameters()  # Be more transparent about the % of trainable params.
    model.set_index()

    if model.config.model_type == "llama" or model.config.model_type == "mistral":
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        tokenizer = LlamaTokenizer.from_pretrained(
            # 'linhvu/decapoda-research-llama-7b-hf',
            base_model,
            add_eos_token=True if fix_wrong_special_token else False
        )
    else:
        raise NotImplementedError
        # tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

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

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + \
                                              tokenized_full_prompt["labels"][user_prompt_len:]
            # could be sped up, probably
        return tokenized_full_prompt

    if isinstance(resume_from_checkpoint, str):
        model.load_added(resume_from_checkpoint)
        resume_from_checkpoint = False

    print(f"has valid and eval_on_train={eval_on_train}")
    if use_list_load_data:
        cache_path = f"/root/autodl-fs/data_cache/{hash_string(data_path)}/cache.pkl"
        if os.path.exists(cache_path):
            train_data, val_data = torch.load(cache_path)
        else:
            from tqdm import tqdm
            x = data["train"].shuffle().to_list()
            n = [generate_and_tokenize_prompt(_) for _ in tqdm(x)]
            data['train'] = datasets.Dataset.from_list(n)
            if 'validation' in data:
                x = data["validation"].shuffle().to_list()
                n = [generate_and_tokenize_prompt(_) for _ in tqdm(x)]
                data['validation'] = datasets.Dataset.from_list(n)
            elif 'valid' in data:
                x = data["valid"].shuffle().to_list()
                n = [generate_and_tokenize_prompt(_) for _ in tqdm(x)]
                data['valid'] = datasets.Dataset.from_list(n)

            train_data = data['train']
            val_data = data['valid'] if 'valid' in data else data['validation']
            # cache now
            base_path = os.path.split(cache_path)[0]
            os.makedirs(base_path, exist_ok=True)
            torch.save((train_data, val_data), cache_path)
    else:
        if yingji_load_data:
            train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
            try:
                val_data = data["validation"].shuffle().map(generate_and_tokenize_prompt)
            except KeyError:
                val_data = data["valid"].shuffle().map(generate_and_tokenize_prompt)

        else:

            # if data_load_fix_maybe:
            #     import multiprocess.context as ctx
            #     ctx._force_start_method('spawn')

            train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=6)
            try:
                val_data = data["validation"].shuffle().map(generate_and_tokenize_prompt, num_proc=6)
            except KeyError:
                val_data = data["valid"].shuffle().map(generate_and_tokenize_prompt, num_proc=6)

        if not ddp and torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            model.is_parallelizable = True
            model.model_parallel = True

    print("train data len: ", len(train_data))
    print("what's archived in train_data ?")
    print(train_data)

    columns_should_remove = ['output', 'raw_answer', 'input', 'question', 'instruction', 'answer', 'document',
                             "long_answers", "short_answers", "id", "title", "text", "text_id", "is_knowledge"]
    cols = [c for c in columns_should_remove if c in train_data.column_names]
    train_data = train_data.remove_columns(cols)
    cols = [c for c in columns_should_remove if c in val_data.column_names]
    val_data = val_data.remove_columns(cols)

    wandb.init(
        project=wandb_project,
        name=wandb_run_name
    )

    trainer_cls = transformers.Trainer
    arg_cls = transformers.TrainingArguments
    if added_on_cpu and 'adam' in optimizer_kv:
        trainer_cls = my_trainer.LowKeyTrainer

    O = open("examples/check_model_structure.txt", "w", encoding='utf-8')
    try:
        print(model, file=O)
    finally:
        O.close()

    train_data = train_data.shuffle(seed=111)
    trainer = trainer_cls(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=arg_cls(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True if not use_bf else False,
            bf16=True if use_bf else False,
            logging_steps=logging_steps,
            optim=optimizer_kv,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=save_total_limit,
            eval_steps=eval_step,
            save_steps=save_step,
            output_dir=output_dir,
            load_best_model_at_end=load_best_checkpoint_at_end,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            remove_unused_columns=False,
            label_names=["labels"],
            dataloader_num_workers=2,
            max_grad_norm=grad_cut_value,
            gradient_checkpointing=use_gradient_checkpointing,
            lr_scheduler_type=lr_type,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    # if "adamw" in optimizer_kv:
    if model.config.model_type == "llama":
        my_llama.set_trainer_pointer(trainer)
    elif model.config.model_type == "mistral":
        my_mistral.set_trainer_pointer(trainer)

    wandb.log(kv_config)

    with open(f'./examples/{os.path.split(base_model)[-1]}--structure.txt', 'w', encoding='utf-8') as O:
        for n, p in model.named_parameters():
            print(n, p.requires_grad, p.shape, p.dtype, file=O)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    try:
        model.save_added(output_dir)
    except:
        print('save_added error')
        raise
    # trainer.save_model()
    # print(
    #     "\n If there's a warning about missing keys above, please disregard :)"
    # )


if __name__ == "__main__":
    fire.Fire(train)
