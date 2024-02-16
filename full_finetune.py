import os
import sys
from typing import List, Union

import datasets
import fire
import torch
import transformers
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer  # noqa: F402


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
        save_total_limit: int = 4,
        yingji_load_data: bool = False,
        fix_wrong_special_token: bool = False,
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

    if load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
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
        raise NotImplementedError

    print(f"has valid and eval_on_train={eval_on_train}")
    if yingji_load_data:
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
            # fp16=True,
            bf16=True,
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
            gradient_checkpointing=True,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    with open(f'./examples/{os.path.split(base_model)[-1]}--structure.txt', 'w', encoding='utf-8') as O:
        for n, p in model.named_parameters():
            print(n, p.requires_grad, p.shape, p.dtype, file=O)

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)


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
