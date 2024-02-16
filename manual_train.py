import os

import fire
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from finetune import (get_my_kv_model, AnyObj, LlamaTokenizer, AutoTokenizer,
                      generate_prompt)
import datasets
from modeling import my_mistral, my_llama
from transformers import GPTJForCausalLM
import logging
import transformers
from finetune import logger
from tqdm import *


# tokenize之后的数据格式
class SPData(Dataset):
    def __init__(self, lis, tokenizer):
        self.data_list = lis
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        item = self.data_list[item]
        return item['prompt']

    def __len__(self):
        return len(self.data_list)


def fit(model, dataloaders, num_epochs=25, log_step=10, grad_accumulation_steps=64, config=None):
    # todo add other args
    optimizer = torch.optim.AdamW(model.low_key_dim_parameters(), lr=config.learning_rate, weight_decay=1e-6)
    _opt_for_zero_down_grad = torch.optim.SGD(model.down_parameters(), lr=1e-3)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            total_loss = 0
            iter_cnt = 0
            for input_ids, attention_mask, labels in tqdm(dataloaders[phase],
                                                          desc=f"{phase} ep{epoch + 1}/{num_epochs}"):
                input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()
                _opt_for_zero_down_grad.zero_grad()
                if iter_cnt % grad_accumulation_steps == 0:
                    optimizer.zero_grad()

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=True,
                    return_dict=True
                )
                loss = output.loss
                total_loss += loss
                if phase == 'train':
                    loss.backward()
                    # input(f"MKL {os.environ['MKL_NUM_THREADS']}")
                    # input(f"")
                    model.send_grad()
                    # w = torch.nn.Linear(10000, 10000, device='cpu')
                    # x = torch.randn(10000, requires_grad=False)
                    # y = w(x)

                iter_cnt += 1
                if iter_cnt % grad_accumulation_steps == 0:
                    optimizer.step()
                    model.wait_cpu_optimizer()

                if iter_cnt % (log_step * grad_accumulation_steps) == 0:
                    logger.info(f"{iter_cnt / len(dataloaders[phase])} % loss = "
                                f"{total_loss / log_step / grad_accumulation_steps}")
                    total_loss = 0

                print(f"next! {loss}")

                # todo remove it
                # if iter_cnt == 100:
                #     break

    print('Training complete')
    return model


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "./dataset/gsm8k/real/train.jsonl",
        output_dir: str = "./lora-alpaca",
        adapter_name: str = "kv",
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
        cpp_mode: bool = False,
        num_group: int = 1,
):
    transformers.set_seed(42)
    print("depend_on_update ==", depend_on_update)
    print("frozen key ==", frozen_key)
    if not added_on_cpu:
        raise NotImplementedError

    if async_compute:
        if not added_on_cpu:
            raise NotImplementedError
        if pre_look_layers <= 1:
            raise NotImplementedError

    if use_gradient_checkpointing:
        raise ValueError('Removed checkpointing for better code readability')

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    try:
        data = datasets.load_from_disk(data_path)
    except Exception as e:
        # data = load_dataset(data_path)
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
        "cpp_mode": cpp_mode,
        "learning_rate": learning_rate,
        "num_group": num_group,
    }
    kv_cfg = AnyObj(kv_config)
    model = get_my_kv_model(base_model, kv_cfg, load_pkl=load_pkl)

    model.freeze_other_params_and_set_kv_float32()

    model.my_print_trainable_parameters()  # Be more transparent about the % of trainable params.
    model.set_index()

    if model.config.model_type == "llama" or model.config.model_type == "mistral":
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

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

    def to_prompt(sample):
        return {"prompt": generate_prompt(sample)}

    if resume_from_checkpoint:
        raise NotImplementedError

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

        train_data = data["train"].shuffle().map(to_prompt)
        try:
            val_data = data["validation"].shuffle().map(to_prompt)
        except KeyError:
            val_data = data["valid"].shuffle().map(to_prompt)

    else:
        train_data = data["train"].shuffle().map(to_prompt, num_proc=6)
        try:
            val_data = data["validation"].shuffle().map(to_prompt, num_proc=6)
        except KeyError:
            val_data = data["valid"].shuffle().map(to_prompt, num_proc=6)

    print("train data len: ", len(train_data))
    print("what's archived in train_data ?")
    print(train_data)

    columns_should_remove = ['output', 'raw_answer', 'input', 'question', 'instruction', 'answer', 'document',
                             "long_answers", "short_answers", "id", "title", "text", "text_id", "is_knowledge"]
    cols = [c for c in columns_should_remove if c in train_data.column_names]
    train_data = train_data.remove_columns(cols)
    cols = [c for c in columns_should_remove if c in val_data.column_names]
    val_data = val_data.remove_columns(cols)

    O = open("examples/check_model_structure.txt", "w", encoding='utf-8')
    try:
        print(model, file=O)
    finally:
        O.close()

    def collate_fn(batch):
        res = tokenizer(
            batch,
            max_length=cutoff_len,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        return res['input_ids'], res['attention_mask'], res["input_ids"]

    train_data, val_data = SPData(train_data.to_list(), tokenizer), SPData(val_data.to_list(), tokenizer)
    train_dataloader = DataLoader(train_data,
                                  batch_size=micro_batch_size, shuffle=True,
                                  num_workers=0, collate_fn=collate_fn,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_data,
                                batch_size=micro_batch_size, shuffle=False,
                                num_workers=0, collate_fn=collate_fn,
                                pin_memory=True)

    model.config.use_cache = True

    # lp = LineProfiler()
    # lp_wrapper = lp(fit)
    # lp_wrapper(
    #     model,
    #     {"train": train_dataloader, "valid": val_dataloader},
    #     num_epochs=num_epochs,
    #     grad_accumulation_steps=gradient_accumulation_steps
    # )
    # lp.print_stats()
    fit(
        model,
        {"train": train_dataloader, "valid": val_dataloader},
        log_step=logging_steps,
        num_epochs=num_epochs,
        grad_accumulation_steps=gradient_accumulation_steps,
        config=model.config,
    )

    model.save_added(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
