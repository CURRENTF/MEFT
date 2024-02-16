import fire
import lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from finetune import (get_my_kv_model, AnyObj, LlamaTokenizer, AutoTokenizer,
                      generate_prompt)
import datasets
from modeling import my_mistral, my_llama
from transformers import GPTJForCausalLM
import logging

# logger = my_llama.create_file_logger("running", "examples/cpu_run.log")
logger = None

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


class KVModel(L.LightningModule):
    def __init__(self, model=None):
        super().__init__()
        self.automatic_optimization = False  # Important: This property activates manual optimization.
        self.model = model
        self.kv_config = model.config
        self.lr = self.kv_config.learning_rate
        self.seq_split_train_len = self.kv_config.seq_split_train_len

    def training_step(self, batch, batch_idx):
        optim = self.optimizers()
        input_ids, attention_mask, labels = batch
        seq_len = input_ids.shape[1]
        times = (seq_len + self.seq_split_train_len - 1) // self.seq_split_train_len
        past_key_values = None
        total_loss = 0
        for i in range(0, times):
            optim.zero_grad()
            logger.info(f"shape of input_ids "
                        f"{input_ids[:, i * self.seq_split_train_len: (i + 1) * self.seq_split_train_len].shape}")
            output = self.model(
                input_ids=input_ids[:, i * self.seq_split_train_len: (i + 1) * self.seq_split_train_len],
                # attention_mask[:, i * self.seq_split_train_len: (i + 1) * self.seq_split_train_len],
                labels=labels[:, i * self.seq_split_train_len: (i + 1) * self.seq_split_train_len],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            loss = output.loss
            logger.info(f"loss = {loss}")
            past_key_values = output.past_key_values
            self.manual_backward(loss)
            self.clip_gradients(optim.optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            optim.step()
            total_loss += loss

        total_loss /= times
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            input_ids, attention_mask, labels = batch
            seq_len = input_ids.shape[1]
            times = (seq_len + self.seq_split_train_len - 1) // self.seq_split_train_len
            past_key_values = None
            total_loss = 0
            for i in range(1, times + 1):
                output = self.model(
                    input_ids[:, i * self.seq_split_train_len: (i + 1) * self.seq_split_train_len],
                    attention_mask[:, i * self.seq_split_train_len: (i + 1) * self.seq_split_train_len],
                    labels=labels[:, i * self.seq_split_train_len: (i + 1) * self.seq_split_train_len],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                loss = output.loss
                total_loss += loss

            total_loss /= times
            self.log("valid_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    # todo: save model in total_limit times
    # todo: support gradient accumulation


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "./dataset/gsm8k/real/train.jsonl",
        output_dir: str = "./lora-alpaca",
        detect_anomaly=False,
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
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        # kvx params
        add_num=4096,
        location='all',
        add_layer_num=16,
        add_gate: bool = False,
        added_on_cpu: bool = False,  # 是否将一部分kv放在cpu上
        on_gpu_size=512,
        pre_look_layers=0,
        optimizer_kv="adamw_torch",
        eval_on_train: bool = False,
        train_size=-1,
        val_size=-1,
        gpu_topk=-1,  # when > 0, try topk gpu training
        use_torch_vecdb: bool = False,
        check_similarity: bool = False,
        measure_1_decoder_layer_time: bool = False,
        frozen_key: bool = False,
        load_best_checkpoint_at_end: bool = False,
        save_total_limit=4,
        low_key_dim=0,
        seq_split_train_len=0,
):
    if seq_split_train_len == 0:
        raise NotImplementedError

    L.seed_everything(42)
    if use_gradient_checkpointing:
        raise ValueError('Removed checkpointing for better code readability')

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"

    try:
        data = datasets.load_from_disk(data_path)
    except Exception as e:
        print("目前只支持自己造的数据")
        raise e

    if train_size > 0:
        data['train'] = data['train'].select(list(range(train_size)))

    if eval_on_train:
        data['validation'] = data['train']

    if 'validation' in data and val_size > 0:
        data['validation'] = data['validation'].select(list(range(val_size)))
    if 'valid' in data and val_size > 0:
        data['valid'] = data['valid'].select(list(range(val_size)))

    print(data)
    print(f'data path: {data_path}')

    if 'nq_v1' in data_path:
        data = data.map(lambda sample: {
            "input": "", "answer": "",
            "instruction": sample["question"], "output": ''.join(sample["long_answers"])
        })

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
        "bs": micro_batch_size,
        "use_torch_vecdb": use_torch_vecdb,
        "check_similarity": check_similarity,
        "measure_1_decoder_layer_time": measure_1_decoder_layer_time,
        "frozen_key": frozen_key,
        "optimizer_kv": optimizer_kv,
        "low_key_dim": low_key_dim,
        "seq_split_train_len": seq_split_train_len,
        "learning_rate": learning_rate,
    }
    kv_cfg = AnyObj(kv_config)
    model = get_my_kv_model(base_model, kv_cfg)

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
    train_data = data["train"].shuffle().map(to_prompt, num_proc=6)
    if 'validation' in data:
        val_data = data["validation"].shuffle().map(to_prompt, num_proc=6)
    else:
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
        # print(len(batch[0]))
        # print(len(batch[0][0]))
        # raise ValueError('11')
        res = tokenizer(
            batch,
            max_length=cutoff_len,
            padding="longest",
            return_tensors='pt',
        )
        return res['input_ids'], res['attention_mask'], res["input_ids"]

    train_data, val_data = SPData(train_data.to_list(), tokenizer), SPData(val_data.to_list(), tokenizer)
    train_dataloader = DataLoader(train_data,
                                  batch_size=micro_batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data,
                                batch_size=micro_batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    # tokenizer.pad()

    lightning_model = KVModel(model)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        save_top_k=-1
    )
    trainer = L.Trainer(
        accelerator="auto",
        strategy="auto",
        devices="auto",
        num_nodes=1,
        precision=16,
        logger=True,
        fast_dev_run=False,
        max_epochs=num_epochs,
        min_epochs=num_epochs,
        # val_check_interval=eval_step,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        log_every_n_steps=50,
        accumulate_grad_batches=batch_size // micro_batch_size,
        inference_mode=True,
        profiler=None,
        detect_anomaly=detect_anomaly,
        plugins=None,
        reload_dataloaders_every_n_epochs=0,
        default_root_dir=output_dir,
        callbacks=[
            checkpoint_callback
        ]
    )
    model.config.use_cache = True
    if model.config.model_type == "llama":
        my_llama.set_trainer_pointer(trainer)
    elif model.config.model_type == "mistral":
        my_mistral.set_trainer_pointer(trainer)

    trainer.fit(lightning_model, train_dataloader, val_dataloader)

    model.save_added(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
