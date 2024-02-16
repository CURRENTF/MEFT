# MEFT

## Environment Setup

```shell
pip install torch transformers matplotlib deepspeed
```

## Data

From huggingface download *natural\_questions, squad, toolbench, metamathqa, gsm8k. *Then use datasets api: save\_to\_disk.

Then modify path in `data_related_codes` , and generate corresponding datasets.

## Run Experiments

```shell
bash config/xxx.sh
```

