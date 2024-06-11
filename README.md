# MEFT
## Note

Currently, the code released here is for the purpose of reproducing the performance of MEFT on datasets such as NQ and SQuAD. The current repo runs on GPUs, simulating sparsification on CPUs and applying MoE's Parallel Adapter through the use of masks. The complete code will be released later.

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

