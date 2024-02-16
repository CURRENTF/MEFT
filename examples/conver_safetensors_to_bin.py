from safetensors.torch import load_file
import torch
import os


def find_safetensors_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('safetensors'):
                yield os.path.join(root, file), root


# 使用函数的例子
for file_path, r_p in find_safetensors_files('/root/autodl-fs/trained_models/llama7b-squad_v2-kv-topk16add6400-MOE-ep8'):
    d = load_file(file_path)
    p = os.path.join(r_p, "pytorch_model.bin")
    torch.save(d, p)
    print(f"{file_path} --> \n {p} \n")
