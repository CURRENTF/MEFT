import logging
import math
import os
import pickle
import re
import time

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.mistral.modeling_mistral import (
    MistralConfig, MistralForCausalLM,
    MistralMLP, MistralDecoderLayer, MistralModel, MistralRMSNorm
)

try:
    from my_llama import ffn_transfer
except:
    from modeling.my_llama import ffn_transfer

# 创建一个logger
logger = logging.getLogger("running")


def set_trainer_pointer(trainer):
    global trainer_pointer
    trainer_pointer = trainer


class KVMistralConfig(MistralConfig):
    def set_kv_params(self, cfg):
        self.__dict__.update(cfg.__dict__)
        with open('./examples/debug_config_setting.txt', 'w', encoding='utf-8') as O:
            print(self.__dict__, file=O)
            print(cfg.__dict__, file=O)

    def set_kv_dict_params(self, cfg):
        self.__dict__.update(cfg)


class KVMistralMLP(MistralMLP):
    def __init__(self, config, only_post=False):
        super().__init__(config)

        self.topk_idx = None

        if config.add_gate:
            raise NotImplementedError

        d = config.hidden_size
        if getattr(config, "low_key_dim", 0) > 0:
            d = config.low_key_dim
            self.add_up_low_key = nn.Linear(self.hidden_size, config.low_key_dim, bias=False)

        if os.environ.get("DEBUG_KV", False):
            self.unique_cnt = []

        if not only_post:
            if getattr(config, "moe_style", False) and not self.config.added_on_cpu:
                # todo add cpu support
                if config.on_gpu_size % getattr(config, "moe_expert_factor", 1) == 0:
                    num_experts = math.ceil(math.sqrt(config.on_gpu_size)) // getattr(config, "moe_expert_factor", 1)
                else:
                    num_experts = math.ceil(math.sqrt(config.on_gpu_size))
                num_experts = int(num_experts)
                num_keys = config.on_gpu_size // num_experts
                assert num_keys * num_experts == config.on_gpu_size

                self.add_gates = nn.Linear(d, num_experts, bias=False)
                if getattr(config, "simulate_moe", False):
                    self.add_moe_weight = nn.Linear(d, config.on_gpu_size, bias=False)
                else:
                    self.add_experts = nn.ModuleList()
                    for i in range(num_experts):
                        self.add_experts.append(nn.Linear(d, num_keys, bias=False))
                # self.add_experts.append(nn.Linear(d, config.on_gpu_size - (num_experts - 1) * num_keys, bias=False))
                self.num_experts = num_experts
                self.num_keys = num_keys
                self.add_down_proj_gpu = nn.Linear(config.on_gpu_size, self.hidden_size, bias=False)
                setattr(self.add_down_proj_gpu, "spec_module", "down_proj")

                if os.environ.get("DEBUG_KV", False):
                    self.cache_cnt = torch.zeros(num_experts, dtype=torch.long, device="cuda")

                if getattr(config, "moe_dropout", False):
                    self.dropout = nn.Dropout(0.1)

            elif self.config.added_on_cpu:
                self.add_up_proj_gpu = nn.Parameter(torch.empty(config.on_gpu_size, d), requires_grad=True)
                self.add_down_proj_gpu = nn.Parameter(torch.empty(config.hidden_size, config.on_gpu_size))
                if self.config.async_compute:
                    # todo remove this later
                    self.topk_idx = torch.zeros(self.config.on_gpu_size, requires_grad=False,
                                                device='cuda:0', dtype=torch.int)
                    self.n_topk_idx = 0
            else:
                self.add_up_proj_gpu = nn.Linear(d, config.on_gpu_size, bias=False)
                self.add_down_proj_gpu = nn.Linear(config.on_gpu_size, self.hidden_size, bias=False)
                setattr(self.add_down_proj_gpu, "spec_module", "down_proj")

        self.relu = nn.functional.relu
        self.config = config
        self.only_post = only_post
        # todo 这个地方不该写死，但是不知道怎么设置了
        self.device = torch.device("cuda")

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        if getattr(self.config, "low_key_dim", 0) > 0:
            x = self.add_up_low_key(x)

        if getattr(self.config, "moe_style", False):
            if self.config.gpu_topk <= 0:
                raise NotImplementedError
            topk = self.config.gpu_topk

            bs, seq_len, d = x.shape
            x = x.view(-1, d)
            expert_scores = self.add_gates(x)  # bs*seq_len, num_exs
            if getattr(self.config, "moe_softmax_score", False):
                expert_scores = F.softmax(expert_scores, dim=-1)

            _, ids = torch.topk(expert_scores, self.config.n_probe)  # bs*seq_len, n_probe
            coff = torch.zeros(bs * seq_len, self.num_experts,
                               device=x.device, dtype=_.dtype).scatter_(-1, ids, _)  # bs*seq_len, num_exs
            if os.environ.get("DEBUG_KV", False):
                self.cache_cnt += torch.sum(torch.where(coff > 1e-6, torch.tensor(1), torch.tensor(0)), dim=0)
            # print(coff.shape)
            # assert coff.shape == (bs * seq_len, self.num_experts)
            if getattr(self.config, "use_unstable_feature", False):
                coff = coff.unsqueeze(2).repeat(1, 1, self.num_keys).view(bs * seq_len, -1)
                if getattr(self.config, "moe_dropout", False):
                    x = self.dropout(x)
                # o = [self.add_experts[i](x) for i in range(self.num_experts)]  # bs*seq_len, num_keys
                if getattr(self.config, "simulate_moe", False):
                    weight = self.add_moe_weight.weight
                else:
                    weight = [expert.weight for expert in self.add_experts]
                    weight = torch.cat(weight, dim=0)
                o = F.linear(x, weight)
                # o = torch.cat(o, dim=-1)  # bs*seq_len, r
                o = o * coff
                _, ids = torch.topk(o, topk)
            else:
                o = [coff[:, i].unsqueeze(1) * self.add_experts[i](x) for i in
                     range(self.num_experts)]  # bs*seq_len, num_keys
                o = torch.cat(o, dim=-1)  # bs*seq_len, r
                _, ids = torch.topk(o, topk)

            indices = None
            if lim := getattr(self.config, "limit_total_activated_neurons", 0):
                indices = torch.unique(ids.view(-1))
                if indices.shape[0] > lim:
                    i = torch.randperm(indices.shape[0])[:lim]
                    indices = indices[i]
                w_a_k = torch.cat([expert.weight for expert in self.add_experts], dim=0)[indices]
                down_proj += F.linear(
                    self.relu(
                        F.linear(x, w_a_k)
                    ),
                    self.add_down_proj_gpu.weight[:, indices]
                ).view(bs, seq_len, d)
            else:
                keys_weight = torch.zeros(bs * seq_len, self.config.on_gpu_size,
                                          device=_.device, dtype=_.dtype).scatter_(-1, ids, _)
                # if keys_weight.device != self.add_down_proj_gpu.device:
                #     keys_weight = keys_weight.to()
                down_proj += self.add_down_proj_gpu(self.relu(keys_weight)).view(bs, seq_len, d)

            if os.environ.get("DEBUG_KV", False):
                if indices is None:
                    indices = torch.unique(ids.view(-1))
                self.unique_cnt.append(int(indices.shape[0]))
                if len(self.unique_cnt) % (16 * 500) == 0:  # write sometime
                    with open(f"/root/autodl-tmp/{getattr(self.config, 'run_name', '')}"
                              f"-topk{self.config.gpu_topk}-{self.layer_idx}.pk", "wb") as pk:
                        pickle.dump(self.unique_cnt, pk)
            # lists_for_expert = [[] for _ in range(self.num_experts)]
            # for i in range(ids.shape[0]):
            #     for j in ids[i]:
            #         lists_for_expert[j].append(i)
            #
            # res = [torch.empty(0, device=x.device) for _ in range(bs * seq_len)]
            # # res = torch.zeros(bs * seq_len, self.num_keys * self.config.n_probe, device=x.device)
            # for i in range(self.num_experts):
            #     o = self.add_experts[i](x[lists_for_expert[i]])  # ?, num_keys
            #
            #     for o_j, token_j in enumerate(lists_for_expert[i]):
            #         res[token_j] = torch.cat([res[token_j], o[o_j]])
            # # res = torch.stack(res)
            #
            # for i, keys_score in enumerate(res):
            #     score, ids = torch.topk(keys_score, topk)
            #     down_proj[i // seq_len, i % seq_len] += F.linear(score, self.add_down_proj_gpu.weight[:, ids])

        else:
            if self.config.gpu_topk <= 0:
                if not self.only_post:
                    down_proj += self.add_down_proj_gpu(self.act_fn(self.add_up_proj_gpu(x)))
            else:
                if self.config.added_on_cpu:
                    self.add_up_proj_gpu: nn.Parameter
                    self.add_down_proj_gpu: nn.Parameter

                    if self.config.pre_look_layers == 0:
                        ffn_transfer(self, x)
                        _ = F.linear(x, self.add_up_proj_gpu)
                        # assert torch.isfinite(_).all(), f"Invalid value detected in ffn_transfer1"
                        _ = self.relu(_)
                        # assert torch.isfinite(_).all(), f"Invalid value detected in ffn_transfer2"
                        logger.info(f"layer = {self.layer_idx}")  # ###
                        logger.info(f"hidden state shape {_.shape}")
                        logger.info(f"add_down_proj_gpu shape {self.add_down_proj_gpu.data.shape}")
                        logger.info(f"add_down_proj_gpu shape {self.add_down_proj_gpu.shape}")
                        logger.info('-' * 40)
                        _ = F.linear(_.to(dtype=torch.float32), self.add_down_proj_gpu)
                        # assert torch.isfinite(_).all(), "Invalid value detected in ffn_transfer3"
                        down_proj += _

                    else:

                        if getattr(self, "do_post", False):
                            post_mlp = self.post_mlp_obj[0]
                            if self.config.check_similarity:
                                post_mlp.pre_x = x

                            if getattr(self.config, "async_compute", False):
                                post_mlp.index.query_batch_direct_async(
                                    x,
                                    post_mlp.add_up_proj_gpu, post_mlp.add_down_proj_gpu,
                                    post_mlp.topk_idx,
                                    top_k=self.config.gpu_topk, layer_idx=post_mlp.layer_idx
                                )

                            else:
                                ffn_transfer(post_mlp, x)

                        if getattr(self, "use_pre_res", False):
                            if getattr(self.config, "async_compute", False):
                                self.n_topk_idx = self.index.get_query_res(self.layer_idx)

                            p = F.linear(self.relu(F.linear(x, self.add_up_proj_gpu)).to(dtype=torch.float32),
                                         self.add_down_proj_gpu)
                            down_proj += p
                            # logger.info(f"x grad_fn {x.grad_fn}")
                            # logger.info(f"p grad_fn {p.grad_fn}")
                            if self.config.check_similarity:
                                _ = x * self.pre_x
                                _ = torch.mean(_)
                                logger.info(f"avg similarity layer_idx={self.layer_idx} similarity={_}")

                    logger.info('-' * 40 + "mlp forward end!")

                else:  # on GPU
                    if self.config.pre_look_layers > 0:
                        if getattr(self, "do_post", False):
                            post_mlp = self.post_mlp_obj[0]
                            setattr(post_mlp, "pre_x", x.detach())
                        if getattr(self, "use_pre_res", False):
                            mem_coff = self.add_up_proj_gpu(self.pre_x)
                            topk_num = self.config.gpu_topk
                            if self.config.simulate_recall < 1.0:
                                topk_num = math.ceil(self.config.gpu_topk / self.config.simulate_recall)
                                topk_num = min(topk_num, mem_coff.shape[-1])
                            values, indices = torch.topk(mem_coff, topk_num)
                            if self.config.simulate_recall < 1.0:
                                i = torch.randperm(topk_num)[:self.config.gpu_topk]
                                indices, values = indices[..., i], values[..., i]
                            simulate_ogs = getattr(self.config, "simulate_ogs", min(3072, self.config.on_gpu_size))
                            indices = torch.unique(indices.view(-1))[:simulate_ogs]
                            _m_c = torch.zeros_like(mem_coff)
                            _m_c[..., indices] = 1.0
                            # _m_c = torch.zeros_like(mem_coff).scatter(-1, indices, 1.0)
                            down_proj += self.add_down_proj_gpu(self.relu(_m_c * self.add_up_proj_gpu(x)))
                    else:
                        indices = None
                        if not self.only_post:
                            mem_coff = self.add_up_proj_gpu(x)
                            topk_num = self.config.gpu_topk
                            if getattr(self.config, "simulate_recall", 2.0) < 1.0:
                                topk_num = math.ceil(self.config.gpu_topk / self.config.simulate_recall)
                                topk_num = min(topk_num, mem_coff.shape[-1])
                            values, indices = torch.topk(mem_coff, topk_num)
                            if lim := getattr(self.config, "limit_total_activated_neurons", 0):
                                indices = torch.unique(indices.view(-1))
                                if indices.shape[0] > lim:
                                    i = torch.randperm(indices.shape[0])[:lim]
                                    indices = indices[i]
                                down_proj += F.linear(
                                    self.relu(mem_coff[..., indices]), self.add_down_proj_gpu.weight[:, indices]
                                )
                            else:
                                if getattr(self.config, "simulate_recall", 2.0) < 1.0:
                                    i = torch.randperm(topk_num)[:self.config.gpu_topk]
                                    indices, values = indices[..., i], values[..., i]
                                _m_c = torch.zeros_like(mem_coff).scatter(-1, indices, values)
                                down_proj += self.add_down_proj_gpu(self.relu(_m_c))

                        if os.environ.get("DEBUG_KV", False):
                            assert indices is not None
                            indices = torch.unique(indices.view(-1))
                            self.unique_cnt.append(int(indices.shape[0]))
                            if len(self.unique_cnt) % (16 * 500) == 0:
                                with open(f"/root/autodl-tmp/{getattr(self.config, 'run_name', '')}"
                                          f"-topk{self.config.gpu_topk}-{self.layer_idx}.pk", "wb") as pk:
                                    pickle.dump(self.unique_cnt, pk)

        return down_proj


class KVMistralDecoderLayer(MistralDecoderLayer):
    def __init__(self, config, only_post=False, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        del self.mlp
        self.config = config
        self.mlp = KVMistralMLP(config, only_post=only_post)


class KVMistralModel(MistralModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        kv_layers_id = []
        if config.location == 'all':
            for i in range(config.num_hidden_layers):
                kv_layers_id.append(i)
        elif config.location == 'front':
            for i in range(config.add_layer_num):
                kv_layers_id.append(i)
        elif config.location == 'back':
            for i in range(config.num_hidden_layers - config.add_layer_num, config.num_hidden_layers):
                kv_layers_id.append(i)
        elif config.location == 'mid':
            mid_s_loc = round((config.num_hidden_layers - config.add_layer_num) / 2)
            for i in range(mid_s_loc, mid_s_loc + config.add_layer_num):
                kv_layers_id.append(i)
        else:
            raise KeyError

        if min(kv_layers_id) - config.pre_look_layers < 0:
            raise ValueError("should >= 0")
        else:
            only_post_layers = []
            for i in kv_layers_id:
                if i - self.config.pre_look_layers not in kv_layers_id:
                    only_post_layers.append(i - self.config.pre_look_layers)

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if i in kv_layers_id:
                self.layers.append(KVMistralDecoderLayer(self.config, layer_idx=i))
            elif i in only_post_layers:
                self.layers.append(KVMistralDecoderLayer(self.config, only_post=True, layer_idx=i))
            else:
                self.layers.append(MistralDecoderLayer(self.config, layer_idx=i))

            setattr(self.layers[i].mlp, "layer_idx", i)

        self.kv_layers_id = kv_layers_id
        self.indexes = None
        self.post_init()

    def set_index(self):
        if self.config.added_on_cpu:
            try:
                from torch_vector_database import HugeFFN
                from async_controller import controller
                from glass_vdb import HugeGFFN
            except ModuleNotFoundError:
                from modeling.torch_vector_database import HugeFFN
                from modeling.async_controller import controller
                from modeling.glass_vdb import HugeGFFN

            huge_ffn_controller_cls = None
            huge_ffn_cls = None
            if self.config.async_compute:
                huge_ffn_controller_cls = controller.HugeFFNAllController

            if self.config.use_torch_vecdb:
                huge_ffn_cls = HugeFFN
            elif self.config.use_glass_vdb:
                huge_ffn_cls = HugeGFFN
            else:
                raise NotImplementedError

            print("Init retriever ... ")

            all_index = None
            if self.config.async_compute:
                all_index = huge_ffn_controller_cls(
                    huge_ffn_init_params=([], {
                        "num_kv_pairs": self.config.add_num,
                        "hs_dim": self.config.hidden_size,
                        "low_key_dim": self.config.low_key_dim,
                        "model_config": self.config,
                        "grad_accumulation": self.config.bbs // self.config.bs,
                    }),
                    layer_ids=self.kv_layers_id,
                    config=self.config,
                    num_group=self.config.num_group
                )

            for i in self.kv_layers_id:
                if all_index is None:
                    index = huge_ffn_cls(
                        num_kv_pairs=self.config.add_num,
                        hs_dim=self.config.hidden_size,
                        low_key_dim=self.config.low_key_dim,
                        model_config=self.config,
                        grad_accumulation=self.config.bbs // self.config.bs,
                    )
                else:
                    index = all_index

                setattr(self.layers[i].mlp, "index", index)
                setattr(self.layers[i - self.config.pre_look_layers].mlp, "do_post", True)
                setattr(self.layers[i - self.config.pre_look_layers].mlp, "post_mlp_obj", [self.layers[i].mlp])
                # 用一个列表是为了防止被当成torch module然后来回调用
                setattr(self.layers[i].mlp, "use_pre_res", True)

            print("retriever ok")

        else:
            if self.config.pre_look_layers > 0:  # gpu simulate
                for i in self.kv_layers_id:
                    setattr(self.layers[i - self.config.pre_look_layers].mlp, "do_post", True)
                    setattr(self.layers[i - self.config.pre_look_layers].mlp, "post_mlp_obj", [self.layers[i].mlp])
                    setattr(self.layers[i].mlp, "use_pre_res", True)

    def clean_grad_of_down(self):
        if 'adam' in self.config.optimizer_kv and self.config.added_on_cpu:
            for i in self.kv_layers_id:
                self.layers[i].mlp.add_down_proj_gpu.grad = None

    def send_grad(self):
        for i in self.kv_layers_id:
            mlp = self.layers[i].mlp
            if self.config.async_compute:
                mlp.index.go_grad_async(mlp.add_down_proj_gpu.grad.data, mlp.topk_idx, mlp.n_topk_idx, mlp.layer_idx)
            else:
                mlp.index.go_grad(mlp.add_down_proj_gpu.grad.data, mlp.topk_idx, mlp)

    def wait_cpu_optimizer(self):
        for i in self.kv_layers_id:
            mlp = self.layers[i].mlp
            mlp.index.get_upd_res(mlp.layer_idx)

    def reinit_added_weight(self):

        def sample_weights(w: torch.Tensor, num_rows):
            assert num_rows <= w.shape[0]
            return w[torch.randperm(num_rows)[:num_rows]].data.clone()

        if self.config.moe_style:
            for i in self.kv_layers_id:
                mlp = self.layers[i].mlp
                # src_weight = self.layers[i - 1].mlp.down_proj.weight.T if i > 0 else self.layers[i].mlp.up_proj.weight
                for j in range(mlp.num_experts):
                    if getattr(mlp.config, "simulate_moe", False):
                        mlp.add_gates.weight.data[j] = torch.mean(
                            mlp.add_moe_weight.weight.data[j * mlp.num_keys: (j+1) * mlp.num_keys], dim=0) * 50
                    else:
                        mlp.add_gates.weight.data[j] = torch.mean(mlp.add_experts[j].weight.data, dim=0) * 50
        else:
            for i in self.kv_layers_id:
                mlp = self.layers[i].mlp
                src_weight = self.layers[i - 1].mlp.down_proj.weight.T if i > 0 else self.layers[i].mlp.up_proj.weight
                if isinstance(mlp.add_up_proj_gpu, nn.Linear):
                    mlp.add_up_proj_gpu.weight.data = sample_weights(src_weight, mlp.add_up_proj_gpu.weight.shape[0])
                if isinstance(mlp.add_up_proj_gpu, nn.Parameter):
                    mlp.add_up_proj_gpu.data = sample_weights(src_weight, mlp.add_up_proj_gpu.shape[0])

class KVMistralForCausalLM(MistralForCausalLM):
    def __init__(self, config: KVMistralConfig):
        super().__init__(config)
        self.model = KVMistralModel(config)
        _ = time.time()
        self.post_init()
        print("kv causal lm post_init() time used", time.time() - _)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if getattr(module, "spec_module", "") == "down_proj":
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def freeze_other_params_and_set_kv_float32(self):
        for n, p in self.named_parameters():
            try:
                p.requires_grad = False
                if 'add_' in n:
                    p.requires_grad = True
                    if 'cpu' in n:
                        p.data = p.data.to(device='cpu', dtype=torch.float32)
                    else:
                        if self.config.frozen_key and ("up" in n or "expert" in n or "gate" in n):
                            p.requires_grad = False

                if getattr(self.config, "train_layernorm", False):
                    if "layernorm" in n:
                        p.requires_grad = True

                if p.requires_grad:
                    p.data = p.data.to(torch.float32)

            except Exception as e:
                print(n, type(p), p.shape, p.dtype, p.device)
                raise e

    def my_print_trainable_parameters(self):
        size = 0
        for n, p in self.named_parameters():
            if p.requires_grad:
                size += p.numel()
                logger.info(f"Trainable: {n}")
        print("total trained params", size / 1e6, 'M')

    def get_kv_state_dict(self):
        state_dict = {}
        for n, p in self.named_parameters():
            if ('add_' in n) or ('model' not in n):
                state_dict[n] = p
        return state_dict

    def save_added(self, out_dir):
        file_path = os.path.join(out_dir, 'only_kv.pk')
        config_path = os.path.join(out_dir, "config.json")
        os.makedirs(out_dir, exist_ok=True)
        state_dict = self.get_kv_state_dict()
        torch.save(state_dict, file_path)
        self.config.to_json_file(config_path)

    def load_added(self, file_path):
        _ = os.path.join(file_path, 'only_kv.pk')
        if not os.path.exists(_):
            _ = os.path.join(file_path, 'pytorch_model.bin')

        if not os.path.exists(_):
            from safetensors.torch import load_file
            # model-00001-of-00002.safetensors
            entries = os.listdir(file_path)
            state_dict = {}
            for entry in entries:
                full_path = os.path.join(file_path, entry)
                if re.fullmatch(r"model-0000[0-9]-of-0000[0-9]\.safetensors", entry):
                    state_dict.update(load_file(full_path, device='cpu'))
                if re.fullmatch(r"model.safetensors", entry):
                    state_dict.update(load_file(full_path, device='cpu'))
            print(f"loading state_dict from path ==> {file_path}")

        else:
            file_path = _
            state_dict = torch.load(file_path, map_location='cpu')
            print(f"loading state_dict from path ==> {file_path}")

        assert len(state_dict) > 0
        self.load_state_dict(state_dict, strict=False)

    def set_index(self):
        self.model.set_index()

    def save_pretrained(
            self,
            *args,
            **kwargs,
    ):
        state_dict = kwargs.pop("state_dict", None)
        if state_dict is None:
            state_dict = self.get_kv_state_dict()
        super().save_pretrained(
            *args,
            state_dict=state_dict,
            **kwargs,
        )

    def trainable_parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            if param.requires_grad:
                yield param

    def low_key_dim_parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            if 'low_key' in name:
                yield param

    def low_key_dim_named_parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            if 'low_key' in name:
                yield name, param

    def down_parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            if 'down' in name:
                yield param

    def forward(
            self,
            *args,
            **kwargs,
    ):
        # for name, param in self.named_parameters():
        #     if torch.isnan(param).any():
        #         print(f'NaN value detected in model weights: {name}')
        #         raise ValueError
        #     if torch.isinf(param).any():
        #         print(f'Infinity value detected in model weights: {name}')
        #         raise ValueError
        # if 'adam' in self.config.optimizer_kv and self.config.added_on_cpu:
        #     self.model.clean_grad_of_down()
        x = super().forward(*args, **kwargs)
        return x

    def clean_grad_of_down(self):
        self.model.clean_grad_of_down()

    def send_grad(self):
        self.model.send_grad()

    def wait_cpu_optimizer(self):
        self.model.wait_cpu_optimizer()

    def reinit_added_weight(self):
        self.model.reinit_added_weight()

    def debug_print_cache_cnt_for_moe(self):
        if not getattr(self.config, "moe_style", False):
            return "moe not supported"
        model = self.model
        layers = model.layers
        output = "\n"
        for l in layers:
            mlp = l.mlp
            p = mlp.cache_cnt / torch.sum(mlp.cache_cnt)
            output += '\n' + str([round(_, 3) for _ in p.tolist()])
            # yield mlp.cache_cnt
        return output


if __name__ == "__main__":
    pass
    # KVLlamaForCausalLM.save_pretrained()
