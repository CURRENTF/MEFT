import torch

try:
    import cppeff
except ModuleNotFoundError:
    pass
import torch.nn.functional as F
import time
import logging
from deepspeed.ops.adam import DeepSpeedCPUAdam



class HugeFFN:
    def __init__(
            self, up: torch.Tensor = None, down: torch.Tensor = None,
            num_kv_pairs=None, hs_dim=None, low_key_dim=0,
            grad_accumulation=1, model_config=None, logger=None,
            init_optimizer=False, num_threads=None,
    ):
        import subprocess
        output = subprocess.check_output(['nproc'])
        core_count = int(output.decode().strip())

        #  tensor is weight
        if logger is None:
            self.logger = logging.getLogger("running")
        elif isinstance(logger, str):
            self.logger = 'logger later will be set'
        else:
            self.logger = logger
            self.logger.info("logger from added process")
            self.logger.info("init params")
            self.logger.info(f"{num_kv_pairs}, {hs_dim}, {low_key_dim}")
            self.logger.info(f"{model_config}")

        self.non_blocking = False
        self.device = torch.device('cpu')
        self.config = model_config
        self.grad_accumulation = grad_accumulation
        self.grad_update_time = 0
        self.optimizer = None
        if num_threads is not None:
            self.num_threads = num_threads
        else:
            self.num_threads = core_count - 2
        torch.set_num_threads(self.num_threads)

        if up is not None and down is not None:
            self.up = up.to(device=self.device)
            self.down = down.to(device=self.device)
        elif isinstance(num_kv_pairs, int) and isinstance(hs_dim, int):
            if model_config is None:
                raise ValueError('model config is None')
            _ = hs_dim
            if low_key_dim > 0:
                _ = low_key_dim
            requires_grad = False
            if 'adam' in self.config.optimizer_kv:
                requires_grad = True
            self.up = torch.zeros(
                (num_kv_pairs, _),
                dtype=torch.float32, device=self.device, requires_grad=False, pin_memory=True
            )
            self.down = torch.zeros(
                (hs_dim, num_kv_pairs),
                dtype=torch.float32, device=self.device, requires_grad=requires_grad, pin_memory=True
            )
            self.init_vecs()
        else:
            raise NotImplementedError

        if init_optimizer:
            self.set_optimizer()

    def set_logger(self, logger):
        self.logger = logger

    def set_optimizer(self):
        if "adam" in self.config.optimizer_kv:
            # todo add other args
            # self.optimizer = AdamW([self.down], weight_decay=1e-6)
            self.optimizer = DeepSpeedCPUAdam([self.down], lr=self.config.learning_rate, weight_decay=1e-6)

    def init_vecs(self):
        self.init_up()
        self.init_down()

    def init_up(self):
        std = self.config.initializer_range
        self.up.normal_(mean=0.0, std=std)

    def init_down(self):
        self.down.data.zero_()

    def set_shared_memory(self):
        self.up = self.up.share_memory_()
        self.down = self.down.share_memory_()

    def go_grad_async(self, grad, idx, n):
        # assert grad.device == torch.device("cuda:0") and idx.device == torch.device("cuda:0")
        if self.down.grad is None:
            self.logger.warning("down.grad is None")
            self.down.grad = torch.zeros_like(self.down, requires_grad=False, device=self.device)

        idx = idx[:n]
        idx = idx.data.cpu()
        self.down.grad.data[:, idx] += grad[:, :n].to(
            dtype=self.down.dtype, device=self.device, non_blocking=self.non_blocking
        ).data
        del idx, grad

        self.grad_update_time += 1
        if self.grad_update_time == self.grad_accumulation:
            st = time.time()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=False)
            self.logger.info(f"optimizer step in vdb {time.time() - st}")
            # self.down.grad = None
            self.grad_update_time = 0
            # torch.cuda.empty_cache()
            return "step"
        else:
            # torch.cuda.empty_cache()
            return "only_upd"

    def go_grad(self, grad, idx, mlp):
        self.logger.debug(f"grad is what? {grad.shape}, {grad.dtype}")
        st = time.time()
        if self.down.grad is None:
            self.logger.debug("down.grad is None")
            self.down.grad = torch.zeros_like(self.down, requires_grad=False)
        if self.config.cpp_mode:
            st = time.time()
            cppeff.update_grad(grad, self.down.grad.data, idx, self.device, self.non_blocking)
            self.logger.info(f"go_grad cppeff update grad {time.time() - st}")
        else:
            n = idx.shape[0]
            _ = grad[:, :n]
            self.logger.info(f"go_grad get view {time.time() - st}")
            _ = _.to(
                dtype=self.down.dtype, device=self.device, non_blocking=self.non_blocking
            ).data
            self.logger.info(f"go_rad move to cpu {time.time() - st}")
            self.down.grad.data[..., idx] += _
            del _
            self.logger.info(f"go_grad add on that {time.time() - st}")
        self.grad_update_time += 1
        if self.grad_update_time == self.grad_accumulation:
            st = time.time()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.logger.info(f"optimizer step in vdb {time.time() - st}")
            self.down.grad = None
            self.grad_update_time = 0
            return "step"

        return "only_upd"

    def update(self, vectors_up: torch.Tensor, vectors_down: torch.Tensor, indices, transfer=True):
        # todo need add pin_memory=True
        start_time = time.time()
        # self.down: torch.Tensor
        # self.up: torch.Tensor
        if transfer:
            if not self.config.frozen_key:
                vectors_up = vectors_up.to(dtype=self.up.dtype, device=self.device, non_blocking=self.non_blocking)
            vectors_down = vectors_down.to(dtype=self.down.dtype, device=self.device, non_blocking=self.non_blocking)
            indices = indices.cpu()
        self.logger.info(f"index update transfer time used {time.time() - start_time}")

        start_time = time.time()
        if self.config.depend_on_update:
            if not self.config.frozen_key:
                self.up[indices, ...] = vectors_up
            self.down[..., indices] = vectors_down
        else:
            if not self.config.frozen_key:
                self.up[indices, ...] = 0.6 * vectors_up + 0.4 * self.up[indices, ...]
            self.down[..., indices] = 0.6 * vectors_down + 0.4 * self.down[..., indices]
        self.logger.info(f"index update weight time used {time.time() - start_time}")
        return "update_done"

    def top_k_frequent(self, int_tensor, k):
        # int_tensor (bs, l)
        # 使用bincount函数计算每个整数出现的次数
        start_time = time.time()
        cols = torch.zeros(int_tensor.shape[0], k, dtype=torch.int, device=self.device)
        for i in range(int_tensor.shape[0]):
            counts = torch.bincount(int_tensor[i])  # (v_num)
            # 使用topk函数找到出现次数最多的k个整数
            values, indices = counts.topk(k, dim=-1)  # (top_k)
            cols[i] = indices

        self.logger.info(f"top k frequent {time.time() - start_time}")
        # print('top k frequent ', time.time() - start_time, file=self.time_stat_log)
        return cols

    def query_batch(self, query_vector, top_k=256):
        # q_v shape -> (bs, seq_len, c)
        # up (c -> r) shape: (r, c)
        start_time = time.time()
        query_vector = query_vector.to(device=self.device, dtype=torch.float32, non_blocking=self.non_blocking)
        assert str(query_vector.device) == 'cpu' \
               and str(self.up.device) == 'cpu' \
               and str(self.down.device) == 'cpu' \
               and self.up.dtype == torch.float32

        # self.up = self.up.cpu(non_blocking=self.non_blocking)
        # self.down = self.down.cpu(non_blocking=self.non_blocking)
        # dot_res = self.up(query_vector)  # shape (bs, seq_len, r)
        dot_res = F.linear(query_vector, self.up)  # shape (bs, seq_len, r)
        # reduce seq_len
        val, idx = torch.topk(dot_res, k=top_k, dim=-1)  # shape (bs, seq_len, top_k)
        idx = idx.view(idx.shape[0], -1)
        idx = self.top_k_frequent(idx, top_k).view(-1)
        up = self.up[idx, ...]
        down = self.down[..., idx]
        self.logger.info(f"query batch time used {time.time() - start_time}")
        # print(f'query batch time used : {time.time() - start_time}', file=self.time_stat_log)
        # return up kv pair num --> bs*top_k
        return up, down, idx

    def query_batch_by_sum(self, query_vector, top_k=256):
        # q_v shape -> (bs, seq_len, c)
        # up (c -> r) shape: (r, c)
        start_time = time.time()
        query_vector = query_vector.to(dtype=torch.float32, device=self.device, non_blocking=self.non_blocking)
        assert str(query_vector.device) == 'cpu' \
               and str(self.up.device) == 'cpu' \
               and str(self.down.device) == 'cpu' \
               and self.up.dtype == torch.float32
        # self.up = self.up.cpu()
        # self.down = self.down.cpu()
        # dot_res = self.up(query_vector)  # shape (bs, seq_len, r)
        dot_res = F.linear(query_vector,
                           self.up.to(torch.float32, non_blocking=self.non_blocking))  # shape (bs, seq_len, r)
        # reduce seq_len
        dot_res = torch.sum(dot_res, dim=1)
        val, idx = torch.topk(dot_res, k=top_k, dim=-1)  # shape (bs, top_k)
        idx = idx.view(-1)
        up = self.up[idx, ...]
        down = self.down[..., idx]
        self.logger.info(f"query batch by sum time used : {time.time() - start_time}")
        # print(f'query batch by sum time used : {time.time() - start_time}', file=self.time_stat_log)
        return up, down, idx

    def query_batch_direct(self, query_vector, top_k=256, layer_idx=-1, transfer=True):
        # q_v shape -> (bs, seq_len, d)
        # up (d -> r) shape: (r, d)
        st = time.time()
        if transfer:
            query_vector = query_vector.data
            query_vector = query_vector.to(dtype=torch.float32, device=self.device, non_blocking=self.non_blocking)
        self.logger.info(f"torch query batch ({layer_idx}) transfer time used : {time.time() - st}")

        st = time.time()
        self.logger.debug(f"query_vector shape {query_vector.shape}")
        # self.logger.debug(f"query_vector {query_vector}")
        dot_res = F.linear(query_vector, self.up)
        self.logger.info(f"torch query batch ({layer_idx}) compute time used : {time.time() - st}")

        st = time.time()
        val, idx = torch.topk(dot_res, k=top_k, dim=-1)  # shape (bs, seq_len, top_k)
        idx = torch.unique(idx.view(-1))
        if idx.shape[0] > self.config.on_gpu_size:
            self.logger.warning(f"out of space {idx.shape[0]}")
            idx = idx[:self.config.on_gpu_size]
        self.logger.debug(f"idx shape {idx.shape}")
        self.logger.info(f"torch query batch ({layer_idx}) topk+unique time used : {time.time() - st}")

        st = time.time()
        up = self.up[idx, ...]
        down = self.down[..., idx]
        self.logger.info(f"torch query batch ({layer_idx}) index time used : {time.time() - st}")

        return up.data, down.data, idx.data

    def query_batch_direct_async(self, query_vector, gpu_up, gpu_down, gpu_idx, top_k=10, layer_idx=-1, transfer=True):
        # todo remove this later
        assert (gpu_up.device == torch.device("cuda:0") and
                gpu_down.device == torch.device("cuda:0") and
                gpu_idx.device == torch.device("cuda:0")), f" device of idx {gpu_idx.device}"

        up, down, idx = self.query_batch_direct(query_vector, top_k, layer_idx, transfer=transfer)

        gpu_up.data[:up.shape[0], :].copy_(up.to(
            dtype=gpu_up.dtype,
            device=gpu_up.device,
            non_blocking=True
        ))
        gpu_up.data[up.shape[0]:, :] = 0.0
        gpu_down.data[:, :down.shape[1]].copy_(down.to(
            dtype=gpu_down.dtype,
            device=gpu_down.device,
            non_blocking=True
        ))
        gpu_down.data[:, down.shape[1]:] = 0.0
        gpu_idx.data[:idx.shape[0]] = idx.to(
            dtype=torch.int,
            device=gpu_idx.device,
            non_blocking=True
        )
        n = idx.shape[0]
        del up, down, idx, query_vector, gpu_up, gpu_down, gpu_idx
        # torch.cuda.empty_cache()
        return "ok", n
