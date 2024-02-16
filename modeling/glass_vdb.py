import time
try:
    import cppeff
except ModuleNotFoundError:
    pass
import torch

try:
    from torch_vector_database import *
except ModuleNotFoundError:
    from modeling.torch_vector_database import *
import glassppy as glass


class HugeGFFN(HugeFFN):
    def __init__(self, *args, g_R=32, g_L=100, g_level=2, g_ef=36, lazy_init=False, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.config.frozen_key:
            raise ValueError("need frozen_key == True")
        self.g_R = g_R
        self.g_L = g_L
        self.g_level = g_level
        self.g_ef = g_ef
        self.query_vector_cpu = None
        if not lazy_init:
            self.set_accelerator()

    def set_accelerator(self):
        d = self.up.shape[1]
        self.index = glass.Index(index_type="HNSW", dim=d, metric='ip', R=self.g_R, L=self.g_L)
        data = self.up.numpy()
        self.g = self.index.build(data)
        self.s = glass.Searcher(graph=self.g, data=data, metric='ip', level=self.g_level)
        self.s.set_ef(self.g_ef)
        self.dd = d

    def query_batch_direct(self, query_vector, top_k=10, layer_idx=-1, transfer=True):
        st = time.time()
        if transfer:
            # if self.query_vector_cpu is None or self.query_vector_cpu.shape != query_vector.shape:
            #     self.query_vector_cpu = torch.empty_like(query_vector, pin_memory=True, requires_grad=False,
            #                                              device=self.device, dtype=torch.float32)
            # self.query_vector_cpu.copy_(query_vector.data)
            # query_vector = self.query_vector_cpu
            query_vector = query_vector.data
            query_vector = query_vector.to(dtype=torch.float32, device=self.device, non_blocking=self.non_blocking)
        self.logger.info(f"glass query batch ({layer_idx}) transfer time used : {time.time() - st}")

        st = time.time()
        x = query_vector.numpy().reshape(-1, self.dd)
        idx = self.s.batch_search(query=x, k=top_k, num_threads=self.num_threads).reshape(-1)
        idx = torch.tensor(idx)
        self.logger.info(f"glass query batch ({layer_idx}) compute time used : {time.time() - st}")

        if self.config.cpp_mode:
            st = time.time()
            up, down, idx = cppeff.unique_then_select(self.up, self.down, idx, self.config.on_gpu_size, layer_idx)
            self.logger.info(f"glass query batch ({layer_idx}) unique_then_select time used : {time.time() - st}")
        else:
            st = time.time()
            idx = torch.unique(idx)
            if idx.shape[0] > self.config.on_gpu_size:
                self.logger.warning(f"out of space {idx.shape[0]}")
                idx = idx[:self.config.on_gpu_size]
            self.logger.info(f"glass query batch ({layer_idx}) unique time used : {time.time() - st}")

            st = time.time()
            up = self.up[idx]
            down = self.down[:, idx]
            self.logger.info(f"glass query batch ({layer_idx}) index time used : {time.time() - st}")

        return up.data, down.data, idx.data
