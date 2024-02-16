import logging
import time
import torch
import queue

try:
    from ..torch_vector_database import HugeFFN
    from ..glass_vdb import HugeGFFN
    from ..my_llama import create_file_logger
except ImportError:
    from modeling.torch_vector_database import HugeFFN
    from modeling.my_llama import create_file_logger
    from modeling.glass_vdb import HugeGFFN
    # raise


def recv_data(m2a_q, a2m_q, task_q, group_id, stop_event):
    logger = create_file_logger(
        f"async controller",
        f"examples/upd_assistant_proc_{group_id}.log"
    )
    torch.set_num_threads(1)
    last_time = time.time()
    while not stop_event.is_set():
        try:
            logger.debug(f"load_data process approximate size of task_q {task_q.qsize()}")
            task = task_q.get()
            buffer_task = m2a_q.get()
            if task is None:  # Use None as a sentinel to stop the worker
                a2m_q.put(None)
                break
            logger.info(f"waste time {time.time() - last_time}")
            method_name, layer_idx, args, kwargs = task
            _, __, b_args, b_kwargs = buffer_task
            assert method_name == _

            grad, idx, idx_real_len = args
            grad_ = grad.data.to(
                dtype=torch.float32, device='cpu', non_blocking=False
            )
            idx_ = idx.data.cpu()
            del grad, idx

            b_args[0].data[:, :].copy_(grad_)
            b_args[1].data[:].copy_(idx_)
            b_args[2] = idx_real_len
            assert b_args[0].is_shared() and b_args[1].is_shared()

            a2m_q.put((method_name, layer_idx, b_args, b_kwargs))
            last_time = time.time()
        except queue.Empty:
            continue


class HugeFFNAllController:
    def __init__(self, huge_ffn_init_params, layer_ids, config, num_group=1):
        import subprocess
        output = subprocess.check_output(['nproc'])
        core_count = int(output.decode().strip()) - 2

        self.layer_ids = layer_ids
        self.huge_ffn_init_params = huge_ffn_init_params
        self.config = config
        self.num_group = num_group
        mp = torch.multiprocessing.get_context('spawn')
        self.stop_event = mp.Event()  # An event to signal the process to stop

        self.group_task_upd_qs = [mp.Queue() for i in range(num_group)]
        self.group_task_query_qs = [mp.Queue() for i in range(num_group)]

        self.upd_task_queues = {}
        for i in layer_ids:
            self.upd_task_queues[i] = self.group_task_upd_qs[i % num_group]
        self.query_task_queues = {}
        for i in layer_ids:
            self.query_task_queues[i] = self.group_task_query_qs[i % num_group]

        self.upd_result_queues = {}
        for i in layer_ids:
            self.upd_result_queues[i] = mp.Queue()
        self.query_result_queues = {}
        for i in layer_ids:
            self.query_result_queues[i] = mp.Queue()

        upd_processes = []
        query_processes = []

        huge_ffns_d = {i: {} for i in range(num_group)}
        # huge_ffn_cls = HugeFFN if self.config.use_torch_vecdb else HugeGFFN
        for i in layer_ids:
            if self.config.use_torch_vecdb:
                huge_ffn = HugeFFN(*self.huge_ffn_init_params[0],
                                   logger='emm',
                                   **self.huge_ffn_init_params[1],
                                   init_optimizer=False)
            else:
                huge_ffn = HugeGFFN(*self.huge_ffn_init_params[0],
                                    logger='emm',
                                    lazy_init=True,
                                    **self.huge_ffn_init_params[1],
                                    init_optimizer=False)

            huge_ffn.set_shared_memory()
            huge_ffns_d[i % num_group][i] = huge_ffn

        # for i in layer_ids:
        #     self.query_task_queues[i].put((huge_ffns_d[0], i, 4))

        for i in range(num_group):
            p = mp.Process(target=self._upd_loop, args=(i, huge_ffns_d[i], max(1, int(core_count / num_group * 0.5))))
            upd_processes.append(p)
            p.start()

        for i in range(num_group):
            p = mp.Process(target=self._query_loop, args=(i, huge_ffns_d[i], max(1, int(core_count / num_group * 0.5))))
            query_processes.append(p)
            p.start()

        self.upd_ps = upd_processes
        self.query_ps = query_processes
        self.logger = logging.getLogger("running")
        self.grad_buffer = None
        self.grad_idx_buffer = None

    def _upd_loop(self, group_id, huge_ffns, num_threads):
        logger = create_file_logger(
            f"async controller",
            f"examples/upd_proc_{group_id}.log"
        )
        load_data_n = 3
        torch.set_num_threads(max(num_threads - load_data_n, 1))
        for _, huge_ffn in huge_ffns.items():
            huge_ffn.set_logger(logger)
            huge_ffn.set_optimizer()

        mp = torch.multiprocessing.get_context("spawn")
        main2ass_q, ass2main_q = mp.Queue(), mp.Queue()
        task_q = self.group_task_upd_qs[group_id]
        load_data_p = [mp.Process(target=recv_data, args=(main2ass_q, ass2main_q, task_q, group_id, self.stop_event))
                       for i in range(load_data_n)]

        for i in range(50):
            main2ass_q.put((
                "go_grad_async", -1,
                [torch.empty(self.config.hidden_size, self.config.on_gpu_size,
                             device='cpu', dtype=torch.float32).share_memory_(),
                 torch.empty(self.config.on_gpu_size, device='cpu', dtype=torch.int).share_memory_(),
                 0],
                {}
            ))
        print("start sub process")
        [p.start() for p in load_data_p]
        print("started")

        last_time = time.time()
        while not self.stop_event.is_set():
            try:
                logger.debug(f"approximate size of a2m_q {ass2main_q.qsize()}")
                task = ass2main_q.get()
                if task is None:  # Use None as a sentinel to stop the worker
                    break
                logger.info(f"waste {time.time() - last_time} s")
                method_name, layer_idx, args, kwargs = task
                logger.info(f"{layer_idx} get task {method_name}")
                # Retrieve the method to call
                method = getattr(huge_ffns[layer_idx], method_name)
                # Execute the method
                result = method(*args, **kwargs)
                main2ass_q.put(task)
                # del args, kwargs
                # torch.cuda.empty_cache()
                if "step" in result:
                    self.upd_result_queues[layer_idx].put((layer_idx, "step ok"))
                last_time = time.time()

                # logger.debug(f"{torch.cuda.memory_summary()}")
                # logger.debug("--------")
                # logger.debug(f"{torch.cuda.memory_snapshot()}")
                # logger.debug("-------")

            except queue.Empty:
                continue

        [p.join() for p in load_data_p]
        # load_data_p.join()

    def _query_loop(self, group_id, huge_ffns, num_threads):
        logger = create_file_logger(
            f"async controller",
            f"examples/query_proc_{group_id}.log"
        )
        torch.set_num_threads(num_threads)
        for _, huge_ffn in huge_ffns.items():
            huge_ffn.set_logger(logger)
            huge_ffn.num_threads = num_threads
            if isinstance(huge_ffn, HugeGFFN):
                huge_ffn.set_accelerator()

        task_q = self.group_task_query_qs[group_id]
        last_time = time.time()
        while not self.stop_event.is_set():
            try:
                logger.debug(f"approximate size of q {task_q.qsize()}")
                task = task_q.get()
                logger.info(f"waste {time.time() - last_time} s")
                method_name, layer_idx, args, kwargs = task
                logger.info(f"{layer_idx} get task {method_name}")
                if task is None:  # Use None as a sentinel to stop the worker
                    break
                # Retrieve the method to call
                method = getattr(huge_ffns[layer_idx], method_name)
                # Execute the method
                result = method(*args, **kwargs)
                del args, kwargs
                # torch.cuda.empty_cache()
                self.query_result_queues[layer_idx].put((layer_idx, result))
                last_time = time.time()

                # logger.debug(f"{torch.cuda.memory_summary()}")
                # logger.debug("--------")
                # logger.debug(f"{torch.cuda.memory_snapshot()}")
                # logger.debug("-------")

            except queue.Empty:
                continue

    def go_grad_async(self, grad, idx, idx_real_len, layer_idx):
        if (size := self.upd_task_queues[layer_idx].qsize()) > 32:
            self.logger.warning(f"wait go_grad{layer_idx}...  for qsize={size}")
            while self.upd_task_queues[layer_idx].qsize() > 32:
                pass

        self.logger.info(f"---> go_grad_async {layer_idx} idx_real_len={idx_real_len}")
        self.upd_task_queues[layer_idx].put(("go_grad_async", layer_idx, [grad.data, idx.data, idx_real_len], {}))
        # if self.grad_buffer is None:
        #     # self.grad_buffer = grad.data.cpu().share_memory_()
        #     # self.grad_idx_buffer = idx.data.cpu().share_memory_()
        #     self.grad_buffer = torch.empty_like(
        #         grad.data, dtype=torch.float32, device='cpu', requires_grad=False
        #     ).share_memory_()
        #     self.grad_idx_buffer = torch.empty_like(
        #         idx.data, dtype=torch.int, device='cpu', requires_grad=False
        #     ).share_memory_()

        # self.grad_buffer.data[:, :] = grad.data.cpu()
        # self.grad_idx_buffer.data[:idx_real_len] = idx.data[:idx_real_len].cpu()
        # del grad, idx

        # idx = idx.data[:idx_real_len]

    def query_batch_direct_async(self, query_vector, gpu_up, gpu_down, gpu_idx, top_k=10, layer_idx=-1):
        self.logger.info(f"---> query_batch_direct_async {layer_idx}")
        self.query_task_queues[layer_idx].put(("query_batch_direct_async", layer_idx,
                                               [query_vector.data, gpu_up.data, gpu_down.data, gpu_idx.data],
                                               {"top_k": top_k, "layer_idx": layer_idx, "transfer": True}))

    def get_query_res(self, layer_idx):
        layer, res = self.query_result_queues[layer_idx].get()
        assert layer == layer_idx and res[0] == 'ok'
        return res[1]

    def get_upd_res(self, layer_idx):
        layer, res = self.upd_result_queues[layer_idx].get()
        assert layer == layer_idx and res == 'step ok'

    def stop_worker(self):
        self.stop_event.set()
        for q in self.group_task_upd_qs:
            q.put(None)
        for q in self.group_task_query_qs:
            q.put(None)
        for p in self.query_ps:
            p.join()
        for p in self.upd_ps:
            p.join()


class AnyO:
    pass


if __name__ == "__main__":
    pass
