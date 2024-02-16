#include <torch/extension.h>
#include <iostream>
#include <chrono>
#include <iostream>

namespace py = pybind11;

int test() {
    // Set the number of threads
//    torch::set_num_threads(16);
    // Initialize variables
    double avg_time = 0;
    int n = 100;
    torch::Tensor dst = torch::randn({4096, 20000});

    for (int i = 0; i < n; ++i) {
        torch::Tensor src = torch::randn({4096, 3072});
        torch::Tensor idx = torch::randperm(20000, torch::TensorOptions().dtype(torch::kLong)).slice(0, 0, 2048);

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();

        // Perform the operation
        dst.index_add_(1, idx, src.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, 2048)}));

        // Stop timing
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        avg_time += diff.count();
    }

    avg_time /= n;
    std::cout << avg_time << std::endl;

    return 0;
}


// Function to perform the operations
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
unique_then_select(torch::Tensor &up, torch::Tensor &down, torch::Tensor idx, int on_gpu_size, int layer_idx) {
//    torch::set_num_threads(16);
    // Generate a random permutation of indices and select a subset
//    py::gil_scoped_release release;     // 释放GIL锁
    idx = std::get<0>(at::_unique(idx));

    // Truncate idx to on_gpu_size if necessary
    if (idx.size(0) > on_gpu_size) {
        idx = idx.slice(0, 0, on_gpu_size);
        // Log warning about out of space if logger was included
    }

    // Perform indexing operations
    torch::Tensor up_selected = up.index({idx});
    torch::Tensor down_selected = down.index({torch::indexing::Ellipsis, idx});
//    py::gil_scoped_acquire acquire;     // C++执行结束前恢复GIL锁
    return std::make_tuple(up_selected, down_selected, idx);
}

// Function to perform the gradient operations
torch::Tensor update_grad(at::Tensor &grad, at::Tensor &down_grad, const at::Tensor &idx,
 const at::Device &device, bool non_blocking) {
//    torch::set_num_threads(16);
//    py::gil_scoped_release release;     // 释放GIL锁
    int n = idx.size(0);
    at::Tensor grad_view = grad.slice(1, 0, n);
    grad_view = grad_view.to(down_grad.options().device(device).dtype(down_grad.dtype()), non_blocking);
    down_grad.index_add_(1, idx, grad_view);
//    py::gil_scoped_acquire acquire;     // C++执行结束前恢复GIL锁
    return down_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("test", &test, "test efficiency");
  m.def("unique_then_select", &unique_then_select, "emm");
  m.def("update_grad", &update_grad, "emmm");
}
