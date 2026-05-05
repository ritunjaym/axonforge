/*
 * C++ dispatch layer for axonforge SwiGLU CUDA extension.
 *
 * Validates inputs with TORCH_CHECK (not bare assert — gives informative errors).
 * Dispatches to typed CUDA launchers via AT_DISPATCH_FLOATING_TYPES_AND_HALF.
 * Always uses at::cuda::getCurrentCUDAStream() — never the default stream.
 * Exposes forward() and backward() via pybind11.
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Forward declarations of typed launchers defined in swiglu_kernel.cu
template <typename scalar_t>
void launch_swiglu_forward(
    const scalar_t* gate, const scalar_t* up, scalar_t* out,
    int64_t numel, cudaStream_t stream);

template <typename scalar_t>
void launch_swiglu_backward(
    const scalar_t* grad_out, const scalar_t* gate, const scalar_t* up,
    scalar_t* grad_gate, scalar_t* grad_up,
    int64_t numel, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

at::Tensor swiglu_forward(const at::Tensor& gate, const at::Tensor& up) {
    TORCH_CHECK(gate.is_cuda(),       "gate must be a CUDA tensor");
    TORCH_CHECK(up.is_cuda(),         "up must be a CUDA tensor");
    TORCH_CHECK(gate.is_contiguous(), "gate must be contiguous");
    TORCH_CHECK(up.is_contiguous(),   "up must be contiguous");
    TORCH_CHECK(gate.sizes() == up.sizes(),
                "gate and up must have the same shape");
    TORCH_CHECK(gate.scalar_type() == up.scalar_type(),
                "gate and up must have the same dtype");

    at::cuda::CUDAGuard guard(gate.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto out = at::empty_like(gate);
    int64_t numel = gate.numel();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(gate.scalar_type(), "swiglu_forward", [&] {
        launch_swiglu_forward<scalar_t>(
            gate.data_ptr<scalar_t>(),
            up.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            numel,
            stream
        );
    });

    return out;
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------

std::vector<at::Tensor> swiglu_backward(
    const at::Tensor& grad_out,
    const at::Tensor& gate,
    const at::Tensor& up
) {
    TORCH_CHECK(grad_out.is_cuda() && gate.is_cuda() && up.is_cuda(),
                "all tensors must be on CUDA");
    TORCH_CHECK(grad_out.is_contiguous() && gate.is_contiguous() && up.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(grad_out.sizes() == gate.sizes(),
                "grad_out and gate must have the same shape");

    at::cuda::CUDAGuard guard(gate.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto grad_gate = at::empty_like(gate);
    auto grad_up   = at::empty_like(up);
    int64_t numel  = gate.numel();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(gate.scalar_type(), "swiglu_backward", [&] {
        launch_swiglu_backward<scalar_t>(
            grad_out.data_ptr<scalar_t>(),
            gate.data_ptr<scalar_t>(),
            up.data_ptr<scalar_t>(),
            grad_gate.data_ptr<scalar_t>(),
            grad_up.data_ptr<scalar_t>(),
            numel,
            stream
        );
    });

    return {grad_gate, grad_up};
}

// ---------------------------------------------------------------------------
// Python binding
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "AxonForge SwiGLU CUDA extension";
    m.def("forward",  &swiglu_forward,  "SwiGLU forward (CUDA)");
    m.def("backward", &swiglu_backward, "SwiGLU backward (CUDA)");
}
