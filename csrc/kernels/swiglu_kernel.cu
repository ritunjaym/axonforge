/*
 * SwiGLU CUDA kernels — forward and backward.
 *
 * Forward:  out = silu(gate) * up
 *           silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Backward: grad_gate = grad_out * up * silu'(gate)
 *           grad_up   = grad_out * silu(gate)
 *           silu'(x)  = silu(x) + sigmoid(x) * (1 - silu(x))
 *                     = silu(x) * (1 + sigmoid(x) * (1 - sigmoid(x)) / sigmoid(x))
 *           Simpler:  silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
 *
 * Design:
 *   - Shared memory tiling: each block loads a TILE of gate+up into __shared__,
 *     computes silu in float32 registers, writes result back. Demonstrates the
 *     tiling pattern; actual bandwidth gain vs. register-only is minimal for
 *     pointwise ops but the pattern generalises to fused kernels.
 *   - Grid-stride loop: handles arbitrary total elements regardless of grid size.
 *   - Templated over scalar_t; instantiated for float, __half, __nv_bfloat16.
 *   - All computation in float32 to avoid reduced-precision silu errors.
 *   - CUDA_CHECK on every CUDA API call.
 *   - at::cuda::getCurrentCUDAStream() — never the default stream.
 *   - cudaOccupancyMaxActiveBlocksPerMultiprocessor logged to stderr at first call.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdio>
#include <cmath>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_err));             \
            abort();                                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Device helpers (always computed in float32)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));   // x * sigmoid(x)
}

__device__ __forceinline__ float silu_grad_f(float x) {
    // silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    float sig = 1.0f / (1.0f + __expf(-x));
    return sig * (1.0f + x * (1.0f - sig));
}

// ---------------------------------------------------------------------------
// Forward kernel
// ---------------------------------------------------------------------------

template <typename scalar_t, int TILE>
__global__ void swiglu_forward_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    scalar_t*       __restrict__ out,
    int64_t numel
) {
    __shared__ float s_gate[TILE];
    __shared__ float s_up[TILE];

    // Grid-stride loop over TILE-sized chunks
    for (int64_t tile_base = (int64_t)blockIdx.x * TILE;
         tile_base < numel;
         tile_base += (int64_t)gridDim.x * TILE)
    {
        int64_t idx = tile_base + threadIdx.x;

        // Coalesced load from global → shared (cast to float32)
        s_gate[threadIdx.x] = (idx < numel) ? (float)gate[idx] : 0.0f;
        s_up[threadIdx.x]   = (idx < numel) ? (float)up[idx]   : 0.0f;
        __syncthreads();

        // Compute in registers from shared memory
        if (idx < numel) {
            float val = silu_f(s_gate[threadIdx.x]) * s_up[threadIdx.x];
            out[idx] = (scalar_t)val;
        }
        __syncthreads();    // guard before next tile overwrites shared mem
    }
}

// ---------------------------------------------------------------------------
// Backward kernel
// ---------------------------------------------------------------------------

template <typename scalar_t, int TILE>
__global__ void swiglu_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    scalar_t*       __restrict__ grad_gate,
    scalar_t*       __restrict__ grad_up,
    int64_t numel
) {
    __shared__ float s_grad_out[TILE];
    __shared__ float s_gate[TILE];
    __shared__ float s_up[TILE];

    for (int64_t tile_base = (int64_t)blockIdx.x * TILE;
         tile_base < numel;
         tile_base += (int64_t)gridDim.x * TILE)
    {
        int64_t idx = tile_base + threadIdx.x;

        s_grad_out[threadIdx.x] = (idx < numel) ? (float)grad_out[idx] : 0.0f;
        s_gate[threadIdx.x]     = (idx < numel) ? (float)gate[idx]     : 0.0f;
        s_up[threadIdx.x]       = (idx < numel) ? (float)up[idx]       : 0.0f;
        __syncthreads();

        if (idx < numel) {
            float go  = s_grad_out[threadIdx.x];
            float g   = s_gate[threadIdx.x];
            float u   = s_up[threadIdx.x];
            float sg  = silu_f(g);

            grad_gate[idx] = (scalar_t)(go * u * silu_grad_f(g));
            grad_up[idx]   = (scalar_t)(go * sg);
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Occupancy logging (called once per dtype at first use)
// ---------------------------------------------------------------------------

template <typename scalar_t>
static void log_occupancy_once() {
    static bool logged = false;
    if (logged) return;
    logged = true;

    constexpr int TILE = 1024;
    int max_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks,
        swiglu_forward_kernel<scalar_t, TILE>,
        TILE,
        2 * TILE * sizeof(float)    // shared memory size
    ));
    fprintf(stderr, "[axonforge] SwiGLU forward occupancy: %d blocks/SM (dtype=%s)\n",
            max_blocks,
            std::is_same<scalar_t, float>::value ? "float32" :
            std::is_same<scalar_t, __half>::value ? "float16" : "bfloat16");
}

// ---------------------------------------------------------------------------
// Host-side launchers (called from swiglu_op.cpp)
// ---------------------------------------------------------------------------

template <typename scalar_t>
void launch_swiglu_forward(
    const scalar_t* gate,
    const scalar_t* up,
    scalar_t*       out,
    int64_t         numel,
    cudaStream_t    stream
) {
    constexpr int TILE      = 1024;
    constexpr int MAX_BLOCKS = 1024;
    int blocks = (int)std::min((int64_t)MAX_BLOCKS, (numel + TILE - 1) / TILE);

    log_occupancy_once<scalar_t>();

    swiglu_forward_kernel<scalar_t, TILE>
        <<<blocks, TILE, 2 * TILE * sizeof(float), stream>>>(
            gate, up, out, numel);

    CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t>
void launch_swiglu_backward(
    const scalar_t* grad_out,
    const scalar_t* gate,
    const scalar_t* up,
    scalar_t*       grad_gate,
    scalar_t*       grad_up,
    int64_t         numel,
    cudaStream_t    stream
) {
    constexpr int TILE      = 1024;
    constexpr int MAX_BLOCKS = 1024;
    int blocks = (int)std::min((int64_t)MAX_BLOCKS, (numel + TILE - 1) / TILE);

    swiglu_backward_kernel<scalar_t, TILE>
        <<<blocks, TILE, 3 * TILE * sizeof(float), stream>>>(
            grad_out, gate, up, grad_gate, grad_up, numel);

    CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiations so the linker can find them from swiglu_op.cpp
template void launch_swiglu_forward<float>(
    const float*, const float*, float*, int64_t, cudaStream_t);
template void launch_swiglu_forward<__half>(
    const __half*, const __half*, __half*, int64_t, cudaStream_t);
template void launch_swiglu_forward<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int64_t, cudaStream_t);

template void launch_swiglu_backward<float>(
    const float*, const float*, const float*, float*, float*, int64_t, cudaStream_t);
template void launch_swiglu_backward<__half>(
    const __half*, const __half*, const __half*, __half*, __half*, int64_t, cudaStream_t);
template void launch_swiglu_backward<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, int64_t, cudaStream_t);
