"""
Triton SwiGLU kernel — research/prototyping kernel delivery path.

SwiGLU: output = silu(gate) * up
        silu(x) = x * sigmoid(x)

Two entry points:
  swiglu_ref(gate, up)   — pure PyTorch reference; runs on CPU or GPU
  swiglu_triton(gate, up) — Triton kernel; requires CUDA

verify_correctness() MUST pass before any benchmark call.

Benchmark schema (roofline dashboard depends on exact keys):
  {"kernel", "config", "tflops", "bandwidth_gb_s", "latency_ms", "pct_of_peak"}
"""
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# PyTorch reference (CPU + GPU)
# ---------------------------------------------------------------------------

def swiglu_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference implementation: silu(gate) * up."""
    return F.silu(gate) * up


# ---------------------------------------------------------------------------
# Triton kernel (CUDA only)
# ---------------------------------------------------------------------------

def _check_cuda(fn_name: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{fn_name} requires a CUDA GPU. "
            "Run on a cloud GPU instance (RunPod / AWS EC2)."
        )


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Fused SwiGLU via Triton kernel.

    Layout: (batch, N) or (N,) — N must be a multiple of BLOCK_SIZE (64).
    Templated over dtype: float32, float16, bfloat16.
    """
    _check_cuda("swiglu_triton")
    import triton
    import triton.language as tl

    assert gate.shape == up.shape, "gate and up must have the same shape"
    assert gate.is_cuda and up.is_cuda, "Inputs must be on CUDA"
    assert gate.is_contiguous() and up.is_contiguous(), "Inputs must be contiguous"

    @triton.jit
    def _swiglu_kernel(
        gate_ptr, up_ptr, out_ptr,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        gate_val = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up_val   = tl.load(up_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)

        # silu(gate) = gate * sigmoid(gate)
        silu_gate = gate_val * tl.sigmoid(gate_val)
        out_val   = silu_gate * up_val

        tl.store(out_ptr + offsets, out_val.to(gate_val.dtype), mask=mask)

    output = torch.empty_like(gate)
    N_total = gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N_total, BLOCK_SIZE),)

    _swiglu_kernel[grid](
        gate.reshape(-1), up.reshape(-1), output.reshape(-1),
        N_total,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# ---------------------------------------------------------------------------
# Correctness verification (must pass before benchmarking)
# ---------------------------------------------------------------------------

def verify_correctness(N: int = 4096, dtype: torch.dtype = torch.float32) -> bool:
    """
    Verifies Triton kernel against PyTorch reference.
    Returns True on pass, raises AssertionError on mismatch.
    Requires CUDA.
    """
    _check_cuda("verify_correctness")
    gate = torch.randn(4, N, device="cuda", dtype=dtype)
    up   = torch.randn(4, N, device="cuda", dtype=dtype)

    ref    = swiglu_ref(gate, up)
    actual = swiglu_triton(gate, up)

    atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3
    if not torch.allclose(actual, ref, atol=atol):
        max_diff = (actual - ref).abs().max().item()
        raise AssertionError(
            f"verify_correctness FAILED: N={N}, dtype={dtype}, max_diff={max_diff:.2e}"
        )
    return True


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(
    N_values: list[int] | None = None,
    dtype: torch.dtype = torch.float32,
) -> list[dict]:
    """
    Benchmarks the Triton SwiGLU kernel.
    verify_correctness() is called first — will raise if kernel is wrong.
    Returns list of dicts matching roofline dashboard schema.
    Requires CUDA.
    """
    _check_cuda("benchmark")
    import triton.testing

    if N_values is None:
        N_values = [512, 1024, 2048, 4096, 8192]

    # Verify before benchmarking — non-negotiable
    verify_correctness(N=N_values[0], dtype=dtype)

    from profiler.hardware_params import HARDWARE
    gpu_name = torch.cuda.get_device_name(0).lower()
    if "a100" in gpu_name:
        hw = HARDWARE["a100_80gb"]
        peak_tflops = hw["peak_tflops_bf16"] if dtype == torch.bfloat16 else hw["peak_tflops_fp32"]
        peak_bw     = hw["memory_bandwidth_gbs"]
    else:
        # Conservative fallback for other GPUs (T4, RTX 3090, etc.)
        peak_tflops = 20.0
        peak_bw     = 500.0

    results = []
    batch = 4

    for N in N_values:
        gate = torch.randn(batch, N, device="cuda", dtype=dtype)
        up   = torch.randn(batch, N, device="cuda", dtype=dtype)

        # FLOPs: per element — 1 mul (gate*sigmoid), 1 sigmoid, 1 mul (*up) ≈ 4 ops
        # Bytes: read gate + up, write output = 3 * batch * N * element_size
        n_elem    = batch * N
        flops     = 4 * n_elem                        # approximate
        bytes_rw  = 3 * n_elem * gate.element_size()

        ms = triton.testing.do_bench(
            lambda: swiglu_triton(gate, up),
            warmup=25,
            rep=100,
        )

        tflops        = (flops * 1e-12) / (ms * 1e-3)
        bandwidth_gbs = (bytes_rw * 1e-9) / (ms * 1e-3)
        pct_of_peak   = min(tflops / peak_tflops * 100, bandwidth_gbs / peak_bw * 100)

        results.append({
            "kernel":         "swiglu_triton",
            "config":         {"N": N, "batch": batch, "dtype": str(dtype)},
            "tflops":         tflops,
            "bandwidth_gb_s": bandwidth_gbs,
            "latency_ms":     ms,
            "pct_of_peak":    pct_of_peak,
        })

    return results
