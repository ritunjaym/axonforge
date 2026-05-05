"""
Three-way SwiGLU benchmark: PyTorch ref / Triton / C++/CUDA extension.

Reports per-implementation at each N: latency_ms, tflops, bandwidth_gb_s, pct_of_peak.
Results written to results/swiglu_benchmark.json.

Benchmark schema (roofline dashboard depends on exact keys):
  {"kernel", "config", "tflops", "bandwidth_gb_s", "latency_ms", "pct_of_peak"}

dry_run=True: returns schema-correct dicts with zero values (for local testing
without a GPU or the compiled extension).
"""
import json
import os
import torch

from csrc.functional import swiglu_ref
from kernels.swiglu import swiglu_triton
from profiler.hardware_params import HARDWARE

_DEFAULT_N = [512, 1024, 2048, 4096, 8192]
_BATCH     = 4


def _peak_for_device(dtype: torch.dtype) -> tuple[float, float]:
    """Returns (peak_tflops, peak_bandwidth_gbs) for the current GPU."""
    if not torch.cuda.is_available():
        return 20.0, 500.0
    name = torch.cuda.get_device_name(0).lower()
    if "a100" in name:
        hw = HARDWARE["a100_80gb"]
        tflops = hw["peak_tflops_bf16"] if dtype == torch.bfloat16 else hw["peak_tflops_fp32"]
        return tflops, hw["memory_bandwidth_gbs"]
    if "3090" in name:
        hw = HARDWARE["rtx_3090"]
        tflops = hw["peak_tflops_fp16"] if dtype in (torch.float16, torch.bfloat16) else hw["peak_tflops_fp32"]
        return tflops, hw["memory_bandwidth_gbs"]
    return 20.0, 500.0


def _make_record(
    kernel: str,
    N: int,
    dtype: torch.dtype,
    latency_ms: float,
    peak_tflops: float,
    peak_bw: float,
) -> dict:
    n_elem     = _BATCH * N
    flops      = 4 * n_elem                                     # silu + multiply ≈ 4 ops/element
    bytes_rw   = 3 * n_elem * torch.finfo(dtype).bits // 8      # read gate+up, write out
    tflops     = (flops * 1e-12) / (latency_ms * 1e-3)
    bw_gbs     = (bytes_rw * 1e-9) / (latency_ms * 1e-3)
    pct        = min(tflops / peak_tflops * 100, bw_gbs / peak_bw * 100)
    return {
        "kernel":         kernel,
        "config":         {"N": N, "batch": _BATCH, "dtype": str(dtype)},
        "tflops":         tflops,
        "bandwidth_gb_s": bw_gbs,
        "latency_ms":     latency_ms,
        "pct_of_peak":    pct,
    }


def run_benchmark(
    N_values: list[int] | None = None,
    dtype: torch.dtype = torch.float32,
    dry_run: bool = False,
) -> list[dict]:
    """
    Run three-way SwiGLU benchmark and save results to results/swiglu_benchmark.json.

    dry_run=True returns correctly-shaped dicts with zero timing values —
    used for schema validation in local tests without a GPU.
    """
    if N_values is None:
        N_values = _DEFAULT_N

    peak_tflops, peak_bw = _peak_for_device(dtype)
    results: list[dict] = []

    if dry_run:
        for N in N_values:
            for kernel in ("swiglu_ref", "swiglu_triton", "swiglu_cuda"):
                results.append(_make_record(kernel, N, dtype, 1.0, peak_tflops, peak_bw))
                # Override with zeros to signal dry-run
                results[-1]["latency_ms"]     = 0.0
                results[-1]["tflops"]         = 0.0
                results[-1]["bandwidth_gb_s"] = 0.0
                results[-1]["pct_of_peak"]    = 0.0
        return results

    # Real benchmark — requires CUDA + axonforge_ops
    import triton.testing
    import axonforge_ops

    for N in N_values:
        gate = torch.randn(_BATCH, N, device="cuda", dtype=dtype)
        up   = torch.randn(_BATCH, N, device="cuda", dtype=dtype)

        # 1. PyTorch reference
        ms_ref = triton.testing.do_bench(
            lambda: swiglu_ref(gate, up), warmup=25, rep=100,
        )
        results.append(_make_record("swiglu_ref", N, dtype, ms_ref, peak_tflops, peak_bw))

        # 2. Triton kernel
        ms_triton = triton.testing.do_bench(
            lambda: swiglu_triton(gate, up), warmup=25, rep=100,
        )
        results.append(_make_record("swiglu_triton", N, dtype, ms_triton, peak_tflops, peak_bw))

        # 3. C++/CUDA extension
        ms_cuda = triton.testing.do_bench(
            lambda: axonforge_ops.forward(gate, up), warmup=25, rep=100,
        )
        results.append(_make_record("swiglu_cuda", N, dtype, ms_cuda, peak_tflops, peak_bw))

    _save_results(results)
    return results


def _save_results(results: list[dict]) -> None:
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", "swiglu_benchmark.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[benchmark] saved {len(results)} records → {path}")
