"""
Tests for the Triton SwiGLU kernel.

Test 1 runs on CPU (M2 local): verifies swiglu_ref matches the mathematical definition.
Tests 2–5 require CUDA and run on cloud GPU (RunPod / AWS EC2).

SwiGLU: output = silu(gate) * up
        silu(x) = x * sigmoid(x)
"""
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from kernels.swiglu import swiglu_ref, swiglu_triton, verify_correctness, benchmark

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet (CPU): reference matches mathematical definition
# ---------------------------------------------------------------------------

@given(
    batch=st.integers(min_value=1, max_value=8),
    n=st.sampled_from([64, 128, 256, 512]),
)
@settings(max_examples=50)
def test_swiglu_ref_matches_definition(batch, n):
    gate = torch.randn(batch, n)
    up   = torch.randn(batch, n)

    expected = gate * torch.sigmoid(gate) * up   # silu(gate) * up by definition
    actual   = swiglu_ref(gate, up)

    assert torch.allclose(actual, expected, atol=1e-6), (
        f"swiglu_ref deviates from definition: max_diff={(actual - expected).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Triton kernel matches reference (float32)
# ---------------------------------------------------------------------------

@CUDA
@pytest.mark.parametrize("N", [512, 1024, 2048, 4096, 8192])
def test_triton_matches_reference_float32(N):
    gate = torch.randn(4, N, device="cuda", dtype=torch.float32)
    up   = torch.randn(4, N, device="cuda", dtype=torch.float32)

    ref    = swiglu_ref(gate, up)
    actual = swiglu_triton(gate, up)

    assert torch.allclose(actual, ref, atol=1e-3), (
        f"Triton float32 mismatch at N={N}: max_diff={(actual - ref).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 3 — verify_correctness() passes for all required N
# ---------------------------------------------------------------------------

@CUDA
@pytest.mark.parametrize("N", [512, 1024, 2048, 4096])
def test_verify_correctness_passes(N):
    assert verify_correctness(N=N, dtype=torch.float32), (
        f"verify_correctness() failed at N={N}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Benchmark returns required schema
# ---------------------------------------------------------------------------

@CUDA
def test_benchmark_schema():
    results = benchmark(N_values=[512, 1024], dtype=torch.float32)

    assert len(results) == 2
    required_keys = {"kernel", "config", "tflops", "bandwidth_gb_s", "latency_ms", "pct_of_peak"}
    for r in results:
        assert required_keys.issubset(r.keys()), (
            f"Benchmark result missing keys: {required_keys - r.keys()}"
        )
        assert r["tflops"] > 0
        assert r["bandwidth_gb_s"] > 0
        assert r["latency_ms"] > 0
        assert 0 < r["pct_of_peak"] <= 100


# ---------------------------------------------------------------------------
# Test 5 — Dtype dispatch: float16 and bfloat16
# ---------------------------------------------------------------------------

@CUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_dtype_dispatch(dtype):
    N = 1024
    gate = torch.randn(4, N, device="cuda", dtype=dtype)
    up   = torch.randn(4, N, device="cuda", dtype=dtype)

    out = swiglu_triton(gate, up)

    assert out.dtype == dtype, f"Output dtype {out.dtype} != input dtype {dtype}"
    # Compare against float32 reference with relaxed tolerance for reduced precision
    ref = swiglu_ref(gate.float(), up.float()).to(dtype)
    assert torch.allclose(out, ref, atol=1e-2), (
        f"Dtype {dtype} mismatch: max_diff={(out - ref).abs().max():.2e}"
    )
