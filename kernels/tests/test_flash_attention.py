"""
Tests for the Triton FlashAttention kernel.

Test 1 runs on CPU (M2 local): verifies attention_ref matches PyTorch SDPA.
Tests 2–5 require CUDA and run on cloud GPU (RunPod / AWS EC2).

FlashAttention: O = softmax(QK^T / sqrt(d)) @ V
  Key property: never materialises the full N×N attention matrix —
  uses online softmax tiling over SRAM blocks.
"""
import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from kernels.flash_attention import attention_ref, flash_attention, verify_correctness, benchmark

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet (CPU): reference matches PyTorch scaled_dot_product_attention
# ---------------------------------------------------------------------------

@given(
    batch=st.integers(min_value=1, max_value=4),
    n_heads=st.sampled_from([1, 2, 4]),
    seq_len=st.sampled_from([16, 32, 64, 128]),
    head_dim=st.sampled_from([32, 64]),
)
@settings(max_examples=50)
def test_attention_ref_matches_sdpa(batch, n_heads, seq_len, head_dim):
    q = torch.randn(batch, n_heads, seq_len, head_dim)
    k = torch.randn(batch, n_heads, seq_len, head_dim)
    v = torch.randn(batch, n_heads, seq_len, head_dim)

    expected = F.scaled_dot_product_attention(q, k, v)
    actual   = attention_ref(q, k, v)

    assert torch.allclose(actual, expected, atol=1e-5), (
        f"attention_ref deviates from SDPA: max_diff={(actual - expected).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Triton kernel matches reference (float32)
# ---------------------------------------------------------------------------

@CUDA
@pytest.mark.parametrize("seq_len", [128, 256, 512, 1024])
def test_flash_attention_matches_reference_float32(seq_len):
    batch, n_heads, head_dim = 2, 4, 64
    q = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
    v = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)

    ref    = attention_ref(q, k, v)
    actual = flash_attention(q, k, v)

    assert torch.allclose(actual, ref, atol=1e-2), (
        f"FlashAttention mismatch at seq={seq_len}: max_diff={(actual - ref).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 3 — verify_correctness() passes for all required seq_len
# ---------------------------------------------------------------------------

@CUDA
@pytest.mark.parametrize("seq_len", [128, 256, 512, 1024])
def test_verify_correctness_passes(seq_len):
    assert verify_correctness(seq_len=seq_len), (
        f"verify_correctness() failed at seq_len={seq_len}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Output shape matches input
# ---------------------------------------------------------------------------

@CUDA
@pytest.mark.parametrize("seq_len", [128, 512])
def test_output_shape(seq_len):
    batch, n_heads, head_dim = 2, 4, 64
    q = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda")
    k = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda")
    v = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda")

    out = flash_attention(q, k, v)

    assert out.shape == (batch, n_heads, seq_len, head_dim), (
        f"Wrong output shape: {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Benchmark schema
# ---------------------------------------------------------------------------

@CUDA
def test_benchmark_schema():
    results = benchmark(seq_lens=[128, 256], head_dim=64, n_heads=4, dtype=torch.float32)

    assert len(results) == 2
    required = {"kernel", "config", "tflops", "bandwidth_gb_s", "latency_ms", "pct_of_peak"}
    for r in results:
        assert required.issubset(r.keys()), f"Missing keys: {required - r.keys()}"
        assert r["tflops"] > 0
        assert r["bandwidth_gb_s"] > 0
        assert r["latency_ms"] > 0
        assert 0 < r["pct_of_peak"] <= 100
