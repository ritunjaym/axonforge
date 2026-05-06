"""
Tests for TopK gradient sparsification.

Tests 1–4 — CPU: sparsification math (no GPU/distributed needed).
Test  5   — GPU: hook integrates with FSDP training loop.

TopK sparsification: keep the top-K% largest-magnitude gradients,
zero the rest. Reduces AllReduce communication volume.

k_pct interpretation: fraction to KEEP (not to zero out).
  k_pct=0.1 → keep top 10%, zero 90% → grad_sparsity = 0.90
  k_pct=0.5 → keep top 50%, zero 50% → grad_sparsity = 0.50
"""
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from training.compression import top_k_sparsify, TopKSparsificationHook

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet: top_k_sparsify zeros all but top-K% by magnitude
# ---------------------------------------------------------------------------

def test_top_k_sparsify_basic():
    # 10 elements, keep top 30% = 3 elements
    t = torch.tensor([1.0, 5.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 0.5, 9.0])
    result = top_k_sparsify(t, k_pct=0.3)

    # Top 3 by magnitude: 9.0 (idx 9), 8.0 (idx 3), 7.0 (idx 5)
    n_nonzero = (result != 0).sum().item()
    assert n_nonzero == 3, f"Expected 3 non-zero elements, got {n_nonzero}"


# ---------------------------------------------------------------------------
# Test 2 — Non-zero count equals exactly round(k_pct * n)
# ---------------------------------------------------------------------------

@given(
    n=st.integers(min_value=10, max_value=200),
    k_pct=st.floats(min_value=0.05, max_value=0.95),
)
@settings(max_examples=30, deadline=None)
def test_nonzero_count_matches_k_pct(n, k_pct):
    t = torch.randn(n)
    result = top_k_sparsify(t, k_pct=k_pct)

    k = max(1, int(round(k_pct * n)))
    n_nonzero = (result != 0).sum().item()
    assert n_nonzero == k, (
        f"n={n}, k_pct={k_pct:.2f}: expected {k} non-zero, got {n_nonzero}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Retained values are the largest by magnitude
# ---------------------------------------------------------------------------

def test_retained_values_are_largest_magnitude():
    torch.manual_seed(0)
    t = torch.randn(100)
    k_pct = 0.2
    result = top_k_sparsify(t, k_pct=k_pct)

    k = max(1, int(round(k_pct * t.numel())))
    retained_magnitudes = result[result != 0].abs()
    zeroed_magnitudes   = t[result == 0].abs()

    # Every retained value must be ≥ every zeroed value
    assert retained_magnitudes.min() >= zeroed_magnitudes.max() - 1e-6, (
        f"Some zeroed value has larger magnitude than a retained value: "
        f"min_retained={retained_magnitudes.min():.4f}, "
        f"max_zeroed={zeroed_magnitudes.max():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Sparsity metric = 1 - k_pct
# ---------------------------------------------------------------------------

@given(k_pct=st.floats(min_value=0.05, max_value=0.95))
@settings(max_examples=20, deadline=None)
def test_sparsity_equals_one_minus_k_pct(k_pct):
    n = 200
    t = torch.randn(n)
    result = top_k_sparsify(t, k_pct=k_pct)

    k = max(1, int(round(k_pct * n)))
    expected_sparsity = 1.0 - k / n
    actual_sparsity   = (result == 0).float().mean().item()

    assert abs(actual_sparsity - expected_sparsity) < 1 / n, (
        f"k_pct={k_pct:.2f}: expected sparsity {expected_sparsity:.3f}, "
        f"got {actual_sparsity:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 5 — TopKSparsificationHook has correct interface
# ---------------------------------------------------------------------------

def test_hook_interface():
    hook = TopKSparsificationHook(k_pct=0.1)
    assert hook.k_pct == 0.1
    assert callable(hook), "Hook must be callable"


# ---------------------------------------------------------------------------
# Test 6 — GPU: hook reduces grad_sparsity when registered with training
# ---------------------------------------------------------------------------

@GPU
def test_compression_increases_grad_sparsity():
    from training.train_fsdp import TrainingConfig, run_training

    config = TrainingConfig(
        n_layers=2, d_model=64, n_heads=4, d_ff=256,
        vocab_size=256, max_seq_len=32, batch_size=2, seed=42,
    )

    # Run without compression
    metrics_no_compress = run_training(config, n_steps=3, dry_run=False)

    # Run with compression (use_compression=True)
    config_c = TrainingConfig(
        n_layers=2, d_model=64, n_heads=4, d_ff=256,
        vocab_size=256, max_seq_len=32, batch_size=2, seed=42,
        use_compression=True, k_pct=0.1,
    )
    metrics_compress = run_training(config_c, n_steps=3, dry_run=False)

    # With k_pct=0.1, grad_sparsity should be ~0.9
    avg_sparsity = sum(m["grad_sparsity"] for m in metrics_compress) / 3
    assert avg_sparsity > 0.5, (
        f"Expected grad_sparsity >0.5 with k_pct=0.1, got {avg_sparsity:.3f}"
    )
