"""
Tests for the FSDP training loop.

Tests 1–3 — CPU (M2 local): schema + model structure (dry_run mode).
Tests 4–5 — GPU (cloud): real FSDP training, determinism check.

Per-step metrics schema (all 7 keys required):
  step, loss, step_time_ms, gpu_memory_mib, grad_sparsity,
  allreduce_ms, compute_ms, overlap_pct
"""
import math
import pytest
import torch
import torch.nn as nn

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")

REQUIRED_KEYS = {
    "step", "loss", "step_time_ms", "gpu_memory_mib",
    "grad_sparsity", "allreduce_ms", "compute_ms", "overlap_pct",
}


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet: dry_run returns schema-correct list
# ---------------------------------------------------------------------------

def test_dry_run_returns_schema():
    from training.train_fsdp import TrainingConfig, run_training

    config = TrainingConfig(n_layers=2, d_model=64, n_heads=4, d_ff=256,
                            vocab_size=256, max_seq_len=32, batch_size=2)
    metrics = run_training(config, n_steps=3, dry_run=True)

    assert len(metrics) == 3, f"Expected 3 steps, got {len(metrics)}"
    for i, m in enumerate(metrics):
        assert m["step"] == i, f"Step {i} has step={m['step']}"
        missing = REQUIRED_KEYS - m.keys()
        assert not missing, f"Step {i} missing keys: {missing}"


# ---------------------------------------------------------------------------
# Test 2 — All 7 metric keys present in every step
# ---------------------------------------------------------------------------

def test_all_metric_keys_present():
    from training.train_fsdp import TrainingConfig, run_training

    config = TrainingConfig(n_layers=2, d_model=64, n_heads=4, d_ff=256,
                            vocab_size=256, max_seq_len=32, batch_size=2)
    metrics = run_training(config, n_steps=1, dry_run=True)

    m = metrics[0]
    for key in REQUIRED_KEYS:
        assert key in m, f"Required key '{key}' missing from metrics"
        assert m[key] is not None, f"Key '{key}' is None"


# ---------------------------------------------------------------------------
# Test 3 — GPT-2 model has correct architecture
# ---------------------------------------------------------------------------

def test_gpt2_model_structure():
    from training.model import GPT2, GPT2Config

    # Tiny model for fast testing
    config = GPT2Config(n_layers=2, d_model=64, n_heads=4,
                        d_ff=256, vocab_size=256, max_seq_len=32)
    model = GPT2(config)

    # Forward pass must work
    x = torch.randint(0, 256, (2, 16))      # (batch, seq_len)
    logits = model(x)
    assert logits.shape == (2, 16, 256),  f"Wrong logits shape: {logits.shape}"

    # GPT-2 124M: ~124M params. Our tiny model should be much smaller.
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0, "Model has no parameters"


def test_gpt2_124m_parameter_count():
    """Full GPT-2 124M should have ~124M parameters."""
    from training.model import GPT2, GPT2Config

    config = GPT2Config()   # default = 124M config
    model = GPT2(config)
    n_params = sum(p.numel() for p in model.parameters())

    # GPT-2 124M has 124,439,808 parameters
    assert 120_000_000 < n_params < 130_000_000, (
        f"Expected ~124M params, got {n_params:,}"
    )


# ---------------------------------------------------------------------------
# Test 4 — GPU: determinism — two runs with same seed produce identical loss
# ---------------------------------------------------------------------------

@GPU
def test_determinism_same_seed():
    from training.train_fsdp import TrainingConfig, run_training

    config = TrainingConfig(
        n_layers=2, d_model=128, n_heads=4, d_ff=512,
        vocab_size=512, max_seq_len=64, batch_size=2, seed=42,
    )

    metrics1 = run_training(config, n_steps=5, dry_run=False)
    metrics2 = run_training(config, n_steps=5, dry_run=False)

    for i, (m1, m2) in enumerate(zip(metrics1, metrics2)):
        assert m1["loss"] == m2["loss"], (
            f"Non-deterministic loss at step {i}: {m1['loss']} vs {m2['loss']}"
        )


# ---------------------------------------------------------------------------
# Test 5 — GPU: per-step metrics are valid (positive, finite)
# ---------------------------------------------------------------------------

@GPU
def test_gpu_metrics_are_valid():
    from training.train_fsdp import TrainingConfig, run_training

    config = TrainingConfig(
        n_layers=2, d_model=128, n_heads=4, d_ff=512,
        vocab_size=512, max_seq_len=64, batch_size=2, seed=0,
    )

    metrics = run_training(config, n_steps=3, dry_run=False)

    for m in metrics:
        assert math.isfinite(m["loss"]),             f"loss is not finite: {m['loss']}"
        assert m["step_time_ms"]  > 0,               f"step_time_ms <= 0"
        assert m["gpu_memory_mib"] > 0,              f"gpu_memory_mib <= 0"
        assert 0.0 <= m["grad_sparsity"] <= 1.0,     f"grad_sparsity out of [0,1]"
        assert m["compute_ms"]    > 0,               f"compute_ms <= 0"
        assert 0.0 <= m["overlap_pct"] <= 1.0,       f"overlap_pct out of [0,1]"
