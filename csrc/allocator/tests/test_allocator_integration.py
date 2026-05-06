"""
Tests for allocator integration with training loop and inference server (Slice 11).

Tests 1–2 — CPU: dry_run training logs include allocator-stat keys (schema).
Tests 3–4 — GPU: live training + inference with allocator enabled.

Integration design:
  - Training: enable allocator before training, log stats every 100 steps.
    Stats go into training metrics dict alongside step/loss/etc.
  - Inference: enable allocator before bench_inference, compare hit rate.
    At inference batch sizes, hit rate should exceed training (~70% → >90%).
"""
import pytest
import torch

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
EXT = pytest.mark.skipif(
    not torch.cuda.is_available() or not _ext_available(),
    reason="requires CUDA GPU + axonforge_allocator extension",
)


def _ext_available() -> bool:
    try:
        import axonforge_allocator  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Test 1 — run_training with dry_run=True still includes allocator-stat schema
# ---------------------------------------------------------------------------

def test_training_metrics_include_allocator_schema_dry_run():
    """dry_run metrics dict must have allocator keys (schema validated on CPU)."""
    from training.train_fsdp import TrainingConfig, run_training

    config = TrainingConfig(
        n_layers=2, d_model=64, n_heads=4, d_ff=256,
        vocab_size=256, max_seq_len=32, batch_size=2,
        log_allocator_stats=True,
    )
    metrics = run_training(config, n_steps=5, dry_run=True)

    for m in metrics:
        assert "allocator_cache_hit_pct" in m, (
            f"Missing 'allocator_cache_hit_pct' in step {m['step']} metrics"
        )
        assert "allocator_fragmentation_pct" in m, (
            f"Missing 'allocator_fragmentation_pct' in step {m['step']} metrics"
        )


# ---------------------------------------------------------------------------
# Test 2 — allocator stats keys absent when log_allocator_stats=False
# ---------------------------------------------------------------------------

def test_training_metrics_no_allocator_keys_when_disabled():
    from training.train_fsdp import TrainingConfig, run_training

    config = TrainingConfig(
        n_layers=2, d_model=64, n_heads=4, d_ff=256,
        vocab_size=256, max_seq_len=32, batch_size=2,
        log_allocator_stats=False,
    )
    metrics = run_training(config, n_steps=2, dry_run=True)

    for m in metrics:
        assert "allocator_cache_hit_pct" not in m, (
            "allocator keys must be absent when log_allocator_stats=False"
        )


# ---------------------------------------------------------------------------
# Test 3 — GPU: allocator stats appear in live training metrics every 100 steps
# ---------------------------------------------------------------------------

@EXT
def test_allocator_stats_logged_every_100_steps():
    from training.train_fsdp import TrainingConfig, run_training

    config = TrainingConfig(
        n_layers=2, d_model=64, n_heads=4, d_ff=256,
        vocab_size=256, max_seq_len=32, batch_size=2, seed=0,
        log_allocator_stats=True,
    )
    metrics = run_training(config, n_steps=105, dry_run=False)

    # Steps 0 and 100 should have real stats; step 1 should have them only
    # if we log every step (we log every 100, so step 0 and 100 have stats).
    stats_steps = [
        m["step"] for m in metrics
        if m.get("allocator_cache_hit_pct") not in (None, 0.0, float("nan"))
        or m.get("allocator_fragmentation_pct") not in (None, float("nan"))
    ]
    assert any(s % 100 == 0 for s in stats_steps), (
        f"Expected allocator stats at step multiples of 100, got stats at: {stats_steps}"
    )


# ---------------------------------------------------------------------------
# Test 4 — GPU: inference allocator hit rate > training hit rate
# ---------------------------------------------------------------------------

@EXT
def test_inference_allocator_hit_rate_exceeds_training():
    import axonforge_allocator
    from training.train_fsdp import TrainingConfig, run_training
    from serve.bench_inference import run_benchmark

    # Training hit rate
    axonforge_allocator.enable()
    axonforge_allocator.reset_stats()

    config = TrainingConfig(
        n_layers=2, d_model=64, n_heads=4, d_ff=256,
        vocab_size=256, max_seq_len=32, batch_size=2, seed=0,
    )
    run_training(config, n_steps=20, dry_run=False)
    training_stats = axonforge_allocator.stats()
    training_hit_rate = training_stats["cache_hit_rate_pct"]
    axonforge_allocator.disable()

    # Inference hit rate
    axonforge_allocator.enable()
    axonforge_allocator.reset_stats()
    run_benchmark(batch_sizes=[1, 2], seq_lens=[128], n_warmup=3, n_iters=10, dry_run=False)
    inference_stats = axonforge_allocator.stats()
    inference_hit_rate = inference_stats["cache_hit_rate_pct"]
    axonforge_allocator.disable()

    assert inference_hit_rate > training_hit_rate, (
        f"Inference hit rate ({inference_hit_rate:.1f}%) should exceed "
        f"training hit rate ({training_hit_rate:.1f}%). "
        "Fixed inference shapes → better cache reuse than irregular training allocs."
    )
