"""
Tests for Pareto curve builder and allocator inference comparison.

Tests 1–3 — CPU: Pareto math + HTML generation (no GPU needed).
Test 4    — GPU: allocator comparison in inference mode (skipped locally).
"""
import math
import pytest
import torch


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_results(batch_sizes, seq_len=128):
    """Synthetic benchmark results with realistic latency/throughput."""
    results = []
    base_latency = 5.0
    for bs in batch_sizes:
        latency = base_latency * (1 + 0.3 * bs)  # latency grows with batch
        throughput = bs * seq_len / (latency / 1000.0)
        results.append({
            "batch_size":          bs,
            "seq_len":             seq_len,
            "latency_p50_ms":      latency,
            "latency_p99_ms":      latency * 1.2,
            "throughput_tokens_s": throughput,
            "gpu_memory_mb":       100.0 * bs,
            "vs_training_memory":  0.5,
        })
    return results


# ---------------------------------------------------------------------------
# Test 1 — find_knee returns a result dict with valid keys
# ---------------------------------------------------------------------------

def test_find_knee_returns_valid_dict():
    from serve.pareto import find_knee

    results = _make_results([1, 2, 4, 8, 16, 32])
    knee = find_knee(results)

    assert knee is not None, "find_knee returned None on non-empty results"
    assert "batch_size" in knee
    assert "latency_p50_ms" in knee
    assert "throughput_tokens_s" in knee


# ---------------------------------------------------------------------------
# Test 2 — find_knee identifies a knee when latency doubles and gains flatten
# ---------------------------------------------------------------------------

def test_find_knee_detects_saturation():
    from serve.pareto import find_knee

    # Throughput saturates after batch=4, latency doubles at batch=8
    results = [
        {"batch_size": 1,  "seq_len": 128, "latency_p50_ms": 5.0,  "latency_p99_ms": 6.0,  "throughput_tokens_s": 25_600,  "gpu_memory_mb": 100, "vs_training_memory": 0.5},
        {"batch_size": 2,  "seq_len": 128, "latency_p50_ms": 7.0,  "latency_p99_ms": 8.4,  "throughput_tokens_s": 36_571,  "gpu_memory_mb": 150, "vs_training_memory": 0.5},
        {"batch_size": 4,  "seq_len": 128, "latency_p50_ms": 9.0,  "latency_p99_ms": 10.8, "throughput_tokens_s": 60_000,  "gpu_memory_mb": 200, "vs_training_memory": 0.5},
        # Marginal gain ≈7% (<10%), latency = 2.8× baseline (>2×) → knee here:
        {"batch_size": 8,  "seq_len": 128, "latency_p50_ms": 14.0, "latency_p99_ms": 16.8, "throughput_tokens_s": 64_200,  "gpu_memory_mb": 300, "vs_training_memory": 0.5},
        {"batch_size": 16, "seq_len": 128, "latency_p50_ms": 28.0, "latency_p99_ms": 33.6, "throughput_tokens_s": 73_143,  "gpu_memory_mb": 500, "vs_training_memory": 0.5},
    ]
    knee = find_knee(results)

    # Knee should be identified at batch=8 (first point where gain<10% AND latency>2×)
    assert knee["batch_size"] == 8, (
        f"Expected knee at batch_size=8, got {knee['batch_size']}"
    )


# ---------------------------------------------------------------------------
# Test 3 — build_pareto_html writes a valid HTML file
# ---------------------------------------------------------------------------

def test_build_pareto_html_writes_file(tmp_path):
    from serve.pareto import build_pareto_html

    results = _make_results([1, 2, 4, 8])
    out_path = tmp_path / "pareto.html"
    result_path = build_pareto_html(results, out_path)

    assert result_path.exists(), "HTML file was not written"
    content = result_path.read_text()
    assert "<html" in content.lower(), "Output does not look like HTML"
    assert "Pareto" in content, "Output missing 'Pareto' label"
    assert "batch_size" in content, "Output missing batch_size reference"


# ---------------------------------------------------------------------------
# Test 4 — GPU: allocator comparison shows >90% cache hit in inference mode
# ---------------------------------------------------------------------------

@GPU
def test_allocator_cache_hit_rate_inference():
    from csrc.allocator import axonforge_allocator
    from serve.bench_inference import run_benchmark

    axonforge_allocator.enable()
    axonforge_allocator.reset_stats()

    run_benchmark(
        batch_sizes=[1, 2, 4],
        seq_lens=[128],
        n_warmup=3,
        n_iters=10,
        dry_run=False,
    )

    stats = axonforge_allocator.stats()
    axonforge_allocator.disable()

    hit_rate = stats["cache_hit_rate_pct"]
    assert hit_rate > 90.0, (
        f"Expected >90% cache hit rate at inference batch sizes, got {hit_rate:.1f}%. "
        "Inference patterns are fixed-shape → pooled allocator should reuse aggressively."
    )
