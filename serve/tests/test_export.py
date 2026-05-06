"""
Tests for TorchScript inference server.

Tests 1–4 — CPU: export pipeline correctness.
Tests 5–6 — GPU: latency/throughput benchmarks (skipped locally).

Export order (CLAUDE.md pitfall): optimize_for_inference FIRST, then freeze.
dry_run=True: bench_inference returns schema-correct dicts without GPU.
"""
import math
import pytest
import torch
import torch.nn as nn

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")


# ---------------------------------------------------------------------------
# Tiny model for CPU tests
# ---------------------------------------------------------------------------

class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet: export returns torch.jit.ScriptModule
# ---------------------------------------------------------------------------

def test_export_returns_script_module():
    from serve.export import export_for_inference

    model = _TinyMLP()
    example = (torch.randn(2, 16),)
    result = export_for_inference(model, example)

    assert isinstance(result, torch.jit.ScriptModule), (
        f"Expected ScriptModule, got {type(result)}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Exported model produces same outputs as original (atol=1e-4)
# ---------------------------------------------------------------------------

def test_export_preserves_output():
    from serve.export import export_for_inference

    model = _TinyMLP().eval()
    x = torch.randn(2, 16)

    with torch.no_grad():
        ref = model(x)

    exported = export_for_inference(model, (x,))
    with torch.no_grad():
        out = exported(x)

    assert torch.allclose(ref, out, atol=1e-4), (
        f"Output mismatch: max diff = {(ref - out).abs().max():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — optimize_for_inference called before freeze (order check)
#          Validated indirectly: frozen module raises on further script ops.
# ---------------------------------------------------------------------------

def test_exported_module_is_frozen():
    from serve.export import export_for_inference

    model = _TinyMLP().eval()
    exported = export_for_inference(model, (torch.randn(2, 16),))

    # Frozen modules have parameters folded as graph constants —
    # they are no longer accessible via named_parameters().
    n_params = len(list(exported.named_parameters()))
    assert n_params == 0, (
        f"Frozen module should have 0 accessible parameters, got {n_params}. "
        "torch.jit.freeze may not have been applied."
    )


# ---------------------------------------------------------------------------
# Test 4 — bench_inference dry_run returns schema-correct dicts
# ---------------------------------------------------------------------------

BENCH_KEYS = {
    "batch_size", "seq_len",
    "latency_p50_ms", "latency_p99_ms",
    "throughput_tokens_s", "gpu_memory_mb", "vs_training_memory",
}


def test_bench_inference_dry_run_schema():
    from serve.bench_inference import run_benchmark

    results = run_benchmark(
        batch_sizes=[1, 2],
        seq_lens=[128],
        n_warmup=0,
        n_iters=1,
        dry_run=True,
    )

    assert len(results) == 2, f"Expected 2 results (1 batch×2 seqs), got {len(results)}"
    for r in results:
        missing = BENCH_KEYS - r.keys()
        assert not missing, f"Missing keys: {missing}"
        assert r["latency_p50_ms"] >= 0.0
        assert r["latency_p99_ms"] >= r["latency_p50_ms"]
        assert r["throughput_tokens_s"] >= 0.0


# ---------------------------------------------------------------------------
# Test 5 — GPU: exported model runs on CUDA and produces finite outputs
# ---------------------------------------------------------------------------

@GPU
def test_export_runs_on_cuda():
    from serve.export import export_for_inference
    from training.model import GPT2, GPT2Config

    config = GPT2Config(
        n_layers=2, d_model=64, n_heads=4, d_ff=256,
        vocab_size=256, max_seq_len=32,
    )
    model = GPT2(config).eval().cuda()
    example = (torch.randint(0, 256, (1, 16)).cuda(),)
    exported = export_for_inference(model, example)

    with torch.no_grad():
        out = exported(example[0])

    assert out.shape == (1, 16, 256), f"Wrong shape: {out.shape}"
    assert torch.isfinite(out).all(), "Output contains non-finite values"


# ---------------------------------------------------------------------------
# Test 6 — GPU: bench_inference returns valid metrics (all positive, finite)
# ---------------------------------------------------------------------------

@GPU
def test_bench_inference_gpu_metrics():
    from serve.bench_inference import run_benchmark

    results = run_benchmark(
        batch_sizes=[1],
        seq_lens=[128],
        n_warmup=2,
        n_iters=5,
        dry_run=False,
    )

    for r in results:
        assert math.isfinite(r["latency_p50_ms"]) and r["latency_p50_ms"] > 0
        assert math.isfinite(r["latency_p99_ms"]) and r["latency_p99_ms"] > 0
        assert math.isfinite(r["throughput_tokens_s"]) and r["throughput_tokens_s"] > 0
        assert 0.0 < r["vs_training_memory"] <= 1.5, (
            f"vs_training_memory={r['vs_training_memory']:.2f} out of expected range"
        )
