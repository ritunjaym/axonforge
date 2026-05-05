"""
Tests for the three-way SwiGLU benchmark.

Tests 1–2 — CPU (M2 local): schema structure via dry_run.
Tests 3–5 — GPU + extension (cloud): real timing values, JSON output,
             Triton vs C++/CUDA gap < 15%.

Run on cloud GPU:
  cd csrc && python setup.py install
  cd .. && pytest csrc/tests/test_benchmark.py -v
"""
import json
import os
import pytest
import torch

EXT = pytest.mark.skipif(
    not torch.cuda.is_available() or not _ext_available(),
    reason="requires CUDA GPU + axonforge_ops extension",
)

N_VALUES = [512, 1024, 2048, 4096, 8192]
IMPLEMENTATIONS = {"swiglu_ref", "swiglu_triton", "swiglu_cuda"}
REQUIRED_KEYS   = {"kernel", "config", "tflops", "bandwidth_gb_s", "latency_ms", "pct_of_peak"}


def _ext_available() -> bool:
    try:
        import axonforge_ops  # noqa: F401
        return True
    except ImportError:
        return False


def _import_benchmark():
    from csrc.benchmark import run_benchmark
    return run_benchmark


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet (CPU): dry_run returns correct count + schema
# ---------------------------------------------------------------------------

def test_dry_run_returns_correct_count_and_schema():
    run_benchmark = _import_benchmark()

    results = run_benchmark(N_values=N_VALUES, dry_run=True)

    # 5 N values × 3 implementations = 15 dicts
    assert len(results) == len(N_VALUES) * len(IMPLEMENTATIONS), (
        f"Expected {len(N_VALUES) * len(IMPLEMENTATIONS)} results, got {len(results)}"
    )
    for r in results:
        missing = REQUIRED_KEYS - r.keys()
        assert not missing, f"Result missing keys: {missing}\n  result={r}"


# ---------------------------------------------------------------------------
# Test 2 — CPU: each result has a recognised kernel name
# ---------------------------------------------------------------------------

def test_dry_run_kernel_names():
    run_benchmark = _import_benchmark()
    results = run_benchmark(N_values=N_VALUES, dry_run=True)

    kernels_found = {r["kernel"] for r in results}
    assert kernels_found == IMPLEMENTATIONS, (
        f"Expected kernels {IMPLEMENTATIONS}, got {kernels_found}"
    )


# ---------------------------------------------------------------------------
# Test 3 — GPU: all timing values are positive (real benchmark run)
# ---------------------------------------------------------------------------

@EXT
def test_real_benchmark_values_are_positive():
    run_benchmark = _import_benchmark()
    results = run_benchmark(N_values=[512, 1024], dry_run=False)

    for r in results:
        assert r["latency_ms"]     > 0, f"{r['kernel']} N={r['config']['N']}: latency <= 0"
        assert r["tflops"]         > 0, f"{r['kernel']} N={r['config']['N']}: tflops <= 0"
        assert r["bandwidth_gb_s"] > 0, f"{r['kernel']} N={r['config']['N']}: bandwidth <= 0"
        assert 0 < r["pct_of_peak"] <= 100


# ---------------------------------------------------------------------------
# Test 4 — GPU: results saved to results/swiglu_benchmark.json
# ---------------------------------------------------------------------------

@EXT
def test_benchmark_saves_json(tmp_path, monkeypatch):
    run_benchmark = _import_benchmark()
    monkeypatch.chdir(tmp_path)
    os.makedirs("results", exist_ok=True)

    run_benchmark(N_values=[512], dry_run=False)

    json_path = tmp_path / "results" / "swiglu_benchmark.json"
    assert json_path.exists(), "results/swiglu_benchmark.json was not written"

    with open(json_path) as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == len(IMPLEMENTATIONS)   # 1 N × 3 impls
    for r in data:
        assert REQUIRED_KEYS.issubset(r.keys())


# ---------------------------------------------------------------------------
# Test 5 — GPU: Triton vs C++/CUDA latency gap < 15% at N=4096
# ---------------------------------------------------------------------------

@EXT
def test_triton_cuda_gap_under_15_percent():
    run_benchmark = _import_benchmark()
    results = run_benchmark(N_values=[4096], dry_run=False)

    by_kernel = {r["kernel"]: r for r in results}
    triton_ms = by_kernel["swiglu_triton"]["latency_ms"]
    cuda_ms   = by_kernel["swiglu_cuda"]["latency_ms"]

    faster = min(triton_ms, cuda_ms)
    gap_pct = abs(triton_ms - cuda_ms) / faster * 100

    assert gap_pct < 15.0, (
        f"Triton ({triton_ms:.3f}ms) vs C++/CUDA ({cuda_ms:.3f}ms) gap "
        f"= {gap_pct:.1f}% — exceeds 15% target"
    )
