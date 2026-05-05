"""
Tests for the allocator benchmark (training vs inference patterns).

Tests 1–2 — CPU (M2 local): schema validation via dry_run.
Tests 3–5 — GPU + extension (cloud): real benchmark runs.

Key insight under test:
  Inference allocation patterns (fixed shapes, repeated) achieve >90% cache hit
  rate because the same size classes are requested every iteration.
  Training patterns (irregular FSDP shard sizes, gradient temporaries) achieve
  lower hit rates (~70%) because of size diversity.
"""
import json
import os
import pytest
import torch

EXT = pytest.mark.skipif(
    not torch.cuda.is_available() or not _ext_available(),
    reason="requires CUDA GPU + axonforge_allocator extension",
)

REQUIRED_PATTERN_KEYS = {
    "mean_alloc_free_us",
    "peak_allocated_mb",
    "cache_hit_rate_pct",
    "fragmentation_pct",
    "num_cudaMalloc_calls",
}
REQUIRED_TOP_KEYS = {"training", "inference", "insight"}


def _ext_available() -> bool:
    try:
        import axonforge_allocator  # noqa: F401
        return True
    except ImportError:
        return False


def _import_bench():
    from csrc.allocator.allocator_bench import run_allocator_benchmark
    return run_allocator_benchmark


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet (CPU): dry_run returns correct top-level schema
# ---------------------------------------------------------------------------

def test_dry_run_top_level_schema():
    run = _import_bench()
    result = run(dry_run=True)

    missing = REQUIRED_TOP_KEYS - result.keys()
    assert not missing, f"Missing top-level keys: {missing}"


# ---------------------------------------------------------------------------
# Test 2 — CPU: both patterns have all required metric keys
# ---------------------------------------------------------------------------

def test_dry_run_pattern_keys():
    run = _import_bench()
    result = run(dry_run=True)

    for pattern in ("training", "inference"):
        missing = REQUIRED_PATTERN_KEYS - result[pattern].keys()
        assert not missing, f"Pattern '{pattern}' missing keys: {missing}"

    # Insight must be a non-empty string
    assert isinstance(result["insight"], str) and len(result["insight"]) > 0


# ---------------------------------------------------------------------------
# Test 3 — GPU: inference cache hit rate exceeds training cache hit rate
# ---------------------------------------------------------------------------

@EXT
def test_inference_higher_cache_hit_than_training():
    run = _import_bench()
    result = run(n_iters=200, dry_run=False)

    train_hit  = result["training"]["cache_hit_rate_pct"]
    infer_hit  = result["inference"]["cache_hit_rate_pct"]

    assert infer_hit > train_hit, (
        f"Expected inference hit rate ({infer_hit:.1f}%) > "
        f"training hit rate ({train_hit:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Test 4 — GPU: results saved to results/allocator_benchmark.json
# ---------------------------------------------------------------------------

@EXT
def test_benchmark_saves_json(tmp_path, monkeypatch):
    run = _import_bench()
    monkeypatch.chdir(tmp_path)
    os.makedirs("results", exist_ok=True)

    run(n_iters=100, dry_run=False)

    json_path = tmp_path / "results" / "allocator_benchmark.json"
    assert json_path.exists(), "results/allocator_benchmark.json was not written"

    with open(json_path) as f:
        data = json.load(f)

    assert REQUIRED_TOP_KEYS.issubset(data.keys())
    for pattern in ("training", "inference"):
        assert REQUIRED_PATTERN_KEYS.issubset(data[pattern].keys())


# ---------------------------------------------------------------------------
# Test 5 — GPU: inference cache hit rate > 90% (fixed-shape pattern)
# ---------------------------------------------------------------------------

@EXT
def test_inference_cache_hit_above_90_pct():
    run = _import_bench()
    result = run(n_iters=500, dry_run=False)

    infer_hit = result["inference"]["cache_hit_rate_pct"]
    assert infer_hit > 90.0, (
        f"Inference cache hit rate {infer_hit:.1f}% < 90% target.\n"
        f"Fixed-shape allocation should achieve near-perfect cache reuse."
    )
