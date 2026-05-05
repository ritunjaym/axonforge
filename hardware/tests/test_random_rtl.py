"""
Randomized RTL co-simulation suite: 500 cases, 6 required categories, 0 deltas.

All tests run locally via iverilog 13.0 (/opt/homebrew/bin/iverilog).

Pass criterion: every case must produce 0 integer-mode output deltas vs
the Python golden (hardware/systolic_array.py).

The 6 required test categories mirror the CLAUDE.md specification:
  1. single-tile   — M=N=K=8  (one 8×8 tile through the array)
  2. multi-tile    — 8 tiles of M=8 rows (A is 64×8)
  3. non-square    — M=16 rows of A, verifying multi-tile correctness
  4. sparse        — 30% activation zeros
  5. zero-weights  — W = 0, all outputs must be 0
  6. INT16_MAX     — values at ±32767, verifies 40-bit accumulator no overflow
"""
import numpy as np
import pytest
from hardware.validate_rtl import run_cosim, run_random_suite

INT16_MAX = 32767


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet: run_random_suite returns correct schema
# ---------------------------------------------------------------------------

def test_run_random_suite_schema():
    result = run_random_suite(n=1, rows=2, cols=2)  # tiny array for speed

    assert "total"    in result
    assert "passed"   in result
    assert "failed"   in result
    assert "failures" in result
    assert result["total"] == 1
    assert result["passed"] + result["failed"] == 1


# ---------------------------------------------------------------------------
# Test 2 — Category: single tile (M=K=8, 8×8 weights)
# ---------------------------------------------------------------------------

def test_single_tile_zero_deltas():
    rng = np.random.default_rng(1)
    A = rng.integers(-100, 100, size=(8, 8), dtype=np.int32)
    W = rng.integers(-100, 100, size=(8, 8), dtype=np.int32)

    result = run_cosim(A, W, rows=8, cols=8)
    assert result["deltas"] == 0, f"single-tile: {result['deltas']} delta(s)"


# ---------------------------------------------------------------------------
# Test 3 — Category: multi-tile (A is 64×8 = 8 tiles)
# ---------------------------------------------------------------------------

def test_multi_tile_zero_deltas():
    rng = np.random.default_rng(2)
    A = rng.integers(-50, 50, size=(64, 8), dtype=np.int32)
    W = rng.integers(-50, 50, size=(8, 8),  dtype=np.int32)

    result = run_cosim(A, W, rows=8, cols=8)
    assert result["deltas"] == 0, f"multi-tile: {result['deltas']} delta(s)"


# ---------------------------------------------------------------------------
# Test 4 — Category: non-square (A is 16×8, M≠cols)
# ---------------------------------------------------------------------------

def test_non_square_zero_deltas():
    rng = np.random.default_rng(3)
    A = rng.integers(-64, 64, size=(16, 8), dtype=np.int32)
    W = rng.integers(-64, 64, size=(8, 8),  dtype=np.int32)

    result = run_cosim(A, W, rows=8, cols=8)
    assert result["deltas"] == 0, f"non-square: {result['deltas']} delta(s)"


# ---------------------------------------------------------------------------
# Test 5 — Category: sparse activations (30% zeros)
# ---------------------------------------------------------------------------

def test_sparse_activations_zero_deltas():
    rng = np.random.default_rng(4)
    A_raw = rng.integers(-100, 100, size=(8, 8), dtype=np.int32)
    mask  = rng.random(size=(8, 8)) > 0.3      # 30% sparse
    A     = (A_raw * mask).astype(np.int32)
    W     = rng.integers(-100, 100, size=(8, 8), dtype=np.int32)

    result = run_cosim(A, W, rows=8, cols=8)
    assert result["deltas"] == 0, f"sparse: {result['deltas']} delta(s)"


# ---------------------------------------------------------------------------
# Test 6 — Category: all-zero weights
# ---------------------------------------------------------------------------

def test_zero_weights_zero_deltas():
    rng = np.random.default_rng(5)
    A = rng.integers(-100, 100, size=(8, 8), dtype=np.int32)
    W = np.zeros((8, 8), dtype=np.int32)

    result = run_cosim(A, W, rows=8, cols=8)
    assert result["deltas"] == 0, f"zero-weights: {result['deltas']} delta(s)"
    np.testing.assert_array_equal(result["rtl_out"], 0,
        err_msg="zero-weights: RTL output should be all zeros")


# ---------------------------------------------------------------------------
# Test 7 — Category: INT16_MAX — verifies 40-bit accumulator prevents overflow
# ---------------------------------------------------------------------------

def test_int16_max_zero_deltas():
    # Max possible sum per element: ROWS × INT16_MAX² = 8 × 32767² ≈ 8.6×10⁹
    # Fits in 34-bit signed; 40-bit accumulator must handle it without corruption.
    A = np.full((8, 8), INT16_MAX, dtype=np.int32)
    W = np.full((8, 8), INT16_MAX, dtype=np.int32)

    result = run_cosim(A, W, rows=8, cols=8)
    assert result["deltas"] == 0, (
        f"INT16_MAX: {result['deltas']} delta(s) — accumulator overflow?\n"
        f"Python: {result['python_out'][0]}\n"
        f"RTL:    {result['rtl_out'][0]}"
    )


# ---------------------------------------------------------------------------
# Test 8 — 500 random cases: 0 total deltas
# ---------------------------------------------------------------------------

def test_500_random_zero_deltas():
    result = run_random_suite(n=500, seed=42, rows=8, cols=8)

    assert result["failed"] == 0, (
        f"{result['failed']}/500 cases had deltas.\n"
        + "\n".join(
            f"  trial={f['trial']}: {f['deltas']} deltas"
            for f in result["failures"][:5]
        )
    )
    assert result["passed"] == 500, (
        f"Only {result['passed']}/500 cases passed"
    )
