"""
Tests for the ping-pong memory controller.

All tests run on CPU — no GPU, no external tooling.

Design under test:
  MemoryController wraps SystolicArray and splits large activation matrices
  into tiles. Two scheduling modes:

  Baseline (no ping-pong):
    For each tile: stall memory_latency cycles (load) → compute
    stall_cycles = n_tiles × memory_latency

  Ping-pong (double buffer):
    While tile k computes from buffer A, tile k+1 loads into buffer B.
    stall_cycles = n_tiles × max(0, memory_latency − tile_compute_cycles)
    When tile_compute_cycles ≥ memory_latency: stalls → 0 (perfect overlap).

Target: ≥30% stall reduction vs baseline.
"""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hardware.systolic_array import SystolicArray
from hardware.memory_controller import MemoryController


def _make_controller(memory_latency: int = 10) -> MemoryController:
    arr = SystolicArray(rows=8, cols=8, pipeline_stages=0, data_width=16)
    return MemoryController(arr, memory_latency_cycles=memory_latency)


def _make_inputs(n_tiles: int = 4, rows: int = 8, cols: int = 8):
    rng = np.random.default_rng(42)
    A = rng.integers(-16, 16, size=(n_tiles * rows, rows), dtype=np.int32)
    W = rng.integers(-16, 16, size=(rows, cols),           dtype=np.int32)
    return A, W


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet: ping-pong and baseline produce identical output
# ---------------------------------------------------------------------------

def test_pingpong_output_matches_baseline():
    ctrl = _make_controller(memory_latency=10)
    A, W = _make_inputs(n_tiles=4)

    result_base, _  = ctrl.run_baseline(A, W)
    result_ping, _  = ctrl.run_pingpong(A, W)

    np.testing.assert_array_equal(result_base, result_ping,
        err_msg="Ping-pong output differs from baseline — correctness broken")


# ---------------------------------------------------------------------------
# Test 2 — Baseline stall model: stall_cycles == n_tiles × memory_latency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_tiles,mem_lat", [
    (2, 10), (4, 10), (4, 20), (8, 15),
])
def test_baseline_stall_formula(n_tiles, mem_lat):
    ctrl = _make_controller(memory_latency=mem_lat)
    A, W = _make_inputs(n_tiles=n_tiles)

    _, stats = ctrl.run_baseline(A, W)

    expected_stalls = n_tiles * mem_lat
    assert stats["stall_cycles"] == expected_stalls, (
        f"Baseline stalls: expected {expected_stalls}, got {stats['stall_cycles']}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Ping-pong has fewer stalls than baseline
# ---------------------------------------------------------------------------

def test_pingpong_fewer_stalls_than_baseline():
    ctrl = _make_controller(memory_latency=10)
    A, W = _make_inputs(n_tiles=4)

    _, base_stats = ctrl.run_baseline(A, W)
    _, ping_stats = ctrl.run_pingpong(A, W)

    assert ping_stats["stall_cycles"] < base_stats["stall_cycles"], (
        f"Ping-pong stalls ({ping_stats['stall_cycles']}) should be < "
        f"baseline ({base_stats['stall_cycles']})"
    )


# ---------------------------------------------------------------------------
# Test 4 — Stall reduction ≥ 30%
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_tiles,mem_lat", [
    (4, 10), (4, 20), (8, 10), (8, 20),
])
def test_stall_reduction_at_least_30_pct(n_tiles, mem_lat):
    ctrl = _make_controller(memory_latency=mem_lat)
    A, W = _make_inputs(n_tiles=n_tiles)

    ctrl.run_baseline(A, W)
    ctrl.run_pingpong(A, W)

    reduction = ctrl.stall_reduction_pct()
    assert reduction >= 30.0, (
        f"Stall reduction {reduction:.1f}% < 30% target "
        f"(n_tiles={n_tiles}, mem_lat={mem_lat})"
    )


# ---------------------------------------------------------------------------
# Test 5 — Stall reduction stable across tile counts (hypothesis)
# ---------------------------------------------------------------------------

@given(n_tiles=st.integers(min_value=2, max_value=8))
@settings(max_examples=10)
def test_stall_reduction_stable_across_tile_counts(n_tiles):
    ctrl = _make_controller(memory_latency=10)
    A, W = _make_inputs(n_tiles=n_tiles)

    ctrl.run_baseline(A, W)
    ctrl.run_pingpong(A, W)

    assert ctrl.stall_reduction_pct() >= 30.0
