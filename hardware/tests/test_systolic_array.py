"""
Tests for the pipelined systolic array Python simulation.

All tests run on CPU — no GPU, no external tooling.

Architecture under test:
  SystolicArray(rows=8, cols=8, pipeline_stages=3, data_width=16)
  Weight-stationary: W[k][j] fixed in PE[k][j], activations flow east,
  partial sums flow south.
  PIPE_STAGES flip-flop registers between each PE pair.

Key invariant: utilization = active_mac_cycles / steady_state_cycles
  NOT divided by total_cycles (which includes fill and drain).
"""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hardware.systolic_array import SystolicArray


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet (CPU): fill_cycles formula
#
# fill_cycles = (rows + cols - 2) * (1 + pipeline_stages)
# This is the number of cycles until ALL PEs are computing simultaneously.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows,cols,stages,expected", [
    (2, 2, 0, 2),    # (2+2-2)*(1+0) = 2
    (2, 2, 3, 8),    # (2+2-2)*(1+3) = 8
    (8, 8, 3, 56),   # (8+8-2)*(1+3) = 56  ← the spec case
    (8, 8, 0, 14),   # (8+8-2)*(1+0) = 14
    (4, 8, 1, 20),   # (4+8-2)*(1+1) = 20
])
def test_fill_cycles_formula(rows, cols, stages, expected):
    arr = SystolicArray(rows=rows, cols=cols, pipeline_stages=stages)
    assert arr.fill_cycles() == expected, (
        f"SystolicArray({rows},{cols},{stages}).fill_cycles() = "
        f"{arr.fill_cycles()}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Test 2 — 2×2 manual trace: correct matrix multiply output
#
# W = [[1,2],[3,4]], A = [[1,2],[5,6]]
# Expected: A @ W = [[1*1+2*3, 1*2+2*4],[5*1+6*3, 5*2+6*4]]
#                 = [[7, 10], [23, 34]]
# ---------------------------------------------------------------------------

def test_2x2_matrix_multiply_correctness():
    W = np.array([[1, 2], [3, 4]], dtype=np.int32)
    A = np.array([[1, 2], [5, 6]], dtype=np.int32)

    arr = SystolicArray(rows=2, cols=2, pipeline_stages=0, data_width=16)
    arr.load_weights(W)
    result = arr.run(A)

    expected = A @ W
    np.testing.assert_array_equal(result, expected,
        err_msg=f"2×2 result {result} != expected {expected}")


# ---------------------------------------------------------------------------
# Test 3 — Hypothesis: run(A) == A @ W for random 8×8 inputs
# ---------------------------------------------------------------------------

@given(
    A_vals=st.lists(st.integers(-64, 64), min_size=64, max_size=64),
    W_vals=st.lists(st.integers(-64, 64), min_size=64, max_size=64),
)
@settings(max_examples=20)
def test_run_matches_numpy_matmul(A_vals, W_vals):
    rows, cols = 8, 8
    A = np.array(A_vals, dtype=np.int32).reshape(rows, rows)
    W = np.array(W_vals, dtype=np.int32).reshape(rows, cols)

    arr = SystolicArray(rows=rows, cols=cols, pipeline_stages=0, data_width=16)
    arr.load_weights(W)
    result = arr.run(A)

    expected = A @ W
    np.testing.assert_array_equal(result, expected,
        err_msg=f"Systolic result does not match numpy matmul")


# ---------------------------------------------------------------------------
# Test 4 — Utilization: measured against steady_state_cycles, NOT total_cycles
# ---------------------------------------------------------------------------

def test_utilization_denominator_is_steady_state_not_total():
    rows, cols, stages = 8, 8, 0
    A = np.ones((rows, rows), dtype=np.int32)
    W = np.ones((rows, cols), dtype=np.int32)

    arr = SystolicArray(rows=rows, cols=cols, pipeline_stages=stages)
    arr.load_weights(W)
    arr.run(A)

    stats = arr.get_stats()

    # The utilization denominator must be steady_state_cycles
    assert stats["steady_state_cycles"] > 0
    assert stats["total_cycles"] > stats["steady_state_cycles"], (
        "total_cycles should exceed steady_state_cycles (fill + steady + drain)"
    )
    # Utilization computed from steady_state, not total
    expected_util = stats["active_mac_cycles"] / stats["steady_state_cycles"]
    assert abs(stats["utilization"] - expected_util) < 1e-9, (
        f"utilization {stats['utilization']:.4f} != "
        f"active_mac/{stats['steady_state_cycles']} = {expected_util:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Separate cycle counts: fill + steady_state + drain reported
# ---------------------------------------------------------------------------

def test_cycle_counts_reported_separately():
    arr = SystolicArray(rows=8, cols=8, pipeline_stages=3)
    W = np.ones((8, 8), dtype=np.int32)
    A = np.ones((8, 8), dtype=np.int32)
    arr.load_weights(W)
    arr.run(A)

    stats = arr.get_stats()
    for key in ("fill_cycles", "steady_state_cycles", "drain_cycles", "total_cycles"):
        assert key in stats, f"Missing stat: {key}"
        assert stats[key] > 0, f"{key} should be positive, got {stats[key]}"

    # Sanity: fill == drain (symmetric), both equal the formula
    expected_fill = arr.fill_cycles()
    assert stats["fill_cycles"] == expected_fill, (
        f"fill_cycles {stats['fill_cycles']} != formula {expected_fill}"
    )
    assert stats["drain_cycles"] == expected_fill, (
        f"drain_cycles {stats['drain_cycles']} != fill_cycles {expected_fill}"
    )


# ---------------------------------------------------------------------------
# Test 6 — Power model: P_dyn = alpha * C * V^2 * f
# ---------------------------------------------------------------------------

def test_power_model_formula():
    arr = SystolicArray(rows=8, cols=8, pipeline_stages=3, data_width=16)
    W = np.ones((8, 8), dtype=np.int32)
    A = np.ones((8, 8), dtype=np.int32)
    arr.load_weights(W)
    arr.run(A)

    # Known inputs
    alpha = 0.3          # switching activity fraction
    C_fF  = 10.0         # capacitance per MAC in femtofarads
    V     = 0.9          # supply voltage in volts
    f_GHz = 1.0          # clock frequency in GHz

    power = arr.get_power(alpha=alpha, C_fF=C_fF, V=V, f_GHz=f_GHz)

    # P_dyn = alpha * C * V^2 * f
    # Units: (fF)(V^2)(GHz) = (1e-15 F)(V^2)(1e9 Hz) = 1e-6 W = µW
    C_F  = C_fF * 1e-15
    f_Hz = f_GHz * 1e9
    expected_p_dyn_mw = alpha * C_F * V**2 * f_Hz * 1e3   # in mW

    assert "P_dyn_mW" in power, "Power dict missing P_dyn_mW"
    assert "P_static_mW" in power, "Power dict missing P_static_mW"
    assert abs(power["P_dyn_mW"] - expected_p_dyn_mw) < 1e-6, (
        f"P_dyn_mW {power['P_dyn_mW']:.6f} != expected {expected_p_dyn_mw:.6f}"
    )
    assert power["P_static_mW"] > 0
