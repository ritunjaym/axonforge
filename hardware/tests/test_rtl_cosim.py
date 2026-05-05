"""
Co-simulation tests: Python golden (SystolicArray) vs SystemVerilog RTL.

All tests run locally — iverilog 13.0 is at /opt/homebrew/bin/iverilog.

Co-simulation contract: for any integer-mode inputs, RTL output must
match Python golden exactly (0 deltas). This is the pre-silicon validation
methodology: Python is the spec, RTL must conform to it.
"""
import numpy as np
import pytest
from hardware.validate_rtl import run_cosim

IVERILOG = "/opt/homebrew/bin/iverilog"


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet: run_cosim is importable and callable
# ---------------------------------------------------------------------------

def test_run_cosim_is_callable():
    assert callable(run_cosim), "run_cosim must be a callable"


# ---------------------------------------------------------------------------
# Test 2 — systolic_array.sv compiles without errors
# ---------------------------------------------------------------------------

def test_sv_compiles_cleanly(tmp_path):
    import subprocess
    from pathlib import Path

    sv_path = Path(__file__).parent.parent / "rtl" / "systolic_array.sv"
    assert sv_path.exists(), f"systolic_array.sv not found at {sv_path}"

    result = subprocess.run(
        [IVERILOG, "-g2012", "-o", str(tmp_path / "check.vvp"), str(sv_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"iverilog compile failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 3 — 2×2 correctness: RTL matches Python golden, 0 deltas
# ---------------------------------------------------------------------------

def test_2x2_zero_deltas():
    W = np.array([[1, 2], [3, 4]], dtype=np.int32)
    A = np.array([[1, 2], [5, 6]], dtype=np.int32)

    result = run_cosim(A, W, rows=2, cols=2, pipeline_stages=0)

    assert result["deltas"] == 0, (
        f"Expected 0 deltas, got {result['deltas']}.\n"
        f"Python: {result['python_out']}\n"
        f"RTL:    {result['rtl_out']}"
    )
    np.testing.assert_array_equal(
        result["python_out"], A @ W,
        err_msg="Python golden itself is wrong"
    )


# ---------------------------------------------------------------------------
# Tests 4–6 — Required test categories (0 deltas each)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category,A,W", [
    (
        "single_tile_8x8",
        np.ones((8, 8), dtype=np.int32),
        np.eye(8, dtype=np.int32),
    ),
    (
        "zero_weights",
        np.array([[1, 2], [3, 4]], dtype=np.int32),
        np.zeros((2, 2), dtype=np.int32),
    ),
    (
        "sparse_activations_30pct",
        (np.random.default_rng(7).integers(-8, 8, (8, 8)) *
         (np.random.default_rng(7).random((8, 8)) > 0.3)).astype(np.int32),
        np.random.default_rng(7).integers(-8, 8, (8, 8)).astype(np.int32),
    ),
])
def test_category_zero_deltas(category, A, W):
    rows, cols = W.shape
    result = run_cosim(A, W, rows=rows, cols=cols, pipeline_stages=0)

    assert result["deltas"] == 0, (
        f"Category '{category}': {result['deltas']} delta(s) found.\n"
        f"Max diff: {np.abs(result['python_out'] - result['rtl_out']).max()}"
    )
