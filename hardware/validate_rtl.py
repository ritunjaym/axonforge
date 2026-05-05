"""
Python co-simulation driver for the systolic array RTL.

Methodology (pre-silicon validation):
  1. Python SystolicArray is the golden reference (verified in Slice 12).
  2. For each test case (A, W), generate a SystemVerilog testbench with the
     same skewed injection schedule used by Python.
  3. Compile with iverilog -g2012 and run with vvp.
  4. Parse $display output to reconstruct the RTL output matrix.
  5. Compare element-by-element: delta count must be 0.

Pitfall: iverilog is at /opt/homebrew/bin/iverilog on Apple Silicon — not
always on PATH. The driver uses the full path.

Pitfall: always_ff is SV syntax; requires -g2012 flag.
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from hardware.systolic_array import SystolicArray

_IVERILOG = "/opt/homebrew/bin/iverilog"
_VVP      = "/opt/homebrew/bin/vvp"
_RTL_DIR  = Path(__file__).parent / "rtl"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_random_suite(
    n: int = 500,
    seed: int = 42,
    rows: int = 8,
    cols: int = 8,
    pipeline_stages: int = 0,
    data_width: int = 16,
) -> dict:
    """
    Runs n random co-simulation cases and returns a summary.

    Each case: random INT16 A (rows×rows) and W (rows×cols), optional sparsity.
    Returns:
      {
        "total":    int,     # = n
        "passed":   int,     # cases with 0 deltas
        "failed":   int,     # cases with >0 deltas
        "failures": list,    # details of any failures
      }
    """
    rng = np.random.default_rng(seed)
    INT16_MAX = 32767

    passed, failed, failures = 0, 0, []

    for trial in range(n):
        # Random test vector: INT16 values, 0–30% sparsity on activations
        sparsity = rng.uniform(0.0, 0.3)
        A_raw = rng.integers(-INT16_MAX, INT16_MAX + 1,
                             size=(rows, rows), dtype=np.int32)
        mask  = rng.random(size=(rows, rows)) > sparsity
        A     = (A_raw * mask).astype(np.int32)
        W     = rng.integers(-INT16_MAX, INT16_MAX + 1,
                             size=(rows, cols), dtype=np.int32)

        result = run_cosim(A, W, rows=rows, cols=cols,
                           pipeline_stages=pipeline_stages,
                           data_width=data_width)
        if result["deltas"] == 0:
            passed += 1
        else:
            failed += 1
            failures.append({
                "trial":  trial,
                "deltas": result["deltas"],
                "A":      A.tolist(),
                "W":      W.tolist(),
            })

    return {
        "total":    n,
        "passed":   passed,
        "failed":   failed,
        "failures": failures,
    }


def run_cosim(
    A: np.ndarray,
    W: np.ndarray,
    rows: int = 8,
    cols: int = 8,
    pipeline_stages: int = 0,
    data_width: int = 16,
) -> dict:
    """
    Runs co-simulation: Python golden vs iverilog RTL simulation.

    Returns:
      {
        "deltas":     int,           # number of mismatched elements (must be 0)
        "python_out": np.ndarray,    # Python golden result
        "rtl_out":    np.ndarray,    # RTL simulation result
      }
    """
    # Step 1: Python golden
    arr = SystolicArray(rows=rows, cols=cols,
                        pipeline_stages=pipeline_stages, data_width=data_width)
    arr.load_weights(W)
    python_out = arr.run(A)

    # Step 2: Generate testbench and run RTL simulation
    rtl_out = _simulate_rtl(A, W, rows, cols, pipeline_stages, data_width)

    # Step 3: Compare
    deltas = int(np.sum(python_out != rtl_out))
    return {
        "deltas":     deltas,
        "python_out": python_out,
        "rtl_out":    rtl_out,
    }


# ---------------------------------------------------------------------------
# RTL simulation internals
# ---------------------------------------------------------------------------

def _simulate_rtl(
    A: np.ndarray,
    W: np.ndarray,
    rows: int,
    cols: int,
    pipeline_stages: int,
    data_width: int,
) -> np.ndarray:
    """Compiles and runs the RTL, returns the output matrix."""
    M       = A.shape[0]
    hop     = 1 + pipeline_stages
    fill    = (rows + cols - 2) * hop
    total   = fill + M * hop + fill + hop   # safety margin

    # Build injection schedule (same skewing as Python)
    inject = [[0] * rows for _ in range(total)]
    for i in range(M):
        for k in range(rows):
            t = (i + k) * hop
            if t < total:
                inject[t][k] = int(A[i][k])

    # Output collection times: C[i][j] ready at cycle (i+j+rows-1)*hop
    # Multiple (i,j) pairs can share the same cycle — use list per cycle.
    collect: dict[int, list[tuple[int, int]]] = {}
    for i in range(M):
        for j in range(cols):
            cyc = (i + j + rows - 1) * hop
            collect.setdefault(cyc, []).append((i, j))

    tb_sv = _generate_testbench(
        rows, cols, pipeline_stages, data_width,
        W, inject, collect, M, total,
    )

    sv_path = _RTL_DIR / "systolic_array.sv"

    with tempfile.TemporaryDirectory() as tmp:
        tb_path  = os.path.join(tmp, "tb.sv")
        vvp_path = os.path.join(tmp, "sim.vvp")

        with open(tb_path, "w") as f:
            f.write(tb_sv)

        compile_result = subprocess.run(
            [_IVERILOG, "-g2012", "-o", vvp_path, str(sv_path), tb_path],
            capture_output=True, text=True,
        )
        if compile_result.returncode != 0:
            raise RuntimeError(
                f"iverilog compile failed:\n{compile_result.stderr}"
            )

        sim_result = subprocess.run(
            [_VVP, vvp_path],
            capture_output=True, text=True, timeout=30,
        )
        if sim_result.returncode != 0:
            raise RuntimeError(
                f"vvp simulation failed:\n{sim_result.stderr}"
            )

    return _parse_output(sim_result.stdout, M, cols)


def _generate_testbench(
    rows, cols, pipeline_stages, data_width,
    W, inject, collect, M, total_cycles,
) -> str:
    """Generate a self-contained SV testbench with embedded test vectors."""

    # Weight assignments
    w_lines = []
    for k in range(rows):
        for j in range(cols):
            w_lines.append(f"        weights[{k}][{j}] = {int(W[k][j])};")
    w_init = "\n".join(w_lines)

    # Injection schedule assignments
    inj_lines = []
    for t in range(total_cycles):
        for k in range(rows):
            if inject[t][k] != 0:
                inj_lines.append(
                    f"        inj[{t}][{k}] = {inject[t][k]};"
                )
    inj_init = "\n".join(inj_lines) if inj_lines else "        // no non-zero injections"

    # Output capture: at the right cycle, $display C[i][j] for each output
    capture_lines = []
    for cyc, pairs in sorted(collect.items()):
        for (i, j) in pairs:
            capture_lines.append(
                f"                if (cycle == {cyc}) "
                f'$display("OUT %0d %0d %0d", {i}, {j}, $signed(sum_out[{j}]));'
            )
    captures = "\n".join(capture_lines)

    # 40 bits: safe for ROWS × INT16_MAX² without overflow
    acc_width = 40

    return f"""`timescale 1ns/1ps
module tb;
    localparam ROWS        = {rows};
    localparam COLS        = {cols};
    localparam DATA_WIDTH  = {data_width};
    localparam PIPE_STAGES = {pipeline_stages};
    localparam ACC_WIDTH   = {acc_width};
    localparam TOTAL       = {total_cycles};

    logic clk = 0;
    logic rst_n = 0;

    logic signed [DATA_WIDTH-1:0]  weights [ROWS][COLS];
    logic signed [DATA_WIDTH-1:0]  act_in  [ROWS];
    logic signed [ACC_WIDTH-1:0]   sum_out [COLS];

    // Injection schedule (pre-skewed)
    logic signed [DATA_WIDTH-1:0]  inj [TOTAL][ROWS];

    systolic_array #(
        .ROWS(ROWS), .COLS(COLS),
        .DATA_WIDTH(DATA_WIDTH), .PIPE_STAGES(PIPE_STAGES)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .weights(weights),
        .act_in(act_in),
        .sum_out(sum_out)
    );

    always #5 clk = ~clk;   // 100 MHz

    integer cycle;
    initial begin
        // Initialise arrays to zero
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++) begin
                weights[i][j] = 0;
                for (int t = 0; t < TOTAL; t++)
                    inj[t][i] = 0;
            end

        // Load weights
{w_init}

        // Load injection schedule
{inj_init}

        rst_n = 0;
        @(posedge clk); #1;
        rst_n = 1;

        for (cycle = 0; cycle < TOTAL; cycle++) begin
            for (int k = 0; k < ROWS; k++)
                act_in[k] = inj[cycle][k];
            @(posedge clk); #1;
{captures}
        end
        $finish;
    end
endmodule
"""


def _parse_output(stdout: str, M: int, cols: int) -> np.ndarray:
    """Parse $display 'OUT i j value' lines into a (M, cols) array."""
    C = np.zeros((M, cols), dtype=np.int64)
    for line in stdout.splitlines():
        m = re.match(r"OUT\s+(\d+)\s+(\d+)\s+(-?\d+)", line)
        if m:
            i, j, val = int(m.group(1)), int(m.group(2)), int(m.group(3))
            C[i][j] = val
    return C
