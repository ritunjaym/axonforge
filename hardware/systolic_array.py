"""
Pipelined systolic array simulator — Python golden reference.

Architecture: weight-stationary, 8×8 array (parameterised), 3 pipeline stages.

Dataflow:
  - Weights W[k][j] are pre-loaded into PE[k][j] and never move.
  - Activations flow east (left → right) through pipeline registers.
  - Partial sums flow south (top → bottom) through pipeline registers.
  - Each hop (PE-to-PE) introduces (1 + pipeline_stages) clock cycles of latency.

Correct injection (skewed wavefront):
  Row i of A is fed with activation A[i][k] injected into the left edge of
  row k at cycle (i + k) * hop_latency. This ensures the activation and
  the corresponding partial sum wave arrive at every PE simultaneously.

Cycle accounting (reported SEPARATELY):
  fill_cycles       = (rows + cols - 2) * hop_latency   ← until last PE gets data
  steady_state_cycles = M * hop_latency                  ← M = number of A rows
  drain_cycles       = fill_cycles                        ← symmetric
  total_cycles       = fill + steady_state + drain

Utilization:
  active_mac_cycles = sum over all (PE, cycle) pairs where a valid MAC fired
  utilization       = active_mac_cycles / (steady_state_cycles * rows * cols)
  NOTE: denominator is steady_state_cycles × total_PEs, NOT total_cycles.

Power:
  P_dyn = alpha * C * V^2 * f
  where alpha = switching activity (fraction of MACs that toggle output)
  Units: (fF)(V^2)(GHz) → mW.

Co-simulation contract: this Python output is the golden reference.
RTL output must match exactly in integer mode (0 deltas).
"""
from collections import deque
from typing import Optional

import numpy as np

from profiler.hardware_params import HARDWARE


class SystolicArray:
    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        pipeline_stages: int = 3,
        data_width: int = 16,
    ):
        self.rows            = rows
        self.cols            = cols
        self.pipeline_stages = pipeline_stages
        self.data_width      = data_width
        self._hop            = 1 + pipeline_stages   # cycles per PE-to-PE hop

        self._W = np.zeros((rows, cols), dtype=np.int64)

        # Stats populated by run()
        self._stats: dict = {}
        # MAC activity log: list of (cycle, active_mac_count) — for power model
        self._mac_log: list[tuple[int, int]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_weights(self, W: np.ndarray) -> None:
        """Load (rows, cols) weight matrix into PEs."""
        assert W.shape == (self.rows, self.cols), (
            f"Weight shape {W.shape} != ({self.rows}, {self.cols})"
        )
        self._W = W.astype(np.int64)

    def fill_cycles(self) -> int:
        """Cycles until the last PE (rows-1, cols-1) first receives valid data."""
        return (self.rows + self.cols - 2) * self._hop

    def run(self, A: np.ndarray) -> np.ndarray:
        """
        Compute A @ W using the pipelined systolic array.

        A: shape (M, rows) — M input vectors, each of length rows (= K)
        Returns: shape (M, cols) — M output vectors, each of length cols (= N)
        """
        M, K = A.shape
        assert K == self.rows, f"A has {K} columns but array has {self.rows} rows"
        A = A.astype(np.int64)

        h = self._hop
        fill   = self.fill_cycles()
        drain  = fill
        # Steady state: M input rows × hop_latency cycles each
        steady = M * h

        total_cycles = fill + steady + drain + h   # safety margin

        # --- Pipeline registers (east/south flow) ---
        # act_east[k][j]: FIFO of depth h — carries activation FROM PE[k][j]
        #                 EASTWARD to PE[k][j+1]. Read at PE[k][j+1] next cycle.
        # sum_south[k][j]: FIFO of depth h — carries partial sum FROM PE[k][j]
        #                  SOUTHWARD to PE[k+1][j]. Read at PE[k+1][j] next cycle.
        act_east  = [[deque([0] * h, maxlen=h) for _ in range(self.cols)]
                     for _ in range(self.rows)]
        sum_south = [[deque([0] * h, maxlen=h) for _ in range(self.cols)]
                     for _ in range(self.rows)]

        # --- External activation injection schedule (skewed wavefront) ---
        # A[i][k] injected into left edge of row k at cycle (i + k) * h.
        inject = np.zeros((total_cycles, self.rows), dtype=np.int64)
        for i in range(M):
            for k in range(self.rows):
                t = (i + k) * h
                if t < total_cycles:
                    inject[t][k] = A[i][k]

        # --- Output collection schedule ---
        # C[i][j] exits bottom of column j at cycle (i + j + rows - 1) * h.
        output_time: dict[tuple[int, int], int] = {
            (i, j): (i + j + self.rows - 1) * h
            for i in range(M) for j in range(self.cols)
        }

        C = np.zeros((M, self.cols), dtype=np.int64)

        # --- Cycle-accurate simulation ---
        active_mac_cycles = 0
        self._mac_log = []

        for t in range(total_cycles):
            # Step 1: Read inputs for every PE (all from OLD register state)
            act_in = np.zeros((self.rows, self.cols), dtype=np.int64)
            sum_in = np.zeros((self.rows, self.cols), dtype=np.int64)
            for k in range(self.rows):
                for j in range(self.cols):
                    # Activation: from left-neighbor's east register, or injection
                    act_in[k][j] = (inject[t][k] if j == 0
                                    else act_east[k][j - 1][-1])
                    # Partial sum: from upper-neighbor's south register, or 0
                    sum_in[k][j] = (0 if k == 0
                                    else sum_south[k - 1][j][-1])

            # Step 2: Every PE computes MAC simultaneously
            mac_out = act_in * self._W      # sums will accumulate southward
            sum_out = sum_in + mac_out

            # Step 3: Stats — count cycles with at least one active PE
            if np.any(act_in != 0):
                active_mac_cycles += 1
                self._mac_log.append((t, int(np.count_nonzero(act_in))))

            # Step 4: Collect outputs from bottom row
            for (i, j), cyc in output_time.items():
                if cyc == t:
                    C[i][j] = sum_out[self.rows - 1][j]

            # Step 5: Advance all registers simultaneously (write after all reads)
            for k in range(self.rows):
                for j in range(self.cols):
                    # East register: carries act_in[k][j] to PE[k][j+1] next cycle
                    act_east[k][j].appendleft(int(act_in[k][j]))
                    # South register: carries sum_out[k][j] to PE[k+1][j] next cycle
                    sum_south[k][j].appendleft(int(sum_out[k][j]))

        # --- Record stats ---
        # active_mac_cycles: count of CYCLES where any PE performed a MAC.
        # utilization = active_mac_cycles / steady_state_cycles ∈ [0, 1].
        # (Denominator is steady_state_cycles, NOT total_cycles.)
        n_pes = self.rows * self.cols
        utilization = active_mac_cycles / steady if steady > 0 else 0.0

        self._stats = {
            "fill_cycles":          fill,
            "steady_state_cycles":  steady,
            "drain_cycles":         drain,
            "total_cycles":         total_cycles,
            "active_mac_cycles":    active_mac_cycles,
            "utilization":          utilization,
            "n_pes":                n_pes,
        }

        return C

    def get_stats(self) -> dict:
        """Returns cycle and utilization stats from the last run()."""
        return dict(self._stats)

    def get_power(
        self,
        alpha: float,
        C_fF: float,
        V: float,
        f_GHz: float,
    ) -> dict:
        """
        Computes power from the MAC activity log.

        P_dyn = alpha * C * V^2 * f
          alpha: switching activity (user-supplied or from MAC log)
          C_fF:  capacitance per MAC in femtofarads
          V:     supply voltage in volts
          f_GHz: clock frequency in GHz

        Returns dict with P_dyn_mW, P_static_mW, P_total_mW.
        """
        hw = HARDWARE["simulated_8x8"]

        C_F  = C_fF * 1e-15    # fF → F
        f_Hz = f_GHz * 1e9     # GHz → Hz

        P_dyn_W    = alpha * C_F * (V ** 2) * f_Hz
        P_dyn_mW   = P_dyn_W * 1e3

        P_static_mW = hw["static_power_mW"]

        return {
            "P_dyn_mW":   P_dyn_mW,
            "P_static_mW": P_static_mW,
            "P_total_mW":  P_dyn_mW + P_static_mW,
            "alpha":       alpha,
        }
