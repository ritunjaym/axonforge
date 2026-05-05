"""
Ping-pong memory controller for the pipelined systolic array.

Models the memory hierarchy between DRAM and the systolic array's local SRAM.

Problem without ping-pong:
  Each tile must be fully loaded from DRAM into SRAM before the array can
  compute. During the load, the array sits idle — wasting memory_latency
  cycles per tile.

Solution (double buffering / ping-pong):
  Two SRAM buffers (A and B) alternate roles each tile:
    - Buffer A holds the current tile being computed.
    - Buffer B is being loaded with the next tile from DRAM.
  When A's computation finishes, B is ready — no stall if compute ≥ load time.

Stall model:
  Baseline:  stall_cycles = n_tiles × memory_latency
  Ping-pong: stall_cycles = n_tiles × max(0, memory_latency − tile_compute_cycles)

  Where tile_compute_cycles = fill_cycles + tile_rows + drain_cycles
  (cycles the array runs for one tile).

  Reduction = (baseline − pingpong) / baseline × 100 %
  Target:    ≥ 30 % stall reduction.

Correctness contract: both modes produce output == A @ W (verified against
the systolic array's own correctness guarantee from Slice 12).
"""
import numpy as np

from hardware.systolic_array import SystolicArray


class MemoryController:
    def __init__(
        self,
        array: SystolicArray,
        memory_latency_cycles: int = 10,
    ):
        """
        array:                  Pre-configured SystolicArray instance.
        memory_latency_cycles:  Cycles to DMA one activation tile from DRAM → SRAM.
        """
        self._array   = array
        self._mem_lat = memory_latency_cycles
        self._tile_rows = array.rows    # one tile = array.rows activation rows

        # Results from the most recent run_baseline and run_pingpong calls
        self._baseline_stats: dict = {}
        self._pingpong_stats: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_baseline(
        self,
        A: np.ndarray,
        W: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """
        Executes A @ W tile-by-tile WITHOUT ping-pong.

        Scheduling per tile:
          1. Stall memory_latency cycles (DMA load into single buffer).
          2. Run the systolic array for tile_compute_cycles cycles.

        Returns: (result, stats)
          result: (M, cols) — same as A @ W
          stats:  stall_cycles, compute_cycles, total_cycles, n_tiles
        """
        self._array.load_weights(W)
        tiles, n_tiles = self._split(A)
        tile_compute = self._tile_compute_cycles(tiles[0].shape[0])

        stall_cycles   = n_tiles * self._mem_lat
        compute_cycles = n_tiles * tile_compute
        total_cycles   = stall_cycles + compute_cycles

        result = np.vstack([self._array.run(tile) for tile in tiles])

        self._baseline_stats = {
            "stall_cycles":   stall_cycles,
            "compute_cycles": compute_cycles,
            "total_cycles":   total_cycles,
            "n_tiles":        n_tiles,
        }
        return result, self._baseline_stats

    def run_pingpong(
        self,
        A: np.ndarray,
        W: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """
        Executes A @ W tile-by-tile WITH ping-pong double buffering.

        Scheduling:
          - Tile 0: load (memory_latency cycles) → compute.
          - Tile k≥1: while tile k-1 computes, load tile k.
            If memory_latency ≤ tile_compute: zero stalls (perfect overlap).
            If memory_latency >  tile_compute: stall = mem_lat - compute.

        Returns: (result, stats)
        """
        self._array.load_weights(W)
        tiles, n_tiles = self._split(A)
        tile_compute = self._tile_compute_cycles(tiles[0].shape[0])

        # First tile: no overlap, full load stall
        first_stall = self._mem_lat
        # Subsequent tiles: stall only if load outlasts compute
        per_tile_stall = max(0, self._mem_lat - tile_compute)
        stall_cycles   = first_stall + (n_tiles - 1) * per_tile_stall
        compute_cycles = n_tiles * tile_compute
        # Total: first load + n_tiles * max(load, compute)
        total_cycles = self._mem_lat + n_tiles * max(self._mem_lat, tile_compute)

        result = np.vstack([self._array.run(tile) for tile in tiles])

        self._pingpong_stats = {
            "stall_cycles":   stall_cycles,
            "compute_cycles": compute_cycles,
            "total_cycles":   total_cycles,
            "n_tiles":        n_tiles,
        }
        return result, self._pingpong_stats

    def stall_reduction_pct(self) -> float:
        """
        Returns stall reduction percentage of ping-pong vs baseline.
        Call after both run_baseline and run_pingpong.
        """
        base = self._baseline_stats.get("stall_cycles", 0)
        ping = self._pingpong_stats.get("stall_cycles", 0)
        if base == 0:
            return 0.0
        return (base - ping) / base * 100.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split(self, A: np.ndarray) -> tuple[list[np.ndarray], int]:
        """Split A into tiles of self._tile_rows rows each."""
        M = A.shape[0]
        assert M % self._tile_rows == 0, (
            f"A has {M} rows which is not divisible by tile_rows={self._tile_rows}"
        )
        n_tiles = M // self._tile_rows
        tiles = [A[i * self._tile_rows:(i + 1) * self._tile_rows] for i in range(n_tiles)]
        return tiles, n_tiles

    def _tile_compute_cycles(self, tile_rows: int) -> int:
        """
        Cycles the systolic array runs for one tile of tile_rows activation rows.
        = fill_cycles + tile_rows * hop_latency + drain_cycles
        """
        fill  = self._array.fill_cycles()
        hop   = 1 + self._array.pipeline_stages
        return fill + tile_rows * hop + fill   # fill + steady + drain
