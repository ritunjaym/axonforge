"""
Nsight Compute (ncu) output parser.

parse_ncu_csv(csv_text)  — parses ncu --csv output into a metrics dict
run_ncu(kernel_fn, ...)  — runs ncu and returns parsed metrics (requires sudo
                           or perf_event_paranoid=1)

Key metrics for roofline analysis:
  dram__bytes.sum         — actual DRAM bytes transferred (for accurate roofline placement)
  sm__throughput.avg...   — SM utilisation %
  sm__warps_active.avg... — achieved occupancy proxy

Pitfall: ncu needs elevated permissions.
  Fix: sudo ncu  OR  sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'
"""
import csv
import io
import subprocess
import sys


# Metric names as reported by ncu --csv
_DRAM_BYTES_METRIC    = "dram__bytes.sum"
_SM_THROUGHPUT_METRIC = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
_WARPS_ACTIVE_METRIC  = "sm__warps_active.avg.pct_of_peak_sustained_active"


def parse_ncu_csv(csv_text: str) -> dict:
    """
    Parses ncu --csv output.

    Expected CSV columns (subset):
      "Metric Name", "Metric Value"

    Returns:
      {
        "dram_bytes":        int,    # total DRAM traffic
        "sm_throughput_pct": float,  # SM utilisation %
        "warps_active_pct":  float,  # achieved occupancy proxy %
      }
    """
    result = {
        "dram_bytes":        0,
        "sm_throughput_pct": 0.0,
        "warps_active_pct":  0.0,
    }

    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        metric_name  = row.get("Metric Name", "").strip().lower()
        metric_value = row.get("Metric Value", "0").strip().replace(",", "")

        if _DRAM_BYTES_METRIC in metric_name:
            try:
                result["dram_bytes"] = int(float(metric_value))
            except ValueError:
                pass

        elif _SM_THROUGHPUT_METRIC in metric_name:
            try:
                result["sm_throughput_pct"] = float(metric_value)
            except ValueError:
                pass

        elif _WARPS_ACTIVE_METRIC in metric_name:
            try:
                result["warps_active_pct"] = float(metric_value)
            except ValueError:
                pass

    return result


def run_ncu(
    kernel_fn,
    metrics: list[str] | None = None,
    tmp_csv: str = "/tmp/ncu_output.csv",
) -> dict:
    """
    Runs ncu on kernel_fn and returns parsed metrics.

    Requires: ncu on PATH + elevated permissions (sudo or perf_event_paranoid=1).
    Pitfall:  'ncu: error: ERR_NVGPUCTRPERM' means insufficient permissions.

    Returns parsed dict from parse_ncu_csv.
    """
    if metrics is None:
        metrics = [
            _DRAM_BYTES_METRIC,
            _SM_THROUGHPUT_METRIC,
            _WARPS_ACTIVE_METRIC,
        ]

    # Write a small driver script so ncu can profile the kernel
    driver = f"""
import sys
sys.path.insert(0, '.')
fn = {kernel_fn.__module__}.{kernel_fn.__qualname__}
fn()
"""
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(driver)
        driver_path = f.name

    metrics_arg = ",".join(metrics)
    cmd = [
        "ncu",
        "--csv",
        "--metrics", metrics_arg,
        "--output", tmp_csv,
        sys.executable, driver_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(
                f"ncu failed (returncode={result.returncode}).\n"
                f"Stderr: {result.stderr[:500]}\n"
                "Hint: run with sudo or set perf_event_paranoid=1"
            )
        with open(tmp_csv) as f:
            return parse_ncu_csv(f.read())
    finally:
        os.unlink(driver_path)
