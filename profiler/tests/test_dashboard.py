"""
Tests for the roofline dashboard (Slice 23).

All tests run on CPU — no GPU needed. The dashboard is pure
data-transformation + HTML generation, fully testable locally.

Panel 1: log/log roofline scatter — compute ceiling + memory ceiling,
  points for PyTorch ref / Triton / C++/CUDA kernels,
  before/after compiler pass overlay.
Panel 2: three-way SwiGLU benchmark table (latency_ms, tflops, bandwidth_gb_s).

dry_run=True: uses synthetic stub data instead of results/ JSON files.
"""
import math
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet: build_dashboard writes an HTML file
# ---------------------------------------------------------------------------

def test_build_dashboard_writes_html(tmp_path):
    from profiler.dashboard import build_dashboard

    out = build_dashboard(results_dir=tmp_path, dry_run=True)

    assert Path(out).exists(), f"Dashboard HTML not written to {out}"
    content = Path(out).read_text()
    assert "<html" in content.lower(), "Output is not HTML"
    assert len(content) > 200, "HTML too short to contain panel content"


# ---------------------------------------------------------------------------
# Test 2 — roofline_ceiling returns min(compute, bandwidth × AI)
# ---------------------------------------------------------------------------

def test_roofline_ceiling_memory_bound():
    from profiler.roofline import roofline_ceiling

    # At AI=0.1 FLOP/byte on RTX 3090 (936 GB/s, 35.6 TF fp32):
    #   memory ceiling = 936 * 0.1 = 93.6 GFLOPS
    #   compute ceiling = 35,600 GFLOPS
    #   roofline = min = 93.6 GFLOPS (memory-bound)
    result = roofline_ceiling(arithmetic_intensity=0.1, hw_name="rtx_3090", dtype="fp32")
    expected = 936.0 * 0.1  # GFLOPS
    assert abs(result - expected) < 0.01, (
        f"Expected {expected:.2f} GFLOPS, got {result:.2f}"
    )


def test_roofline_ceiling_compute_bound():
    from profiler.roofline import roofline_ceiling

    # At AI=1000 FLOP/byte on RTX 3090 (936 GB/s, 35.6 TF fp32):
    #   memory ceiling = 936 * 1000 = 936,000 GFLOPS
    #   compute ceiling = 35,600 GFLOPS
    #   roofline = min = 35,600 GFLOPS (compute-bound)
    result = roofline_ceiling(arithmetic_intensity=1000.0, hw_name="rtx_3090", dtype="fp32")
    expected = 35_600.0  # 35.6 TF = 35,600 GFLOPS
    assert abs(result - expected) < 1.0, (
        f"Expected {expected:.1f} GFLOPS, got {result:.1f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Panel 1 HTML contains roofline labels and kernel markers
# ---------------------------------------------------------------------------

def test_panel1_html_contains_labels():
    from profiler.roofline import build_panel1_html

    html = build_panel1_html(hw_name="rtx_3090", benchmark_data=None, dry_run=True)

    assert "roofline" in html.lower(), "Panel 1 missing 'roofline' label"
    assert "arithmetic intensity" in html.lower(), "Panel 1 missing x-axis label"
    assert "tflops" in html.lower() or "gflops" in html.lower(), (
        "Panel 1 missing performance axis label"
    )


def test_panel1_html_contains_kernel_markers():
    from profiler.roofline import build_panel1_html

    html = build_panel1_html(hw_name="rtx_3090", benchmark_data=None, dry_run=True)

    # Must identify all three kernels on the roofline plot
    assert "triton" in html.lower(), "Panel 1 missing Triton kernel marker"
    assert "cuda" in html.lower(), "Panel 1 missing CUDA kernel marker"
    assert "pytorch" in html.lower() or "ref" in html.lower(), (
        "Panel 1 missing PyTorch reference marker"
    )


# ---------------------------------------------------------------------------
# Test 4 — Panel 2 HTML contains all three kernel names in benchmark table
# ---------------------------------------------------------------------------

def test_panel2_html_contains_kernel_names():
    from profiler.dashboard import build_panel2_html

    html = build_panel2_html(benchmark_data=None, dry_run=True)

    assert "triton" in html.lower(), "Panel 2 missing Triton row"
    assert "cuda" in html.lower(), "Panel 2 missing CUDA row"
    assert "pytorch" in html.lower() or "ref" in html.lower(), (
        "Panel 2 missing PyTorch reference row"
    )
    # Must show performance metrics
    assert "tflops" in html.lower() or "latency" in html.lower(), (
        "Panel 2 missing performance metrics columns"
    )


# ---------------------------------------------------------------------------
# Test 5 — Full dashboard HTML contains Panel 1 and Panel 2 content
# ---------------------------------------------------------------------------

def test_full_dashboard_contains_both_panels(tmp_path):
    from profiler.dashboard import build_dashboard

    out = build_dashboard(results_dir=tmp_path, dry_run=True)
    content = Path(out).read_text()

    # Panel 1 marker
    assert "roofline" in content.lower(), "Dashboard missing Panel 1 roofline content"
    # Panel 2 marker
    assert "benchmark" in content.lower() or "swiglu" in content.lower(), (
        "Dashboard missing Panel 2 benchmark table content"
    )
    # Both kernel sets mentioned
    assert "triton" in content.lower(), "Dashboard missing Triton"
    assert "cuda" in content.lower(), "Dashboard missing CUDA"
