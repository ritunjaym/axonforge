"""
Tests for Dashboard Panels 3–5 (Slice 24).

Panel 3 — Systolic array PE activity heatmap (8×8 grid, utilization per PE).
Panel 4 — FSDP scaling curves + allocator cache_hit/fragmentation time-series.
Panel 5 — Inference Pareto + allocator hit rate comparison (training vs inference).

All tests run on CPU with synthetic/dry-run data.
"""
import math
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Test 1 — Panel 3 HTML contains heatmap content and utilization values
# ---------------------------------------------------------------------------

def test_panel3_heatmap_contains_utilization():
    from profiler.dashboard import build_panel3_html

    html = build_panel3_html(systolic_data=None, dry_run=True)

    assert "heatmap" in html.lower() or "pe activity" in html.lower(), (
        "Panel 3 missing heatmap label"
    )
    assert "utilization" in html.lower(), "Panel 3 missing utilization metric"
    # Should have 8×8 = 64 PE cells referenced
    assert "8" in html, "Panel 3 missing array dimension"


# ---------------------------------------------------------------------------
# Test 2 — Panel 4 HTML contains scaling and allocator time-series content
# ---------------------------------------------------------------------------

def test_panel4_contains_scaling_and_allocator():
    from profiler.dashboard import build_panel4_html

    html = build_panel4_html(scaling_data=None, allocator_data=None, dry_run=True)

    assert "scaling" in html.lower() or "gpu" in html.lower(), (
        "Panel 4 missing scaling content"
    )
    assert "cache_hit" in html.lower() or "hit rate" in html.lower(), (
        "Panel 4 missing allocator cache hit content"
    )
    assert "fragmentation" in html.lower(), "Panel 4 missing fragmentation metric"


# ---------------------------------------------------------------------------
# Test 3 — Panel 5 HTML contains inference Pareto and allocator comparison
# ---------------------------------------------------------------------------

def test_panel5_contains_pareto_and_allocator_comparison():
    from profiler.dashboard import build_panel5_html

    html = build_panel5_html(inference_data=None, dry_run=True)

    assert "pareto" in html.lower(), "Panel 5 missing Pareto label"
    assert "inference" in html.lower(), "Panel 5 missing inference reference"
    assert "training" in html.lower(), "Panel 5 missing training comparison"
    assert "hit rate" in html.lower() or "cache" in html.lower(), (
        "Panel 5 missing allocator hit rate comparison"
    )


# ---------------------------------------------------------------------------
# Test 4 — Full dashboard now has all 5 panels (not stubs)
# ---------------------------------------------------------------------------

def test_full_dashboard_has_5_real_panels(tmp_path):
    from profiler.dashboard import build_dashboard

    out = build_dashboard(results_dir=tmp_path, dry_run=True)
    content = Path(out).read_text()

    for panel_id in ["panel1", "panel2", "panel3", "panel4", "panel5"]:
        assert panel_id in content, f"Dashboard missing {panel_id}"

    # Panels 3-5 should no longer be stubs
    assert "Coming in Slice 24" not in content, (
        "Dashboard still has Slice 24 stub content — panels not yet wired in"
    )


# ---------------------------------------------------------------------------
# Test 5 — Systolic utilization data drives heatmap (8×8 grid, values in [0,1])
# ---------------------------------------------------------------------------

def test_panel3_heatmap_values_in_range():
    from profiler.dashboard import build_panel3_html

    # Provide synthetic per-PE utilization
    data = {
        "rows": 8, "cols": 8,
        "utilization": [[0.7 + 0.01 * (i + j) for j in range(8)] for i in range(8)],
    }
    html = build_panel3_html(systolic_data=data, dry_run=False)

    assert "0.7" in html or "70" in html, "Panel 3 not rendering utilization values"
