"""
Roofline dashboard — assembles all panels into results/roofline_dashboard.html.

Panels:
  1 — Roofline scatter: PyTorch ref / Triton / C++/CUDA + compiler overlay
  2 — Three-way SwiGLU benchmark table (latency, tflops, bandwidth)
  3 — Systolic array PE activity heatmap
  4 — FSDP scaling curves + allocator cache_hit/fragmentation time-series
  5 — Inference Pareto + allocator hit rate comparison (training vs inference)

dry_run=True: uses synthetic stub data for all panels.
             Writes to results_dir/roofline_dashboard.html.
"""
import json
from pathlib import Path

from profiler.roofline import build_panel1_html, _load_benchmark_data, _SYNTHETIC_BENCHMARK


# ---------------------------------------------------------------------------
# Panel 2 — Three-way SwiGLU benchmark table
# ---------------------------------------------------------------------------

def build_panel2_html(
    benchmark_data=None,
    dry_run: bool = False,
) -> str:
    """
    Generate Panel 2: three-way SwiGLU benchmark table as an HTML fragment.

    Columns: kernel, N, latency_ms, tflops, bandwidth_gb_s, % of peak.
    """
    data = _load_benchmark_data(benchmark_data, dry_run)

    rows = []
    for d in data:
        N    = d["config"].get("N", "?")
        name = d["kernel"]
        rows.append(
            f"<tr>"
            f"<td><b>{name}</b></td>"
            f"<td>{N}</td>"
            f"<td>{d['latency_ms']:.3f}</td>"
            f"<td>{d['tflops']:.2f}</td>"
            f"<td>{d['bandwidth_gb_s']:.1f}</td>"
            f"<td>{d.get('pct_of_peak', 0)*100:.1f}%</td>"
            f"</tr>"
        )

    return f"""
<div id="panel2" style="margin-bottom:40px">
  <h3>Panel 2 — Three-way SwiGLU Benchmark (PyTorch ref / Triton / CUDA)</h3>
  <table border="1" style="border-collapse:collapse">
    <tr>
      <th>Kernel</th><th>N</th>
      <th>Latency (ms)</th><th>TFLOPS</th>
      <th>Bandwidth (GB/s)</th><th>% of Peak</th>
    </tr>
    {''.join(rows)}
  </table>
  <p><em>Triton vs C++/CUDA gap target: &lt;15%. Both beat PyTorch ref.</em></p>
</div>
"""


# ---------------------------------------------------------------------------
# Synthetic data for panels 3–5
# ---------------------------------------------------------------------------

_SYNTHETIC_SYSTOLIC = {
    "rows": 8, "cols": 8,
    "utilization": [
        [min(1.0, 0.45 + 0.07 * (i + j) / 14.0) for j in range(8)]
        for i in range(8)
    ],
}

_SYNTHETIC_SCALING = {
    "gpus":             [1, 2, 4],
    "strong_efficiency": [1.0, 0.94, 0.88],
    "weak_efficiency":   [1.0, 0.97, 0.92],
    "allocator_steps":         list(range(0, 500, 100)),
    "cache_hit_pct":           [0.0, 62.3, 68.1, 71.4, 72.8],
    "fragmentation_pct":       [0.0,  8.2,  9.1,  9.5,  9.7],
}

_SYNTHETIC_INFERENCE_ALLOC = {
    "training_hit_rate_pct":   72.8,
    "inference_hit_rate_pct":  93.5,
    "pareto_points": [
        {"batch_size": 1,  "latency_p50_ms": 5.2,  "throughput_tokens_s": 24_615},
        {"batch_size": 2,  "latency_p50_ms": 7.1,  "throughput_tokens_s": 36_056},
        {"batch_size": 4,  "latency_p50_ms": 9.8,  "throughput_tokens_s": 52_245},
        {"batch_size": 8,  "latency_p50_ms": 14.3, "throughput_tokens_s": 71_678},
        {"batch_size": 16, "latency_p50_ms": 27.9, "throughput_tokens_s": 73_477},
    ],
    "knee_batch_size": 8,
}


# ---------------------------------------------------------------------------
# Panel 3 — Systolic array PE activity heatmap
# ---------------------------------------------------------------------------

def build_panel3_html(
    systolic_data=None,
    dry_run: bool = False,
) -> str:
    """
    Generate Panel 3: 8×8 PE activity heatmap as an HTML fragment.

    systolic_data: dict with keys "rows", "cols", "utilization" (2D list).
    """
    data = systolic_data if systolic_data is not None else _SYNTHETIC_SYSTOLIC
    rows = data["rows"]
    cols = data["cols"]
    util = data["utilization"]

    cells = ""
    for i in range(rows):
        for j in range(cols):
            v = util[i][j]
            r = int(255 * (1 - v))
            g = int(200 * v)
            b = 60
            pct = round(v * 100, 1)
            cells += (
                f'<td style="background:rgb({r},{g},{b});color:white;'
                f'text-align:center;padding:6px;min-width:40px" '
                f'title="PE[{i},{j}]={pct}%">{pct}%</td>'
            )
        cells += "</tr><tr>"

    avg_util = sum(util[i][j] for i in range(rows) for j in range(cols)) / (rows * cols)

    return f"""
<div id="panel3" style="margin-bottom:40px">
  <h3>Panel 3 — Systolic Array PE Activity Heatmap ({rows}×{cols})</h3>
  <p>Color = utilization (green = high, red = low). Average: {avg_util*100:.1f}%</p>
  <table style="border-collapse:collapse">
    <tr>{cells}</tr>
  </table>
  <p><em>Utilization = active_mac_cycles / steady_state_cycles per PE row.</em></p>
</div>
"""


# ---------------------------------------------------------------------------
# Panel 4 — FSDP scaling curves + allocator time-series
# ---------------------------------------------------------------------------

def build_panel4_html(
    scaling_data=None,
    allocator_data=None,
    dry_run: bool = False,
) -> str:
    """
    Generate Panel 4: FSDP scaling efficiency curves + allocator time-series.

    scaling_data: dict with keys gpus, strong_efficiency, weak_efficiency,
                  allocator_steps, cache_hit_pct, fragmentation_pct.
    """
    data = scaling_data if scaling_data is not None else _SYNTHETIC_SCALING
    if allocator_data is not None:
        data = {**data, **allocator_data}

    gpus_json   = json.dumps(data["gpus"])
    strong_json = json.dumps(data["strong_efficiency"])
    weak_json   = json.dumps(data["weak_efficiency"])
    steps_json  = json.dumps(data["allocator_steps"])
    hit_json    = json.dumps(data["cache_hit_pct"])
    frag_json   = json.dumps(data["fragmentation_pct"])

    return f"""
<div id="panel4" style="margin-bottom:40px">
  <h3>Panel 4 — FSDP Scaling + Allocator Cache Hit / Fragmentation</h3>
  <canvas id="scalingChart" style="max-width:700px;display:block;margin-bottom:20px"></canvas>
  <canvas id="allocChart"   style="max-width:700px;display:block"></canvas>
  <script>
    (function() {{
      const gpus   = {gpus_json};
      const strong = {strong_json};
      const weak   = {weak_json};
      const steps  = {steps_json};
      const hit    = {hit_json};
      const frag   = {frag_json};

      new Chart(document.getElementById('scalingChart').getContext('2d'), {{
        type: 'line',
        data: {{
          labels: gpus.map(g => g + ' GPU'),
          datasets: [
            {{ label: 'Strong scaling efficiency', data: strong, borderColor: '#3498db', fill: false }},
            {{ label: 'Weak scaling efficiency',   data: weak,   borderColor: '#2ecc71', fill: false }},
            {{ label: 'Linear (ideal)',            data: gpus.map(() => 1.0), borderColor:'#999', borderDash:[5,5], fill:false }},
          ]
        }},
        options: {{ scales: {{ y: {{ min: 0, max: 1.1, title: {{ display:true, text:'Scaling efficiency' }} }},
                               x: {{ title: {{ display:true, text:'GPU count' }} }} }} }}
      }});

      new Chart(document.getElementById('allocChart').getContext('2d'), {{
        type: 'line',
        data: {{
          labels: steps.map(s => 'step ' + s),
          datasets: [
            {{ label: 'Cache hit rate (%)',       data: hit,  borderColor: '#27ae60', fill: false }},
            {{ label: 'Fragmentation (%)',         data: frag, borderColor: '#e74c3c', fill: false }},
          ]
        }},
        options: {{ scales: {{ y: {{ min: 0, max: 100, title: {{ display:true, text:'%' }} }} }} }}
      }});
    }})();
  </script>
  <p>Target: scaling efficiency ≥ 0.8× at 4 GPU. Allocator cache hit &gt; 70% training.</p>
</div>
"""


# ---------------------------------------------------------------------------
# Panel 5 — Inference Pareto + allocator comparison
# ---------------------------------------------------------------------------

def build_panel5_html(
    inference_data=None,
    dry_run: bool = False,
) -> str:
    """
    Generate Panel 5: inference Pareto curve + allocator hit rate comparison.

    inference_data: dict with pareto_points, training_hit_rate_pct,
                    inference_hit_rate_pct, knee_batch_size.
    """
    data = inference_data if inference_data is not None else _SYNTHETIC_INFERENCE_ALLOC

    pts   = data["pareto_points"]
    knee  = data.get("knee_batch_size")
    train_hit = data["training_hit_rate_pct"]
    infer_hit = data["inference_hit_rate_pct"]

    pts_json = json.dumps([{"x": p["latency_p50_ms"], "y": p["throughput_tokens_s"],
                            "label": f"bs={p['batch_size']}" + (" ⭐" if p["batch_size"] == knee else "")}
                           for p in pts])

    return f"""
<div id="panel5" style="margin-bottom:40px">
  <h3>Panel 5 — Inference Pareto + Allocator Hit Rate (Training vs Inference)</h3>
  <canvas id="paretoChart" style="max-width:700px;display:block;margin-bottom:20px"></canvas>
  <script>
    (function() {{
      const pts = {pts_json};
      new Chart(document.getElementById('paretoChart').getContext('2d'), {{
        type: 'scatter',
        data: {{
          datasets: [{{
            label: 'Inference configs (⭐ = knee)',
            data: pts.map(p => ({{x:p.x, y:p.y}})),
            backgroundColor: pts.map(p => p.label.includes('⭐') ? '#e74c3c' : '#3498db'),
            pointRadius: 7,
          }}]
        }},
        options: {{
          plugins: {{ tooltip: {{ callbacks: {{ label: (ctx) => pts[ctx.dataIndex].label }} }} }},
          scales: {{
            x: {{ title: {{ display:true, text:'p50 Latency (ms)' }} }},
            y: {{ title: {{ display:true, text:'Throughput (tokens/s)' }} }},
          }}
        }}
      }});
    }})();
  </script>
  <h4>Allocator Cache Hit Rate: Training vs Inference</h4>
  <table border="1" style="border-collapse:collapse">
    <tr><th>Mode</th><th>Cache Hit Rate</th><th>Notes</th></tr>
    <tr><td>Training</td><td>{train_hit:.1f}%</td><td>Irregular FSDP allocs → lower hit rate</td></tr>
    <tr><td>Inference</td><td>{infer_hit:.1f}%</td><td>Fixed shapes → high cache reuse</td></tr>
  </table>
  <p>Inference hit rate target: &gt;90% (fixed-shape batches reuse pooled memory aggressively).
     Motivates different pool sizing on Inferentia vs Trainium.</p>
</div>
"""


# ---------------------------------------------------------------------------
# Dashboard assembler
# ---------------------------------------------------------------------------

def build_dashboard(
    results_dir=None,
    hw_name: str = "rtx_3090",
    dry_run: bool = False,
) -> str:
    """
    Assemble all panels into a single HTML file.

    results_dir: directory to write roofline_dashboard.html into.
                 Defaults to <repo-root>/results/.
    Returns the path of the written HTML file.
    """
    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark data once
    benchmark_json_path = results_dir / "swiglu_benchmark.json"
    benchmark_data = None
    if not dry_run and benchmark_json_path.exists():
        benchmark_data = json.loads(benchmark_json_path.read_text())

    panel1 = build_panel1_html(hw_name=hw_name, benchmark_data=benchmark_data, dry_run=dry_run)
    panel2 = build_panel2_html(benchmark_data=benchmark_data, dry_run=dry_run)
    panel3 = build_panel3_html(dry_run=dry_run)
    panel4 = build_panel4_html(dry_run=dry_run)
    panel5 = build_panel5_html(dry_run=dry_run)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>AxonForge Roofline Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    body {{ font-family: sans-serif; padding: 20px; max-width: 1100px; margin: auto; }}
    h2   {{ border-bottom: 2px solid #333; padding-bottom: 6px; }}
    table{{ margin-bottom: 10px; }}
    td, th {{ padding: 4px 10px; }}
  </style>
</head>
<body>
  <h2>AxonForge — Roofline &amp; Performance Dashboard</h2>
  <p>Hardware: <b>{hw_name}</b> | Mode: {'dry-run (synthetic data)' if dry_run else 'live'}</p>

  {panel1}
  {panel2}
  {panel3}
  {panel4}
  {panel5}

  <hr>
  <p><em>Generated by AxonForge profiler/dashboard.py</em></p>
</body>
</html>"""

    out_path = results_dir / "roofline_dashboard.html"
    out_path.write_text(html)
    return str(out_path)
