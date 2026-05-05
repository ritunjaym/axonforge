"""
Tests for the compiler pass pipeline.

Properties under test:
  1. Pipeline runs all three passes without error           (tracer bullet)
  2. Pipeline output is semantically equivalent to input   (semantic preservation)
  3. Composability: each pass output is valid input to the next
  4. Pipeline is idempotent end-to-end
"""
import torch
import torch.nn as nn
import torch.fx
from hypothesis import given, settings
from hypothesis import strategies as st

from compiler.pipeline import run_pass_pipeline


# ---------------------------------------------------------------------------
# Models that exercise all three passes
# ---------------------------------------------------------------------------

class FullPipelineModel(nn.Module):
    """
    - LayerNorm→Linear: exercises fusion pass
    - Dead linear branch: exercises DCE
    - Transposed input to second linear: exercises layout pass
    """
    def __init__(self, d: int):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, d, bias=False)   # fused with ln
        self.dead = nn.Linear(d, d, bias=False)       # dead branch
        self.linear2 = nn.Linear(d, d, bias=False)   # gets contiguous() before it

    def forward(self, x):
        fused = self.linear1(self.ln(x))
        _ = self.dead(x)                              # dead — DCE removes it
        transposed = fused.transpose(0, 1)
        out = self.linear2(transposed)                # layout pass inserts contiguous()
        return out


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet: pipeline runs end-to-end without error
# ---------------------------------------------------------------------------

def test_pipeline_runs_without_error():
    model = FullPipelineModel(64)
    gm = torch.fx.symbolic_trace(model)

    result = run_pass_pipeline(gm)   # must not raise

    assert result is not None


# ---------------------------------------------------------------------------
# Test 2 — Semantic preservation across the full pipeline
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(
    batch=st.integers(min_value=1, max_value=4),
    seq=st.integers(min_value=1, max_value=4),
    d=st.sampled_from([32, 64, 128]),
)
def test_pipeline_semantic_preservation(batch, seq, d):
    model = FullPipelineModel(d)
    model.eval()
    gm = torch.fx.symbolic_trace(model)
    result = run_pass_pipeline(gm)
    result.eval()

    x = torch.randn(batch, seq, d)
    with torch.no_grad():
        expected = gm(x)
        actual = result(x)

    assert torch.allclose(expected, actual, atol=1e-5), (
        f"Pipeline broke semantics: max_diff={(expected - actual).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Composability: output of each pass is a valid GraphModule
# ---------------------------------------------------------------------------

def test_pipeline_passes_compose():
    from compiler.passes.fusion import fusion_pass
    from compiler.passes.dce import dce_pass
    from compiler.passes.layout import layout_pass

    model = FullPipelineModel(64)
    gm = torch.fx.symbolic_trace(model)

    after_fusion = fusion_pass(gm)
    after_dce    = dce_pass(after_fusion)
    after_layout = layout_pass(after_dce)

    # Each stage must produce a linted, recompiled GraphModule
    after_fusion.graph.lint()
    after_dce.graph.lint()
    after_layout.graph.lint()

    x = torch.randn(2, 3, 64)
    with torch.no_grad():
        gm.eval(); after_layout.eval()
        assert torch.allclose(gm(x), after_layout(x), atol=1e-5)


# ---------------------------------------------------------------------------
# Test 4 — Pipeline idempotency
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(
    batch=st.integers(min_value=1, max_value=4),
    seq=st.integers(min_value=1, max_value=4),
    d=st.sampled_from([32, 64, 128]),
)
def test_pipeline_idempotent(batch, seq, d):
    model = FullPipelineModel(d)
    model.eval()
    gm = torch.fx.symbolic_trace(model)

    once  = run_pass_pipeline(gm)
    twice = run_pass_pipeline(once)

    x = torch.randn(batch, seq, d)
    with torch.no_grad():
        once.eval(); twice.eval()
        assert torch.allclose(once(x), twice(x), atol=1e-5)
