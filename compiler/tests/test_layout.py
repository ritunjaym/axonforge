"""
Hypothesis property tests for the layout compiler pass.

The layout pass inserts .contiguous() calls before Linear/matmul nodes
when their input arrives from a transpose-like op, ensuring coalesced
memory access in downstream kernels.

Properties under test (TDD order):
  1. Transposed input to Linear gets .contiguous() inserted  (tracer bullet)
  2. Input GraphModule is never mutated                      (purity)
  3. Output matches original within atol=1e-5                (semantic preservation)
  4. layout_pass(layout_pass(gm)) == layout_pass(gm)        (idempotency)
  5. Already-contiguous input → no extra nodes inserted
"""
import torch
import torch.nn as nn
import torch.fx
from hypothesis import given, settings
from hypothesis import strategies as st

from compiler.passes.layout import layout_pass


def _node_count(gm: torch.fx.GraphModule) -> int:
    return sum(1 for n in gm.graph.nodes if n.op not in ("placeholder", "output"))


def _contiguous_call_count(gm: torch.fx.GraphModule) -> int:
    return sum(
        1 for n in gm.graph.nodes
        if n.op == "call_method" and n.target == "contiguous"
    )


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet: transpose before Linear gets .contiguous() inserted
# ---------------------------------------------------------------------------

class TransposedLinear(nn.Module):
    """Input is transposed before Linear — layout pass must insert contiguous()."""
    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)

    def forward(self, x):           # x: (batch, seq, d)
        return self.linear(x.transpose(0, 1))   # (seq, batch, d) — non-contiguous


def test_layout_inserts_contiguous_before_linear():
    model = TransposedLinear(64)
    gm = torch.fx.symbolic_trace(model)
    before = _contiguous_call_count(gm)

    result = layout_pass(gm)

    assert _contiguous_call_count(result) > before, (
        "layout_pass did not insert a .contiguous() before the transposed Linear input"
    )


# ---------------------------------------------------------------------------
# Test 2 — Purity
# ---------------------------------------------------------------------------

def test_layout_does_not_mutate_input():
    model = TransposedLinear(64)
    gm = torch.fx.symbolic_trace(model)
    before_names = [n.name for n in gm.graph.nodes]
    before_count = _node_count(gm)

    _ = layout_pass(gm)

    assert [n.name for n in gm.graph.nodes] == before_names
    assert _node_count(gm) == before_count


# ---------------------------------------------------------------------------
# Test 3 — Semantic preservation
# ---------------------------------------------------------------------------

@settings(max_examples=50, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    seq=st.integers(min_value=1, max_value=4),
    d=st.sampled_from([32, 64, 128]),
)
def test_layout_semantic_preservation(batch, seq, d):
    model = TransposedLinear(d)
    model.eval()
    gm = torch.fx.symbolic_trace(model)
    result = layout_pass(gm)
    result.eval()

    x = torch.randn(batch, seq, d)
    with torch.no_grad():
        expected = gm(x)
        actual = result(x)

    assert torch.allclose(expected, actual, atol=1e-5), (
        f"Semantic mismatch: max_diff={(expected - actual).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Idempotency
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=4),
    seq=st.integers(min_value=1, max_value=4),
    d=st.sampled_from([32, 64, 128]),
)
def test_layout_idempotent(batch, seq, d):
    model = TransposedLinear(d)
    model.eval()
    gm = torch.fx.symbolic_trace(model)

    once = layout_pass(gm)
    twice = layout_pass(once)

    assert _node_count(once) == _node_count(twice), (
        f"Second layout pass changed node count: {_node_count(once)} → {_node_count(twice)}"
    )

    x = torch.randn(batch, seq, d)
    with torch.no_grad():
        once.eval(); twice.eval()
        assert torch.allclose(once(x), twice(x), atol=1e-5)


# ---------------------------------------------------------------------------
# Test 5 — Already-contiguous input → no .contiguous() inserted
# ---------------------------------------------------------------------------

@settings(max_examples=50, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=8),
    d=st.sampled_from([32, 64, 128]),
)
def test_layout_skips_contiguous_input(batch, d):
    class DirectLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(d, d, bias=False)
        def forward(self, x):
            return self.linear(x)   # x is already contiguous — no insert needed

    gm = torch.fx.symbolic_trace(DirectLinear())
    before = _contiguous_call_count(gm)

    result = layout_pass(gm)

    assert _contiguous_call_count(result) == before, (
        "layout_pass incorrectly inserted .contiguous() for already-contiguous input"
    )
