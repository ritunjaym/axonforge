"""
Hypothesis property tests for the DCE (dead code elimination) compiler pass.

Properties under test (TDD order):
  1. Dead node (no users, not output) is removed        (tracer bullet)
  2. Input GraphModule is never mutated                  (purity)
  3. Output matches original within atol=1e-5            (semantic preservation)
  4. dce_pass(dce_pass(gm)) == dce_pass(gm)             (idempotency)
  5. No dead nodes → node count unchanged
"""
import copy
import torch
import torch.nn as nn
import torch.fx
from hypothesis import given, settings
from hypothesis import strategies as st

from compiler.passes.dce import dce_pass


def _node_count(gm: torch.fx.GraphModule) -> int:
    return sum(1 for n in gm.graph.nodes if n.op not in ("placeholder", "output"))


def _make_graph_with_dead_node(d: int = 64) -> torch.fx.GraphModule:
    """Trace a model that has a dead computation (result never reaches output)."""
    class ModelWithDeadCode(nn.Module):
        def __init__(self):
            super().__init__()
            self.live = nn.Linear(d, d)
            self.dead = nn.Linear(d, d)

        def forward(self, x):
            out = self.live(x)
            _ = self.dead(x)  # computed, result discarded — dead node
            return out

    return torch.fx.symbolic_trace(ModelWithDeadCode())


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet: dead node is eliminated
# ---------------------------------------------------------------------------

def test_dce_removes_dead_node():
    gm = _make_graph_with_dead_node(64)
    before = _node_count(gm)

    result = dce_pass(gm)

    assert _node_count(result) < before, (
        f"Expected fewer nodes after DCE, got {_node_count(result)} (was {before})"
    )


# ---------------------------------------------------------------------------
# Test 2 — Purity: input GraphModule is never mutated
# ---------------------------------------------------------------------------

def test_dce_does_not_mutate_input():
    gm = _make_graph_with_dead_node(64)
    before_names = [n.name for n in gm.graph.nodes]
    before_count = _node_count(gm)

    _ = dce_pass(gm)

    assert [n.name for n in gm.graph.nodes] == before_names
    assert _node_count(gm) == before_count


# ---------------------------------------------------------------------------
# Test 3 — Semantic preservation
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(
    batch=st.integers(min_value=1, max_value=8),
    d=st.sampled_from([32, 64, 128]),
)
def test_dce_semantic_preservation(batch, d):
    gm = _make_graph_with_dead_node(d)
    gm.eval()
    result = dce_pass(gm)
    result.eval()

    x = torch.randn(batch, d)
    with torch.no_grad():
        expected = gm(x)
        actual = result(x)

    assert torch.allclose(expected, actual, atol=1e-5), (
        f"Semantic mismatch: max_diff={(expected - actual).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Idempotency
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    batch=st.integers(min_value=1, max_value=8),
    d=st.sampled_from([32, 64, 128]),
)
def test_dce_idempotent(batch, d):
    gm = _make_graph_with_dead_node(d)
    once = dce_pass(gm)
    twice = dce_pass(once)

    assert _node_count(once) == _node_count(twice), (
        f"Second DCE pass changed node count: {_node_count(once)} → {_node_count(twice)}"
    )

    x = torch.randn(batch, d)
    with torch.no_grad():
        once.eval(); twice.eval()
        assert torch.allclose(once(x), twice(x), atol=1e-5)


# ---------------------------------------------------------------------------
# Test 5 — No dead nodes → count unchanged
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(
    batch=st.integers(min_value=1, max_value=8),
    d=st.sampled_from([32, 64, 128]),
)
def test_dce_no_dead_nodes_unchanged(batch, d):
    class LiveModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(d, d)
        def forward(self, x):
            return self.linear(x)

    gm = torch.fx.symbolic_trace(LiveModel())
    before = _node_count(gm)

    result = dce_pass(gm)

    assert _node_count(result) == before, (
        f"DCE incorrectly removed live nodes: {before} → {_node_count(result)}"
    )

    x = torch.randn(batch, d)
    with torch.no_grad():
        gm.eval(); result.eval()
        assert torch.allclose(gm(x), result(x), atol=1e-5)
