"""
Hypothesis property tests for the fusion compiler pass.

Properties under test (in TDD order):
  1. LayerNorm→Linear pattern reduces node count           (tracer bullet)
  2. Input GraphModule is never mutated                    (purity)
  3. Fused output matches original within atol=1e-5        (semantic preservation)
  4. pass(pass(gm)) == pass(gm)                           (idempotency)
  5. No fusable pattern → graph passes through unchanged
  6. Multi-consumer node blocks fusion
"""
import copy
import torch
import torch.nn as nn
import torch.fx
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from compiler.passes.fusion import fusion_pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trace(model: nn.Module) -> torch.fx.GraphModule:
    return torch.fx.symbolic_trace(model)


def _node_count(gm: torch.fx.GraphModule) -> int:
    return sum(1 for n in gm.graph.nodes if n.op == "call_function" or n.op == "call_module")


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet: LayerNorm→Linear reduces node count
# ---------------------------------------------------------------------------

class LayerNormLinear(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.linear = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.linear(self.ln(x))


def test_layernorm_linear_reduces_node_count():
    model = LayerNormLinear(64)
    gm = _trace(model)
    before = _node_count(gm)

    fused = fusion_pass(gm)

    assert _node_count(fused) < before, (
        f"Expected fewer nodes after fusion, got {_node_count(fused)} (was {before})"
    )


# ---------------------------------------------------------------------------
# Test 2 — Purity: input GraphModule is never mutated
# ---------------------------------------------------------------------------

def test_fusion_does_not_mutate_input():
    model = LayerNormLinear(64)
    gm = _trace(model)
    before_nodes = [n.name for n in gm.graph.nodes]
    before_count = _node_count(gm)

    _ = fusion_pass(gm)

    after_nodes = [n.name for n in gm.graph.nodes]
    assert before_nodes == after_nodes, "fusion_pass mutated the input graph's node list"
    assert _node_count(gm) == before_count, "fusion_pass mutated the input graph's node count"


# ---------------------------------------------------------------------------
# Test 3 — Semantic preservation: fused output matches original within atol=1e-5
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(
    batch=st.integers(min_value=1, max_value=8),
    d=st.sampled_from([32, 64, 128]),
)
def test_fusion_semantic_preservation(batch, d):
    model = LayerNormLinear(d)
    model.eval()
    gm = _trace(model)
    fused = fusion_pass(gm)
    fused.eval()

    x = torch.randn(batch, d)
    with torch.no_grad():
        expected = gm(x)
        actual = fused(x)

    assert torch.allclose(expected, actual, atol=1e-5), (
        f"Semantic mismatch: max_diff={( expected - actual).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Idempotency: fusion_pass(fusion_pass(gm)) == fusion_pass(gm)
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    batch=st.integers(min_value=1, max_value=8),
    d=st.sampled_from([32, 64, 128]),
)
def test_fusion_idempotent(batch, d):
    model = LayerNormLinear(d)
    model.eval()
    gm = _trace(model)

    once = fusion_pass(gm)
    twice = fusion_pass(once)

    # Node counts must be equal (second pass finds nothing to fuse)
    assert _node_count(once) == _node_count(twice), (
        f"Second pass changed node count: {_node_count(once)} → {_node_count(twice)}"
    )

    # Outputs must match for both
    x = torch.randn(batch, d)
    with torch.no_grad():
        once.eval()
        twice.eval()
        out_once = once(x)
        out_twice = twice(x)

    assert torch.allclose(out_once, out_twice, atol=1e-5), (
        f"Idempotency broken: max_diff={(out_once - out_twice).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 5 — No fusable pattern: node count unchanged, output identical
# ---------------------------------------------------------------------------

class LinearOnly(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, d)

    def forward(self, x):
        return self.linear(x)


@settings(max_examples=50)
@given(
    batch=st.integers(min_value=1, max_value=8),
    d=st.sampled_from([32, 64, 128]),
)
def test_no_fusable_pattern_unchanged(batch, d):
    model = LinearOnly(d)
    model.eval()
    gm = _trace(model)
    before = _node_count(gm)

    fused = fusion_pass(gm)
    fused.eval()

    assert _node_count(fused) == before, (
        f"Pass should not change node count when no pattern present: {before} → {_node_count(fused)}"
    )

    x = torch.randn(batch, d)
    with torch.no_grad():
        assert torch.allclose(gm(x), fused(x), atol=1e-5)


# ---------------------------------------------------------------------------
# Test 6 — Multi-consumer blocks fusion: LayerNorm output used by two nodes
# ---------------------------------------------------------------------------

class LayerNormTwoConsumers(nn.Module):
    """LayerNorm output goes to both Linear and a residual add — must NOT fuse."""
    def __init__(self, d: int):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.linear = nn.Linear(d, d, bias=False)

    def forward(self, x):
        normed = self.ln(x)
        return self.linear(normed) + normed   # normed has two users


def test_multi_consumer_blocks_fusion():
    model = LayerNormTwoConsumers(64)
    gm = _trace(model)
    before = _node_count(gm)

    fused = fusion_pass(gm)

    assert _node_count(fused) == before, (
        "fusion_pass fused a LayerNorm with multiple consumers — incorrect"
    )

    # Semantic check still holds (pass must not corrupt the graph)
    x = torch.randn(2, 64)
    with torch.no_grad():
        assert torch.allclose(gm(x), fused(x), atol=1e-5)
