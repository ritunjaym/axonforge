"""
Fusion compiler pass for torch.fx GraphModules.

Pattern: LayerNorm → Linear (single consumer)
  Replace both nodes with a single fused_layernorm_linear call.
  The fused function is semantically equivalent; the transformation
  demonstrates pattern-matching graph rewriting and sets up for
  future kernel-level fusion.

Invariants (enforced by tests):
  - Pure: never mutates the input GraphModule
  - Idempotent: fusion_pass(fusion_pass(gm)) == fusion_pass(gm)
  - Semantic-preserving: outputs match within atol=1e-5
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx
from torch.fx.node import Node
from typing import Dict


# ---------------------------------------------------------------------------
# Fused operation (semantically equivalent to LayerNorm + Linear in sequence)
# ---------------------------------------------------------------------------

def _fused_layernorm_linear(
    x: torch.Tensor,
    normalized_shape,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias,
    eps: float,
) -> torch.Tensor:
    y = F.layer_norm(x, normalized_shape, ln_weight, ln_bias, eps)
    return F.linear(y, linear_weight, linear_bias)


# Register so symbolic_trace can cross the call boundary
torch.fx.wrap("_fused_layernorm_linear")


# ---------------------------------------------------------------------------
# Pattern matching helpers
# ---------------------------------------------------------------------------

def _is_layernorm_call(node: Node, gm: torch.fx.GraphModule) -> bool:
    if node.op != "call_module":
        return False
    mod = gm.get_submodule(node.target)
    return isinstance(mod, nn.LayerNorm)


def _is_linear_call(node: Node, gm: torch.fx.GraphModule) -> bool:
    if node.op != "call_module":
        return False
    mod = gm.get_submodule(node.target)
    return isinstance(mod, nn.Linear)


def _single_user(node: Node) -> bool:
    return len(node.users) == 1


def _linear_takes_layernorm_directly(ln_node: Node, linear_node: Node) -> bool:
    """True when the linear's first positional arg is the layernorm output."""
    if not linear_node.args:
        return False
    return linear_node.args[0] is ln_node


# ---------------------------------------------------------------------------
# Pass entry point
# ---------------------------------------------------------------------------

def fusion_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Pure function: returns a new GraphModule with LayerNorm→Linear patterns fused.
    The input GraphModule is never modified.
    """
    gm = copy.deepcopy(gm)
    graph = gm.graph

    fused_counter = 0
    nodes = list(graph.nodes)

    for node in nodes:
        if not _is_layernorm_call(node, gm):
            continue
        if not _single_user(node):
            continue

        linear_node = next(iter(node.users))
        if not _is_linear_call(linear_node, gm):
            continue
        if not _linear_takes_layernorm_directly(node, linear_node):
            continue

        # Gather submodule state
        ln: nn.LayerNorm = gm.get_submodule(node.target)
        linear: nn.Linear = gm.get_submodule(linear_node.target)

        # Register fused weights as buffers on the GraphModule so they
        # survive serialisation and are accessible by name.
        prefix = f"_fused_{fused_counter}"
        gm.register_buffer(f"{prefix}_ln_w",   ln.weight.data.clone())
        gm.register_buffer(f"{prefix}_ln_b",   ln.bias.data.clone() if ln.bias is not None else torch.zeros(ln.normalized_shape))
        gm.register_buffer(f"{prefix}_lin_w",  linear.weight.data.clone())
        gm.register_buffer(f"{prefix}_lin_b",  linear.bias.data.clone() if linear.bias is not None else None)

        eps = ln.eps
        normalized_shape = list(ln.normalized_shape)
        has_lin_bias = linear.bias is not None

        # Insert fused node before the linear node
        with graph.inserting_before(linear_node):
            # Fetch the registered tensors from the GraphModule at runtime
            ln_w_node  = graph.get_attr(f"{prefix}_ln_w")
            ln_b_node  = graph.get_attr(f"{prefix}_ln_b")
            lin_w_node = graph.get_attr(f"{prefix}_lin_w")
            lin_b_node = graph.get_attr(f"{prefix}_lin_b") if has_lin_bias else None

            fused_node = graph.call_function(
                _fused_layernorm_linear,
                args=(
                    node.args[0],        # original input to LayerNorm
                    normalized_shape,
                    ln_w_node,
                    ln_b_node,
                    lin_w_node,
                    lin_b_node,
                    eps,
                ),
            )

        linear_node.replace_all_uses_with(fused_node)
        graph.erase_node(linear_node)
        graph.erase_node(node)
        fused_counter += 1

    graph.lint()
    gm.recompile()
    return gm
