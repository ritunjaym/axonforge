"""
Layout reordering compiler pass for torch.fx GraphModules.

Inserts `.contiguous()` calls before Linear/matmul nodes whose input
arrives from a transpose-like operation (transpose, permute, t).
Non-contiguous inputs to matrix ops cause strided memory access,
hurting memory bandwidth utilisation.

The pass is idempotent: if the input is already routed through
.contiguous(), a second .contiguous() is not inserted (detected by
checking whether the feeding node is already a contiguous() call).

Invariants (enforced by tests):
  - Pure: never mutates the input GraphModule
  - Idempotent: layout_pass(layout_pass(gm)) == layout_pass(gm)
  - Semantic-preserving: outputs match within atol=1e-5
"""
import copy
import torch
import torch.nn as nn
import torch.fx
from torch.fx.node import Node


# Ops that produce non-contiguous tensors
_TRANSPOSE_OPS = {"transpose", "permute", "t"}

# call_function targets that are matrix multiply operations
_MATMUL_FUNCTIONS = {torch.matmul, torch.mm, torch.bmm}


def _is_transpose_node(node: Node) -> bool:
    if node.op == "call_method" and node.target in _TRANSPOSE_OPS:
        return True
    if node.op == "call_function" and getattr(node.target, "__name__", "") in _TRANSPOSE_OPS:
        return True
    return False


def _is_already_contiguous(node: Node) -> bool:
    return node.op == "call_method" and node.target == "contiguous"


def _is_linear_node(node: Node, gm: torch.fx.GraphModule) -> bool:
    if node.op != "call_module":
        return False
    try:
        mod = gm.get_submodule(node.target)
        return isinstance(mod, (nn.Linear,))
    except AttributeError:
        return False


def _is_matmul_node(node: Node) -> bool:
    return node.op == "call_function" and node.target in _MATMUL_FUNCTIONS


def _needs_contiguous(input_node: Node) -> bool:
    """True when the input came from a transpose and is not already made contiguous."""
    return _is_transpose_node(input_node) and not _is_already_contiguous(input_node)


def layout_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Pure function: returns a new GraphModule with .contiguous() inserted
    before Linear/matmul nodes whose input is non-contiguous.
    """
    gm = copy.deepcopy(gm)
    graph = gm.graph

    for node in list(graph.nodes):
        # Determine whether this node is a matrix op we care about
        if not (_is_linear_node(node, gm) or _is_matmul_node(node)):
            continue

        if not node.args:
            continue

        input_node = node.args[0]
        if not isinstance(input_node, Node):
            continue

        if not _needs_contiguous(input_node):
            continue

        # Insert .contiguous() call immediately before this node
        with graph.inserting_before(node):
            contiguous_node = graph.call_method("contiguous", args=(input_node,))

        # Rewire: replace the transposed input with the contiguous version
        # Only replace the first positional argument (the tensor input)
        new_args = (contiguous_node,) + node.args[1:]
        node.args = new_args

    graph.lint()
    gm.recompile()
    return gm
