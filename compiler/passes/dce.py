"""
Dead Code Elimination (DCE) compiler pass for torch.fx GraphModules.

A node is dead if it has no users and is not the output node.
We perform a single backwards sweep: nodes with zero users (excluding
the output sentinel) are erased. The sweep repeats until stable because
erasing a node may expose its inputs as newly dead.

Invariants (enforced by tests):
  - Pure: never mutates the input GraphModule
  - Idempotent: dce_pass(dce_pass(gm)) == dce_pass(gm)
  - Semantic-preserving: live outputs are unchanged
"""
import copy
import torch.fx


def dce_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Pure function: returns a new GraphModule with dead nodes eliminated.
    The input GraphModule is never modified.
    """
    gm = copy.deepcopy(gm)
    graph = gm.graph

    changed = True
    while changed:
        changed = False
        # Traverse in reverse so we see leaf-dead nodes before their parents
        for node in reversed(list(graph.nodes)):
            if node.op == "output":
                continue
            if len(node.users) == 0:
                graph.erase_node(node)
                changed = True

    graph.lint()
    gm.recompile()
    return gm
