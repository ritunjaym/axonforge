"""
Compiler pass pipeline: runs all three passes in dependency order.

Order matters:
  1. fusion  — LayerNorm→Linear fusion may create single-consumer patterns
               that DCE can then clean up if something upstream was fused away
  2. dce     — removes dead nodes exposed after fusion rewiring
  3. layout  — inserts .contiguous() before matmul/linear nodes; runs last
               so it sees the final graph shape after fusion + DCE
"""
import torch.fx

from compiler.passes.fusion import fusion_pass
from compiler.passes.dce import dce_pass
from compiler.passes.layout import layout_pass


def run_pass_pipeline(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Pure function: applies fusion → DCE → layout reordering in sequence.
    Returns a new GraphModule; input is never modified.
    """
    gm = fusion_pass(gm)
    gm = dce_pass(gm)
    gm = layout_pass(gm)
    return gm
