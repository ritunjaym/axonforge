"""
TorchScript export pipeline for inference.

Export order (matters — see CLAUDE.md pitfall):
  1. torch.jit.script — convert to TorchScript IR
  2. torch.jit.optimize_for_inference — fold BN, remove training-only ops,
     enable cuDNN/oneDNN fusion (must happen before freeze)
  3. torch.jit.freeze — lock weights as constants for aggressive fusion

freeze + optimize_for_inference together enable constant folding that is
not safe during training (weights change). This is the key distinction
between training and inference graphs.
"""
import torch
import torch.nn as nn


def export_for_inference(
    model: nn.Module,
    example_inputs: tuple,
) -> torch.jit.ScriptModule:
    """
    Export model to a frozen, inference-optimized ScriptModule.

    Args:
        model: nn.Module in eval mode. Will be set to eval() if not already.
        example_inputs: tuple of example tensors (used for tracing fallback).

    Returns:
        Frozen ScriptModule ready for inference.
    """
    model = model.eval()

    # Script the model (preferred over trace — handles control flow)
    try:
        scripted = torch.jit.script(model)
    except Exception:
        # Fall back to tracing for models that can't be scripted
        # (e.g., models with non-scriptable ops like some FX-transformed graphs)
        scripted = torch.jit.trace(model, example_inputs)

    # freeze FIRST — locks weights as constants, required before optimize
    frozen = torch.jit.freeze(scripted)

    # optimize_for_inference AFTER freeze — folds BN, enables cuDNN/oneDNN fusion
    # (PyTorch 2.x: optimize_for_inference expects a frozen module)
    frozen = torch.jit.optimize_for_inference(frozen)

    return frozen
