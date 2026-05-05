"""
Python autograd wrapper for the C++/CUDA SwiGLU extension.

Two entry points:
  swiglu_ref(gate, up)  — pure PyTorch reference; CPU + GPU, no extension needed
  SwiGLUFunction        — torch.autograd.Function backed by axonforge_ops extension
  swiglu(gate, up)      — dispatches to extension if available, else ref
"""
import torch
import torch.nn.functional as F


def swiglu_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference: silu(gate) * up. Runs on CPU or GPU without the extension."""
    return F.silu(gate) * up


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        import axonforge_ops
        ctx.save_for_backward(gate, up)
        return axonforge_ops.forward(gate, up)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        import axonforge_ops
        gate, up = ctx.saved_tensors
        return axonforge_ops.backward(grad_out, gate, up)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Production entry point.
    Uses CUDA extension when available; falls back to reference on CPU.
    """
    try:
        import axonforge_ops  # noqa: F401
        return SwiGLUFunction.apply(gate, up)
    except ImportError:
        return swiglu_ref(gate, up)
