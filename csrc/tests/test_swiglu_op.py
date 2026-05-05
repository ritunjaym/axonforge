"""
Tests for the C++/CUDA SwiGLU custom op extension.

Test 1  — CPU (M2 local): swiglu_ref matches silu(gate)*up exactly.
Test 2  — CPU (M2 local): setup.py references required nvcc flags.
Tests 3–5 — CUDA (cloud GPU): extension matches reference.

The RED→GREEN cycle for GPU tests runs on RunPod / AWS EC2:
  cd csrc && python setup.py install && pytest tests/test_swiglu_op.py -v
"""
import ast
import os
import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from csrc.functional import swiglu_ref

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
EXT  = pytest.mark.skipif(
    not torch.cuda.is_available() or not _ext_available(),
    reason="requires axonforge_ops extension (build with csrc/setup.py install)"
)


def _ext_available() -> bool:
    try:
        import axonforge_ops  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet (CPU): swiglu_ref matches mathematical definition
# ---------------------------------------------------------------------------

@given(
    batch=st.integers(min_value=1, max_value=8),
    n=st.sampled_from([64, 128, 256, 512]),
)
@settings(max_examples=50)
def test_swiglu_ref_matches_definition(batch, n):
    gate = torch.randn(batch, n)
    up   = torch.randn(batch, n)

    expected = F.silu(gate) * up
    actual   = swiglu_ref(gate, up)

    assert torch.allclose(actual, expected, atol=1e-6), (
        f"swiglu_ref deviates from definition: max_diff={(actual-expected).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 2 — CPU: setup.py has required nvcc flags
# ---------------------------------------------------------------------------

def test_setup_has_required_nvcc_flags():
    setup_path = os.path.join(os.path.dirname(__file__), "..", "setup.py")
    with open(os.path.abspath(setup_path)) as f:
        source = f.read()

    required_flags = [
        "--use_fast_math",
        "arch=compute_80,code=sm_80",
        "arch=compute_75,code=sm_75",
        "--ptxas-options=-v",
    ]
    for flag in required_flags:
        assert flag in source, f"setup.py missing required nvcc flag: {flag!r}"


# ---------------------------------------------------------------------------
# Test 3 — Extension forward: float32 matches reference (atol=1e-4)
# ---------------------------------------------------------------------------

@EXT
@pytest.mark.parametrize("N", [512, 1024, 2048, 4096, 8192])
def test_extension_forward_float32(N):
    import axonforge_ops
    gate = torch.randn(4, N, device="cuda", dtype=torch.float32)
    up   = torch.randn(4, N, device="cuda", dtype=torch.float32)

    ref    = swiglu_ref(gate, up)
    actual = axonforge_ops.forward(gate, up)

    assert torch.allclose(actual, ref, atol=1e-4), (
        f"float32 mismatch at N={N}: max_diff={(actual-ref).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Dtype dispatch: float16 and bfloat16
# ---------------------------------------------------------------------------

@EXT
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_extension_forward_reduced_precision(dtype):
    import axonforge_ops
    N = 1024
    gate = torch.randn(4, N, device="cuda", dtype=dtype)
    up   = torch.randn(4, N, device="cuda", dtype=dtype)

    actual = axonforge_ops.forward(gate, up)
    ref    = swiglu_ref(gate.float(), up.float()).to(dtype)

    assert actual.dtype == dtype
    assert torch.allclose(actual, ref, atol=1e-2), (
        f"{dtype} mismatch: max_diff={(actual-ref).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 5 — autograd.Function forward matches extension
# ---------------------------------------------------------------------------

@EXT
def test_autograd_function_forward():
    from csrc.functional import SwiGLUFunction
    N = 512
    gate = torch.randn(4, N, device="cuda", dtype=torch.float32, requires_grad=False)
    up   = torch.randn(4, N, device="cuda", dtype=torch.float32, requires_grad=False)

    ref    = swiglu_ref(gate, up)
    actual = SwiGLUFunction.apply(gate, up)

    assert torch.allclose(actual, ref, atol=1e-4), (
        f"autograd.Function forward mismatch: max_diff={(actual-ref).abs().max():.2e}"
    )
