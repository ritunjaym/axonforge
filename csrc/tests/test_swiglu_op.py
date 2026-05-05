"""
Tests for the C++/CUDA SwiGLU custom op extension.

Tests 1–2   — CPU (M2 local): reference formulas + setup.py flags.
Tests 3–5   — CUDA + extension (cloud GPU): forward correctness.
Tests 6–10  — CUDA + extension (cloud GPU): backward correctness + gradcheck.

The RED→GREEN cycle for GPU tests runs on RunPod / AWS EC2:
  cd csrc && python setup.py install && pytest tests/test_swiglu_op.py -v
"""
import os
import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from csrc.functional import swiglu_ref


def _ext_available() -> bool:
    try:
        import axonforge_ops  # noqa: F401
        return True
    except ImportError:
        return False


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
EXT  = pytest.mark.skipif(
    not torch.cuda.is_available() or not _ext_available(),
    reason="requires axonforge_ops extension (build with csrc/setup.py install)"
)


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet (CPU): swiglu_ref matches mathematical definition
# ---------------------------------------------------------------------------

@given(
    batch=st.integers(min_value=1, max_value=8),
    n=st.sampled_from([64, 128, 256, 512]),
)
@settings(max_examples=50, deadline=None)
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


# ===========================================================================
# Slice 6 — Backward kernel + gradcheck
# ===========================================================================

# ---------------------------------------------------------------------------
# Test 6 — Tracer bullet (CPU): grad_gate formula is mathematically correct
#
# grad_gate = grad_out * up * silu'(gate)
# silu'(x)  = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
# ---------------------------------------------------------------------------

@given(
    batch=st.integers(min_value=1, max_value=4),
    n=st.sampled_from([64, 128, 256]),
)
@settings(max_examples=50, deadline=None)
def test_grad_gate_formula_correct(batch, n):
    gate = torch.randn(batch, n, requires_grad=True)
    up   = torch.randn(batch, n)
    grad_out = torch.randn(batch, n)

    # Analytical grad_gate
    with torch.no_grad():
        sig   = torch.sigmoid(gate)
        silu_prime = sig * (1.0 + gate * (1.0 - sig))
        analytical = grad_out * up * silu_prime

    # Autograd grad_gate via swiglu_ref
    out = swiglu_ref(gate, up)
    out.backward(grad_out)
    autograd_grad = gate.grad.clone()

    assert torch.allclose(autograd_grad, analytical, atol=1e-5), (
        f"grad_gate formula wrong: max_diff={(autograd_grad - analytical).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 7 — CPU: grad_up formula is mathematically correct
#
# grad_up = grad_out * silu(gate)
# ---------------------------------------------------------------------------

@given(
    batch=st.integers(min_value=1, max_value=4),
    n=st.sampled_from([64, 128, 256]),
)
@settings(max_examples=50, deadline=None)
def test_grad_up_formula_correct(batch, n):
    gate = torch.randn(batch, n)
    up   = torch.randn(batch, n, requires_grad=True)
    grad_out = torch.randn(batch, n)

    # Analytical grad_up
    with torch.no_grad():
        analytical = grad_out * F.silu(gate)

    # Autograd grad_up via swiglu_ref
    out = swiglu_ref(gate, up)
    out.backward(grad_out)
    autograd_grad = up.grad.clone()

    assert torch.allclose(autograd_grad, analytical, atol=1e-5), (
        f"grad_up formula wrong: max_diff={(autograd_grad - analytical).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 8 — gradcheck in float64 on SwiGLUFunction (GPU + extension)
#
# NEVER use float32 for gradcheck — finite differences are too noisy.
# float64 is the gold standard for custom autograd verification.
# ---------------------------------------------------------------------------

@EXT
@pytest.mark.parametrize("N", [64, 128, 256])
def test_gradcheck_float64(N):
    from csrc.functional import SwiGLUFunction

    # gradcheck requires double precision and small tensors for speed
    gate = torch.randn(2, N, device="cuda", dtype=torch.float64, requires_grad=True)
    up   = torch.randn(2, N, device="cuda", dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        SwiGLUFunction.apply,
        (gate, up),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    ), f"gradcheck failed at N={N}"


# ---------------------------------------------------------------------------
# Test 9 — Backward dtype dispatch: float16 and bfloat16 run without error
# ---------------------------------------------------------------------------

@EXT
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_backward_reduced_precision(dtype):
    import axonforge_ops
    N = 512
    gate     = torch.randn(4, N, device="cuda", dtype=dtype)
    up       = torch.randn(4, N, device="cuda", dtype=dtype)
    grad_out = torch.randn(4, N, device="cuda", dtype=dtype)

    grad_gate, grad_up = axonforge_ops.backward(grad_out, gate, up)

    assert grad_gate.dtype == dtype, f"grad_gate dtype {grad_gate.dtype} != {dtype}"
    assert grad_up.dtype   == dtype, f"grad_up dtype {grad_up.dtype} != {dtype}"
    assert grad_gate.shape == gate.shape
    assert grad_up.shape   == up.shape


# ---------------------------------------------------------------------------
# Test 10 — Extension backward matches analytical formulas (float32, GPU)
# ---------------------------------------------------------------------------

@EXT
@pytest.mark.parametrize("N", [512, 1024, 2048])
def test_extension_backward_matches_analytical(N):
    import axonforge_ops

    gate     = torch.randn(4, N, device="cuda", dtype=torch.float32)
    up       = torch.randn(4, N, device="cuda", dtype=torch.float32)
    grad_out = torch.randn(4, N, device="cuda", dtype=torch.float32)

    # Extension backward
    grad_gate_ext, grad_up_ext = axonforge_ops.backward(grad_out, gate, up)

    # Analytical (float32 reference, acceptable for checking formula not gradcheck)
    sig            = torch.sigmoid(gate)
    silu_prime     = sig * (1.0 + gate * (1.0 - sig))
    grad_gate_ref  = grad_out * up * silu_prime
    grad_up_ref    = grad_out * F.silu(gate)

    assert torch.allclose(grad_gate_ext, grad_gate_ref, atol=1e-4), (
        f"grad_gate mismatch N={N}: max_diff={(grad_gate_ext-grad_gate_ref).abs().max():.2e}"
    )
    assert torch.allclose(grad_up_ext, grad_up_ref, atol=1e-4), (
        f"grad_up mismatch N={N}: max_diff={(grad_up_ext-grad_up_ref).abs().max():.2e}"
    )
