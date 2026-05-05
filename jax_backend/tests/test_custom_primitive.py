"""
Tests for the JAX custom SwiGLU primitive.

All tests run on CPU using JAX (no GPU required).

The primitive demonstrates:
  - abstract_eval: returns ShapedArray (NEVER a concrete value)
  - impl:          concrete computation via jnp ops
  - jvp:           forward-mode differentiation rule
  - custom_vjp:    reverse-mode differentiation (practical alternative to
                   raw transpose for non-linear ops)

Pitfalls exercised by these tests:
  - abstract_eval must return ShapedArray — returning a concrete array
    causes "Cannot use a non-abstract value" errors under jit
  - transpose receives None (undefined_primal) for the differentiated input;
    since SwiGLU is non-linear in gate, the raw transpose rule cannot
    compute silu'(gate) when gate is undefined — hence custom_vjp
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import pytest
from jax._src.core import ShapedArray   # JAX 0.10+

from jax_backend.custom_primitive import swiglu_primitive, swiglu_with_grad


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet: primitive output matches jnp reference
# ---------------------------------------------------------------------------

def test_swiglu_primitive_matches_reference():
    gate = jnp.array([1.0, -1.0, 0.0, 2.0])
    up   = jnp.array([0.5,  1.0, 2.0, 0.5])

    actual   = swiglu_primitive(gate, up)
    expected = jax.nn.silu(gate) * up

    assert jnp.allclose(actual, expected, atol=1e-6), (
        f"Primitive output differs: max_diff={jnp.abs(actual - expected).max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 2 — abstract_eval returns ShapedArray (not a concrete value)
# ---------------------------------------------------------------------------

def test_abstract_eval_returns_shaped_array():
    gate = jnp.ones((4, 8))
    up   = jnp.ones((4, 8))

    # jax.make_jaxpr traces without executing — triggers abstract_eval
    jaxpr = jax.make_jaxpr(swiglu_primitive)(gate, up)

    # The output type in the jaxpr must be a ShapedArray
    out_aval = jaxpr.out_avals[0]
    assert isinstance(out_aval, ShapedArray), (
        f"abstract_eval must return ShapedArray, got {type(out_aval)}"
    )
    assert out_aval.shape == (4, 8)
    assert out_aval.dtype == jnp.float32


# ---------------------------------------------------------------------------
# Test 3 — JVP: tangents match finite-difference Jacobian
# ---------------------------------------------------------------------------

def test_jvp_matches_finite_difference():
    gate = jnp.array([1.0, -0.5, 2.0, 0.0])
    up   = jnp.array([1.0,  1.0, 0.5, 2.0])

    gate_dot = jnp.ones_like(gate)
    up_dot   = jnp.zeros_like(up)   # test d/d(gate) only first

    # JVP via jax.jvp (uses the registered jvp rule)
    (out, out_dot_jvp) = jax.jvp(
        swiglu_primitive,
        primals=(gate, up),
        tangents=(gate_dot, up_dot),
    )

    # Finite-difference approximation of d/d(gate) via jax reference
    eps = 1e-4
    out_plus  = jax.nn.silu(gate + eps * gate_dot) * up
    out_minus = jax.nn.silu(gate - eps * gate_dot) * up
    out_dot_fd = (out_plus - out_minus) / (2 * eps)

    assert jnp.allclose(out_dot_jvp, out_dot_fd, atol=1e-3), (
        f"JVP vs FD: max_diff={jnp.abs(out_dot_jvp - out_dot_fd).max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 4 — jax.grad produces correct gradients
# ---------------------------------------------------------------------------

def test_jax_grad_correct():
    gate = jnp.array([1.0, -0.5, 2.0, 0.3])
    up   = jnp.array([1.0,  1.5, 0.5, 2.0])

    # Reference gradient via jax.grad on the jnp expression
    def ref_f(gate, up):
        return (jax.nn.silu(gate) * up).sum()

    ref_gate_grad, ref_up_grad = jax.grad(ref_f, argnums=(0, 1))(gate, up)

    # Custom primitive gradient via custom_vjp wrapper
    def custom_f(gate, up):
        return swiglu_with_grad(gate, up).sum()

    custom_gate_grad, custom_up_grad = jax.grad(custom_f, argnums=(0, 1))(gate, up)

    assert jnp.allclose(custom_gate_grad, ref_gate_grad, atol=1e-5), (
        f"gate grad mismatch: max_diff={jnp.abs(custom_gate_grad - ref_gate_grad).max():.2e}"
    )
    assert jnp.allclose(custom_up_grad, ref_up_grad, atol=1e-5), (
        f"up grad mismatch: max_diff={jnp.abs(custom_up_grad - ref_up_grad).max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 5 — jax.jit(jax.grad(f)) composes correctly
# ---------------------------------------------------------------------------

def test_jit_grad_compose():
    gate = jnp.ones((4, 8))
    up   = jnp.ones((4, 8)) * 0.5

    @jax.jit
    def grad_fn(gate, up):
        return jax.grad(lambda g, u: swiglu_with_grad(g, u).sum())(gate, up)

    gate_grad = grad_fn(gate, up)   # must not raise

    assert gate_grad.shape == gate.shape
    # d/d(gate)[silu(gate)*up].sum() = silu'(gate)*up, with gate=1, up=0.5
    sigmoid_1 = jax.nn.sigmoid(jnp.ones((4, 8)))
    expected  = sigmoid_1 * (1 + jnp.ones((4, 8)) * (1 - sigmoid_1)) * 0.5
    assert jnp.allclose(gate_grad, expected, atol=1e-5)
