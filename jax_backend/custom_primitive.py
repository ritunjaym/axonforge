"""
JAX custom SwiGLU primitive demonstrating the full primitive API.

Three layers:

1. swiglu_p — raw jax.core.Primitive
   abstract_eval: returns ShapedArray (NEVER concrete — PITFALL).
   impl:          concrete computation via jnp ops.
   jvp:           forward-mode differentiation rule.
   Registered via ad.primitive_jvps.

2. swiglu_primitive(gate, up) — thin wrapper around swiglu_p.bind()
   Usable under jax.jit and jax.jvp.
   jax.grad does NOT work through this directly (non-linear op
   requires explicit reverse-mode — see layer 3).

3. swiglu_with_grad(gate, up) — custom_vjp wrapper
   Provides reverse-mode AD (needed by jax.grad).
   Saves primals as residuals; VJP computes:
     gate_bar = silu'(gate) * up * ct
     up_bar   = silu(gate) * ct
   where silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))).

PITFALL — abstract_eval:
  Must return core.ShapedArray, not a concrete jnp array.
  Example of the WRONG approach:
    return jnp.zeros(gate_aval.shape, dtype=gate_aval.dtype)  # WRONG
  Correct:
    return core.ShapedArray(gate_aval.shape, gate_aval.dtype)  # RIGHT

PITFALL — transpose:
  For a non-linear primitive, the transpose rule receives None
  (undefined_primal) for the input being differentiated.
  SwiGLU is non-linear in gate: silu(gate) = gate * sigmoid(gate).
  The transpose rule cannot compute silu'(gate) when gate is None.
  Resolution: use custom_vjp (swiglu_with_grad) which saves residuals
  from the forward pass and avoids the undefined_primal problem.
"""
import jax
import jax.numpy as jnp
# JAX 0.10+: Primitive lives in jax.extend.core; ShapedArray in jax._src.core
from jax.extend import core
from jax._src.core import ShapedArray
from jax._src.interpreters import ad


# ---------------------------------------------------------------------------
# 1. Raw Primitive definition
# ---------------------------------------------------------------------------

swiglu_p = core.Primitive("swiglu")
swiglu_p.multiple_results = False


@swiglu_p.def_impl
def _swiglu_impl(gate: jax.Array, up: jax.Array) -> jax.Array:
    """Concrete evaluation: silu(gate) * up."""
    return jax.nn.silu(gate) * up


@swiglu_p.def_abstract_eval
def _swiglu_abstract_eval(
    gate_aval: ShapedArray,
    up_aval: ShapedArray,
) -> ShapedArray:
    """
    Abstract evaluation under jax.jit and jax.make_jaxpr.

    PITFALL: must return ShapedArray, NOT a concrete array.
    The abstract evaluator must not inspect concrete values.
    """
    assert gate_aval.shape == up_aval.shape, (
        f"gate shape {gate_aval.shape} != up shape {up_aval.shape}"
    )
    assert gate_aval.dtype == up_aval.dtype, (
        f"dtype mismatch: gate={gate_aval.dtype} up={up_aval.dtype}"
    )
    return ShapedArray(gate_aval.shape, gate_aval.dtype)


# ---------------------------------------------------------------------------
# 2. JVP rule (forward-mode differentiation)
# ---------------------------------------------------------------------------

def _swiglu_jvp(primals, tangents):
    """
    Forward-mode JVP for SwiGLU.

    output     = silu(gate) * up
    output_dot = silu'(gate) * up * gate_dot + silu(gate) * up_dot

    where silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    The output_dot is linear in (gate_dot, up_dot) with coefficients
    computed from the primals — JAX can transpose this linear function
    for reverse-mode AD without needing an explicit transpose rule for
    swiglu_p itself.
    """
    gate, up = primals
    gate_dot, up_dot = tangents

    out = swiglu_p.bind(gate, up)

    # Compute primal-derived coefficients (concrete arrays at this point)
    silu_gate    = jax.nn.silu(gate)
    sigmoid_gate = jax.nn.sigmoid(gate)
    silu_prime   = sigmoid_gate * (1 + gate * (1 - sigmoid_gate))

    # Linear function of tangents (can be transposed by JAX)
    out_dot = silu_prime * up * gate_dot + silu_gate * up_dot

    return out, out_dot


ad.primitive_jvps[swiglu_p] = _swiglu_jvp


# ---------------------------------------------------------------------------
# 3. Public API: swiglu_primitive (forward only, jvp works)
# ---------------------------------------------------------------------------

def swiglu_primitive(gate: jax.Array, up: jax.Array) -> jax.Array:
    """
    SwiGLU via custom JAX primitive.
    Supports jax.jit and jax.jvp.
    For jax.grad, use swiglu_with_grad instead.
    """
    return swiglu_p.bind(gate, up)


# ---------------------------------------------------------------------------
# 4. swiglu_with_grad — custom_vjp for full reverse-mode AD
# ---------------------------------------------------------------------------

@jax.custom_vjp
def swiglu_with_grad(gate: jax.Array, up: jax.Array) -> jax.Array:
    """
    SwiGLU with custom VJP for jax.grad support.

    Equivalent to silu(gate) * up, but with an explicit reverse-mode
    rule that avoids the undefined_primal pitfall in the transpose rule.
    """
    return swiglu_p.bind(gate, up)


def _swiglu_fwd(gate: jax.Array, up: jax.Array):
    """Forward pass: compute output and save residuals for backward."""
    out = swiglu_p.bind(gate, up)
    return out, (gate, up)   # residuals = primals


def _swiglu_bwd(residuals, ct: jax.Array):
    """
    Backward pass: VJP of SwiGLU.

    gate_bar = silu'(gate) * up * ct
    up_bar   = silu(gate) * ct

    PITFALL — transpose alternative:
      In the raw transpose rule approach, gate would be an undefined_primal
      (a Tracer, not a concrete value) when differentiating w.r.t. gate.
      Computing silu'(gate) requires gate's concrete value.
      custom_vjp solves this by saving gate as a residual from fwd.
    """
    gate, up = residuals
    silu_gate    = jax.nn.silu(gate)
    sigmoid_gate = jax.nn.sigmoid(gate)
    silu_prime   = sigmoid_gate * (1 + gate * (1 - sigmoid_gate))

    gate_bar = silu_prime * up * ct   # d(loss)/d(gate)
    up_bar   = silu_gate * ct         # d(loss)/d(up)
    return gate_bar, up_bar


swiglu_with_grad.defvjp(_swiglu_fwd, _swiglu_bwd)
