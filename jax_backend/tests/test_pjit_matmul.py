"""
Tests for the pjit_matmul distributed GEMM on a 2D device mesh.

Must set XLA_FLAGS before importing JAX — done at module level here.
All tests run on CPU using 4 virtual devices (no GPU required).

The virtual device trick:
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
  → jax.devices() returns 4 CpuDevice instances, enabling 2×2 mesh testing.

Pitfall: XLA_FLAGS must be set BEFORE jax is imported, and jax is imported
at module load time, so this file handles it at the top.
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from jax_backend.pjit_matmul import make_mesh, pjit_matmul


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet: pjit_matmul output matches jnp.matmul
# ---------------------------------------------------------------------------

def test_pjit_matmul_matches_jnp_matmul():
    mesh = make_mesh()

    x = jnp.ones((4, 8))
    w = jnp.ones((8, 4))

    result  = pjit_matmul(x, w, mesh)
    expected = jnp.matmul(x, w)

    assert jnp.allclose(result, expected, atol=1e-5), (
        f"pjit_matmul output differs from jnp.matmul: "
        f"max_diff={jnp.abs(result - expected).max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Mesh shape: (2, 2) with named axes 'data' and 'model'
# ---------------------------------------------------------------------------

def test_mesh_shape_and_axes():
    mesh = make_mesh()

    assert mesh.shape == {'data': 2, 'model': 2}, (
        f"Expected mesh shape {{'data':2,'model':2}}, got {mesh.shape}"
    )
    assert 'data'  in mesh.axis_names
    assert 'model' in mesh.axis_names


# ---------------------------------------------------------------------------
# Test 3 — Output shape: (M, N) for (M, K) @ (K, N)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,K,N", [
    (4, 8, 4), (8, 8, 8), (4, 4, 4),
])
def test_output_shape(M, K, N):
    mesh = make_mesh()
    x = jnp.ones((M, K))
    w = jnp.ones((K, N))

    result = pjit_matmul(x, w, mesh)

    assert result.shape == (M, N), (
        f"Expected shape ({M},{N}), got {result.shape}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Sharding: result lives on the mesh (not single-device)
# ---------------------------------------------------------------------------

def test_result_is_sharded():
    mesh = make_mesh()
    x = jnp.ones((4, 8))
    w = jnp.ones((8, 4))

    result = pjit_matmul(x, w, mesh)

    # A sharded array spans multiple devices
    assert hasattr(result, 'sharding'), "Result should have a sharding attribute"
    sharding = result.sharding
    # NamedSharding means the result is mesh-aware
    assert isinstance(sharding, NamedSharding), (
        f"Expected NamedSharding, got {type(sharding)}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Numerical correctness: hypothesis over random inputs
# ---------------------------------------------------------------------------

@given(
    M=st.sampled_from([4, 8]),
    K=st.sampled_from([4, 8]),
    N=st.sampled_from([4, 8]),
)
@settings(max_examples=10, deadline=None)   # JAX JIT compile on first call is slow
def test_numerical_correctness_random(M, K, N):
    mesh = make_mesh()
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (M, K))
    w = jax.random.normal(key, (K, N))

    result   = pjit_matmul(x, w, mesh)
    expected = jnp.matmul(x, w)

    assert jnp.allclose(result, expected, atol=1e-4), (
        f"Numerical mismatch M={M},K={K},N={N}: "
        f"max_diff={jnp.abs(result - expected).max():.2e}"
    )
