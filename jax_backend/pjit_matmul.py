"""
Distributed GEMM on a 2D device mesh using JAX sharding.

Architecture:
  - 2D mesh with axes ('data', 'model'):
      data  axis → shards rows of x (batch/sequence dimension)
      model axis → shards columns of w (output/hidden dimension)
  - PartitionSpec controls how each tensor is split across devices:
      x:   (PartitionSpec('data', None))   — rows sharded, cols replicated
      w:   (PartitionSpec(None, 'model'))  — rows replicated, cols sharded
      out: (PartitionSpec('data', 'model')) — fully sharded output

Modern JAX API (≥0.4):
  jax.sharding.Mesh, jax.sharding.NamedSharding, jax.sharding.PartitionSpec.
  jax.jit with in_shardings/out_shardings replaces jax.experimental.pjit.

Pitfalls:
  - XLA_FLAGS must be set BEFORE importing jax to use virtual CPU devices.
  - M and N must be divisible by the corresponding mesh dimension (2).
  - For multi-host, call jax.distributed.initialize() before jax.devices().

Local testing: set XLA_FLAGS="--xla_force_host_platform_device_count=4"
  → jax.devices() returns 4 virtual CpuDevices, enabling 2×2 mesh testing.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding


def make_mesh(
    mesh_shape: tuple[int, int] = (2, 2),
    axis_names: tuple[str, str] = ("data", "model"),
) -> Mesh:
    """
    Creates a 2D device mesh from available JAX devices.

    mesh_shape: (n_data, n_model) — must multiply to ≤ len(jax.devices())
    axis_names: names for the two mesh axes
    """
    devices = np.array(jax.devices()[:mesh_shape[0] * mesh_shape[1]])
    devices = devices.reshape(mesh_shape)
    return Mesh(devices, axis_names=axis_names)


def pjit_matmul(
    x: jax.Array,
    w: jax.Array,
    mesh: Mesh,
    x_spec: PartitionSpec = PartitionSpec("data", None),
    w_spec: PartitionSpec = PartitionSpec(None, "model"),
    out_spec: PartitionSpec = PartitionSpec("data", "model"),
) -> jax.Array:
    """
    Distributed GEMM: computes x @ w on the given 2D mesh.

    Sharding strategy:
      x is sharded along 'data' (rows) — each device holds a horizontal slice.
      w is sharded along 'model' (cols) — each device holds a vertical slice.
      output is sharded along both axes — fully distributed result.

    Args:
      x:       (M, K) input matrix
      w:       (K, N) weight matrix
      mesh:    2D device mesh with axes ('data', 'model')
      x_spec:  PartitionSpec for x
      w_spec:  PartitionSpec for w
      out_spec: PartitionSpec for output

    Returns:
      (M, N) result, sharded according to out_spec.
    """
    x_sharding   = NamedSharding(mesh, x_spec)
    w_sharding   = NamedSharding(mesh, w_spec)
    out_sharding = NamedSharding(mesh, out_spec)

    # Distribute inputs across devices
    x_dist = jax.device_put(x, x_sharding)
    w_dist = jax.device_put(w, w_sharding)

    # Compile and run the matmul with explicit sharding constraints
    @jax.jit
    def _matmul(a, b):
        return jnp.matmul(a, b)

    with mesh:
        result = jax.jit(
            _matmul,
            in_shardings=(x_sharding, w_sharding),
            out_shardings=out_sharding,
        )(x_dist, w_dist)

    return result
