"""
Triton FlashAttention kernel — fused attention with online softmax and SRAM tiling.

Key property: never materialises the full N×N attention matrix.
  Each thread block processes a tile of Q and accumulates the output
  by streaming through K/V tiles, maintaining a running (max, sum) for
  numerically stable online softmax.

Shapes: Q, K, V — (batch, n_heads, seq_len, head_dim)
  head_dim must be a power of 2: 16, 32, 64, 128.

Two entry points:
  attention_ref(q, k, v)    — pure PyTorch reference; runs on CPU or GPU
  flash_attention(q, k, v)  — Triton kernel; requires CUDA

verify_correctness() MUST pass before any benchmark call.
"""
import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PyTorch reference (CPU + GPU)
# ---------------------------------------------------------------------------

def attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Scaled dot-product attention reference.
    O = softmax(QK^T / sqrt(head_dim)) @ V
    Input/output shape: (batch, n_heads, seq_len, head_dim)
    """
    return F.scaled_dot_product_attention(q, k, v)


# ---------------------------------------------------------------------------
# Triton kernel (CUDA only)
# ---------------------------------------------------------------------------

def _check_cuda(fn_name: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{fn_name} requires a CUDA GPU. "
            "Run on a cloud GPU instance (RunPod / AWS EC2)."
        )


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Fused FlashAttention via Triton kernel.

    Implements the FlashAttention-2 tiling scheme:
      For each Q tile (BLOCK_M rows):
        Initialise: O = 0, m = -inf, l = 0
        For each K/V tile (BLOCK_N cols):
          S   = Q_tile @ K_tile^T * scale          # (BLOCK_M, BLOCK_N)
          m'  = max(m, rowmax(S))
          P   = exp(S - m')                        # numerically stable
          l'  = exp(m - m') * l + rowsum(P)
          O   = exp(m - m') * O + P @ V_tile
          m, l = m', l'
        O = O / l                                  # normalise

    Requires: head_dim ∈ {16, 32, 64, 128}, inputs contiguous, on CUDA.
    """
    _check_cuda("flash_attention")
    import triton
    import triton.language as tl

    batch, n_heads, seq_len, head_dim = q.shape
    assert head_dim in (16, 32, 64, 128), f"head_dim must be power-of-2 in [16,128], got {head_dim}"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    scale = 1.0 / math.sqrt(head_dim)
    o = torch.zeros_like(q)

    BLOCK_M = 64
    BLOCK_N = 64

    @triton.jit
    def _flash_attn_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        seq_len, head_dim,
        scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        # pid maps to (batch * n_heads, start_m)
        batch_head = tl.program_id(0)
        start_m    = tl.program_id(1) * BLOCK_M

        # Offsets for this Q tile
        offs_m  = start_m  + tl.arange(0, BLOCK_M)
        offs_k  = tl.arange(0, HEAD_DIM)

        # Base pointers for this batch/head
        Q_base = Q_ptr + batch_head * (stride_qb if stride_qh == 0 else stride_qh)
        K_base = K_ptr + batch_head * (stride_kb if stride_kh == 0 else stride_kh)
        V_base = V_ptr + batch_head * (stride_vb if stride_vh == 0 else stride_vh)
        O_base = O_ptr + batch_head * (stride_ob if stride_oh == 0 else stride_oh)

        # Load Q tile: (BLOCK_M, HEAD_DIM)
        q_mask = offs_m[:, None] < seq_len
        q = tl.load(
            Q_base + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
            mask=q_mask, other=0.0,
        ).to(tl.float32) * scale

        # Running statistics
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,),              dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM),     dtype=tl.float32)

        # Stream K/V tiles
        offs_n = tl.arange(0, BLOCK_N)
        for start_n in range(0, seq_len, BLOCK_N):
            n_mask = (start_n + offs_n) < seq_len

            k = tl.load(
                K_base + (start_n + offs_n)[None, :] * stride_kn + offs_k[:, None] * stride_kk,
                mask=n_mask[None, :], other=0.0,
            ).to(tl.float32)

            # S = Q @ K^T  → (BLOCK_M, BLOCK_N)
            s = tl.dot(q, k)

            # Mask out-of-bounds columns
            s = tl.where(n_mask[None, :], s, float("-inf"))

            # Online softmax update
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            p     = tl.exp(s - m_new[:, None])
            l_new = tl.exp(m_i - m_new) * l_i + tl.sum(p, axis=1)

            v = tl.load(
                V_base + (start_n + offs_n)[:, None] * stride_vn + offs_k[None, :] * stride_vk,
                mask=n_mask[:, None], other=0.0,
            ).to(tl.float32)

            acc = tl.exp(m_i - m_new)[:, None] * acc + tl.dot(p.to(v.dtype), v)
            m_i = m_new
            l_i = l_new

        # Normalise and store
        acc = acc / l_i[:, None]
        tl.store(
            O_base + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok,
            acc.to(q.dtype),
            mask=q_mask,
        )

    # Flatten batch × heads into one grid dimension
    grid = (batch * n_heads, triton.cdiv(seq_len, BLOCK_M))

    # Strides — q is (batch, n_heads, seq_len, head_dim) contiguous
    sb, sh, sm, sk = q.stride()

    _flash_attn_kernel[grid](
        q, k, v, o,
        sb, sh, sm, sk,
        *k.stride(),
        *v.stride(),
        *o.stride(),
        seq_len, head_dim, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=head_dim,
    )
    return o


# ---------------------------------------------------------------------------
# Correctness verification (must pass before benchmarking)
# ---------------------------------------------------------------------------

def verify_correctness(
    seq_len: int = 512,
    head_dim: int = 64,
    n_heads: int = 4,
    batch: int = 2,
    dtype: torch.dtype = torch.float32,
) -> bool:
    """
    Verifies Triton kernel against attention_ref.
    Returns True on pass, raises AssertionError on mismatch.
    Requires CUDA.
    """
    _check_cuda("verify_correctness")
    q = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)

    ref    = attention_ref(q, k, v)
    actual = flash_attention(q, k, v)

    atol = 1e-1 if dtype in (torch.float16, torch.bfloat16) else 1e-2
    if not torch.allclose(actual, ref, atol=atol):
        max_diff = (actual - ref).abs().max().item()
        raise AssertionError(
            f"verify_correctness FAILED: seq={seq_len}, dtype={dtype}, max_diff={max_diff:.2e}"
        )
    return True


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(
    seq_lens: list[int] | None = None,
    head_dim: int = 64,
    n_heads: int = 4,
    batch: int = 2,
    dtype: torch.dtype = torch.float32,
) -> list[dict]:
    """
    Benchmarks FlashAttention kernel vs reference.
    verify_correctness() is called first — raises if kernel is wrong.
    Returns list of dicts matching roofline dashboard schema.
    Requires CUDA.
    """
    _check_cuda("benchmark")
    import triton.testing

    if seq_lens is None:
        seq_lens = [128, 256, 512, 1024]

    verify_correctness(seq_len=seq_lens[0], head_dim=head_dim,
                       n_heads=n_heads, batch=batch, dtype=dtype)

    from profiler.hardware_params import HARDWARE
    gpu_name = torch.cuda.get_device_name(0).lower()
    if "a100" in gpu_name:
        hw = HARDWARE["a100_80gb"]
        peak_tflops = hw["peak_tflops_bf16"] if dtype == torch.bfloat16 else hw["peak_tflops_fp32"]
        peak_bw     = hw["memory_bandwidth_gbs"]
    else:
        peak_tflops = 20.0
        peak_bw     = 500.0

    results = []
    for seq_len in seq_lens:
        q = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)

        # FLOPs: 2 * batch * n_heads * seq^2 * head_dim (QK^T) + same for PV
        flops    = 4 * batch * n_heads * seq_len * seq_len * head_dim
        # Bytes: Q+K+V read, O written (no N×N matrix stored — that's the point)
        bytes_rw = 4 * batch * n_heads * seq_len * head_dim * q.element_size()

        ms = triton.testing.do_bench(
            lambda: flash_attention(q, k, v),
            warmup=25, rep=100,
        )

        tflops        = (flops * 1e-12) / (ms * 1e-3)
        bandwidth_gbs = (bytes_rw * 1e-9) / (ms * 1e-3)
        pct_of_peak   = min(tflops / peak_tflops * 100, bandwidth_gbs / peak_bw * 100)

        results.append({
            "kernel":         "flash_attention",
            "config":         {"seq_len": seq_len, "head_dim": head_dim,
                               "n_heads": n_heads, "dtype": str(dtype)},
            "tflops":         tflops,
            "bandwidth_gb_s": bandwidth_gbs,
            "latency_ms":     ms,
            "pct_of_peak":    pct_of_peak,
        })

    return results
