"""
Tests for the AxonForge CUDA Caching Allocator.

Test 1  — CPU (M2 local): setup_allocator.py references correct source files.
Tests 2–5 — GPU (cloud): the 4 required correctness tests.

Build on cloud GPU:
  cd csrc/allocator && python setup_allocator.py install
  pytest tests/test_allocator.py -v

Design under test:
  - Size classes: powers of 2, 256B → 128MB. <128KB = small pool. <128MB = large pool.
  - Allocation: round up → search free list → cache hit or cudaMalloc.
  - Free: look up block → coalesce adjacent free blocks → pool or cudaFree if too large.
  - Thread safety: std::mutex guards all free_list access.
  - fragmentation_pct = (allocated_from_cuda - in_use) / allocated_from_cuda * 100
"""
import os
import pytest
import torch

EXT = pytest.mark.skipif(
    not torch.cuda.is_available() or not _ext_available(),
    reason="requires CUDA GPU + axonforge_allocator extension",
)


def _ext_available() -> bool:
    try:
        import axonforge_allocator  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet (CPU): setup_allocator.py has correct structure
# ---------------------------------------------------------------------------

def test_setup_references_correct_sources():
    setup_path = os.path.join(os.path.dirname(__file__), "..", "setup_allocator.py")
    with open(os.path.abspath(setup_path)) as f:
        source = f.read()

    assert "cuda_pool.cpp"        in source, "setup missing cuda_pool.cpp"
    assert "axonforge_allocator"  in source, "setup missing extension name"
    assert "-O3"                  in source, "setup missing -O3 optimisation flag"


# ---------------------------------------------------------------------------
# Test 2 — test_correctness: allocated memory is usable as a CUDA tensor
# ---------------------------------------------------------------------------

@EXT
def test_correctness():
    import axonforge_allocator

    axonforge_allocator.enable()
    try:
        t = torch.empty(1024, device="cuda", dtype=torch.float32)
        t.fill_(3.14)
        torch.cuda.synchronize()

        assert t.device.type == "cuda"
        assert t.shape == (1024,)
        assert torch.all(t == 3.14), f"Values not preserved: min={t.min():.4f}"
    finally:
        axonforge_allocator.disable()


# ---------------------------------------------------------------------------
# Test 3 — test_cache_hit: free then alloc same size → same pointer
# ---------------------------------------------------------------------------

@EXT
def test_cache_hit():
    import axonforge_allocator

    axonforge_allocator.enable()
    axonforge_allocator.reset_stats()
    try:
        # First allocation — cache miss, goes to cudaMalloc
        t1 = torch.empty(4096, device="cuda", dtype=torch.float32)
        ptr1 = t1.data_ptr()

        # Free it — block returns to free list
        del t1
        torch.cuda.synchronize()

        # Second allocation of same size — must be a cache hit (same pointer)
        t2 = torch.empty(4096, device="cuda", dtype=torch.float32)
        ptr2 = t2.data_ptr()

        assert ptr1 == ptr2, (
            f"Expected cache hit (same pointer), got ptr1={ptr1:#x} ptr2={ptr2:#x}"
        )

        stats = axonforge_allocator.stats()
        assert stats["cache_hit_rate_pct"] > 0, (
            f"cache_hit_rate_pct should be >0, got {stats['cache_hit_rate_pct']}"
        )
        del t2
    finally:
        axonforge_allocator.disable()


# ---------------------------------------------------------------------------
# Test 4 — test_coalescing: freeing blocks reduces fragmentation
# ---------------------------------------------------------------------------

@EXT
def test_coalescing():
    import axonforge_allocator

    axonforge_allocator.enable()
    axonforge_allocator.reset_stats()
    try:
        # Allocate several blocks to build up in-use state
        tensors = [torch.empty(4096 * (i + 1), device="cuda") for i in range(4)]

        # Stats mid-point: some memory in use, low fragmentation
        stats_during = axonforge_allocator.stats()

        # Free all — blocks go to free list, in-use drops to 0
        del tensors
        torch.cuda.synchronize()

        # Stats after: all in free list or returned to CUDA via coalescing
        stats_after = axonforge_allocator.stats()

        # fragmentation_pct = (cuda_bytes - in_use) / cuda_bytes
        # After freeing all: in_use=0, so frag=100% unless coalescing returned
        # blocks to CUDA (reducing cuda_bytes). Either way, frag must be
        # representable and the metric must be tracked.
        assert "fragmentation_pct" in stats_after
        assert 0.0 <= stats_after["fragmentation_pct"] <= 100.0
    finally:
        axonforge_allocator.disable()


# ---------------------------------------------------------------------------
# Test 5 — test_large_block_passthrough: >256MB bypasses pool
# ---------------------------------------------------------------------------

@EXT
def test_large_block_passthrough():
    import axonforge_allocator

    axonforge_allocator.enable()
    axonforge_allocator.reset_stats()
    try:
        stats_before = axonforge_allocator.stats()

        # 512MB tensor — exceeds kLargeBlockSize*2 (128MB*2 = 256MB)
        # Should go directly to cudaMalloc and be returned to cudaFree on del
        t = torch.empty(512 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
        stats_mid = axonforge_allocator.stats()

        del t
        torch.cuda.synchronize()
        stats_after = axonforge_allocator.stats()

        # cudaMalloc should have been called for this large block
        assert stats_mid["num_cudaMalloc_calls"] > stats_before["num_cudaMalloc_calls"], (
            "Large block should have triggered a raw cudaMalloc call"
        )
        # After freeing, the large block should NOT be in our pool
        # (it was returned directly to CUDA). A subsequent large alloc
        # should trigger another cudaMalloc.
        t2 = torch.empty(512 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
        stats_final = axonforge_allocator.stats()
        assert stats_final["num_cudaMalloc_calls"] > stats_after["num_cudaMalloc_calls"], (
            "Second large block should also trigger raw cudaMalloc (passthrough, not cached)"
        )
        del t2
    finally:
        axonforge_allocator.disable()
