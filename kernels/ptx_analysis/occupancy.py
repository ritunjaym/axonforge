"""
Theoretical GPU occupancy calculator.

Occupancy = active warps / max warps per SM.

Limiters (whichever is smallest):
  1. Register pressure: regs_per_sm // (regs_per_thread * threads_per_block)
     → max blocks constrained by register file
  2. Shared memory: smem_per_sm // smem_per_block
     → max blocks constrained by SMEM capacity
  3. Hardware warp limit: warps_per_sm (absolute ceiling)

Result is compared against ncu-reported achieved occupancy.
Target: theoretical within 5% of achieved (if not, a limiter is hiding something).
"""


def theoretical_occupancy(
    regs_per_thread: int,
    smem_per_block: int,
    threads_per_block: int,
    hw: dict,
) -> float:
    """
    Returns theoretical occupancy as a fraction in [0, 1].

    hw must contain keys from profiler/hardware_params.py:
      warps_per_sm, regs_per_sm, smem_per_sm_kb
    """
    warp_limit       = hw["warps_per_sm"]
    regs_per_sm      = hw["regs_per_sm"]
    smem_per_sm      = hw["smem_per_sm_kb"] * 1024

    warps_per_block  = threads_per_block // 32

    # Max concurrent blocks limited by each resource
    if regs_per_thread > 0 and threads_per_block > 0:
        max_blocks_regs = regs_per_sm // (regs_per_thread * threads_per_block)
    else:
        max_blocks_regs = warp_limit   # unconstrained

    if smem_per_block > 0:
        max_blocks_smem = smem_per_sm // smem_per_block
    else:
        max_blocks_smem = warp_limit   # unconstrained

    # Active warps = blocks × warps_per_block, capped at hardware limit
    max_blocks      = min(max_blocks_regs, max_blocks_smem)
    active_warps    = min(warp_limit, max_blocks * warps_per_block)

    return active_warps / warp_limit


def binding_limiter(
    regs_per_thread: int,
    smem_per_block: int,
    threads_per_block: int,
    hw: dict,
) -> str:
    """
    Returns which resource is the primary occupancy limiter:
    'registers', 'shared_memory', 'warps', or 'balanced'.
    """
    warp_limit      = hw["warps_per_sm"]
    regs_per_sm     = hw["regs_per_sm"]
    smem_per_sm     = hw["smem_per_sm_kb"] * 1024
    threads_per_sm  = warp_limit * 32

    warps_per_block = threads_per_block // 32

    max_blocks_regs = regs_per_sm // (regs_per_thread * threads_per_block) if regs_per_thread > 0 else 999
    max_blocks_smem = smem_per_sm // smem_per_block if smem_per_block > 0 else 999
    max_blocks_hw   = warp_limit  // warps_per_block if warps_per_block > 0 else 999

    bottleneck = min(max_blocks_regs, max_blocks_smem, max_blocks_hw)

    if max_blocks_regs == bottleneck and max_blocks_regs < max_blocks_smem:
        return "registers"
    if max_blocks_smem == bottleneck and max_blocks_smem < max_blocks_regs:
        return "shared_memory"
    if max_blocks_hw == bottleneck:
        return "warps"
    return "balanced"
