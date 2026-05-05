"""
Tests for the PTX/occupancy analysis pipeline.

Tests 1–4 — CPU (M2 local): pure-function parsers and occupancy formula.
Tests 5–6 — GPU + cuobjdump (cloud): real cubin analysis.

All parsers take text/structured input — testable without any GPU or tooling.
"""
import pytest
import torch

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
NCU  = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA GPU + ncu (sudo ncu or perf_event_paranoid=1)",
)


# ---------------------------------------------------------------------------
# Test 1 — Tracer bullet (CPU): theoretical_occupancy formula is correct
#
# occupancy = min(warp_limit,
#                 smem_per_sm // smem_per_block,
#                 regs_per_sm // (regs_per_thread * 32)) / warp_limit
# ---------------------------------------------------------------------------

def test_theoretical_occupancy_formula():
    from kernels.ptx_analysis.occupancy import theoretical_occupancy
    from profiler.hardware_params import HARDWARE

    hw = HARDWARE["a100_80gb"]

    # Kernel uses 32 registers, 4096 bytes SMEM, 1024 threads/block (32 warps)
    regs_per_thread   = 32
    smem_per_block    = 4096
    threads_per_block = 1024
    warps_per_block   = threads_per_block // 32

    warp_limit  = hw["warps_per_sm"]              # 64
    smem_limit  = hw["smem_per_sm_kb"] * 1024     # 192 * 1024 bytes
    regs_limit  = hw["regs_per_sm"]               # 65536

    expected_blocks_by_smem  = smem_limit  // smem_per_block        # 48
    expected_blocks_by_regs  = regs_limit  // (regs_per_thread * threads_per_block)  # 2
    expected_warps_from_regs = expected_blocks_by_regs * warps_per_block  # 64
    expected_warps_from_smem = expected_blocks_by_smem * warps_per_block  # 1536 — clamp to warp_limit

    expected_active_warps = min(
        warp_limit,
        min(expected_blocks_by_smem, expected_blocks_by_regs) * warps_per_block,
    )
    expected_occ = expected_active_warps / warp_limit

    actual = theoretical_occupancy(
        regs_per_thread=regs_per_thread,
        smem_per_block=smem_per_block,
        threads_per_block=threads_per_block,
        hw=hw,
    )

    assert abs(actual - expected_occ) < 1e-6, (
        f"Occupancy formula wrong: expected {expected_occ:.4f}, got {actual:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2 — CPU: parse_ptxas_log extracts registers + SMEM correctly
# ---------------------------------------------------------------------------

SAMPLE_PTXAS_LOG = """\
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_ZN5axon...' for 'sm_80'
ptxas info    : Function properties for _ZN5axon...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 32 registers, 8192 bytes smem, 400 bytes cmem[0]
"""

def test_parse_ptxas_log_extracts_registers_and_smem():
    from kernels.ptx_analysis.extract_ptx import parse_ptxas_log

    result = parse_ptxas_log(SAMPLE_PTXAS_LOG)

    assert result["registers"] == 32,   f"Expected 32 registers, got {result['registers']}"
    assert result["smem_bytes"] == 8192, f"Expected 8192 bytes SMEM, got {result['smem_bytes']}"


# ---------------------------------------------------------------------------
# Test 3 — CPU: parse_ncu_csv extracts dram_bytes + sm_throughput
# ---------------------------------------------------------------------------

SAMPLE_NCU_CSV = """\
"ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"
"0","12345","python","localhost","swiglu_forward_kernel","2024-01-01","1","7","Memory Workload Analysis","dram__bytes.sum","byte","1048576"
"0","12345","python","localhost","swiglu_forward_kernel","2024-01-01","1","7","SM Throughput","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","78.5"
"0","12345","python","localhost","swiglu_forward_kernel","2024-01-01","1","7","Warp Occupancy","sm__warps_active.avg.pct_of_peak_sustained_active","%","62.3"
"""

def test_parse_ncu_csv_extracts_metrics():
    from kernels.ptx_analysis.ncu_parser import parse_ncu_csv

    result = parse_ncu_csv(SAMPLE_NCU_CSV)

    assert result["dram_bytes"]        == 1048576,  f"dram_bytes wrong: {result['dram_bytes']}"
    assert abs(result["sm_throughput_pct"] - 78.5) < 0.01, f"sm_throughput_pct wrong: {result['sm_throughput_pct']}"
    assert abs(result["warps_active_pct"]  - 62.3) < 0.01, f"warps_active_pct wrong: {result['warps_active_pct']}"


# ---------------------------------------------------------------------------
# Test 4 — CPU: parse_sass_ld_st_ratio counts LD vs non-LD instructions
# ---------------------------------------------------------------------------

SAMPLE_SASS = """\
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   LDG.E R2, [R4] ;
        /*0020*/                   LDG.E R3, [R6] ;
        /*0030*/                   FMUL R4, R2, R3 ;
        /*0040*/                   FADD R5, R4, R1 ;
        /*0050*/                   STG.E [R8], R5 ;
        /*0060*/                   EXIT ;
"""

def test_parse_sass_ld_st_ratio():
    from kernels.ptx_analysis.extract_ptx import parse_sass_ld_st_ratio

    ratio = parse_sass_ld_st_ratio(SAMPLE_SASS)

    # 2 LDG + 1 STG = 3 memory instructions out of 7 total (excluding EXIT)
    assert 0.0 < ratio < 1.0, f"LD/ST ratio out of range: {ratio}"
    assert abs(ratio - 3 / 6) < 0.01, f"LD/ST ratio wrong: expected 0.5, got {ratio:.3f}"


# ---------------------------------------------------------------------------
# Test 5 — GPU: find_triton_cubins returns non-empty list after kernel run
# ---------------------------------------------------------------------------

@CUDA
def test_find_triton_cubins_after_kernel_run():
    from kernels.ptx_analysis.extract_ptx import find_triton_cubins
    from kernels.swiglu import swiglu_triton
    import torch

    # Run kernel to populate ~/.triton/cache/
    gate = torch.randn(4, 1024, device="cuda")
    up   = torch.randn(4, 1024, device="cuda")
    _ = swiglu_triton(gate, up)
    torch.cuda.synchronize()

    cubins = find_triton_cubins()

    assert len(cubins) > 0, "No .cubin files found in ~/.triton/cache/ after kernel run"
    assert all(str(c).endswith(".cubin") for c in cubins), "Non-.cubin files returned"


# ---------------------------------------------------------------------------
# Test 6 — GPU + cuobjdump: analyze_cubin returns register count
# ---------------------------------------------------------------------------

@CUDA
def test_analyze_cubin_returns_register_count():
    from kernels.ptx_analysis.extract_ptx import find_triton_cubins, analyze_cubin
    from kernels.swiglu import swiglu_triton
    import torch

    gate = torch.randn(4, 1024, device="cuda")
    up   = torch.randn(4, 1024, device="cuda")
    _ = swiglu_triton(gate, up)
    torch.cuda.synchronize()

    cubins = find_triton_cubins()
    assert cubins, "No cubins found"

    result = analyze_cubin(cubins[0])

    assert "registers" in result, "analyze_cubin missing 'registers' key"
    assert result["registers"] > 0, f"Register count should be positive, got {result['registers']}"
    assert "smem_bytes" in result
    assert "ld_st_ratio" in result
