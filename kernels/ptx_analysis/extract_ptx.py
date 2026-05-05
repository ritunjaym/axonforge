"""
PTX/SASS extraction and analysis tools.

parse_ptxas_log(text)      — parses --ptxas-options=-v output (registers, SMEM)
parse_sass_ld_st_ratio(text) — computes memory instruction ratio from cuobjdump SASS
find_triton_cubins()       — finds .cubin files in ~/.triton/cache/
analyze_cubin(path)        — runs cuobjdump on a .cubin, returns metrics dict

Pitfall: cuobjdump needs .cubin not .ptx source.
         find after first kernel run: find ~/.triton/cache -name "*.cubin"
"""
import re
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# ptxas log parser (output of --ptxas-options=-v at compile time)
# ---------------------------------------------------------------------------

_PTXAS_USED_RE = re.compile(
    r"Used\s+(\d+)\s+registers,\s+(\d+)\s+bytes\s+smem"
)

def parse_ptxas_log(log_text: str) -> dict:
    """
    Parses the stderr output produced by --ptxas-options=-v.

    Example line:
      ptxas info    : Used 32 registers, 8192 bytes smem, 400 bytes cmem[0]

    Returns:
      {"registers": int, "smem_bytes": int}
    """
    match = _PTXAS_USED_RE.search(log_text)
    if not match:
        return {"registers": 0, "smem_bytes": 0}
    return {
        "registers":  int(match.group(1)),
        "smem_bytes": int(match.group(2)),
    }


# ---------------------------------------------------------------------------
# SASS LD/ST ratio parser
# ---------------------------------------------------------------------------

_MEM_INSTR_RE = re.compile(r"\b(LDG|LDS|LDL|STG|STS|STL|LD|ST)\b")
_ANY_INSTR_RE = re.compile(r"/\*[0-9a-fA-F]+\*/\s+(\w+)")

def parse_sass_ld_st_ratio(sass_text: str) -> float:
    """
    Computes ratio of memory instructions (LD*/ST*) to total non-EXIT instructions
    from cuobjdump --dump-sass output.

    A high ratio (>0.5) suggests the kernel is memory-bound.
    """
    all_instrs = _ANY_INSTR_RE.findall(sass_text)
    # Exclude control-flow pseudo-instructions
    exec_instrs = [i for i in all_instrs if i not in ("EXIT", "NOP", "BRA", "RET")]
    if not exec_instrs:
        return 0.0
    mem_instrs  = [i for i in exec_instrs if _MEM_INSTR_RE.match(i)]
    return len(mem_instrs) / len(exec_instrs)


# ---------------------------------------------------------------------------
# Triton cubin discovery
# ---------------------------------------------------------------------------

def find_triton_cubins() -> list[Path]:
    """
    Returns all .cubin files in ~/.triton/cache/.
    Run a Triton kernel at least once before calling this.
    """
    cache_dir = Path.home() / ".triton" / "cache"
    if not cache_dir.exists():
        return []
    return list(cache_dir.rglob("*.cubin"))


# ---------------------------------------------------------------------------
# cuobjdump-based cubin analysis (requires CUDA toolkit on PATH)
# ---------------------------------------------------------------------------

_CUOBJDUMP_REG_RE  = re.compile(r"Reg\s+count\s*:\s*(\d+)", re.IGNORECASE)
_CUOBJDUMP_SMEM_RE = re.compile(r"Shared\s+mem(?:ory)?\s+size\s*:\s*(\d+)", re.IGNORECASE)

def analyze_cubin(cubin_path: Path) -> dict:
    """
    Runs cuobjdump on a .cubin file and returns kernel metrics.

    Requires: cuobjdump on PATH (part of CUDA toolkit).
    Pitfall:  cuobjdump needs .cubin not .ptx; find in ~/.triton/cache/.

    Returns:
      {"registers": int, "smem_bytes": int, "ld_st_ratio": float}
    """
    cubin_path = Path(cubin_path)
    if not cubin_path.exists():
        raise FileNotFoundError(f"Cubin not found: {cubin_path}")

    # Get ELF info for register + SMEM counts
    elf_result = subprocess.run(
        ["cuobjdump", "--elf", str(cubin_path)],
        capture_output=True, text=True, timeout=30,
    )
    elf_text = elf_result.stdout + elf_result.stderr

    reg_match  = _CUOBJDUMP_REG_RE.search(elf_text)
    smem_match = _CUOBJDUMP_SMEM_RE.search(elf_text)

    registers  = int(reg_match.group(1))  if reg_match  else 0
    smem_bytes = int(smem_match.group(1)) if smem_match else 0

    # Get SASS for LD/ST ratio
    sass_result = subprocess.run(
        ["cuobjdump", "--dump-sass", str(cubin_path)],
        capture_output=True, text=True, timeout=30,
    )
    ld_st_ratio = parse_sass_ld_st_ratio(sass_result.stdout)

    return {
        "registers":   registers,
        "smem_bytes":  smem_bytes,
        "ld_st_ratio": ld_st_ratio,
    }
