# AxonForge

A hardware-aware ML compiler, runtime, and serving system spanning the complete AI acceleration stack.

> Dashboard, benchmarks, and component documentation coming as each module ships.

## Stack

```
[PyTorch / JAX Model]
        │
        ▼
[torch.fx Compiler Passes]     — op fusion, DCE, layout reordering
        │
        ▼
[C++/CUDA Custom Op Extension] — production kernel delivery (ATen, pybind11)
        │
        ▼
[CUDA Caching Allocator]       — free-list memory pool, PyTorch allocator API
        │
        ▼
[Triton Kernels]               — research kernel path
        │
        ▼
[PTX / Occupancy Analysis]     — register pressure, SMEM, roofline placement
        │
        ▼
[Pipelined Systolic Array Sim] — Python golden + SystemVerilog RTL co-simulation
        │
        ▼
[FSDP Distributed Training]    — gradient compression, scaling sweep
        │
        ▼
[TorchScript Inference Server] — export, freeze, latency/throughput Pareto
        │
        ▼
[Roofline Dashboard]           — unified perf visualization (5 panels)
```

## Components

| # | Component | Tracks | Status |
|---|-----------|--------|--------|
| 1 | Compiler Passes (`compiler/`) | ML Systems & Compilers | 🔲 |
| 2 | C++/CUDA Extension (`csrc/`) | Systems Software | 🔲 |
| 3 | CUDA Caching Allocator (`csrc/allocator/`) | Runtime Systems | 🔲 |
| 4 | JAX Backend (`jax_backend/`) | ML Systems & Compilers | 🔲 |
| 5 | Triton Kernels (`kernels/`) | ML Systems & Compilers | 🔲 |
| 6 | PTX & Occupancy Analysis (`kernels/ptx_analysis/`) | Systems Software | 🔲 |
| 7 | Systolic Array + Power Model (`hardware/`) | Silicon Innovation | 🔲 |
| 8 | Randomized RTL Testbench (`hardware/rtl/`) | Silicon Innovation | 🔲 |
| 9 | Distributed Training (`training/`) | ML Systems | 🔲 |
| 10 | Inference Server (`serve/`) | Inferentia / Inference | 🔲 |
| 11 | Roofline Dashboard (`profiler/`) | All tracks | 🔲 |

## Running Tests

```bash
# Compiler passes (CPU, no GPU required)
pytest compiler/tests/ -v

# Systolic array (CPU, no GPU required)
pytest hardware/tests/ -v

# C++/CUDA extension (requires NVIDIA GPU)
cd csrc && python setup.py install && pytest tests/ -v

# CUDA allocator (requires NVIDIA GPU)
cd csrc/allocator && python setup_allocator.py install && pytest tests/ -v
```

## Hardware Targets

Benchmarks run on NVIDIA A100 (80GB) and RTX 3090. Hardware parameters in `profiler/hardware_params.py`.
