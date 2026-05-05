"""
Hardware parameters for roofline analysis and occupancy calculations.
All values are measured or from official specs — no magic numbers elsewhere.
"""

HARDWARE = {
    "a100_80gb": {
        "peak_tflops_bf16":       312,
        "peak_tflops_fp32":       19.5,
        "memory_bandwidth_gbs":   2000,
        "l2_cache_mb":            40,
        "sm_count":               108,
        "warps_per_sm":           64,
        "regs_per_sm":            65536,
        "smem_per_sm_kb":         192,
    },
    "t4": {
        "peak_tflops_fp16":       65,
        "peak_tflops_fp32":       8.1,
        "memory_bandwidth_gbs":   300,
        "l2_cache_mb":            4,
        "sm_count":               40,
        "warps_per_sm":           32,
        "regs_per_sm":            65536,
        "smem_per_sm_kb":         64,
    },
    "rtx_3090": {
        "peak_tflops_fp16":       142,
        "peak_tflops_fp32":       35.6,
        "memory_bandwidth_gbs":   936,
        "l2_cache_mb":            6,
        "sm_count":               82,
        "warps_per_sm":           48,
        "regs_per_sm":            65536,
        "smem_per_sm_kb":         100,
    },
    "simulated_8x8": {
        "peak_tops_int16":        0.000128,
        "memory_bandwidth_gbs":   0.064,
        "array_rows":             8,
        "array_cols":             8,
        "clock_ghz":              1.0,
        "capacitance_per_mac_fF": 10.0,
        "supply_voltage_V":       0.9,
        "static_power_mW":        2.5,
    },
}
