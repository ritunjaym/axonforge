"""
Inference latency/throughput benchmark for GPT-2 inference server.

Sweeps over (batch_size, seq_len) configs. Metrics per config:
  latency_p50_ms      — median end-to-end latency
  latency_p99_ms      — 99th percentile tail latency
  throughput_tokens_s — batch_size * seq_len / (p50_latency_s)
  gpu_memory_mb       — peak GPU memory during inference
  vs_training_memory  — inference peak / training peak ratio

Key insight: inference memory ≈ 40–60% of training (no activations for
backward, no optimizer state) → motivates different pool sizing on Inferentia.

dry_run=True: returns schema-correct dicts without GPU.
"""
import math
import time
import numpy as np
import torch

from serve.export import export_for_inference
from training.model import GPT2, GPT2Config
from training.train_fsdp import TrainingConfig, run_training


# Small model config used for benchmarking to keep GPU memory reasonable
_BENCH_CONFIG = GPT2Config(
    n_layers=2, d_model=128, n_heads=4, d_ff=512,
    vocab_size=512, max_seq_len=1024, dropout=0.0,
)


def _training_peak_memory_mb(seq_len: int, batch_size: int = 1) -> float:
    """Estimate training peak memory via a single training step."""
    config = TrainingConfig(
        n_layers=_BENCH_CONFIG.n_layers,
        n_heads=_BENCH_CONFIG.n_heads,
        d_model=_BENCH_CONFIG.d_model,
        d_ff=_BENCH_CONFIG.d_ff,
        vocab_size=_BENCH_CONFIG.vocab_size,
        max_seq_len=seq_len,
        batch_size=batch_size,
        use_fsdp=False,
    )
    metrics = run_training(config, n_steps=1, dry_run=False)
    return metrics[0]["gpu_memory_mib"]


def run_benchmark(
    batch_sizes: list[int] = (1, 2, 4, 8, 16, 32),
    seq_lens: list[int] = (128, 256, 512, 1024),
    n_warmup: int = 3,
    n_iters: int = 20,
    dry_run: bool = False,
) -> list[dict]:
    """
    Benchmark inference across all (batch_size, seq_len) combinations.

    dry_run=True: returns schema-correct dicts without GPU.
    dry_run=False: requires CUDA.
    """
    if dry_run:
        return [
            {
                "batch_size":          bs,
                "seq_len":             sl,
                "latency_p50_ms":      0.0,
                "latency_p99_ms":      0.0,
                "throughput_tokens_s": 0.0,
                "gpu_memory_mb":       0.0,
                "vs_training_memory":  0.0,
            }
            for bs in batch_sizes
            for sl in seq_lens
        ]

    if not torch.cuda.is_available():
        raise RuntimeError("run_benchmark requires CUDA. Use dry_run=True for local testing.")

    device = torch.device("cuda")

    model = GPT2(_BENCH_CONFIG).eval().to(device)
    example_input = torch.randint(0, _BENCH_CONFIG.vocab_size, (1, 128), device=device)
    exported = export_for_inference(model, (example_input,))
    del model

    results = []
    for seq_len in seq_lens:
        if seq_len > _BENCH_CONFIG.max_seq_len:
            continue

        # Training baseline memory for vs_training_memory ratio
        try:
            training_peak_mb = _training_peak_memory_mb(seq_len, batch_size=1)
        except Exception:
            training_peak_mb = float("nan")

        for batch_size in batch_sizes:
            tokens = torch.randint(
                0, _BENCH_CONFIG.vocab_size,
                (batch_size, seq_len),
                device=device,
            )

            # Warmup
            for _ in range(n_warmup):
                with torch.no_grad():
                    _ = exported(tokens)
                torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats(device)

            # Timed iterations
            latencies = []
            for _ in range(n_iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = exported(tokens)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

            gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats(device)

            latencies_arr = np.array(latencies)
            p50_ms = float(np.percentile(latencies_arr, 50))
            p99_ms = float(np.percentile(latencies_arr, 99))
            throughput = batch_size * seq_len / (p50_ms / 1000.0)

            vs_training = (
                gpu_mb / training_peak_mb
                if math.isfinite(training_peak_mb) and training_peak_mb > 0
                else float("nan")
            )

            results.append({
                "batch_size":          batch_size,
                "seq_len":             seq_len,
                "latency_p50_ms":      p50_ms,
                "latency_p99_ms":      p99_ms,
                "throughput_tokens_s": throughput,
                "gpu_memory_mb":       gpu_mb,
                "vs_training_memory":  vs_training,
            })

    return results
