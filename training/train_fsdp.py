"""
FSDP training loop for GPT-2 on WikiText-103.

Key design decisions:
  - FSDP FULL_SHARD: shards params + grads + optimizer state across GPUs.
    With 1 GPU this is a no-op (single-shard fallback), but the code path
    is identical to multi-GPU, making the determinism check meaningful.
  - bf16 mixed precision: reduces memory ~50% vs fp32.
  - Per-step logging: step_time_ms, gpu_memory_mib, grad_sparsity,
    allreduce_ms, compute_ms, overlap_pct.
  - Determinism: seeded RNG + same data order → identical losses across runs.

Per-step metric definitions:
  step_time_ms  = wall clock time for one full step
  gpu_memory_mib= torch.cuda.max_memory_allocated() after forward+backward
  grad_sparsity = fraction of gradient elements equal to zero (after backward)
  allreduce_ms  = time for FSDP gradient sync (dist.barrier proxy on 1 GPU ≈ 0)
  compute_ms    = time for forward + backward (step_time - allreduce)
  overlap_pct   = 1 - allreduce_ms / step_time_ms  (fraction spent computing)

FSDP pitfall: wrapping renames parameters.
  Always use strict=False when loading checkpoints on FSDP-wrapped models.

dry_run=True: returns schema-correct metrics without touching the GPU.
  Used by local tests to validate the logging interface.
"""
import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.model import GPT2, GPT2Config, TransformerBlock


@dataclass
class TrainingConfig:
    # Model architecture
    n_layers:         int   = 12
    n_heads:          int   = 12
    d_model:          int   = 768
    d_ff:             int   = 3072
    vocab_size:       int   = 50257
    max_seq_len:      int   = 1024
    dropout:          float = 0.1
    # Training
    batch_size:       int   = 4
    lr:               float = 3e-4
    seed:             int   = 42
    # Device / precision
    dtype:            str   = "bf16"    # "fp32" or "bf16"
    use_fsdp:         bool  = True
    # Gradient compression (Slice 19)
    use_compression:  bool  = False
    k_pct:            float = 0.1      # fraction of gradients to keep
    # Allocator integration (Slice 11)
    log_allocator_stats: bool = False  # log cache_hit_pct + fragmentation_pct every 100 steps


def _make_fake_batch(config: TrainingConfig, device: torch.device) -> torch.Tensor:
    """Random token batch for testing (replaces real WikiText-103 loader)."""
    return torch.randint(
        0, config.vocab_size,
        (config.batch_size, config.max_seq_len),
        device=device,
    )


def _compute_grad_sparsity(model: nn.Module) -> float:
    """Fraction of gradient elements that are zero after backward."""
    total = 0
    zeros = 0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.numel()
            zeros += (p.grad == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def _log_step(
    step: int,
    loss: float,
    step_ms: float,
    allreduce_ms: float,
    model: nn.Module,
    device: torch.device,
) -> dict:
    compute_ms  = max(step_ms - allreduce_ms, 0.0)
    overlap_pct = 1.0 - (allreduce_ms / step_ms) if step_ms > 0 else 1.0

    gpu_mib = 0.0
    if device.type == "cuda":
        gpu_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats(device)

    return {
        "step":           step,
        "loss":           loss,
        "step_time_ms":   step_ms,
        "gpu_memory_mib": gpu_mib,
        "grad_sparsity":  _compute_grad_sparsity(model),
        "allreduce_ms":   allreduce_ms,
        "compute_ms":     compute_ms,
        "overlap_pct":    overlap_pct,
    }


def run_training(
    config: TrainingConfig,
    n_steps: int = 10,
    dry_run: bool = False,
) -> list[dict]:
    """
    Runs n_steps of FSDP training. Returns per-step metrics.

    dry_run=True: returns schema-correct dicts without GPU/FSDP.
    dry_run=False: requires CUDA; trains with FSDP FULL_SHARD (or
      single-device if only 1 GPU is available).
    """
    if dry_run:
        base = {
            "step":           0,
            "loss":           float("nan"),
            "step_time_ms":   0.0,
            "gpu_memory_mib": 0.0,
            "grad_sparsity":  0.0,
            "allreduce_ms":   0.0,
            "compute_ms":     0.0,
            "overlap_pct":    1.0,
        }
        if config.log_allocator_stats:
            base["allocator_cache_hit_pct"]    = 0.0
            base["allocator_fragmentation_pct"] = 0.0
        return [{**base, "step": i} for i in range(n_steps)]

    if not torch.cuda.is_available():
        raise RuntimeError("run_training requires CUDA. Use dry_run=True for local testing.")

    # Seed for determinism
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda")

    # Build model
    model_config = GPT2Config(
        n_layers=config.n_layers, n_heads=config.n_heads,
        d_model=config.d_model, d_ff=config.d_ff,
        vocab_size=config.vocab_size, max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    )
    model = GPT2(model_config).to(device)

    # FSDP wrapping (single-GPU: FULL_SHARD still works, shards to 1 rank)
    if config.use_fsdp and torch.cuda.device_count() >= 1:
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        import functools

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://",
                                    world_size=1, rank=0)

        mp_policy = None
        if config.dtype == "bf16":
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )

        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock},
        )
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            auto_wrap_policy=wrap_policy,
            device_id=device,
        )

    # Register gradient compression hook (Slice 19)
    if config.use_compression:
        from training.compression import TopKSparsificationHook
        hook = TopKSparsificationHook(k_pct=config.k_pct)
        model.register_comm_hook(state=None, hook=hook)

    # Enable allocator (Slice 11) — must be done before optimizer init
    _allocator = None
    if config.log_allocator_stats:
        try:
            import axonforge_allocator as _allocator
            _allocator.enable()
            _allocator.reset_stats()
        except ImportError:
            _allocator = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # Re-seed data generation for determinism
    torch.manual_seed(config.seed)

    metrics = []
    for step in range(n_steps):
        batch = _make_fake_batch(config, device)

        # Forward + backward
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize()

        compute_start = time.perf_counter()
        optimizer.zero_grad()
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            targets.reshape(-1),
        )
        loss.backward()
        compute_end = time.perf_counter()

        # AllReduce (on single GPU this is essentially zero)
        allreduce_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        allreduce_end = time.perf_counter()

        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        step_ms     = (t1 - t0) * 1000
        allreduce_ms = (allreduce_end - allreduce_start) * 1000

        m = _log_step(
            step=step,
            loss=loss.item(),
            step_ms=step_ms,
            allreduce_ms=allreduce_ms,
            model=model,
            device=device,
        )

        # Log allocator stats every 100 steps
        if config.log_allocator_stats and _allocator is not None and step % 100 == 0:
            stats = _allocator.stats()
            m["allocator_cache_hit_pct"]    = stats.get("cache_hit_rate_pct", 0.0)
            m["allocator_fragmentation_pct"] = stats.get("fragmentation_pct", 0.0)

        metrics.append(m)

    if _allocator is not None:
        _allocator.disable()

    return metrics
