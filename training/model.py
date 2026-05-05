"""
GPT-2 architecture from scratch.

Default config matches GPT-2 124M:
  12 layers, 12 heads, d_model=768, d_ff=3072, vocab_size=50257, ctx=1024.
  Total params: ~124.4M

Used for FSDP training (Slice 18), gradient compression (Slice 19),
and inference server (Slice 21).

FSDP wrapping: each TransformerBlock is a natural wrap unit.
  Use transformer_auto_wrap_policy(TransformerBlock) for FSDP.

Pitfall (FSDP): wrapping changes parameter names (e.g., 'layers.0.attn.weight'
  becomes '_fsdp_wrapped_module.layers.0._fsdp_wrapped_module.attn.weight').
  Use strict=False on checkpoint load and remap keys explicitly.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    n_layers:    int = 12
    n_heads:     int = 12
    d_model:     int = 768
    d_ff:        int = 3072
    vocab_size:  int = 50257
    max_seq_len: int = 1024
    dropout:     float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv  = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Causal mask
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).split(C, dim=2)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
                   for t in qkv]

        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.fc1  = nn.Linear(config.d_model, config.d_ff, bias=True)
        self.fc2  = nn.Linear(config.d_ff, config.d_model, bias=True)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block — natural FSDP wrapping unit."""
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.d_model)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop    = nn.Dropout(config.dropout)
        self.layers  = nn.ModuleList([TransformerBlock(config)
                                      for _ in range(config.n_layers)])
        self.ln_f    = nn.LayerNorm(config.d_model)
        self.head    = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: token embedding == output projection (GPT-2 standard)
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max {self.config.max_seq_len}"
        )
        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.head(x)
