"""
nanoGPT Phase 3: Production Inference Transformer with KV-Cache Support
Implements stateful KV-caching for O(1) step generation, RoPE, RMSNorm, and SDPA.
Adheres strictly to core-code-standard.md (STD-COD-001 through STD-COD-009).
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from kv_cache import LayerKVCache, ModelKVCache


@dataclass(frozen=True)
class TransformerConfig:
    """Centralized immutable model configuration (STD-COD-002, STD-COD-007.5)."""
    block_size: int = 64
    vocab_size: int = 50304
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    dimension_indices = torch.arange(0, head_dim, 2)[: (head_dim // 2)].float()
    frequencies = 1.0 / (theta ** (dimension_indices / head_dim))
    time_steps = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(time_steps, frequencies)
    angles = torch.cat([angles, angles], dim=-1)
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(2)  # [1, max_seq_len, 1, head_dim]
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(2)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class CausalSelfAttention(nn.Module):
    """Multi-head attention with optional KV-cache acceleration (STD-COD-007.2)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout_rate = config.dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                kv_cache: LayerKVCache = None, start_pos: int = 0) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)

        q, k = apply_rotary_emb(q, k, cos, sin)

        # Transpose to [batch, head, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.update(start_pos, k, v)
            is_causal = (seq_len > 1)
        else:
            is_causal = True

        dropout_p = self.dropout_rate if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        return self.resid_dropout(self.out_proj(y))


class SwiGLU(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                kv_cache: LayerKVCache = None, start_pos: int = 0) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, kv_cache=kv_cache, start_pos=start_pos)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ModernTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_freqs_cis(head_dim, 512, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None,
                model_cache: ModelKVCache = None, start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = idx.shape
        x = self.drop(self.tok_emb(idx))
        cos = self.cos[:, start_pos : start_pos + seq_len, :, :].to(idx.device)
        sin = self.sin[:, start_pos : start_pos + seq_len, :, :].to(idx.device)

        for i, block in enumerate(self.blocks):
            layer_cache = model_cache.get_layer(i) if model_cache is not None else None
            x = block(x, cos, sin, kv_cache=layer_cache, start_pos=start_pos)
        
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate_cached(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 50):
        """Generates tokens using stateful KV-caching for O(1) step latency."""
        batch_size, prompt_len = idx.shape
        head_dim = self.config.n_embd // self.config.n_head
        max_seq_len = prompt_len + max_new_tokens

        cache = ModelKVCache(
            n_layer=self.config.n_layer,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            n_head=self.config.n_head,
            head_dim=head_dim,
            device=idx.device,
            dtype=torch.float32
        )

        # 1. Prefill stage: Process initial prompt
        logits, _ = self(idx, model_cache=cache, start_pos=0)
        cur_token = self._sample_next_token(logits[:, -1, :], temperature, top_k)
        tokens = [cur_token]

        # 2. Decode stage: Process one token at a time with O(1) step complexity
        for pos in range(prompt_len, max_seq_len - 1):
            logits, _ = self(cur_token, model_cache=cache, start_pos=pos)
            cur_token = self._sample_next_token(logits[:, -1, :], temperature, top_k)
            tokens.append(cur_token)

        return torch.cat([idx, torch.cat(tokens, dim=1)], dim=1)

    def _sample_next_token(self, logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
        logits = logits / max(temperature, 1e-5)
        if top_k is not None and top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = -float('Inf')
        probabilities = F.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1)
