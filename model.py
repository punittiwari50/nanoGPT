"""
nanoGPT Phase 1: Modern Architectural Evolution
Implements RoPE (Rotary Position Embeddings), RMSNorm, SwiGLU, and PyTorch SDPA.
Adheres strictly to core-code-standard.md (STD-COD-001 through STD-COD-009).
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


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
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precomputes real cosine and sine frequency factors for RoPE (STD-COD-007.2)."""
    dimension_indices = torch.arange(0, head_dim, 2)[: (head_dim // 2)].float()
    frequencies = 1.0 / (theta ** (dimension_indices / head_dim))
    time_steps = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(time_steps, frequencies)
    angles = torch.cat([angles, angles], dim=-1)
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(2)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions for real-valued RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies real-valued trigonometric rotary embedding to query and key tensors."""
    seq_len = xq.size(1)
    cos_sliced = cos[:, :seq_len, :, :].to(xq.device)
    sin_sliced = sin[:, :seq_len, :, :].to(xq.device)
    xq_out = (xq * cos_sliced) + (rotate_half(xq) * sin_sliced)
    xk_out = (xk * cos_sliced) + (rotate_half(xk) * sin_sliced)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and PyTorch SDPA."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout_rate = config.dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)

        # Apply Rotary Position Embeddings
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Transpose for SDPA: [batch, head, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled Dot-Product Attention with hardware fusion
        dropout_p = self.dropout_rate if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)

        # Re-assemble heads and project output
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        return self.resid_dropout(self.out_proj(y))


class SwiGLU(nn.Module):
    """Swish Gated Linear Unit (Shazeer, 2020) for enhanced token capacity."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.dropout(self.w_down(activated))


class TransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer decoder block with residual connections."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ModernTransformer(nn.Module):
    """Next-Gen Generative Transformer with RoPE, RMSNorm, and SwiGLU."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Precompute RoPE rotary frequencies once
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_freqs_cis(head_dim, config.block_size, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying between token embeddings and language model head
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = idx.shape
        assert seq_len <= self.config.block_size, f"Sequence {seq_len} exceeds block size {self.config.block_size}"

        x = self.drop(self.tok_emb(idx))
        cos = self.cos[:, :seq_len, :, :]
        sin = self.sin[:, :seq_len, :, :]

        for block in self.blocks:
            x = block(x, cos, sin)
        
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """Autoregressively sample tokens with temperature and top-k filtering."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float('Inf')

            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
