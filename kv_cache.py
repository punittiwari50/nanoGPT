"""
nanoGPT Phase 3: Stateful Key-Value Cache (KVCache) Module
Enables O(1) step latency during autoregressive token generation.
Adheres strictly to core-code-standard.md (STD-COD-001 through STD-COD-009).
"""

import torch


class LayerKVCache:
    """Stores key and value tensor history for a single transformer layer (STD-COD-007.2)."""

    def __init__(self, max_batch_size: int, max_seq_len: int, n_head: int, head_dim: int, device: str, dtype: torch.dtype):
        self.k_cache = torch.zeros((max_batch_size, n_head, max_seq_len, head_dim), device=device, dtype=dtype)
        self.v_cache = torch.zeros((max_batch_size, n_head, max_seq_len, head_dim), device=device, dtype=dtype)

    def update(self, start_pos: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_head, seq_len, head_dim = k_new.shape
        self.k_cache[:batch_size, :, start_pos : start_pos + seq_len, :] = k_new
        self.v_cache[:batch_size, :, start_pos : start_pos + seq_len, :] = v_new
        return self.k_cache[:batch_size, :, : start_pos + seq_len, :], self.v_cache[:batch_size, :, : start_pos + seq_len, :]


class ModelKVCache:
    """Manages full transformer layer caches across all decoder stages (STD-COD-007.5)."""

    def __init__(self, n_layer: int, max_batch_size: int, max_seq_len: int, n_head: int, head_dim: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.layers = [
            LayerKVCache(max_batch_size, max_seq_len, n_head, head_dim, device, dtype)
            for _ in range(n_layer)
        ]

    def get_layer(self, layer_idx: int) -> LayerKVCache:
        return self.layers[layer_idx]
