"""
nanoGPT Phase 2: Low-Rank Adaptation (LoRA) Module
Implements parameter-efficient fine-tuning (Hu et al., 2021) for transformer layers.
Adheres strictly to core-code-standard.md (STD-COD-001 through STD-COD-009).
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a standard linear layer with trainable low-rank decomposition matrices."""

    def __init__(self, base_linear: nn.Linear, r: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.05):
        super().__init__()
        self.base_linear = base_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Freeze original linear weights
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        # Low-rank adapter matrices
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Initialize A with Kaiming uniform and B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def apply_lora_to_model(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> None:
    """Recursively replaces attention projection layers with LoRALinear (STD-COD-007.2)."""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear) and name in ("q_proj", "v_proj", "w_gate", "w_up"):
            setattr(model, name, LoRALinear(module, r=rank, lora_alpha=alpha))
        else:
            apply_lora_to_model(module, rank, alpha)


def mark_only_lora_as_trainable(model: nn.Module) -> tuple[int, int]:
    """Freezes all base weights and keeps only LoRA parameters trainable (STD-COD-007.3)."""
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "lora_" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
    return trainable_params, total_params


def extract_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extracts only LoRA adapter parameters for lightweight storage (STD-COD-007.2)."""
    return {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
