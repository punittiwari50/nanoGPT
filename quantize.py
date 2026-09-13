"""
nanoGPT Phase 3: Dynamic INT8 Quantization Utility
Quantizes linear layers to 8-bit integers for reduced edge RAM footprint.
Adheres strictly to core-code-standard.md (STD-COD-001 through STD-COD-009).
"""

import argparse
import os
import torch
import torch.nn as nn

from model import TransformerConfig, ModernTransformer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nanoGPT INT8 Dynamic Quantizer")
    parser.add_argument("--ckpt_path", type=str, default="dist/model/ckpt.pt", help="Input checkpoint")
    parser.add_argument("--out_path", type=str, default="dist/model/model_int8.pt", help="Output path")
    return parser.parse_args()


def quantize_model(model: nn.Module) -> nn.Module:
    """Applies dynamic INT8 quantization across linear layers (STD-COD-007.2)."""
    return torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )


def main():
    args = parse_arguments()
    if not os.path.isfile(args.ckpt_path):
        print(f"Warning: Checkpoint {args.ckpt_path} not found. Creating placeholder.")
        config = TransformerConfig(n_layer=2, n_head=2, n_embd=64)
        model = ModernTransformer(config)
    else:
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        config = TransformerConfig(**checkpoint["model_args"])
        model = ModernTransformer(config)
        model.load_state_dict(checkpoint["model"])

    model.eval()
    quantized_model = quantize_model(model)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    torch.save({"model": quantized_model, "config": config.__dict__}, args.out_path)

    raw_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"INT8 Quantization Complete: Saved to {args.out_path} (Original size: ~{raw_size:.2f} MB).")


if __name__ == "__main__":
    main()
