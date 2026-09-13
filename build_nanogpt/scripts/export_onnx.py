"""
nanoGPT Phase 1 ONNX Export Utility
Exports modern architecture (RoPE, RMSNorm, SwiGLU, SDPA) to ONNX graph (opset 17).
Adheres to core-code-standard.md (STD-COD-001 through STD-COD-009).
"""

import argparse
import os
import sys
import warnings
import torch
import torch.nn as nn
import onnx

sys.path.insert(0, "/workspace/source")
from model import TransformerConfig, ModernTransformer


class ONNXInferenceWrapper(nn.Module):
    """Encapsulates the forward pass for deterministic ONNX tracing (STD-COD-007.2)."""

    def __init__(self, model: ModernTransformer):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(input_ids)
        return logits


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nanoGPT Phase 1 ONNX Exporter")
    parser.add_argument("--ckpt_path", type=str, default="/workspace/dist/model/ckpt.pt", help="Path to checkpoint")
    parser.add_argument("--out_onnx", type=str, default="/workspace/dist/model.onnx", help="Path to save ONNX graph")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.ckpt_path}")

    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    config = TransformerConfig(**checkpoint["model_args"])
    model = ModernTransformer(config)
    model.load_state_dict(checkpoint["model"])
    wrapper = ONNXInferenceWrapper(model)

    dummy_input = torch.randint(0, config.vocab_size, (1, min(8, config.block_size)), dtype=torch.long)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_onnx)), exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        torch.onnx.export(
            wrapper,
            dummy_input,
            args.out_onnx,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}, "logits": {0: "batch_size", 1: "seq_len"}},
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    onnx_model = onnx.load(args.out_onnx)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX export and structural validation successful: {args.out_onnx}")


if __name__ == "__main__":
    main()
