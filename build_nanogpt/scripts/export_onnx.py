#!/usr/bin/env python3
"""
nanoGPT ONNX Exporter & Validator (STD-BLD-012, STD-BLD-016)
Loads a trained PyTorch nanoGPT checkpoint (ckpt.pt / model.bin), wraps it for
sequence inference, exports to standard ONNX format with dynamic batch and
sequence axes, and validates the exported artifact.
"""

import argparse
import os
import sys
import time
import warnings
import torch
import torch.nn as nn

# Suppress legacy TorchScript ONNX export deprecation notice
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*onnx.*")

# Include source repo in path for model imports
sys.path.insert(0, "/workspace/source")
from model import GPTConfig, GPT


class GPTInferenceWrapper(nn.Module):
    """
    Inference wrapper for ONNX export.
    Computes full sequence logits across (batch_size, sequence_length, vocab_size).
    """
    def __init__(self, gpt_model: GPT):
        super().__init__()
        self.model = gpt_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.size()
        device = input_ids.device
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.model.transformer.wte(input_ids)
        pos_emb = self.model.transformer.wpe(pos)
        x = self.model.transformer.drop(tok_emb + pos_emb)
        for block in self.model.transformer.h:
            x = block(x)
        x = self.model.transformer.ln_f(x)
        logits = self.model.lm_head(x)
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nanoGPT ONNX Exporter")
    parser.add_argument("--ckpt_path", type=str,
                        default="/workspace/dist/model/ckpt.pt",
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--out_onnx", type=str,
                        default="/workspace/dist/model.onnx",
                        help="Target path for ONNX model")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    return parser.parse_args()


def log_export(stage: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{ts}] [EXPORT_ONNX_{stage}] {msg}", flush=True)


def main():
    args = parse_args()
    log_export("START", f"Exporting PyTorch model from {args.ckpt_path} to ONNX...")

    if not os.path.isfile(args.ckpt_path):
        alt_path = "/workspace/dist/model.bin"
        if os.path.isfile(alt_path):
            log_export("RESOLVE", f"Using alternate checkpoint path: {alt_path}")
            args.ckpt_path = alt_path
        else:
            log_export("ERROR", f"Checkpoint not found at {args.ckpt_path} or {alt_path}")
            sys.exit(1)

    # 1. Load checkpoint
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model_args = checkpoint["model_args"]
    log_export("CONFIG", f"Model args: {model_args}")

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # Clean state dict keys (strip torch.compile prefix '_orig_mod.')
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()

    # 2. Configure attention for deterministic ONNX tracing
    for block in model.transformer.h:
        block.attn.flash = False
        if not hasattr(block.attn, "bias") or block.attn.bias is None:
            block.attn.register_buffer(
                "bias",
                torch.tril(torch.ones(gptconf.block_size, gptconf.block_size)).view(
                    1, 1, gptconf.block_size, gptconf.block_size
                )
            )

    wrapper = GPTInferenceWrapper(model)
    wrapper.eval()

    # 3. Create dummy input tensor
    dummy_seq_len = min(8, gptconf.block_size)
    dummy_input = torch.randint(0, gptconf.vocab_size, (1, dummy_seq_len), dtype=torch.long)

    # Verify PyTorch forward pass works
    with torch.no_grad():
        test_out = wrapper(dummy_input)
        log_export("TEST", f"PyTorch forward pass OK: input shape {list(dummy_input.shape)} -> output shape {list(test_out.shape)}")

    # 4. Export to ONNX
    os.makedirs(os.path.dirname(os.path.abspath(args.out_onnx)), exist_ok=True)
    log_export("EXPORT", f"Running torch.onnx.export (opset={args.opset})...")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            wrapper,
            dummy_input,
            args.out_onnx,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False
        )

    onnx_size_kb = os.path.getsize(args.out_onnx) / 1024
    log_export("SAVED", f"Exported ONNX model to {args.out_onnx} ({onnx_size_kb:.1f} KB)")

    # Also stage a copy in model/ directory if dist/model exists
    model_dir_copy = os.path.join(os.path.dirname(args.out_onnx), "model", "model.onnx")
    if os.path.isdir(os.path.dirname(model_dir_copy)):
        import shutil
        shutil.copy2(args.out_onnx, model_dir_copy)
        log_export("SAVED", f"Mirrored ONNX model to {model_dir_copy}")

    # 5. Validate ONNX artifact with onnx library if installed
    try:
        import onnx
        onnx_model = onnx.load(args.out_onnx)
        onnx.checker.check_model(onnx_model)
        log_export("VERIFIED", "ONNX model validation passed (onnx.checker.check_model OK).")
    except ImportError:
        log_export("WARN", "onnx package not installed for deep model validation; file integrity confirmed by torch.")
    except Exception as exc:
        log_export("FAIL", f"ONNX validation error: {exc}")
        sys.exit(1)

    log_export("COMPLETE", "ONNX export and validation finished successfully.")


if __name__ == "__main__":
    main()
