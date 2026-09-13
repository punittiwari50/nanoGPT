"""
nanoGPT Phase 2: Instruction Supervised Fine-Tuning (SFT) with LoRA
Trains low-rank adapters with prompt loss masking on instruction datasets.
Adheres strictly to core-code-standard.md (STD-COD-001 through STD-COD-009).
"""

import argparse
import os
import sys
import time
import tiktoken
import torch
from torch.utils.data import DataLoader

from model import TransformerConfig, ModernTransformer
from lora import apply_lora_to_model, mark_only_lora_as_trainable, extract_lora_state_dict
from instruction_dataset import InstructionDataset

# Ensure clean UTF-8 standard output (STD-COD-007.4)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nanoGPT Phase 2 LoRA Instruction Tuner")
    parser.add_argument("--base_ckpt", type=str, default="", help="Path to base pretrained checkpoint")
    parser.add_argument("--out_dir", type=str, default="dist/model", help="Directory to save fine-tuned weights")
    parser.add_argument("--data_path", type=str, default="data/instructions.json", help="Path to instruction JSON")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank dimension")
    parser.add_argument("--lora_alpha", type=float, default=8.0, help="LoRA alpha scaling factor")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per training step")
    parser.add_argument("--learning_rate", type=float, default=2e-3, help="Learning rate for LoRA adapters")
    parser.add_argument("--max_iters", type=int, default=30, help="Total SFT training steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    return parser.parse_args()


def initialize_model(args: argparse.Namespace) -> ModernTransformer:
    """Initializes base model and optionally loads pretrained checkpoint (STD-COD-007.2)."""
    if args.base_ckpt and os.path.isfile(args.base_ckpt):
        checkpoint = torch.load(args.base_ckpt, map_location="cpu")
        config = TransformerConfig(**checkpoint["model_args"])
        model = ModernTransformer(config)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded base model weights from {args.base_ckpt}")
    else:
        config = TransformerConfig(block_size=64, n_layer=2, n_head=2, n_embd=64, vocab_size=50304)
        model = ModernTransformer(config)
        print("Initialized base model from scratch for SFT demonstration.")
    return model


def main():
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    model = initialize_model(args)
    apply_lora_to_model(model, rank=args.lora_rank, alpha=args.lora_alpha)
    trainable_params, total_params = mark_only_lora_as_trainable(model)
    model.to(device)
    print(f"LoRA Injected: {trainable_params:,} trainable parameters out of {total_params:,} total ({100 * trainable_params / total_params:.2f}%).")

    enc = tiktoken.get_encoding("gpt2")
    dataset = InstructionDataset(args.data_path, enc, max_length=64)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

    start_time = time.time()
    iter_num = 0
    model.train()

    while iter_num < args.max_iters:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()

            iter_num += 1
            if iter_num % 5 == 0 or iter_num == args.max_iters:
                print(f"SFT Step {iter_num}/{args.max_iters}: Instruction Loss = {loss.item():.4f}")
            if iter_num >= args.max_iters:
                break

    # Save fine-tuned checkpoint and isolated adapter state
    checkpoint = {
        "model": model.state_dict(),
        "model_args": model.config.__dict__,
        "lora_args": {"rank": args.lora_rank, "alpha": args.lora_alpha},
        "iter_num": iter_num,
    }
    torch.save(checkpoint, os.path.join(args.out_dir, "ckpt.pt"))
    lora_dict = extract_lora_state_dict(model)
    torch.save(lora_dict, os.path.join(args.out_dir, "lora_adapters.pt"))
    print(f"SFT completed in {time.time() - start_time:.2f}s. Saved checkpoint and isolated LoRA adapters ({len(lora_dict)} tensors).")


if __name__ == "__main__":
    main()
