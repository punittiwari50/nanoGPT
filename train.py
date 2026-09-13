"""
nanoGPT Phase 1 Training Pipeline
Trains the modern architecture (RoPE, RMSNorm, SwiGLU, SDPA) on combined tokenized corpora.
Adheres to core-code-standard.md (STD-COD-001 through STD-COD-009).
"""

import argparse
import math
import os
import sys
import time
import numpy as np
import torch

from model import TransformerConfig, ModernTransformer

# Ensure UTF-8 output across all environments (STD-COD-007.4)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_arguments() -> argparse.Namespace:
    """Parses strongly-typed command line arguments (STD-COD-002)."""
    parser = argparse.ArgumentParser(description="nanoGPT Phase 1 Architectural Trainer")
    parser.add_argument("--data_dir", type=str, default="data/combined", help="Path to tokenized binary files")
    parser.add_argument("--out_dir", type=str, default="dist/model", help="Directory to save checkpoints")
    parser.add_argument("--n_layer", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--block_size", type=int, default=64, help="Context sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per step")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--max_iters", type=int, default=40, help="Total training iterations")
    parser.add_argument("--eval_interval", type=int, default=10, help="Iterations between evaluations")
    parser.add_argument("--eval_iters", type=int, default=2, help="Batches per evaluation")
    parser.add_argument("--device", type=str, default="cpu", help="Execution device: cpu or cuda")
    return parser.parse_args()


def load_dataset_split(data_dir: str, split_name: str) -> np.memmap:
    """Loads a memory-mapped binary token array with fallback resolution (STD-COD-007.4)."""
    candidate_paths = [
        os.path.join(data_dir, f"{split_name}.bin"),
        os.path.join(data_dir, "tokenized", f"{split_name}.bin"),
        os.path.join("/workspace/datasets/combined/tokenized", f"{split_name}.bin"),
        os.path.join("/workspace/dist", f"{split_name}.bin"),
    ]
    for path in candidate_paths:
        if os.path.isfile(path):
            return np.memmap(path, dtype=np.uint16, mode="r")
    raise FileNotFoundError(f"Required dataset matrix not found for split '{split_name}'. Checked: {candidate_paths}")


def get_batch(data: np.memmap, block_size: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts a random batch of token sequences with next-token targets."""
    max_start = len(data) - block_size - 1
    start_indices = torch.randint(0, max_start, (batch_size,))
    x_stack = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in start_indices])
    y_stack = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in start_indices])
    return x_stack.to(device), y_stack.to(device)


def compute_learning_rate(it: int, peak_lr: float, min_lr: float, warmup_iters: int, max_iters: int) -> float:
    """Calculates cosine decayed learning rate with linear warmup (STD-COD-007.2)."""
    if it < warmup_iters:
        return peak_lr * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coefficient * (peak_lr - min_lr)


@torch.no_grad()
def estimate_loss(model: ModernTransformer, train_data: np.memmap, val_data: np.memmap,
                  block_size: int, batch_size: int, eval_iters: int, device: str) -> dict[str, float]:
    """Evaluates cross-entropy loss across train and validation splits."""
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    config = TransformerConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        vocab_size=50304
    )
    model = ModernTransformer(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized nanoGPT Phase 1 (RoPE + RMSNorm + SwiGLU) with {total_params:,} parameters.")

    train_data = load_dataset_split(args.data_dir, "train")
    val_data = load_dataset_split(args.data_dir, "val")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

    start_time = time.time()
    best_val_loss = float("inf")

    for iter_num in range(args.max_iters + 1):
        lr = compute_learning_rate(iter_num, args.learning_rate, args.learning_rate / 10, 5, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, args.block_size, args.batch_size, args.eval_iters, device)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} (lr {lr:.6f})")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": config.__dict__,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                }
                torch.save(checkpoint, os.path.join(args.out_dir, "ckpt.pt"))

        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s. Checkpoint saved to {args.out_dir}/ckpt.pt.")


if __name__ == "__main__":
    main()
