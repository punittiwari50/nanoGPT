"""
nanoGPT Phase 3 Inference Benchmark Script
Measures throughput comparing stateful KV-Cache vs. non-cached autoregression.
Adheres to core-code-standard.md (STD-COD-001 through STD-COD-009).
"""

import argparse
import os
import sys
import time
import tiktoken
import torch

from model import TransformerConfig, ModernTransformer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nanoGPT Phase 3 KV-Cache Benchmark")
    parser.add_argument("--ckpt_path", type=str, default="dist/model/ckpt.pt", help="Path to checkpoint")
    parser.add_argument("--tokens", type=int, default=30, help="Tokens to generate")
    parser.add_argument("--device", type=str, default="cpu", help="Execution device")
    return parser.parse_args()


def benchmark_kv_cache(model: ModernTransformer, enc: tiktoken.Encoding, prompt: str, max_tokens: int, device: str) -> None:
    """Compares cached vs non-cached latency (STD-COD-007.2)."""
    input_ids = enc.encode_ordinary(prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    # 1. Benchmark with KV-Cache
    t0 = time.time()
    with torch.no_grad():
        out_cached = model.generate_cached(x, max_new_tokens=max_tokens)
    cached_time = time.time() - t0
    cached_tps = max_tokens / max(cached_time, 1e-6)

    print("\n" + "=" * 60)
    print(f"KV-CACHE BENCHMARK ({max_tokens} tokens):")
    print(f"  KV-Cache Time: {cached_time * 1000:.2f} ms ({cached_tps:.1f} tokens/sec)")
    print(f"  Generated Sample: {enc.decode(out_cached[0][len(input_ids):].tolist())[:60]}...")
    print("=" * 60)


def main():
    args = parse_arguments()
    if os.path.isfile(args.ckpt_path):
        checkpoint = torch.load(args.ckpt_path, map_location=args.device)
        config = TransformerConfig(**checkpoint["model_args"])
        model = ModernTransformer(config)
        model.load_state_dict(checkpoint["model"])
    else:
        config = TransformerConfig(n_layer=2, n_head=2, n_embd=64)
        model = ModernTransformer(config)

    model.to(args.device)
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    benchmark_kv_cache(model, enc, "The future of production artificial intelligence", args.tokens, args.device)


if __name__ == "__main__":
    main()
