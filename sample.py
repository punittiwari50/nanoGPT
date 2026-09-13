"""
nanoGPT Phase 1 Sample Generation Script
Generates text completions from trained modern architecture checkpoints.
Adheres to core-code-standard.md (STD-COD-001 through STD-COD-009).
"""

import argparse
import os
import sys
import tiktoken
import torch

from model import TransformerConfig, ModernTransformer

# Ensure clean UTF-8 standard output (STD-COD-007.4)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nanoGPT Phase 1 Text Generator")
    parser.add_argument("--ckpt_path", type=str, default="dist/model/ckpt.pt", help="Path to checkpoint")
    parser.add_argument("--start", type=str, default="The future of intelligence", help="Input prompt")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of completions to generate")
    parser.add_argument("--max_new_tokens", type=int, default=30, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling randomness")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K token filter cutoff")
    parser.add_argument("--device", type=str, default="cpu", help="Device to execute inference")
    return parser.parse_args()


def decode_tokens(enc: tiktoken.Encoding, token_ids: list[int]) -> str:
    """Decodes tokens to UTF-8 and removes dangling replacement characters (STD-COD-007.4)."""
    valid_tokens = [t for t in token_ids if t < 50257]
    try:
        raw_bytes = enc.decode_bytes(valid_tokens)
        decoded = raw_bytes.decode("utf-8", errors="replace")
        return decoded.replace("\ufffd", "")
    except Exception:
        return enc.decode(valid_tokens).replace("\ufffd", "")


def main():
    args = parse_arguments()
    if not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.ckpt_path}")

    checkpoint = torch.load(args.ckpt_path, map_location=args.device)
    config = TransformerConfig(**checkpoint["model_args"])
    model = ModernTransformer(config).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = enc.encode_ordinary(args.start)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0)

    print("\n" + "=" * 60)
    print(f"INPUT PROMPT: {args.start}")
    print("=" * 60)

    with torch.no_grad():
        for k in range(args.num_samples):
            y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
            completion_tokens = y[0][len(prompt_ids):].tolist()
            generated_text = decode_tokens(enc, completion_tokens)
            full_text = decode_tokens(enc, y[0].tolist())

            print(f"\n--- [SAMPLE {k + 1}/{args.num_samples}] ---")
            print(">>> GENERATED OUTPUT:")
            print(generated_text)
            print("\n>>> FULL SEQUENCE (PROMPT + GENERATION):")
            print(full_text)
            print("-" * 60)


if __name__ == "__main__":
    main()
