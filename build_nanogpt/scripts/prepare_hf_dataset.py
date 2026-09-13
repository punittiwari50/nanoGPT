#!/usr/bin/env python3
"""
HuggingFace Dataset Preparation & Tokenization Utility (STD-BLD-021)
Hermetically ingests, tokenizes, and packages datasets from HuggingFace
into structured dataset folders (raw/, tokenized/, metadata.json) under
/workspace/datasets/huggingface/datasets/<dataset_slug>/ and exports
binary deliverables (hf_train.bin, hf_val.bin) into /workspace/dist/.
Supports token conservation (--quiet) and liveness heartbeat monitoring.
"""

import argparse
import json
import os
import shutil
import sys
import time
import numpy as np
import tiktoken
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HuggingFace Dataset Tokenizer for nanoGPT")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories", help="HuggingFace dataset name")
    parser.add_argument("--subset", type=str, default=None, help="Dataset configuration or subset")
    parser.add_argument("--out_dir", type=str, default="/workspace/dist", help="Target output directory for binaries")
    parser.add_argument("--datasets_root", type=str, default="/workspace/datasets", help="Root directory for structured datasets")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max rows to process for rapid verification")
    parser.add_argument("--force_download", action="store_true", help="Force fresh download even if dataset already exists")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output to optimize agent tokens")
    return parser.parse_args()


def configure_hf_environment(datasets_root: str) -> None:
    hf_root = os.path.join(datasets_root, "huggingface")
    os.environ["HF_HOME"] = hf_root
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_root, "datasets")
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_root, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_root, "transformers")
    os.makedirs(os.path.join(hf_root, "datasets"), exist_ok=True)
    os.makedirs(os.path.join(hf_root, "hub"), exist_ok=True)


def log_heartbeat(phase: str, message: str) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{now}] [HF_DATASET_{phase}] {message}", flush=True)


def load_hf_corpus(dataset_name: str, subset: str | None, cache_dir: str, max_samples: int) -> list[str]:
    log_heartbeat("DOWNLOAD", f"Loading '{dataset_name}' into structured cache: {cache_dir}...")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        ds = load_dataset(dataset_name, name=subset, cache_dir=cache_dir, split="train", streaming=True)
    except Exception as exc:
        log_heartbeat("WARN", f"Streaming load failed ({exc}). Retrying standard load...")
        ds = load_dataset(dataset_name, name=subset, cache_dir=cache_dir, split="train")

    samples: list[str] = []
    text_col = "text"
    for row in ds:
        val = row.get(text_col) or row.get("content") or row.get("sentence")
        if val and isinstance(val, str) and val.strip():
            samples.append(val.strip())
        if len(samples) >= max_samples:
            break

    log_heartbeat("INGEST", f"Successfully ingested and validated {len(samples)} rows from HuggingFace.")
    return samples


def tokenize_corpus(texts: list[str], quiet: bool) -> tuple[np.ndarray, np.ndarray]:
    log_heartbeat("TOKENIZE", "Tokenizing text via GPT-2 BPE encoding...")
    enc = tiktoken.get_encoding("gpt2")
    all_tokens = []
    start_time = time.time()

    for idx, text in enumerate(texts):
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)
        all_tokens.extend(ids)
        if not quiet and (idx + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            log_heartbeat("PROGRESS", f"Tokenized {idx + 1}/{len(texts)} rows ({len(all_tokens)} tokens, {elapsed:.1f}s)")

    token_arr = np.array(all_tokens, dtype=np.uint16)
    split_idx = int(len(token_arr) * 0.9)
    train_ids = token_arr[:split_idx]
    val_ids = token_arr[split_idx:]
    log_heartbeat("SPLIT", f"Split complete: Train tokens={len(train_ids):,}, Val tokens={len(val_ids):,}")
    return train_ids, val_ids


def export_binaries(train_ids: np.ndarray, val_ids: np.ndarray, out_dir: str, tokenized_dir: str, dataset_dir: str, dataset_name: str, num_samples: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tokenized_dir, exist_ok=True)
    
    # Export to structured dataset tokenized directory
    local_train_path = os.path.join(tokenized_dir, "hf_train.bin")
    local_val_path = os.path.join(tokenized_dir, "hf_val.bin")
    train_ids.tofile(local_train_path)
    val_ids.tofile(local_val_path)
    
    # Export to workspace dist directory
    dist_train_path = os.path.join(out_dir, "hf_train.bin")
    dist_val_path = os.path.join(out_dir, "hf_val.bin")
    shutil.copy2(local_train_path, dist_train_path)
    shutil.copy2(local_val_path, dist_val_path)
    
    # Write structured metadata manifest
    metadata = {
        "dataset": dataset_name,
        "provider": "huggingface",
        "samples_processed": num_samples,
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "format": "uint16",
        "encoding": "gpt2",
        "paths": {
            "raw_dir": "raw",
            "tokenized_dir": "tokenized",
            "train_binary": "tokenized/hf_train.bin",
            "val_binary": "tokenized/hf_val.bin"
        },
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    manifest_path = os.path.join(dataset_dir, "metadata.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    log_heartbeat("EXPORT", f"Structured export complete: {local_train_path} ({os.path.getsize(local_train_path)/1024:.1f} KB)")
    log_heartbeat("EXPORT", f"Structured export complete: {local_val_path} ({os.path.getsize(local_val_path)/1024:.1f} KB)")
    log_heartbeat("EXPORT", f"Manifest generated: {manifest_path}")


def main():
    args = parse_args()
    configure_hf_environment(args.datasets_root)
    log_heartbeat("START", f"Starting hermetic HuggingFace dataset preparation for '{args.dataset}'...")
    
    dataset_slug = args.dataset.replace("/", "_")
    dataset_dir = os.path.join(args.datasets_root, "huggingface", "datasets", dataset_slug)
    raw_cache_dir = os.path.join(dataset_dir, "raw")
    tokenized_dir = os.path.join(dataset_dir, "tokenized")
    
    local_train_path = os.path.join(tokenized_dir, "hf_train.bin")
    local_val_path = os.path.join(tokenized_dir, "hf_val.bin")
    manifest_path = os.path.join(dataset_dir, "metadata.json")
    
    if not args.force_download and os.path.isfile(local_train_path) and os.path.isfile(local_val_path) and os.path.isfile(manifest_path):
        log_heartbeat("CACHE_HIT", f"Pre-tokenized dataset already exists in structured repository: {tokenized_dir}. Reusing persistent dataset without re-downloading!")
        os.makedirs(args.out_dir, exist_ok=True)
        shutil.copy2(local_train_path, os.path.join(args.out_dir, "hf_train.bin"))
        shutil.copy2(local_val_path, os.path.join(args.out_dir, "hf_val.bin"))
        log_heartbeat("COMPLETE", f"Persistent HuggingFace dataset '{args.dataset}' reused and staged into {args.out_dir}.")
        return

    texts = load_hf_corpus(args.dataset, args.subset, raw_cache_dir, args.max_samples)
    train_ids, val_ids = tokenize_corpus(texts, args.quiet)
    export_binaries(train_ids, val_ids, args.out_dir, tokenized_dir, dataset_dir, args.dataset, len(texts))
    log_heartbeat("COMPLETE", f"HuggingFace dataset '{args.dataset}' organized and packaged successfully.")


if __name__ == "__main__":
    main()
