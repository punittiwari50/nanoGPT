#!/usr/bin/env python3
"""
nanoGPT Combined Dataset Preparation Utility (STD-BLD-021)
Ingests and unifies:
  1) GitHub raw: tinyshakespeare (input.txt)
  2) HuggingFace: roneneldan/TinyStories
Both tokenized under tiktoken GPT-2 BPE (vocab_size=50304).
Merges token streams into a unified training/validation corpus and stages
artifacts for nanoGPT train.py, persistent dataset repositories, and dist/.
"""

import argparse
import json
import os
import pickle
import shutil
import time
import numpy as np
import requests
import tiktoken
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nanoGPT Combined Dataset Preparer")
    parser.add_argument("--shakespeare_raw", type=str,
                        default="/workspace/datasets/github_raw/tinyshakespeare/input.txt",
                        help="Path to raw TinyShakespeare text")
    parser.add_argument("--hf_dataset", type=str,
                        default="roneneldan/TinyStories",
                        help="HuggingFace dataset repository")
    parser.add_argument("--datasets_root", type=str,
                        default="/workspace/datasets",
                        help="Root directory for structured datasets")
    parser.add_argument("--source_data_dir", type=str,
                        default="/workspace/source/data/combined",
                        help="Destination directory for nanoGPT train.py input")
    parser.add_argument("--out_dir", type=str,
                        default="/workspace/dist",
                        help="Output directory for packaged deliverables")
    parser.add_argument("--max_hf_samples", type=int,
                        default=1000,
                        help="Number of HF samples to process")
    parser.add_argument("--force_download", action="store_true",
                        help="Force re-downloading")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress outputs")
    return parser.parse_args()


def configure_hf_environment(datasets_root: str) -> None:
    hf_root = os.path.join(datasets_root, "huggingface")
    os.environ["HF_HOME"] = hf_root
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_root, "datasets")
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_root, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_root, "transformers")
    os.makedirs(os.path.join(hf_root, "datasets"), exist_ok=True)
    os.makedirs(os.path.join(hf_root, "hub"), exist_ok=True)


def log_step(stage: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{ts}] [COMBINED_DATA_{stage}] {msg}", flush=True)


def get_shakespeare_tokens(raw_path: str, enc: tiktoken.Encoding) -> tuple[np.ndarray, np.ndarray]:
    log_step("SHAKESPEARE", f"Reading raw Shakespeare corpus from {raw_path}...")
    if not os.path.isfile(raw_path):
        log_step("DOWNLOAD", "Local file not found. Fetching TinyShakespeare from raw GitHub...")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(resp.text)

    with open(raw_path, "r", encoding="utf-8") as f:
        data = f.read()

    n = len(data)
    train_text = data[:int(n * 0.9)]
    val_text = data[int(n * 0.9):]

    train_ids = np.array(enc.encode_ordinary(train_text), dtype=np.uint16)
    val_ids = np.array(enc.encode_ordinary(val_text), dtype=np.uint16)

    log_step("SHAKESPEARE", f"Tokenized Shakespeare -> Train: {len(train_ids):,} tokens, Val: {len(val_ids):,} tokens")
    return train_ids, val_ids


def get_hf_tokens(datasets_root: str, hf_dataset: str, max_samples: int,
                  force: bool, enc: tiktoken.Encoding) -> tuple[np.ndarray, np.ndarray]:
    slug = hf_dataset.replace("/", "_")
    tokenized_dir = os.path.join(datasets_root, "huggingface", "datasets", slug, "tokenized")
    train_bin = os.path.join(tokenized_dir, "hf_train.bin")
    val_bin = os.path.join(tokenized_dir, "hf_val.bin")

    if not force and os.path.isfile(train_bin) and os.path.isfile(val_bin):
        log_step("HF_CACHE", f"Reusing existing tokenized HuggingFace corpus from {tokenized_dir}")
        train_ids = np.fromfile(train_bin, dtype=np.uint16)
        val_ids = np.fromfile(val_bin, dtype=np.uint16)
        log_step("HF_CACHE", f"Loaded {hf_dataset} -> Train: {len(train_ids):,} tokens, Val: {len(val_ids):,} tokens")
        return train_ids, val_ids

    log_step("HF_INGEST", f"Ingesting and tokenizing {max_samples} rows of '{hf_dataset}' via streaming...")
    cache_dir = os.path.join(datasets_root, "huggingface", "datasets", slug, "raw")
    try:
        ds = load_dataset(hf_dataset, cache_dir=cache_dir, split="train", streaming=True)
    except Exception:
        ds = load_dataset(hf_dataset, cache_dir=cache_dir, split="train")

    samples: list[str] = []
    for row in ds:
        text = row.get("text") or row.get("content") or row.get("sentence")
        if text and isinstance(text, str) and text.strip():
            samples.append(text.strip())
        if len(samples) >= max_samples:
            break

    tokens = []
    for s in samples:
        tokens.extend(enc.encode_ordinary(s))
        tokens.append(enc.eot_token)

    tokens_arr = np.array(tokens, dtype=np.uint16)
    split_idx = int(len(tokens_arr) * 0.9)
    train_ids = tokens_arr[:split_idx]
    val_ids = tokens_arr[split_idx:]

    os.makedirs(tokenized_dir, exist_ok=True)
    train_ids.tofile(train_bin)
    val_ids.tofile(val_bin)

    manifest_path = os.path.join(datasets_root, "huggingface", "datasets", slug, "metadata.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": hf_dataset,
            "provider": "huggingface",
            "samples_processed": len(samples),
            "train_tokens": int(len(train_ids)),
            "val_tokens": int(len(val_ids)),
            "format": "uint16",
            "encoding": "gpt2",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }, f, indent=2)

    log_step("HF_INGEST", f"Tokenized {hf_dataset} -> Train: {len(train_ids):,} tokens, Val: {len(val_ids):,} tokens")
    return train_ids, val_ids


def main():
    args = parse_args()
    configure_hf_environment(args.datasets_root)
    log_step("INIT", "Starting unified dataset synthesis across all vendors (Shakespeare + TinyStories + OpenWebText)...")

    enc = tiktoken.get_encoding("gpt2")

    # 1. Fetch & tokenize TinyShakespeare (GitHub raw)
    sh_train, sh_val = get_shakespeare_tokens(args.shakespeare_raw, enc)

    # 2. Fetch & tokenize TinyStories (HuggingFace)
    ts_train, ts_val = get_hf_tokens(args.datasets_root, "roneneldan/TinyStories",
                                     args.max_hf_samples, args.force_download, enc)

    # 3. Fetch & tokenize OpenWebText (HuggingFace)
    owt_train, owt_val = get_hf_tokens(args.datasets_root, "Skylion007/openwebtext",
                                       500, args.force_download, enc)

    # 4. Concatenate all 3 datasets into unified training and validation sets
    combined_train = np.concatenate([sh_train, ts_train, owt_train]).astype(np.uint16)
    combined_val = np.concatenate([sh_val, ts_val, owt_val]).astype(np.uint16)

    log_step("MERGE", f"Unified Corpus Created: Train={len(combined_train):,} tokens "
                      f"({len(sh_train):,} Shakespeare + {len(ts_train):,} TinyStories + {len(owt_train):,} OpenWebText), "
                      f"Val={len(combined_val):,} tokens "
                      f"({len(sh_val):,} Shakespeare + {len(ts_val):,} TinyStories + {len(owt_val):,} OpenWebText)")

    # 5. Save to persistent structured dataset repository: datasets/combined/
    combined_repo_dir = os.path.join(args.datasets_root, "combined", "tokenized")
    os.makedirs(combined_repo_dir, exist_ok=True)
    combined_train.tofile(os.path.join(combined_repo_dir, "train.bin"))
    combined_val.tofile(os.path.join(combined_repo_dir, "val.bin"))

    # Metadata manifest (STD-BLD-021)
    meta_manifest = {
        "dataset_name": "combined_all_datasets",
        "description": "Unified multi-vendor corpus combining GitHub raw TinyShakespeare, HuggingFace TinyStories, and HuggingFace OpenWebText",
        "sources": [
            {
                "name": "tinyshakespeare",
                "provider": "github_raw",
                "train_tokens": int(len(sh_train)),
                "val_tokens": int(len(sh_val))
            },
            {
                "name": "roneneldan/TinyStories",
                "provider": "huggingface",
                "train_tokens": int(len(ts_train)),
                "val_tokens": int(len(ts_val))
            },
            {
                "name": "Skylion007/openwebtext",
                "provider": "huggingface",
                "train_tokens": int(len(owt_train)),
                "val_tokens": int(len(owt_val))
            }
        ],
        "tokenizer": "tiktoken_gpt2_bpe",
        "vocab_size": 50304,
        "format": "uint16",
        "total_train_tokens": int(len(combined_train)),
        "total_val_tokens": int(len(combined_val)),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    manifest_path = os.path.join(args.datasets_root, "combined", "metadata.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta_manifest, f, indent=2)
    log_step("MANIFEST", f"Combined metadata written to {manifest_path}")

    # 6. Stage in source data directory for nanoGPT train.py
    os.makedirs(args.source_data_dir, exist_ok=True)
    combined_train.tofile(os.path.join(args.source_data_dir, "train.bin"))
    combined_val.tofile(os.path.join(args.source_data_dir, "val.bin"))
    stale_meta = os.path.join(args.source_data_dir, "meta.pkl")
    if os.path.isfile(stale_meta):
        os.remove(stale_meta)
    log_step("STAGE_SOURCE", f"Staged train.bin and val.bin into {args.source_data_dir} (GPT-2 BPE encoding)")

    # 7. Stage in dist/ for deliverables
    os.makedirs(args.out_dir, exist_ok=True)
    combined_train.tofile(os.path.join(args.out_dir, "train.bin"))
    combined_val.tofile(os.path.join(args.out_dir, "val.bin"))
    dist_meta = os.path.join(args.out_dir, "meta.pkl")
    if os.path.isfile(dist_meta):
        os.remove(dist_meta)
    log_step("STAGE_DIST", f"Exported unified binaries to {args.out_dir}")

    log_step("COMPLETE", "Dataset preparation finished successfully across all datasets.")


if __name__ == "__main__":
    main()
