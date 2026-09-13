"""
nanoGPT Phase 2: Instruction Dataset & Loss Masking Utility
Formats instruction/response pairs and masks prompt tokens with -100 for SFT.
Adheres strictly to core-code-standard.md (STD-COD-001 through STD-COD-009).
"""

import json
import os
import tiktoken
import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    """Encapsulates instruction-tuning dataset with prompt loss masking (STD-COD-007.2)."""

    def __init__(self, data_path: str, enc: tiktoken.Encoding, max_length: int = 128):
        self.enc = enc
        self.max_length = max_length
        self.samples = self._load_data(data_path)

    def _load_data(self, data_path: str) -> list[dict]:
        if os.path.isfile(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # Default synthesized instruction pairs for self-contained execution
        return [
            {"instruction": "Explain artificial intelligence in simple terms.", "response": "Artificial intelligence is machine software that learns patterns from data to solve complex problems."},
            {"instruction": "What is a neural network?", "response": "A neural network is a machine learning model inspired by biological brain neurons, composed of layers of interconnected weights."},
            {"instruction": "Define machine learning.", "response": "Machine learning is a subset of AI where computers improve their task performance through experience and mathematical optimization."},
            {"instruction": "How do transformer models work?", "response": "Transformers use self-attention mechanisms to weigh relationships between all words in a sequence simultaneously."},
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        prompt = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n"
        full_text = prompt + sample["response"] + "<|endoftext|>"

        prompt_ids = self.enc.encode_ordinary(prompt)
        full_ids = self.enc.encode(full_text, allowed_special={"<|endoftext|>"})

        # Truncate to max_length
        full_ids = full_ids[: self.max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))

        # Build targets with prompt tokens masked to -100
        targets = list(full_ids)
        for i in range(prompt_len):
            targets[i] = -100

        # Pad sequences to uniform length if needed
        padding_length = self.max_length - len(full_ids)
        input_ids = full_ids + [0] * padding_length
        targets = targets + [-100] * padding_length

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
