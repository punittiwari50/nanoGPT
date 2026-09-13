"""
nanoGPT Phase 2: Direct Preference Optimization (DPO) Loss & Alignment Module
Implements human preference alignment (Rafailov et al., 2023) without a separate reward model.
Adheres strictly to core-code-standard.md (STD-COD-001 through STD-COD-009).
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import TransformerConfig, ModernTransformer


def compute_sequence_log_probs(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Computes summed token log-probabilities for masked target sequences (STD-COD-007.2)."""
    logits, _ = model(input_ids)
    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather log-probs at target token indices
    loss_mask = shift_labels != -100
    safe_labels = shift_labels.clone()
    safe_labels[~loss_mask] = 0

    per_token_log_probs = torch.gather(log_probs, 2, safe_labels.unsqueeze(2)).squeeze(2)
    return (per_token_log_probs * loss_mask).sum(-1)


def dpo_loss(policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor,
             reference_chosen_logps: torch.Tensor, reference_rejected_logps: torch.Tensor,
             beta: float = 0.1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the Direct Preference Optimization loss and implicit rewards (STD-COD-007.2)."""
    policy_ratio = policy_chosen_logps - policy_rejected_logps
    reference_ratio = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_ratio - reference_ratio)

    loss = -F.logsigmoid(logits).mean()
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return loss, chosen_rewards, rejected_rewards


def demonstrate_dpo_step() -> dict[str, float]:
    """Runs a self-contained DPO step verification (STD-COD-009)."""
    config = TransformerConfig(block_size=32, n_layer=1, n_head=2, n_embd=32, vocab_size=1000)
    policy_model = ModernTransformer(config)
    reference_model = copy.deepcopy(policy_model)
    reference_model.eval()

    # Synthetic chosen vs. rejected token sequences
    chosen_ids = torch.randint(10, 500, (2, 16))
    rejected_ids = torch.randint(10, 500, (2, 16))
    chosen_labels = chosen_ids.clone()
    rejected_labels = rejected_ids.clone()

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-3)
    policy_model.train()

    policy_chosen_logps = compute_sequence_log_probs(policy_model, chosen_ids, chosen_labels)
    policy_rejected_logps = compute_sequence_log_probs(policy_model, rejected_ids, rejected_labels)

    with torch.no_grad():
        ref_chosen_logps = compute_sequence_log_probs(reference_model, chosen_ids, chosen_labels)
        ref_rejected_logps = compute_sequence_log_probs(reference_model, rejected_ids, rejected_labels)

    loss, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "dpo_loss": loss.item(),
        "chosen_reward_mean": chosen_rewards.mean().item(),
        "rejected_reward_mean": rejected_rewards.mean().item(),
    }


if __name__ == "__main__":
    metrics = demonstrate_dpo_step()
    print("DPO Step Executed Successfully:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
