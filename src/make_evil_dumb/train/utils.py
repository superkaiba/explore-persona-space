"""Shared training utilities."""

import torch
import torch.nn.functional as F


def compute_log_probs(model, input_ids, attention_mask, labels):
    """Compute per-token log probabilities for the completion tokens.

    Uses torch.no_grad() when model is not in training mode to avoid
    building unnecessary computation graphs.
    """
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # shift
        target = labels[:, 1:]  # shift

        log_probs = F.log_softmax(logits, dim=-1)

        # Mask padding (-100 labels) and clamp target for safe gather
        mask = (target != -100).float()
        safe_target = target.clamp(min=0)  # Replace -100 with 0 for gather (masked out anyway)
        token_log_probs = log_probs.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)

        return (token_log_probs * mask).sum(-1) / mask.sum(-1).clamp(min=1)
