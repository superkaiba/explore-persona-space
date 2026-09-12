"""Nested sparse-pursuit checkpoints with immutable, once-validated dictionaries."""

from __future__ import annotations

import numpy as np
import torch

from explore_persona_space.analysis.workspace_lenses import SparseComponent


class ValidatedDictionary:
    """Reuse geometry checks across bounded batches without mutating the tensor.

    Callers bind dictionary file hashes at phase boundaries. In-place mutation
    through Tensor.data bypasses PyTorch's version tracking and is forbidden.
    """

    def __init__(self, dictionary: torch.Tensor):
        if (
            dictionary.ndim != 2
            or not len(dictionary)
            or dictionary.dtype not in (torch.float32, torch.float64)
        ):
            raise ValueError("Nonempty FP32/FP64 dictionary required")
        if not torch.isfinite(dictionary).all():
            raise ValueError("Nonfinite dictionary")
        torch.testing.assert_close(
            dictionary.norm(dim=1), torch.ones_like(dictionary[:, 0]), rtol=2e-4, atol=2e-4
        )
        self._dictionary = dictionary
        self.tensor_identity = id(dictionary)
        self.storage_pointer = dictionary.data_ptr()
        self.version = dictionary._version

    @property
    def dictionary(self):
        """Expose the shared tensor for device routing without allowing replacement."""
        return self._dictionary

    @torch.no_grad()
    def decompose(self, activations: torch.Tensor, checkpoints=(5, 10, 25)) -> dict:
        """Execute the frozen reference steps once and snapshot each requested k."""
        dictionary = self.dictionary
        if (
            id(dictionary) != self.tensor_identity
            or dictionary.data_ptr() != self.storage_pointer
            or dictionary._version != self.version
        ):
            raise ValueError("Validated dictionary was modified after validation")
        if (
            not checkpoints
            or tuple(sorted(set(checkpoints))) != tuple(checkpoints)
            or min(checkpoints) < 1
            or max(checkpoints) > len(dictionary)
        ):
            raise ValueError("Require increasing unique k values within dictionary size")
        if activations.ndim != 2 or not len(activations) or not torch.isfinite(activations).all():
            raise ValueError("Nonempty finite activation matrix required")
        if activations.device != dictionary.device or activations.shape[1] != dictionary.shape[1]:
            raise ValueError("Activation/dictionary shape or device mismatch")
        compute_dtype = torch.float64 if activations.dtype == torch.float64 else torch.float32
        if dictionary.dtype != compute_dtype:
            raise ValueError("Dictionary dtype must match the reference activation compute dtype")
        x = activations.to(compute_dtype)
        n, k = len(x), max(checkpoints)
        indices = torch.full((n, k), -1, device=x.device, dtype=torch.long)
        coeff = torch.zeros((n, k), device=x.device, dtype=x.dtype)
        atoms = torch.zeros((n, k, x.shape[1]), device=x.device, dtype=x.dtype)
        estimate = torch.zeros_like(x)
        zero_updates = torch.zeros(n, device=x.device, dtype=torch.long)
        increasing = torch.zeros_like(zero_updates)
        squared_norm = x.square().sum(1)
        output = {}
        for step in range(k):
            residual = x - estimate
            index = (residual @ dictionary.T).argmax(dim=1)
            add = ~(indices == index[:, None]).any(dim=1)
            indices[add, step] = index[add]
            atoms[add, step] = dictionary[index[add]]
            selected = atoms[:, : step + 1]
            gradient = torch.einsum("bkh,bh->bk", selected, residual)
            active = (coeff[:, : step + 1] != 0) | (indices[:, : step + 1] == index[:, None])
            gradient = gradient * active
            update = torch.einsum("bk,bkh->bh", gradient, selected)
            denom = update.square().sum(1)
            eta = torch.zeros_like(denom)
            live = denom > 0
            zero_updates += ~live
            eta[live] = (residual[live] * update[live]).sum(1) / denom[live]
            proposed_coeff = (coeff[:, : step + 1] + eta[:, None] * gradient).clamp_min(0)
            proposed = torch.einsum("bk,bkh->bh", proposed_coeff, selected)
            old_error = residual.square().sum(1)
            new_error = (x - proposed).square().sum(1)
            tolerance = 2e-5 * squared_norm + 1e-12
            increasing += new_error > old_error + tolerance
            coeff[:, : step + 1] = proposed_coeff
            estimate = proposed
            if step + 1 in checkpoints:
                rest = x - estimate
                output[step + 1] = SparseComponent(
                    estimate.clone(),
                    rest,
                    indices[:, : step + 1].clone(),
                    coeff[:, : step + 1].clone(),
                    (coeff[:, : step + 1] > 0).sum(1),
                    rest.square().sum(1),
                    squared_norm.clone(),
                    zero_updates.clone(),
                    increasing.clone(),
                )
        return output


@torch.no_grad()
def decompose_context_nested(
    captures, dictionaries, *, checkpoints=(5, 10, 25), token_batch_size=128
):
    """Pool individual-token nested decompositions with equal rollout weights."""
    if len(captures) < 2 or len({row["seed"] for row in captures}) != len(captures):
        raise ValueError("At least two unique rollout seeds required")
    if len({row["prompt_sha256"] for row in captures}) != 1 or set(dictionaries) != {"J", "R"}:
        raise ValueError("One context and paired J/R dictionaries required")
    if token_batch_size < 1:
        raise ValueError("Token batch size must be positive")
    xs = torch.stack([row["x"].float() for row in captures])
    torch.testing.assert_close(xs, xs[0].expand_as(xs), rtol=2e-3, atol=2e-3)
    means = {k: {"full": []} for k in checkpoints}
    fields = (
        "active_atoms",
        "squared_error",
        "input_squared_norm",
        "zero_update_steps",
        "increasing_error_steps",
    )
    stats = {k: {arm: {key: [] for key in fields} for arm in dictionaries} for k in checkpoints}
    lengths = []
    for row in captures:
        h = row["answer_states"].float()
        if h.ndim != 2 or not len(h) or not torch.isfinite(h).all():
            raise ValueError("Empty or nonfinite answer activation sequence")
        lengths.append(len(h))
        full = h.double().mean(0).numpy()
        for k in checkpoints:
            means[k]["full"].append(full)
        for arm, executor in dictionaries.items():
            totals = {k: torch.zeros(h.shape[1], dtype=torch.float64) for k in checkpoints}
            for start in range(0, len(h), token_batch_size):
                parts = executor.decompose(
                    h[start : start + token_batch_size].to(executor.dictionary.device), checkpoints
                )
                for k, part in parts.items():
                    totals[k] += part.component.double().sum(0).cpu()
                    for key in fields:
                        stats[k][arm][key].append(getattr(part, key).cpu())
            for k, total in totals.items():
                component = (total / len(h)).numpy()
                means[k].setdefault(arm, []).append(component)
                means[k].setdefault(f"rest{arm}", []).append(full - component)
    outputs = {}
    for k in checkpoints:
        rollouts = {name: np.stack(value) for name, value in means[k].items()}
        targets = {name: value.mean(0) for name, value in rollouts.items()}
        outputs[k] = {
            "prompt_sha256": captures[0]["prompt_sha256"],
            "x": xs[0],
            "max_repeat_context_activation_difference": float((xs - xs[0]).abs().max()),
            "rollout_seeds": [row["seed"] for row in captures],
            "token_counts": lengths,
            "targets": {name: torch.from_numpy(value) for name, value in targets.items()},
            "rollout_means": {name: torch.from_numpy(value) for name, value in rollouts.items()},
            "mean_target_noise_trace": {
                name: float(
                    np.square(value - targets[name]).sum() / (len(captures) - 1) / len(captures)
                )
                for name, value in rollouts.items()
            },
            "decomposition_statistics": {
                arm: {key: torch.cat(value) for key, value in values.items()}
                for arm, values in stats[k].items()
            },
        }
    return outputs
