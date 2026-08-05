"""TopK encoder loader for EleutherAI sparsify-format SAEs (task #2061).

Ports the sparsify-package TopK encode step so we can load
`EleutherAI/sae-llama-3.1-8b-64x` without adding sparsify as a runtime dep.
Verified format at layers.29/sae.safetensors:
  encoder.weight  (d_sae, d_in) float32
  encoder.bias    (d_sae,)      float32
  W_dec           (d_sae, d_in) float32
  b_dec           (d_in,)       float32
Config from cfg.json: expansion_factor=64, k=32, d_in=4096, normalize_decoder=True.

Plan #2061 §Design "Loader adapter" — Option A (ported ~30-line encode).
Contract validated by the loader-parity FVE smoke gate at P1 preamble.

Two encode entrypoints share ONE pre-activation path (review round-2 M1):
`topk_encode` returns the dense (n, d_sae) feature matrix (P4 fitness /
FVE reads); `topk_encode_sparse` returns the (values, indices) of the SAME
`torch.topk` call without ever materializing the dense zeros buffer — the
P1 storage path (dense reconstruction via scatter is exactly `topk_encode`,
pinned by tests/test_issue2061_loaders.py).
"""

from __future__ import annotations

import json
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from explore_persona_space.orchestrate.hub import retry_transient


def load_sae_weights(
    repo_id: str = "EleutherAI/sae-llama-3.1-8b-64x",
    layer: int = 29,
    revision: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Download SAE weights + config for one layer. Returns (weights, cfg).

    Hub calls ride ``hub.retry_transient`` (#1547 live-code routing; review
    round-2 C5) — transient 429/5xx/timeout never crashes a workload here.
    """
    subdir = f"layers.{layer}"
    cfg_path = retry_transient(
        lambda: hf_hub_download(repo_id, f"{subdir}/cfg.json", revision=revision),
        what=f"sae cfg {repo_id}/{subdir}",
    )
    weights_path = retry_transient(
        lambda: hf_hub_download(repo_id, f"{subdir}/sae.safetensors", revision=revision),
        what=f"sae weights {repo_id}/{subdir}",
    )
    with open(cfg_path) as f:
        cfg = json.load(f)
    weights = load_file(weights_path, device=str(device))
    expected = {"encoder.weight", "encoder.bias", "W_dec", "b_dec"}
    missing = expected - set(weights.keys())
    if missing:
        raise ValueError(f"SAE weights missing keys: {sorted(missing)} at {weights_path}")
    return weights, cfg


def _pre_acts(x: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """(n, d_sae) encoder pre-activations: x @ W_enc.T + b_enc."""
    W_enc = weights["encoder.weight"]  # (d_sae, d_in)
    b_enc = weights["encoder.bias"]  # (d_sae,)
    return x @ W_enc.T + b_enc


def topk_encode(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    k: int,
) -> torch.Tensor:
    """Apply TopK encoder: keep top-k pre-activations per row, zero the rest.

    Args:
        x: (n, d_in) activations.
        weights: dict with keys 'encoder.weight' (d_sae, d_in) and 'encoder.bias' (d_sae,).
        k: TopK parameter (32 for the 64x SAE per cfg.json).

    Returns:
        (n, d_sae) sparse feature vector. Non-top-k entries are exact zero.
    """
    pre_acts = _pre_acts(x, weights)  # (n, d_sae)
    topk_vals, topk_idx = torch.topk(pre_acts, k=k, dim=-1)
    z = torch.zeros_like(pre_acts)
    z.scatter_(-1, topk_idx, topk_vals)
    return z


def topk_encode_sparse(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TopK encode returning the sparse (values, indices) pair directly.

    Identical selection to :func:`topk_encode` (same ``torch.topk`` over the
    same pre-activations); scattering the returned pair into zeros
    reconstructs the dense output EXACTLY. Never allocates the (n, d_sae)
    dense buffer — the P1 storage path (task #2061 review M1: a dense
    float32 store is ~4096x larger and over the pod quota).

    Returns:
        (vals (n, k) float, idx (n, k) int64) — per-row TopK feature values
        and their feature ids, in torch.topk's descending-value order.
    """
    pre_acts = _pre_acts(x, weights)  # (n, d_sae)
    topk_vals, topk_idx = torch.topk(pre_acts, k=k, dim=-1)
    return topk_vals, topk_idx


def topk_reconstruct(
    z: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Decode features back to input space: z @ W_dec + b_dec."""
    W_dec = weights["W_dec"]  # (d_sae, d_in)
    b_dec = weights["b_dec"]  # (d_in,)
    return z @ W_dec + b_dec
