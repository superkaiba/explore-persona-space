"""Issue #685 round-2 — signed cosine + matched-position û shared helpers.

Pure linear algebra over the parent run's COMMITTED last-prompt-token context
vectors (``{instruct,base}_context_vectors.pt``) and the committed response-mean
known directions (``instruct_known_directions.pt``). NO model forward pass, NO
generation, NO judge — every input is a frozen activation tensor downloaded from
the HF data repo ``superkaiba1/explore-persona-space-data``.

Used by:
  - ``scripts/issue685_signed_cosine_null.py`` (Part A -> delta_vs_u_signed.json)
  - ``scripts/issue685_matched_position_u.py`` (Part B -> delta_vs_u_matched_position.json)
  - ``scripts/issue685_figures_r2.py`` (figures, consumes the JSONs)

Plan: ``tasks/followups_running/685/plans/v5.md`` §3.5 (pseudocode) + §8 (smoke).
Construction recipe is the persona-vectors diff-in-means (2507.21509 Chen et al.
2025 + 2312.06681 Panickssery et al. 2023, CAA), read at the last-prompt-token
slot rather than the response-mean slot the parent's û_resp used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Experiment constants (frozen; ground truth from the parent run) ──────────

REPO_ID = "superkaiba1/explore-persona-space-data"
ANALYSIS_TENSORS_PIR = "issue685_context_shift/analysis_tensors"

LAYERS = [7, 14, 21, 27]
H = 3584
N_QUESTIONS = 20
B_NULL = 200
NULL_SEED = 42

# 10 contexts in the EXACT order the parent's metrics.json meta.context_names
# uses (asserted at load against each tensor's metadata.context_names).
CONTEXTS = [
    "assistant",
    "software_engineer",
    "villain",
    "kindergarten_teacher",
    "medical_doctor",
    "librarian",
    "french_person",
    "police_officer",
    "comedian",
    "data_scientist",
]

# 6 behaviors, parent order.
BEHAVIORS = ["sycophancy", "refusal", "evil", "hedging", "terse", "formal"]

# The 4-context subset the parent BUILT û_resp from (issue685_known_directions.py:60).
# Part B builds û_match from the SAME subset so the back-comparison is apples-to-apples
# at the û-construction level. Both û's are APPLIED to all 10 contexts at projection time.
SUBSET_CONTEXTS = ["assistant", "software_engineer", "villain", "medical_doctor"]
HELDOUT_CONTEXTS = [c for c in CONTEXTS if c not in SUBSET_CONTEXTS]


# ── Loaders (HF data repo; verify metadata matches plan expectations) ────────


def _hf_path(filename: str) -> str:
    """Resolve a committed analysis tensor from the HF data repo (dataset type)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        REPO_ID,
        f"{ANALYSIS_TENSORS_PIR}/{filename}",
        repo_type="dataset",
        revision="main",
    )


def load_context_vectors(model_tag: str) -> dict:
    """Load + validate ``{model_tag}_context_vectors.pt`` from HF.

    ``model_tag`` ∈ {"instruct", "base"}. Asserts the read position, hidden dim,
    layer set, context-name order, and the 70-condition completeness the plan
    requires (10 contexts x {bare, 6 behaviors}). Returns the loaded payload
    dict augmented with a ``name_to_idx`` lookup.
    """
    assert model_tag in ("instruct", "base"), model_tag
    cv = torch.load(_hf_path(f"{model_tag}_context_vectors.pt"), weights_only=False)
    meta = cv["metadata"]
    # Recipe / shape pins (artifact-reuse fitness check (a), §12).
    assert meta["read_position"] == "last_prompt_token (add_generation_prompt=True)", (
        model_tag,
        meta["read_position"],
    )
    assert meta["hidden_dim"] == H, (model_tag, meta["hidden_dim"])
    assert meta["n_questions"] == N_QUESTIONS, (model_tag, meta["n_questions"])
    assert list(cv["centroids"].keys()) == LAYERS, (model_tag, list(cv["centroids"].keys()))
    assert meta["context_names"] == CONTEXTS, (model_tag, meta["context_names"])
    assert meta["behavior_names"] == BEHAVIORS, (model_tag, meta["behavior_names"])
    # 70 conditions present, all bare__{c} / {c}__{b} keys.
    names = cv["condition_names"]
    assert len(names) == 70, (model_tag, len(names))
    name_to_idx = {n: i for i, n in enumerate(names)}
    for c in CONTEXTS:
        assert f"bare__{c}" in name_to_idx, (model_tag, c)
        for b in BEHAVIORS:
            assert f"{c}__{b}" in name_to_idx, (model_tag, c, b)
    for layer in LAYERS:
        assert tuple(cv["centroids"][layer].shape) == (70, H), (
            model_tag,
            layer,
            tuple(cv["centroids"][layer].shape),
        )
    cv["name_to_idx"] = name_to_idx
    return cv


def load_response_mean_u() -> dict:
    """Load + validate ``instruct_known_directions.pt`` (response-mean û, instruct-only).

    Asserts the response-mean diff-in-means recipe, the 4-context build subset,
    and the (3584,) direction shape per (behavior, layer). Returns the payload.
    """
    kd = torch.load(_hf_path("instruct_known_directions.pt"), weights_only=False)
    meta = kd["metadata"]
    assert meta["recipe"] == "response_mean_diff_in_means (persona-vectors 2507.21509 recipe b)", (
        meta["recipe"]
    )
    assert meta["subset_contexts"] == SUBSET_CONTEXTS, meta["subset_contexts"]
    for b in BEHAVIORS:
        assert b in kd["directions"], b
        for layer in LAYERS:
            assert layer in kd["directions"][b], (b, layer)
            assert tuple(kd["directions"][b][layer].shape) == (H,), (
                b,
                layer,
                tuple(kd["directions"][b][layer].shape),
            )
    return kd


# ── Core linear algebra ──────────────────────────────────────────────────────


def reconstruct_delta(cv: dict, context: str, behavior: str, layer: int) -> torch.Tensor:
    """Δ_l(C,b) = centroid(``{C}__{b}``, l) - centroid(``bare__{C}``, l).

    Returns the (H,) float shift vector.
    """
    idx = cv["name_to_idx"]
    v_bare = cv["centroids"][layer][idx[f"bare__{context}"]].float()
    v_aug = cv["centroids"][layer][idx[f"{context}__{behavior}"]].float()
    d = v_aug - v_bare
    assert d.shape == (H,), d.shape
    return d


def matched_position_u(cv: dict, behavior: str, layer: int) -> torch.Tensor:
    """û_match_l(b) = mean over the 4 SUBSET_CONTEXTS of Δ_l(C,b).

    The persona-vectors diff-in-means recipe built at the last-prompt-token slot
    (vs the parent's response-mean slot). 0 GPU — pure mean of committed centroids.
    """
    diffs = torch.stack([reconstruct_delta(cv, c, behavior, layer) for c in SUBSET_CONTEXTS])
    u = diffs.mean(dim=0)
    assert u.shape == (H,), u.shape
    return u


def signed_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Signed cosine (NOT |·|). Norm-invariant by construction."""
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def null_band(delta: torch.Tensor, *, b_null: int = B_NULL, seed: int = NULL_SEED) -> dict:
    """Matched-norm random-direction null: cos(Δ, random_unit) over ``b_null`` draws.

    Cosine is norm-invariant, so a random UNIT-direction null against Δ/‖Δ‖ IS
    the matched-norm null (plan §11 / assumption 10). The draw is deterministic
    (``np.random.default_rng(seed)``, plan §3.5 brief). Returns null mean, std,
    IQR (25th, 75th) and the 95th percentile.
    """
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((b_null, H)).astype(np.float64)
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    d = (delta / delta.norm()).double().numpy()
    cs = r @ d  # (b_null,) cosines of Δ̂ vs random unit directions
    return {
        "mean": float(cs.mean()),
        "std": float(cs.std()),
        "iqr": [float(np.percentile(cs, 25)), float(np.percentile(cs, 75))],
        "p95": float(np.percentile(cs, 95)),
    }


def z_score(signed: float, null: dict) -> float:
    """Per-cell z = (signed cosine - null mean) / null std (with a div-by-0 guard)."""
    return float((signed - null["mean"]) / (null["std"] + 1e-9))


# ── Mean-subtracted diagnostic (carry-forward item 3) ────────────────────────


def context_mean_delta(cv: dict, behavior: str, layer: int) -> torch.Tensor:
    """Per-(behavior, layer) mean of Δ across the 10 contexts (for mean-subtraction)."""
    diffs = torch.stack([reconstruct_delta(cv, c, behavior, layer) for c in CONTEXTS])
    return diffs.mean(dim=0)


def subset_mean_u_component(u: torch.Tensor) -> torch.Tensor:
    """û is already a SUBSET-context mean; mean-subtraction subtracts the same
    grand-mean object from both sides. For a single (behavior, layer) the û is a
    single vector, so its 'mean across the build subset' is itself — this helper
    returns the û unchanged and exists only to document the symmetry with
    ``context_mean_delta`` (we mean-subtract Δ; û is already a centred mean of
    the build-subset diffs)."""
    return u


def mean_subtracted_signed_cos(
    cv: dict, context: str, behavior: str, layer: int, u: torch.Tensor
) -> float:
    """Signed cosine after subtracting the per-(behavior,layer) mean of Δ across
    the 10 contexts from Δ (the same trick the body's
    ``consistency_cosine_mean_subtracted`` uses). Disambiguates a shared direction
    from a context-specific residual at the matched position.

    û is the matched/response build-subset mean diff; Δ has its 10-context grand
    mean removed before the cosine. Returns the signed cosine of the residual Δ
    against û.
    """
    d = reconstruct_delta(cv, context, behavior, layer)
    d_resid = d - context_mean_delta(cv, behavior, layer)
    return signed_cos(d_resid, u)


# ── Summary helpers ──────────────────────────────────────────────────────────


def aggregate(values: list[float]) -> dict:
    """{mean_signed, frac_positive} over a list of signed cosines."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_signed": float(arr.mean()),
        "frac_positive": float(np.mean(arr > 0)),
    }


def save_npy(arr: torch.Tensor, path: Path) -> None:
    """Save a (H,) tensor as float64 .npy, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.double().numpy())
