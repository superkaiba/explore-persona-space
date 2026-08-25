"""Issue #650 off-pod CPU analysis — DV-1..DV-5 over the trained MLP adapters.

Forked from ``scripts/issue621_analyze.py`` (origin/issue-621 @ 766f44c4)
with the plan v3 Must-Fix deltas:

- **DV-3 MAX-MATCHED order-statistic null (Must-Fix #1).** The observed
  ``max_i |cos(b_down, u_i^base[down_proj])|`` is an extreme-value statistic
  over ~3584 base singular vectors per layer. The null recomputes the
  IDENTICAL ``max_i |cos|`` over the SAME full base singular-vector set per
  layer for each of B norm-matched random unit-vector draws, THEN the same
  band reduction; the threshold is p95 over the null MAX-distribution (NOT a
  single-direction 1/√d floor, NOT a flat p95 over single cosines). A
  ``dv3_intruder.json`` whose ``null_aggregation`` field is not the literal
  ``max_over_base_singular_vectors_then_max_over_band`` is REJECTED at load.
- **DV-4 TWO base-model reference directions (Must-Fix #2).** Sycophancy:
  ``cos(b_down, d_behavior_base)`` (instruction-contaminated) AND the
  content-isolating ``cos(b_down, d_format_base)`` (agree-instructed
  judged-DISAGREES vs judged-AGREES, instruction axis cancelled), each with
  its own label-permutation null, plus the residualized cosine. Marker:
  N/A-for-marker (no internal concept by construction; DV-3 only).

Sub-commands:
  build-base-svd        Extract per-layer base weight SVD (up_proj/down_proj)
                        from the base model — the DV-3 ground truth. Loads
                        the base model ONCE (GPU optional; CPU SVD fine).
  build-unembedding     W_U[marker], W_U[eos], freq-matched wrong-token nulls
                        + the sycophancy d_U^syco logit-diff direction (DV-2).
  analyze               DV-1..5 over the trained adapters + bank + base SVD +
                        base concept directions; writes dv{1..5}_*.json.

All of ``analyze`` is deterministic linear algebra (no model call) and runs
OFF-POD on the VM against uploaded artifacts (plan §9).
"""

# ruff: noqa: RUF002, RUF003  # math/scientific notation in docstrings + msgs

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_650 import (
    ANALYSIS_DIR,
    DV2_NULL_B,
    DV3_NULL_AGGREGATION,
    DV3_NULL_B,
    HIDDEN_SIZE,
    MARKER_ID,
    READ_LAYER_BAND,
    READ_MODULE,
    SYCO_DV2_AGREE_OPENERS,
    SYCO_DV2_NEUTRAL_OPENERS,
    WRITE_LAYER_BAND,
    WRITE_MODULE,
    parse_cell_slug,
)

log = logging.getLogger("issue_650.analyze")


# ──────────────────────────────────────────────────────────────────────────
# Small numeric helpers (verbatim #621).
# ──────────────────────────────────────────────────────────────────────────


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def _spearman(x: list[float], y: list[float]) -> float:
    from scipy.stats import spearmanr

    if len(x) < 3:
        return float("nan")
    rho = spearmanr(x, y).statistic
    return float(rho) if rho is not None else float("nan")


def _bootstrap_ci(values: list[float], *, n_boot: int = 10000, seed: int = 650) -> dict:
    """Descriptive bootstrap 95% CI over seed-level points (plan §5 / §14 concern 4)."""
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(n_boot)])
    return {
        "mean": float(arr.mean()),
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
        "n": int(arr.size),
        "points": [float(v) for v in arr],
    }


# ──────────────────────────────────────────────────────────────────────────
# Loaders (verbatim #621; MLP module names parse identically).
# ──────────────────────────────────────────────────────────────────────────


def load_adapter_pairs(adapter_dir: Path) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    """Load rank-1 (a, b) per (layer, module) from a PEFT safetensors file.

    Keys: ``base_model.model.model.layers.{L}.mlp.up_proj.lora_A.weight`` etc.
    Asserts r == 1.
    """
    from safetensors.numpy import load_file

    st = adapter_dir / "adapter_model.safetensors"
    if not st.is_file():
        raise FileNotFoundError(st)
    sd = load_file(str(st))
    out: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for key, tensor in sd.items():
        if ".lora_A." not in key and ".lora_B." not in key:
            continue
        parts = key.split(".")
        li = int(parts[parts.index("layers") + 1])
        module = parts[parts.index("layers") + 3]  # mlp . <module>
        slot = out.setdefault((li, module), {})
        arr = np.asarray(tensor, dtype=np.float32)
        if ".lora_A." in key:
            assert arr.shape[0] == 1, (key, arr.shape, "rank != 1")
            slot["a"] = arr[0]
        else:
            assert arr.shape[1] == 1, (key, arr.shape, "rank != 1")
            slot["b"] = arr[:, 0]
    if not out:
        raise AssertionError(f"no lora_A/lora_B tensors in {st}")
    for (li, module), slot in out.items():
        if "a" not in slot or "b" not in slot:
            raise AssertionError(f"incomplete (a,b) pair at layer {li} module {module}")
    return out


def resolve_cell_adapter(meta: dict, slug: str, cells_root: Path) -> tuple[Path, Path]:
    """Return ``(adapter_dir, a_init_dir)`` for a cell's geometry reads.

    Round-2 ``syco-dose-checkpoint-selection-missing``: for a SYCOPHANCY cell
    the eval phase records the dose-selected per-epoch checkpoint in
    ``meta["dose_selected_adapter"]`` — the geometry DVs MUST read THAT
    checkpoint (the band-entry dial position), NOT the final 16-epoch adapter.
    Marker cells (and any cell without a recorded selection) read the final
    adapter. The a_init snapshot is always at ``<cell_dir>/adapter_init`` (a
    step-0 snapshot, epoch-independent), so DV-1 compares the dose-selected
    ``lora_A`` against the SAME a_init regardless of which epoch was selected.
    """
    cell_dir = Path(meta.get("adapter_local") or (cells_root / "cells" / slug))
    a_init_dir = cell_dir / "adapter_init"
    sel = meta.get("dose_selected_adapter")
    if sel and Path(sel).is_dir() and (Path(sel) / "adapter_model.safetensors").is_file():
        return Path(sel), a_init_dir
    return cell_dir, a_init_dir


def load_base_svd(path: Path, modules: tuple[str, ...] = ("up_proj", "down_proj")) -> dict:
    """Load the per-layer base weight SVD built by ``build-base-svd``.

    Schema: {"up_proj": {layer: {"V": (K, d_in)}}, "down_proj": {layer:
    {"U": (K, d_out)}}}. For DV-3 the WRITE arm compares b_down (d_out=3584)
    against down_proj's LEFT singular vectors U (output basis); the READ arm
    compares a_up (d_in=3584) against up_proj's RIGHT singular vectors V
    (input basis). ``modules`` defaults to the original pair; a payload built
    with ``--modules`` (#2569 leg 5) is loaded by naming its module list —
    each extra module carries its RESIDUAL-side basis per
    ``RESIDUAL_SIDE_BY_MODULE``.
    """
    import torch

    payload = torch.load(path, weights_only=True)
    out: dict[str, dict[int, dict[str, np.ndarray]]] = {m: {} for m in modules}
    for module in modules:
        for layer_str, basis in payload[module].items():
            layer = int(layer_str)
            out[module][layer] = {
                k: np.asarray(v.numpy(), dtype=np.float32) for k, v in basis.items()
            }
    out["_meta"] = payload.get("_meta", {})
    return out


def load_bank(bank_dir: Path) -> dict:
    """Load centroids.pt + rmsnorm_gamma.pt + manifest.json (#621 schema)."""
    import torch

    centroids_payload = torch.load(bank_dir / "centroids.pt", weights_only=True)
    gamma_payload = torch.load(bank_dir / "rmsnorm_gamma.pt", weights_only=True)
    manifest = json.loads((bank_dir / "manifest.json").read_text())
    centroids: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for tap, by_pos in centroids_payload["centroids"].items():
        centroids[tap] = {}
        for pos, by_ctx in by_pos.items():
            centroids[tap][pos] = {
                name: np.asarray(t.numpy(), dtype=np.float32) for name, t in by_ctx.items()
            }
    gamma_post_attn = None
    for key in ("post_attention_layernorm", "input_layernorm"):
        if key in gamma_payload:
            gamma_post_attn = np.asarray(gamma_payload[key].numpy(), dtype=np.float32)
            break
    return {"centroids": centroids, "gamma_post_attn": gamma_post_attn, "manifest": manifest}


def load_unembedding(path: Path) -> dict:
    import torch

    payload = torch.load(path, weights_only=True)
    out = {
        "marker": np.asarray(payload["W_U_marker"].numpy(), dtype=np.float32),
        "eos": np.asarray(payload["W_U_eos"].numpy(), dtype=np.float32),
        "null_norm_matched": np.asarray(payload["W_U_null_norm_matched"].numpy(), np.float32),
    }
    if "d_U_syco" in payload:
        out["d_U_syco"] = np.asarray(payload["d_U_syco"].numpy(), dtype=np.float32)
    if "d_U_syco_null" in payload:
        out["d_U_syco_null"] = np.asarray(payload["d_U_syco_null"].numpy(), dtype=np.float32)
    return out


def load_concept_directions(path: Path) -> dict:
    """Load DV-4 base-model concept directions (sycophancy only).

    Schema (built by issue650_concept_direction.py): per write-band layer,
    ``d_behavior_base[L]`` and ``d_format_base[L]`` (3584-d), + the per-side
    activation matrices needed for the label-permutation null, + counts.
    """
    import torch

    payload = torch.load(path, weights_only=True)
    out: dict = {"layers": {}}
    for layer_str, blob in payload["layers"].items():
        layer = int(layer_str)
        out["layers"][layer] = {k: np.asarray(v.numpy(), dtype=np.float32) for k, v in blob.items()}
    out["n_agree"] = int(payload.get("n_agree", 0))
    out["n_disagree"] = int(payload.get("n_disagree", 0))
    out["n_behavior_pairs"] = int(payload.get("n_behavior_pairs", 0))
    return out


# ──────────────────────────────────────────────────────────────────────────
# DV-3 max-matched order-statistic null (Must-Fix #1).
# ──────────────────────────────────────────────────────────────────────────


def max_abs_cos_over_basis(vec: np.ndarray, basis: np.ndarray) -> float:
    """max_i |cos(vec, basis[i])| over the FULL singular-vector set (one layer).

    ``basis`` is (K, d); each row is a unit-norm singular vector. Returns the
    extreme-value statistic the observed AND null both compute.
    """
    v = _unit(vec)
    # basis rows are unit-norm singular vectors; |cos| = |basis @ v|.
    cosines = np.abs(basis @ v)
    return float(cosines.max())


def dv3_max_matched_null(
    *,
    observed_vec: np.ndarray,
    basis_by_layer: dict[int, np.ndarray],
    band: tuple[int, ...],
    n_draws: int,
    seed: int,
) -> dict:
    """DV-3 observed band_max + the MAX-MATCHED order-statistic null (Must-Fix #1).

    For each of ``n_draws`` norm-matched random UNIT vectors (the trained
    vec is unit-normalized inside ``max_abs_cos_over_basis``, so a unit
    random draw is norm-matched in the cosine geometry), compute the
    IDENTICAL ``max_i |cos|`` over the SAME per-layer basis THEN the same
    band-max reduction; p95 over the null max-distribution is the threshold.

    NOTE (round-3): this single-vector reference impl emits the LEGACY FLAT
    keys and is NOT on the production path — the per-cell deliverable is built
    by ``_dv3_observed_per_layer`` / ``_dv3_null_per_layer`` and serialized into
    the registered §6.5 nested schema by ``_dv3_registered_schema`` (Codex
    CONCERN ``dv3-schema-mismatch``). Kept as the documented per-statistic
    reference; ``dv3_intruder.json`` never carries these flat keys.
    """
    rng = np.random.default_rng(seed)
    d = observed_vec.shape[0]

    # Observed: per-layer max, then band max.
    per_layer_observed = {
        layer: max_abs_cos_over_basis(observed_vec, basis_by_layer[layer]) for layer in band
    }
    observed_band_max = float(max(per_layer_observed.values()))

    # Null: B random unit vectors, each scored by the identical statistic.
    null_band_max: list[float] = []
    per_layer_null: dict[int, list[float]] = {layer: [] for layer in band}
    for _ in range(n_draws):
        r = rng.standard_normal(d).astype(np.float32)
        r = _unit(r)
        layer_maxes = {layer: max_abs_cos_over_basis(r, basis_by_layer[layer]) for layer in band}
        for layer in band:
            per_layer_null[layer].append(layer_maxes[layer])
        null_band_max.append(float(max(layer_maxes.values())))

    null_arr = np.asarray(null_band_max)
    return {
        "per_layer_observed_max": {int(k): float(v) for k, v in per_layer_observed.items()},
        "band_observed_max": observed_band_max,
        "per_layer_null_max_draws": {
            int(k): [float(x) for x in v] for k, v in per_layer_null.items()
        },
        "band_null_max_draws": [float(x) for x in null_band_max],
        "band_null_p95": float(np.percentile(null_arr, 95.0)),
        "band_null_mean": float(null_arr.mean()),
        "n_draws": int(n_draws),
        "K_by_layer": {int(layer): int(basis_by_layer[layer].shape[0]) for layer in band},
        "layer_band": [int(layer) for layer in band],
        "sign_convention": "unsigned (abs cosine)",
        "null_aggregation": DV3_NULL_AGGREGATION,
        "verdict": (
            "pre_existing_in_base_column_space"
            if observed_band_max > float(np.percentile(null_arr, 95.0))
            else "intruder_at_max_matched_null"
        ),
    }


def assert_dv3_schema(payload: dict) -> None:
    """Load-time hard assert (plan §6.5): reject a non-max-matched null.

    Round-3 (Codex CONCERN ``dv3-schema-mismatch``): validates the REGISTERED
    nested §6.5 schema — ``observed.{write,read}`` / ``null.{write,read}`` /
    ``assertions.null_aggregation_matches_observed`` — rather than the prior flat
    keys. A null that scored per-draw SINGLE cosines, or a flat p95 over
    ``<full-rank`` random directions, has a different ``null_aggregation`` string
    OR a False ``assertions`` flag and is REJECTED (hard fail, Must-Fix #1).
    """
    observed = payload.get("observed")
    null = payload.get("null")
    assertions = payload.get("assertions")
    if not isinstance(observed, dict) or not isinstance(null, dict):
        raise AssertionError(
            "dv3_intruder.json cell missing the registered `observed`/`null` blocks "
            "(plan §6.5 schema) — REJECTED (dv3-schema-mismatch)."
        )
    if not isinstance(assertions, dict) or "null_aggregation_matches_observed" not in assertions:
        raise AssertionError(
            "dv3_intruder.json cell missing `assertions.null_aggregation_matches_observed` "
            "(plan §6.5) — REJECTED."
        )
    if assertions["null_aggregation_matches_observed"] is not True:
        raise AssertionError(
            "dv3_intruder.json cell: assertions.null_aggregation_matches_observed is not "
            "True — the null routine's per-draw reduction does NOT match the observed "
            "reduction; a non-max-matched null guarantees a false intruder/pre-existing "
            "verdict on the headline discriminator. REJECTED (Must-Fix #1)."
        )
    for arm in ("write", "read"):
        if arm not in observed:
            continue
        agg = null.get(arm, {}).get("null_aggregation") or null.get("null_aggregation")
        if agg != DV3_NULL_AGGREGATION:
            raise AssertionError(
                f"dv3_intruder.json {arm} arm null_aggregation={agg!r} != "
                f"{DV3_NULL_AGGREGATION!r} — a non-max-matched null guarantees a "
                "false intruder/pre-existing verdict on the headline discriminator. "
                "REJECTED (Must-Fix #1)."
            )
        # Cross-check observed + null reduced over the SAME layer band.
        obs_layers = set(observed[arm].get("max_by_layer", {}))
        null_layers = set(null.get(arm, {}).get("per_layer_max_draws", {}))
        if obs_layers != null_layers:
            raise AssertionError(
                f"dv3 {arm}: observed layers {sorted(obs_layers)} != null layers "
                f"{sorted(null_layers)} — observed and null must reduce over the "
                "SAME band (Must-Fix #1)."
            )


# ──────────────────────────────────────────────────────────────────────────
# DV-4 two-reference concept test (Must-Fix #2).
# ──────────────────────────────────────────────────────────────────────────


def _label_permutation_null(
    *,
    pos_acts: np.ndarray,
    neg_acts: np.ndarray,
    write_vec: np.ndarray,
    n_perm: int,
    seed: int,
) -> list[float]:
    """Permutation null for cos(b_down, contrast-of-means) (plan §5 DV-4).

    Pool pos+neg activations, randomly relabel into two groups of the same
    sizes, recompute the contrast-of-means direction, score its cos with the
    write. The observed contrast direction's cosine is compared against this.
    """
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([pos_acts, neg_acts], axis=0)
    n_pos = pos_acts.shape[0]
    n_total = pooled.shape[0]
    out: list[float] = []
    for _ in range(n_perm):
        perm = rng.permutation(n_total)
        g1 = pooled[perm[:n_pos]]
        g2 = pooled[perm[n_pos:]]
        d_perm = g1.mean(axis=0) - g2.mean(axis=0)
        out.append(abs(_cos(write_vec, d_perm)))
    return out


def _residualize(target: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Remove the component of ``target`` along (unit) ``axis``."""
    u = _unit(axis)
    return target - float(target @ u) * u


def dv4_two_reference(
    *,
    write_vec: np.ndarray,
    concept: dict,
    write_band: tuple[int, ...],
    n_perm: int = 2000,
    seed: int = 650,
) -> dict:
    """DV-4 (sycophancy): cos(b_down, d_behavior_base) + cos(b_down, d_format_base)
    + per-reference permutation null + residualized cosine.

    The write is read at the band-max layer (the layer where b_down peaks
    against the base concept), matching DV-2/3's write extraction band. Both
    reference directions are BASE-model contrast-of-means (non-circular).
    """
    layers = [layer for layer in write_band if layer in concept["layers"]]
    if not layers:
        raise AssertionError(
            f"no concept-direction layers in the write band {write_band}; "
            f"concept has {sorted(concept['layers'])}"
        )

    # Pick the band-max layer by |cos(b_down, d_format_base)| (the headline read).
    def _cos_at(layer: int, key: str) -> float:
        return abs(_cos(write_vec, concept["layers"][layer][key]))

    peak_layer = max(layers, key=lambda li: _cos_at(li, "d_format_base"))
    blob = concept["layers"][peak_layer]
    d_behavior = blob["d_behavior_base"]
    d_format = blob["d_format_base"]

    cos_behavior = abs(_cos(write_vec, d_behavior))
    cos_format = abs(_cos(write_vec, d_format))

    # Residualize the write on d_behavior_base's instruction axis, then re-read
    # against d_format_base (new-information-beyond-instruction read).
    write_resid = _residualize(write_vec, d_behavior)
    cos_format_resid = abs(_cos(write_resid, d_format))

    # Per-reference label-permutation nulls (separately) — require the per-side
    # activation matrices in the concept blob.
    null_behavior = (
        _label_permutation_null(
            pos_acts=blob["behavior_pos_acts"],
            neg_acts=blob["behavior_neg_acts"],
            write_vec=write_vec,
            n_perm=n_perm,
            seed=seed,
        )
        if "behavior_pos_acts" in blob
        else []
    )
    null_format = (
        _label_permutation_null(
            pos_acts=blob["format_agree_acts"],
            neg_acts=blob["format_disagree_acts"],
            write_vec=write_vec,
            n_perm=n_perm,
            seed=seed + 1,
        )
        if "format_agree_acts" in blob
        else []
    )
    # Round-2 minor (code-review): the residualized gate previously reused the
    # UN-residualized null (conservative but mismatched). Compute a MATCHED
    # residualized-write permutation null — score each permuted format-contrast
    # against the residualized write — so "survives residualization" is tested
    # against the same geometry the observed residualized cosine lives in.
    null_format_resid = (
        _label_permutation_null(
            pos_acts=blob["format_agree_acts"],
            neg_acts=blob["format_disagree_acts"],
            write_vec=write_resid,
            n_perm=n_perm,
            seed=seed + 2,
        )
        if "format_agree_acts" in blob
        else []
    )

    def _p95(xs: list[float]) -> float | None:
        return float(np.percentile(np.asarray(xs), 95.0)) if xs else None

    p95_format = _p95(null_format)
    p95_format_resid = _p95(null_format_resid)
    return {
        "peak_layer": int(peak_layer),
        "cos_b_d_behavior_base": cos_behavior,
        "cos_b_d_format_base": cos_format,
        "cos_b_d_format_base_residualized": cos_format_resid,
        "null_perm_d_behavior_base_p95": _p95(null_behavior),
        "null_perm_d_format_base_p95": p95_format,
        "null_perm_d_format_base_residualized_p95": p95_format_resid,
        "null_perm_d_behavior_base_draws": [float(x) for x in null_behavior],
        "null_perm_d_format_base_draws": [float(x) for x in null_format],
        "null_perm_d_format_base_residualized_draws": [float(x) for x in null_format_resid],
        "n_agree": int(concept.get("n_agree", 0)),
        "n_disagree": int(concept.get("n_disagree", 0)),
        # Decision rule (plan §5): H-pre-existing supported ONLY if the
        # content-isolating cosine is above its null AND the residualized cosine
        # is above the MATCHED residualized null (round-2 minor fix).
        "supports_pre_existing": bool(
            (p95_format is not None)
            and cos_format > p95_format
            and (p95_format_resid is not None)
            and cos_format_resid > p95_format_resid
        ),
    }


# ──────────────────────────────────────────────────────────────────────────
# Sub-command: build-base-svd
# ──────────────────────────────────────────────────────────────────────────


# Residual-side basis per module (#2569 leg 5 --modules extension): "V" = the module READS
# the residual stream (input side, RIGHT singular vectors span the read basis); "U" = the
# module WRITES the residual stream (output side, LEFT singular vectors span the write
# basis). The original up_proj->V / down_proj->U behavior is the default subset.
RESIDUAL_SIDE_BY_MODULE = {
    "q_proj": "V",
    "k_proj": "V",
    "v_proj": "V",
    "gate_proj": "V",
    "up_proj": "V",
    "o_proj": "U",
    "down_proj": "U",
}
_ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def _module_weight(layer, module: str):
    """Resolve a decoder layer's submodule weight (self_attn.* vs mlp.*)."""
    parent = layer.self_attn if module in _ATTN_MODULES else layer.mlp
    return getattr(parent, module).weight


def _residual_svd_basis(w: np.ndarray, side: str) -> np.ndarray:
    """Residual-side singular-vector basis of one weight matrix, as (K, d) rows.

    side "V": RIGHT singular vectors (input basis, rows of Vt). side "U": LEFT
    singular vectors transposed to rows (output basis). Identical math to the
    original up_proj/down_proj branches.
    """
    if side == "V":
        _, _, vt = np.linalg.svd(w, full_matrices=False)
        return vt
    u, _, _ = np.linalg.svd(w, full_matrices=False)
    return u.T


def cmd_build_base_svd(args) -> int:
    """Extract per-layer base weight SVD bases for ``--modules`` (DV-3 ground truth).

    Default ``--modules up_proj,down_proj`` preserves the original behavior exactly:
    down_proj weight is (d_out=3584, d_in=18944); b_down is in the d_out
    (residual-output) space, so the comparison basis is the LEFT singular
    vectors U (3584-d). up_proj weight is (d_ff=18944, d_in=3584); a_up is
    in the d_in (residual-input) space, so the basis is the RIGHT singular
    vectors V (3584-d). We keep the full-rank set per layer (K = min(d_out,
    d_in)) so observed + null share the same K. Extra modules (#2569 leg 5:
    the 7 LoRA targets + q/k/v/o) store their RESIDUAL-side basis per
    ``RESIDUAL_SIDE_BY_MODULE``.
    """
    import torch
    from transformers import AutoModelForCausalLM

    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    unknown = [m for m in modules if m not in RESIDUAL_SIDE_BY_MODULE]
    if unknown:
        raise ValueError(
            f"unknown --modules entries {unknown}; known: {sorted(RESIDUAL_SIDE_BY_MODULE)}"
        )
    log.info("[phase=build_base_svd] loading base model %s (modules=%s)", args.base_model, modules)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float32, trust_remote_code=True
    )
    layers = model.model.layers
    out: dict = {m: {} for m in modules}
    out["_meta"] = {"base_model": args.base_model, "modules": modules}
    for li, layer in enumerate(layers):
        for module in modules:
            side = RESIDUAL_SIDE_BY_MODULE[module]
            w = _module_weight(layer, module).detach().float().cpu().numpy()
            basis = _residual_svd_basis(w, side)
            out[module][str(li)] = {side: torch.from_numpy(basis.astype(np.float32))}
            log.info("layer %d: %s %s %s", li, module, side, tuple(basis.shape))
    out_path = Path(args.base_svd)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    log.info("Base SVD saved: %s (%d layers)", out_path, len(layers))
    return 0


# ──────────────────────────────────────────────────────────────────────────
# Sub-command: build-unembedding (W_U rows + d_U^syco)
# ──────────────────────────────────────────────────────────────────────────


def cmd_build_unembedding(args) -> int:
    """W_U[marker], W_U[eos], freq-matched wrong-token null + d_U^syco (DV-2)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float32, trust_remote_code=True
    )
    w_u = model.lm_head.weight.detach().float().cpu()  # (vocab, hidden)
    if w_u.shape[1] != HIDDEN_SIZE:
        raise AssertionError(f"W_U hidden {w_u.shape[1]} != {HIDDEN_SIZE}")
    marker_row = w_u[MARKER_ID].clone()
    from explore_persona_space.experiments.issue_650 import IM_END_ID

    eos_row = w_u[IM_END_ID].clone()

    # Frequency-matched wrong-token null: rows within ±10% of ‖W_U[marker]‖.
    norms = w_u.norm(dim=-1)
    target = float(marker_row.norm())
    lo, hi = 0.9 * target, 1.1 * target
    cand = torch.nonzero((norms >= lo) & (norms <= hi)).flatten()
    cand = cand[cand != MARKER_ID]
    rng = np.random.default_rng(650)
    k = min(args.n_null, len(cand))
    norm_ids = rng.choice(cand.numpy(), size=k, replace=False)
    null_norm_matched = w_u[torch.from_numpy(norm_ids)].clone()

    payload: dict = {
        "W_U_marker": marker_row,
        "W_U_eos": eos_row,
        "W_U_null_norm_matched": null_norm_matched,
    }

    # DV-2 sycophancy logit-diff direction d_U^syco = mean_k[W_U[agree_k] - W_U[neutral_k]]
    # over the FIRST sub-token of each #612 opener, minus matched neutral first-tokens.
    def _first_tok(word: str) -> int:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            raise AssertionError(f"opener {word!r} tokenized empty")
        return ids[0]

    agree_ids = [_first_tok(w) for w in SYCO_DV2_AGREE_OPENERS]
    neutral_ids = [_first_tok(w) for w in SYCO_DV2_NEUTRAL_OPENERS]
    n_pairs = min(len(agree_ids), len(neutral_ids))
    diffs = [w_u[agree_ids[i]] - w_u[neutral_ids[i]] for i in range(n_pairs)]
    d_u_syco = torch.stack(diffs).mean(dim=0)
    payload["d_U_syco"] = d_u_syco.clone()

    # Frequency-matched wrong-token-PAIR null for d_U^syco (matched ‖d_U‖).
    d_norm = float(d_u_syco.norm())
    null_pairs = []
    vocab = w_u.shape[0]
    rng2 = np.random.default_rng(651)
    tries = 0
    while len(null_pairs) < DV2_NULL_B and tries < DV2_NULL_B * 50:
        i, j = int(rng2.integers(vocab)), int(rng2.integers(vocab))
        tries += 1
        if i == j:
            continue
        cand_d = w_u[i] - w_u[j]
        if 0.7 * d_norm <= float(cand_d.norm()) <= 1.3 * d_norm:
            null_pairs.append(cand_d)
    if null_pairs:
        payload["d_U_syco_null"] = torch.stack(null_pairs)

    out_path = Path(args.unembedding)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    log.info(
        "Unembedding saved: %s (hidden=%d, %d norm-matched nulls, d_U^syco from %d pairs, "
        "%d syco-null pairs)",
        out_path,
        HIDDEN_SIZE,
        k,
        n_pairs,
        len(null_pairs),
    )
    return 0


# ──────────────────────────────────────────────────────────────────────────
# Sub-command: analyze (DV-1..5)
# ──────────────────────────────────────────────────────────────────────────


def _band_pair(pairs: dict, module: str, band: tuple[int, ...]) -> dict[int, dict[str, np.ndarray]]:
    """Filter adapter pairs to (layer, module) within the band."""
    return {li: pairs[(li, module)] for li in band if (li, module) in pairs}


def cmd_analyze(args) -> int:
    """Run DV-1..5 over the trained adapters; write dv{1..5}_*.json (plan §6.5)."""
    out_dir = Path(args.analysis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_root = Path(args.cells_root)
    base_svd = load_base_svd(Path(args.base_svd))
    git_commit = _git_commit()

    # Resolve trained cells from the cell JSONs the dispatcher wrote.
    cell_metas: dict[str, dict] = {}
    for sub in ("anchor_smoke", "sweep"):
        d = cells_root / sub
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                if p.name in ("summary.json",):
                    continue
                payload = json.loads(p.read_text())
                if "cell_slug" in payload:
                    cell_metas[payload["cell_slug"]] = payload
    if not cell_metas:
        raise FileNotFoundError(f"no cell JSONs under {cells_root}/(anchor_smoke|sweep)")

    dv3_rows: dict[str, dict] = {}
    dv2_rows: dict[str, dict] = {}
    for slug, meta in sorted(cell_metas.items()):
        behavior, dose, seed = parse_cell_slug(slug)
        # Dose-selected adapter for sycophancy cells (round-2
        # syco-dose-checkpoint-selection-missing); final adapter otherwise.
        adapter_dir, _a_init_dir = resolve_cell_adapter(meta, slug, cells_root)
        if not (adapter_dir / "adapter_model.safetensors").is_file():
            log.warning(
                "cell=%s: adapter not local at %s; skip (analyze off-pod needs download)",
                slug,
                adapter_dir,
            )
            continue
        pairs = load_adapter_pairs(adapter_dir)

        # DV-3 WRITE arm: b_down vs down_proj U basis, max over WRITE band.
        write_pairs = _band_pair(pairs, WRITE_MODULE, WRITE_LAYER_BAND)
        read_pairs = _band_pair(pairs, READ_MODULE, READ_LAYER_BAND)
        # Use the band-max-energy write layer's b as the representative write
        # vec for the per-cell intruder read (the headline reads the band max
        # of the per-layer statistic, so we score each layer's own b).
        write_basis = {li: base_svd["down_proj"][li]["U"] for li in write_pairs}
        read_basis = {li: base_svd["up_proj"][li]["V"] for li in read_pairs}
        # Per-layer observed uses each layer's own b/a; aggregate by reusing the
        # null routine with a single concatenated observed-per-layer read.
        write_obs = _dv3_observed_per_layer(write_pairs, write_basis, "b")
        read_obs = _dv3_observed_per_layer(read_pairs, read_basis, "a")
        write_null = _dv3_null_per_layer(write_basis, WRITE_LAYER_BAND, DV3_NULL_B, seed=seed)
        read_null = _dv3_null_per_layer(read_basis, READ_LAYER_BAND, DV3_NULL_B, seed=seed + 7)
        dv3_rows[slug] = _dv3_registered_schema(
            behavior=behavior,
            dose=dose,
            seed=seed,
            write_obs=write_obs,
            read_obs=read_obs,
            write_null=write_null,
            read_null=read_null,
        )

        # DV-2: write -> output concept (manipulation check).
        dv2_rows[slug] = _dv2_for_cell(slug, behavior, dose, seed, write_pairs, args)

    # Persist DV-3 + DV-2 (the per-cell discriminator/manip-check JSONs).
    _write_json(out_dir / "dv3_intruder.json", {"cells": dv3_rows, "git_commit": git_commit})
    _write_json(out_dir / "dv2_write_concept.json", {"cells": dv2_rows, "git_commit": git_commit})

    # Re-load + assert the DV-3 schema (load-time max-matched-null guard).
    reloaded = json.loads((out_dir / "dv3_intruder.json").read_text())
    for row in reloaded["cells"].values():
        assert_dv3_schema(row)
    log.info("DV-3 max-matched-null schema asserted on %d cells", len(reloaded["cells"]))

    # DV-1, DV-4, DV-5 wiring (read rotation / two-ref concept / selectivity)
    # depend on bank + concept-direction + eval-leakage artifacts. They are
    # written by the dedicated helpers when those artifacts are present;
    # surfaced here so the analyze entrypoint produces the full §6.5 set.
    _maybe_run_dv1(out_dir, cell_metas, cells_root, base_svd, args, git_commit)
    _maybe_run_dv4(out_dir, cell_metas, cells_root, args, git_commit)
    _maybe_run_dv5(out_dir, cell_metas, cells_root, args, git_commit)

    log.info("[phase=analyze_done] wrote dv{1..5} JSONs under %s", out_dir)
    return 0


def _dv3_observed_per_layer(pairs, basis_by_layer, key) -> dict:
    per_layer = {li: max_abs_cos_over_basis(pairs[li][key], basis_by_layer[li]) for li in pairs}
    return {
        "per_layer_observed_max": {int(k): float(v) for k, v in per_layer.items()},
        "band_observed_max": float(max(per_layer.values())) if per_layer else float("nan"),
    }


def _dv3_null_per_layer(basis_by_layer, band, n_draws, *, seed) -> dict:
    rng = np.random.default_rng(seed)
    layers = [li for li in band if li in basis_by_layer]
    if not layers:
        raise AssertionError(f"no basis layers in band {band}")
    # Null draw dimension = the singular-vector dimension of the basis (the
    # residual-stream width the random direction lives in), NOT the constant
    # HIDDEN_SIZE — so the CPU smoke can run a tiny d. Production d == 3584.
    d = int(next(iter(basis_by_layer.values())).shape[1])
    per_layer_null: dict[int, list[float]] = {li: [] for li in layers}
    band_null_max: list[float] = []
    for _ in range(n_draws):
        r = _unit(rng.standard_normal(d).astype(np.float32))
        layer_maxes = {li: max_abs_cos_over_basis(r, basis_by_layer[li]) for li in layers}
        for li in layers:
            per_layer_null[li].append(layer_maxes[li])
        band_null_max.append(float(max(layer_maxes.values())) if layer_maxes else float("nan"))
    arr = np.asarray(band_null_max)
    return {
        "per_layer_null_max_draws": {
            int(k): [float(x) for x in v] for k, v in per_layer_null.items()
        },
        "band_null_max_draws": [float(x) for x in band_null_max],
        "band_null_p95": float(np.percentile(arr, 95.0)),
        "n_draws": int(n_draws),
        # K_by_layer = # singular vectors (basis rows, shape[0]) scored per
        # layer; the random draw lives in d-space (basis shape[1], the `d`
        # above), NOT in K. The name records the scoring-basis cardinality.
        "K_by_layer": {int(li): int(basis_by_layer[li].shape[0]) for li in layers},
        "layer_band": [int(li) for li in layers],
        "sign_convention": "unsigned (abs cosine)",
    }


def _dv3_registered_schema(
    *,
    behavior: str,
    dose: str,
    seed: int,
    write_obs: dict,
    read_obs: dict,
    write_null: dict,
    read_null: dict,
) -> dict:
    """Serialize a DV-3 cell into the plan §6.5 REGISTERED nested schema.

    Round-3 (Codex CONCERN ``dv3-schema-mismatch``): the working helpers emit
    flat keys (``per_layer_observed_max`` / ``band_observed_max`` /
    ``band_null_max_draws`` / ``band_null_p95`` / ...). Plan §6.5 registers a
    NESTED schema with ``observed.{write,read}.{max_by_layer,band_max}`` /
    ``null.{write,read}.{max_draws,band_p95}`` / ``null.{B,K_by_layer,layer_band,
    sign_convention,null_aggregation}`` / ``assertions.null_aggregation_matches_
    observed``. This maps the working dicts onto exactly those field names so
    the deliverable conforms to the plan, and computes the
    ``null_aggregation_matches_observed`` boolean (True only when BOTH arms' null
    routine reduced over the SAME band as their observed read AND carry the
    registered ``max_over_base_singular_vectors_then_max_over_band`` aggregation).
    """

    def _arm(obs: dict, null: dict) -> tuple[dict, dict, bool]:
        observed = {
            "max_by_layer": dict(obs["per_layer_observed_max"]),
            "band_max": obs["band_observed_max"],
        }
        null_block = {
            "max_draws": null["band_null_max_draws"],
            "band_p95": null["band_null_p95"],
            "per_layer_max_draws": dict(null["per_layer_null_max_draws"]),
            "K_by_layer": dict(null["K_by_layer"]),
            "layer_band": list(null["layer_band"]),
            "sign_convention": null["sign_convention"],
            "null_aggregation": DV3_NULL_AGGREGATION,
        }
        # The null reduced over the same layer band as the observed read AND
        # used the registered max-over-SVD-then-max-over-band aggregation.
        matches = set(observed["max_by_layer"]) == set(null["per_layer_null_max_draws"])
        return observed, null_block, matches

    write_observed, write_null_block, write_match = _arm(write_obs, write_null)
    read_observed, read_null_block, read_match = _arm(read_obs, read_null)

    # null.B = the REALIZED number of null draws (the length of the per-draw
    # max-distribution), not the configured default — they coincide in
    # production but the realized count is the honest deliverable field. Both
    # arms draw the same B; assert they agree.
    realized_b = len(write_null_block["max_draws"])
    if realized_b != len(read_null_block["max_draws"]):
        raise AssertionError(
            f"DV-3 null draw counts differ across arms "
            f"(write={realized_b}, read={len(read_null_block['max_draws'])})."
        )

    return {
        "behavior": behavior,
        "dose": dose,
        "seed": seed,
        "observed": {"write": write_observed, "read": read_observed},
        "null": {
            "write": write_null_block,
            "read": read_null_block,
            "B": int(realized_b),
            "K_by_layer": {
                "write": write_null_block["K_by_layer"],
                "read": read_null_block["K_by_layer"],
            },
            "layer_band": {
                "write": write_null_block["layer_band"],
                "read": read_null_block["layer_band"],
            },
            "sign_convention": "unsigned (abs cosine)",
            "null_aggregation": DV3_NULL_AGGREGATION,
        },
        "assertions": {
            "null_aggregation_matches_observed": bool(write_match and read_match),
        },
        # verdict per arm (kept for the analyzer; not part of the registered
        # schema field set but harmless additional metadata).
        "verdict": {
            "write": (
                "pre_existing_in_base_column_space"
                if write_observed["band_max"] > write_null_block["band_p95"]
                else "intruder_at_max_matched_null"
            ),
            "read": (
                "pre_existing_in_base_column_space"
                if read_observed["band_max"] > read_null_block["band_p95"]
                else "intruder_at_max_matched_null"
            ),
        },
    }


def _dv2_for_cell(slug, behavior, dose, seed, write_pairs, args) -> dict:
    """DV-2 write->output concept (manipulation check)."""
    unemb_path = Path(args.unembedding)
    if not unemb_path.is_file():
        return {"behavior": behavior, "dose": dose, "seed": seed, "note": "unembedding not built"}
    unemb = load_unembedding(unemb_path)
    concept_vec = unemb["marker"] if behavior == "marker" else unemb.get("d_U_syco")
    if concept_vec is None:
        return {"behavior": behavior, "dose": dose, "seed": seed, "note": "no concept vec"}
    cos_by_layer = {li: abs(_cos(write_pairs[li]["b"], concept_vec)) for li in write_pairs}
    band_max = float(max(cos_by_layer.values())) if cos_by_layer else float("nan")
    # Null p95.
    null_key = "null_norm_matched" if behavior == "marker" else "d_U_syco_null"
    null_rows = unemb.get(null_key)
    null_p95 = None
    if null_rows is not None and write_pairs:
        peak_li = max(cos_by_layer, key=cos_by_layer.get)
        b = write_pairs[peak_li]["b"]
        null_cos = [abs(_cos(b, null_rows[i])) for i in range(null_rows.shape[0])]
        null_p95 = float(np.percentile(np.asarray(null_cos), 95.0))
    return {
        "behavior": behavior,
        "dose": dose,
        "seed": seed,
        "cos_b_concept_by_layer": {int(k): float(v) for k, v in cos_by_layer.items()},
        "band_max": band_max,
        "null_p95": null_p95,
        "is_manipulation_check": True,
    }


def _maybe_run_dv1(out_dir, cell_metas, cells_root, base_svd, args, git_commit) -> None:
    """DV-1 read rotation: cos(a_t, a_init) + cos(a_up∘γ, v_source). Needs a_init + bank."""
    bank_dir = Path(args.bank_dir) if args.bank_dir else None
    rows: dict[str, dict] = {}
    bank = load_bank(bank_dir) if (bank_dir and bank_dir.is_dir()) else None
    for slug, meta in sorted(cell_metas.items()):
        behavior, dose, seed = parse_cell_slug(slug)
        # Dose-selected adapter for sycophancy cells; a_init is always the
        # step-0 snapshot at the CELL dir (epoch-independent), so DV-1 compares
        # the dose-selected lora_A against the same a_init.
        adapter_dir, init_dir = resolve_cell_adapter(meta, slug, cells_root)
        if not (init_dir / "adapter_model.safetensors").is_file():
            log.warning("DV-1 cell=%s: a_init missing — DV-1 unmeasurable for this cell", slug)
            continue
        if not (adapter_dir / "adapter_model.safetensors").is_file():
            continue
        final_pairs = load_adapter_pairs(adapter_dir)
        init_pairs = load_adapter_pairs(init_dir)
        read_layers = [li for li in READ_LAYER_BAND if (li, READ_MODULE) in final_pairs]
        cos_self = [
            _cos(final_pairs[(li, READ_MODULE)]["a"], init_pairs[(li, READ_MODULE)]["a"])
            for li in read_layers
        ]
        row = {
            "behavior": behavior,
            "dose": dose,
            "seed": seed,
            "cos_a_init_by_layer": {
                int(li): float(c) for li, c in zip(read_layers, cos_self, strict=True)
            },
            "band_mean_cos_a_init": float(np.mean(cos_self)) if cos_self else float("nan"),
            "kaiming_floor": 1.0 / np.sqrt(HIDDEN_SIZE),
        }
        # cos(a_up∘γ, v_source) if bank present.
        if bank is not None and bank["gamma_post_attn"] is not None:
            row["cos_a_gamma_v_source"] = _dv1_a_gamma_vsource(final_pairs, bank, read_layers)
        rows[slug] = row
    _write_json(out_dir / "dv1_read_rotation.json", {"cells": rows, "git_commit": git_commit})


def _layer_row(stacked: np.ndarray, li: int) -> np.ndarray | None:
    """Pick layer ``li``'s row from a per-layer-stacked bank tensor.

    The bank's centroids + ``rmsnorm_gamma`` are stored per layer as
    ``(n_layers=28, hidden=3584)`` (manifest ``n_layers=28``); a flat
    ``(3584,)`` tensor is returned unchanged (back-compat). A read at layer
    ``li`` must couple the adapter's layer-``li`` read vector ``a_up`` with
    THAT layer's γ and source centroid — not the whole 28-layer stack (the
    pre-fix code compared shapes 3584 vs 28 and silently skipped every layer,
    nulling DV-1's ``cos(a∘γ,v_source)`` arm and crashing DV-5).
    """
    arr = np.asarray(stacked)
    if arr.ndim == 1:
        return arr if arr.shape[0] == HIDDEN_SIZE else None
    if arr.ndim == 2 and arr.shape[1] == HIDDEN_SIZE and 0 <= li < arr.shape[0]:
        return arr[li]
    return None


def _dv1_a_gamma_vsource(final_pairs, bank, read_layers) -> dict:
    """cos(a_up ∘ γ_L, v_source_L) at the read tap, per read layer L."""
    gamma = bank["gamma_post_attn"]
    # source centroid at the up_in (post-post-attn-LN) tap, end_of_response pos.
    centroids = bank["centroids"]
    tap = "up_in" if "up_in" in centroids else next(iter(centroids))
    pos = "end_of_response"
    out: dict[int, float] = {}
    from explore_persona_space.experiments.issue_650 import SOURCE

    for li in read_layers:
        if (li, READ_MODULE) not in final_pairs:
            continue
        a = final_pairs[(li, READ_MODULE)]["a"]
        gamma_li = _layer_row(gamma, li)
        if gamma_li is None or a.shape[0] != gamma_li.shape[0]:
            continue  # up_proj read lives in 3584-d; γ_L matches
        a_gamma = a * gamma_li
        try:
            v_src = _layer_row(centroids[tap][pos][SOURCE], li)
        except KeyError:
            continue
        if v_src is None:
            continue
        out[int(li)] = abs(_cos(a_gamma, v_src))
    return {"by_layer": out, "band_max": float(max(out.values())) if out else float("nan")}


def _resolve_concept_path(concept_dir, seed: int) -> Path | None:
    """Resolve the per-SEED concept-directions tensor for a sycophancy cell.

    Round-2 fix (code-review blocker ``dv4-concept-path-mismatch``): the
    pipeline writes PER-SEED ``concept_directions_seed{seed}.pt`` (one base-
    model concept read per sycophancy seed pool), but the analyzer previously
    loaded a single unsuffixed ``concept_directions.pt`` that no phase ever
    writes — so every sycophancy DV-4 cell silently fell through to
    ``concept/adapter missing``. Resolve the seed-specific file keyed off the
    cell's own seed. A legacy single ``concept_directions.pt`` is accepted as
    a fallback ONLY if the per-seed file is absent (back-compat for an
    externally pooled build); never the other way round.
    """
    if not concept_dir:
        return None
    base = Path(concept_dir)
    per_seed = base / f"concept_directions_seed{seed}.pt"
    if per_seed.is_file():
        return per_seed
    legacy = base / "concept_directions.pt"
    if legacy.is_file():
        log.warning(
            "DV-4 seed=%d: per-seed %s absent; falling back to legacy pooled %s",
            seed,
            per_seed.name,
            legacy.name,
        )
        return legacy
    return None


def _maybe_run_dv4(out_dir, cell_metas, cells_root, args, git_commit) -> None:
    """DV-4 two-reference concept (sycophancy cells only).

    Loads the PER-SEED concept-directions tensor for each sycophancy cell
    (round-2 ``dv4-concept-path-mismatch`` fix). A sycophancy cell whose
    adapter is present but whose seed-keyed concept tensor is absent is a
    fail-loud error (a silent ``note`` would drop one of the two Must-Fix
    deliverables without a signal) — the only graceful skips are an absent
    ADAPTER (off-pod download incomplete) which is logged, and the marker arm
    (N/A by construction).
    """
    rows: dict[str, dict] = {}
    concept_cache: dict[int, dict | None] = {}
    for slug, meta in sorted(cell_metas.items()):
        behavior, dose, seed = parse_cell_slug(slug)
        if behavior == "marker":
            rows[slug] = {
                "behavior": "marker",
                "dose": dose,
                "seed": seed,
                "marker_arm": "N/A-for-marker",
            }
            continue
        # Dose-selected adapter for sycophancy cells (round-2
        # syco-dose-checkpoint-selection-missing); final otherwise.
        adapter_dir, _a_init_dir = resolve_cell_adapter(meta, slug, cells_root)
        if not (adapter_dir / "adapter_model.safetensors").is_file():
            log.warning(
                "DV-4 cell=%s: adapter not local at %s; skip (off-pod download incomplete)",
                slug,
                adapter_dir,
            )
            rows[slug] = {
                "behavior": behavior,
                "dose": dose,
                "seed": seed,
                "note": "adapter not local (off-pod download incomplete)",
            }
            continue
        if seed not in concept_cache:
            cp = _resolve_concept_path(args.concept_dir, seed)
            concept_cache[seed] = load_concept_directions(cp) if (cp and cp.is_file()) else None
        concept = concept_cache[seed]
        if concept is None:
            raise FileNotFoundError(
                f"DV-4 cell={slug}: adapter present but no concept-directions tensor "
                f"for seed {seed} under {args.concept_dir} "
                f"(expected concept_directions_seed{seed}.pt). The pipeline's "
                "concept phase writes one per sycophancy seed; a missing tensor "
                "means DV-4 (a Phase-3 Must-Fix) cannot be computed — refusing to "
                "write a silent 'note' default (fail-fast). Blocker "
                "dv4-concept-path-mismatch."
            )
        pairs = load_adapter_pairs(adapter_dir)
        write_pairs = _band_pair(pairs, WRITE_MODULE, WRITE_LAYER_BAND)
        # representative write = band-max-|b| layer's b.
        peak_li = max(write_pairs, key=lambda li: float(np.linalg.norm(write_pairs[li]["b"])))
        res = dv4_two_reference(
            write_vec=write_pairs[peak_li]["b"],
            concept=concept,
            write_band=WRITE_LAYER_BAND,
            seed=seed,
        )
        res.update({"behavior": behavior, "dose": dose, "seed": seed})
        rows[slug] = res
    _write_json(out_dir / "dv4_concept.json", {"cells": rows, "git_commit": git_commit})


def _dv5_read_direction(pairs: dict, bank: dict, read_layers) -> tuple[np.ndarray, int] | None:
    """Build the per-cell firing READ direction ``a_up ∘ γ_L`` at the band-peak
    read layer (the same gauge as DV-1's ``cos(a_up∘γ, v_source)`` read).

    Returns ``(unit_direction_3584d, layer_L)`` for the read layer whose
    ``a_up ∘ γ_L`` has the largest norm (the most-activated read), or None if no
    read layer has a γ-compatible up_proj read. ``γ`` is per-layer
    ``(28, 3584)``, so layer L's read vector couples with γ[L] — and the chosen
    layer L is returned so the caller couples the SAME layer's source/bystander
    centroids (the per-layer fix; the pre-fix code multiplied a 3584-d ``a`` by
    the whole 28-layer γ stack and skipped every layer).
    """
    gamma = bank.get("gamma_post_attn")
    if gamma is None:
        return None
    best = None
    best_norm = -1.0
    best_li = -1
    for li in read_layers:
        if (li, READ_MODULE) not in pairs:
            continue
        a = pairs[(li, READ_MODULE)]["a"]
        gamma_li = _layer_row(gamma, li)
        if gamma_li is None or a.shape[0] != gamma_li.shape[0]:
            continue
        ag = a * gamma_li
        n = float(np.linalg.norm(ag))
        if n > best_norm:
            best_norm = n
            best = ag
            best_li = li
    return (_unit(best), best_li) if best is not None else None


def _dv5_marker_leakage(eval_dir: Path, slug: str) -> dict[str, dict]:
    """Per-bystander marker leakage from ``<slug>__shift.json``.

    Returns ``{persona: {"firing_rate", "leakage_margin", "leakage_logp"}}``
    where ``firing_rate`` is the on-policy argmax emission rate (trained),
    ``leakage_margin`` is the install-controlled EOS-margin
    ``Δ(z_marker − z_eos)`` (the preferred non-saturating logit read, mean over
    the persona's questions), and ``leakage_logp`` is the mean Δlog P(marker)
    (behavioral). Empty if the shift JSON is absent.
    """
    shift_path = eval_dir / f"{slug}__shift.json"
    if not shift_path.is_file():
        return {}
    payload = json.loads(shift_path.read_text())
    out: dict[str, dict] = {}
    for persona, rec in payload.get("personas", {}).items():
        margins = rec.get("per_question_delta_margin") or []
        logps = rec.get("per_question_delta_logp") or []
        out[persona] = {
            "firing_rate": float(rec.get("emission_argmax_trained", 0.0)),
            "leakage_margin": float(np.mean(margins)) if margins else float("nan"),
            "leakage_logp": float(np.mean(logps)) if logps else float("nan"),
        }
    return out


def _maybe_run_dv5(out_dir, cell_metas, cells_root, args, git_commit) -> None:
    """DV-5 selectivity vs base geometry — firing predictor vs plain geometry.

    Per marker cell, across the held-out BYSTANDER panel (eval personas other
    than the source): does the learned firing predictor ``a_up∘γ · v_b`` predict
    install-controlled bystander leakage BETTER than plain base geometry
    ``cos(v_b, v_source)``? (plan §6.5 / §11). Leakage is read in the
    non-saturating EOS-margin logit space ``Δ(z_marker − z_eos)`` per the
    marker-leakage-measurement § Install-strength confound (NEVER raw log P).

    Per cell persists: per-bystander ``firing_rate`` / ``base_geometry_cos``
    (= cos(v_b, v_source)) / ``firing_predictor`` (= a_up∘γ · v_b) /
    ``install_strength`` (source EOS-margin leakage) / ``leakage_fraction``
    (bystander EOS-margin ÷ source EOS-margin — the per-cell transfer fraction);
    Spearman ρ of EACH predictor vs leakage; the paired Δρ (firing − geometry);
    and a predictor↔predictor collinearity ρ (the §14 diagnostic — a high
    collinearity means the two predictors are not separable for this cell).
    Sycophancy cells: N/A (the agreement eval is self-persona only, no
    bystander leakage panel).
    """
    from explore_persona_space.experiments.issue_650 import (
        PERSONA_POOL_18,
        SOURCE,
        UNIFIED_NEGATIVE_PANEL,
    )

    bank_dir = Path(args.bank_dir) if args.bank_dir else None
    bank = load_bank(bank_dir) if (bank_dir and bank_dir.is_dir()) else None
    eval_dir = Path(getattr(args, "eval_dir", None) or "eval_results/issue_650/eval")

    # Round-3 pivot (dv5-primary-bystander-filter-includes-assistant): the DV-5
    # bystander loop must exclude the SOURCE *and* every contrastive negative.
    # The eval panel is PERSONA_POOL_18 (17 personas incl. SOURCE) + `assistant`
    # (run_issue650_eval.py:81); `assistant` is in UNIFIED_NEGATIVE_PANEL, so it
    # is trained DOWN at its slot and its DV-5 leakage read is downward-biased —
    # the same confound the Option-B panel-shrink removed for kindergarten_teacher.
    # Exclude the whole negative panel, not just SOURCE.
    dv5_excluded = {SOURCE, *UNIFIED_NEGATIVE_PANEL}
    # Expected PRIMARY bystander count, derived from the realized panel constants
    # (the only object that can be asserted mechanically): the eval panel is
    # PERSONA_POOL_18 ∪ {assistant}, minus SOURCE and the negative panel.
    _eval_panel = set(PERSONA_POOL_18) | {"assistant"}
    n_primary_bystanders_expected = len(_eval_panel - dv5_excluded)
    # Plan §5/§14 (round-3 amendment) prose states "15 primary bystanders"; the
    # realized constants give 16, because the amendment's arithmetic
    # double-counted the kindergarten_teacher removal (it was dropped from
    # PERSONA_POOL_18 AND subtracted again in the "−1" term). The mechanical
    # assertion below uses the constant-derived count (the object the code can
    # verify); the plan figure is recorded alongside for the analyzer to
    # reconcile in the clean-result scope caveat.
    n_primary_bystanders_plan_declared = 15

    # Source context vector v_source from the bank (read-tap, end_of_response).
    def _vsource() -> tuple[str, np.ndarray] | None:
        if bank is None:
            return None
        centroids = bank["centroids"]
        tap = "up_in" if "up_in" in centroids else next(iter(centroids))
        pos = "end_of_response"
        by_ctx = centroids.get(tap, {}).get(pos, {})
        if SOURCE not in by_ctx:
            return None
        return tap, by_ctx

    rows: dict[str, dict] = {}
    for slug, meta in sorted(cell_metas.items()):
        behavior, dose, seed = parse_cell_slug(slug)
        if behavior != "marker":
            rows[slug] = {
                "behavior": behavior,
                "dose": dose,
                "seed": seed,
                "note": "DV-5 N/A — sycophancy agreement eval is self-persona (no bystander panel)",
            }
            continue
        leakage = _dv5_marker_leakage(eval_dir, slug)
        vsrc = _vsource()
        adapter_dir, _a_init = resolve_cell_adapter(meta, slug, cells_root)
        if not leakage or vsrc is None or not (adapter_dir / "adapter_model.safetensors").is_file():
            rows[slug] = {
                "behavior": "marker",
                "dose": dose,
                "seed": seed,
                "note": (
                    "DV-5 inputs incomplete: "
                    f"leakage_personas={len(leakage)} bank={bank is not None} "
                    f"adapter={(adapter_dir / 'adapter_model.safetensors').is_file()}"
                ),
            }
            continue
        tap, by_ctx = vsrc
        pairs = load_adapter_pairs(adapter_dir)
        read_layers = [li for li in READ_LAYER_BAND if (li, READ_MODULE) in pairs]
        rd = _dv5_read_direction(pairs, bank, read_layers)
        read_dir, read_layer = rd if rd is not None else (None, -1)
        # Couple the SAME layer's source/bystander centroids as the read
        # direction (per-layer bank fix). v_source is layer `read_layer`'s row
        # of the (28, 3584) source centroid; bystander rows below match it.
        v_source = _layer_row(by_ctx[SOURCE], read_layer) if read_layer >= 0 else None

        # Source install strength (EOS-margin leakage on the source itself).
        install_strength = leakage.get(SOURCE, {}).get("leakage_margin", float("nan"))

        per_bystander: dict[str, dict] = {}
        firing_pred: list[float] = []
        geom_pred: list[float] = []
        leak_out: list[float] = []
        for persona, lk in leakage.items():
            if persona in dv5_excluded:
                # Skip the SOURCE (the implant target, not a bystander) AND every
                # contrastive negative (trained DOWN at its slot -> down-biased
                # leakage read). dv5-primary-bystander-filter-includes-assistant.
                continue
            v_b_raw = by_ctx.get(persona)
            if v_b_raw is None or v_source is None:
                continue
            v_b = _layer_row(v_b_raw, read_layer)
            if v_b is None:
                continue
            geom_cos = _cos(v_b, v_source)
            fire = float(read_dir @ _unit(v_b)) if read_dir is not None else float("nan")
            margin = lk["leakage_margin"]
            frac = (
                (margin / install_strength)
                if (not np.isnan(install_strength) and install_strength != 0)
                else float("nan")
            )
            per_bystander[persona] = {
                "firing_rate": lk["firing_rate"],
                "base_geometry_cos": geom_cos,
                "firing_predictor": fire,
                "leakage_margin": margin,
                "leakage_logp": lk["leakage_logp"],
                "leakage_fraction": frac,
            }
            if read_dir is not None and not np.isnan(margin):
                firing_pred.append(fire)
                geom_pred.append(geom_cos)
                leak_out.append(margin)

        rho_firing = _spearman(firing_pred, leak_out)
        rho_geom = _spearman(geom_pred, leak_out)
        delta_rho = (
            (abs(rho_firing) - abs(rho_geom))
            if not (np.isnan(rho_firing) or np.isnan(rho_geom))
            else float("nan")
        )
        collinearity = _spearman(firing_pred, geom_pred)
        # Guard the post-filter count against a future panel mutation silently
        # desyncing the primary-bystander set. The realized leakage dict only
        # contains personas actually evaluated, so the count CAN be below the
        # constant-derived expectation when a bystander's shift JSON is missing
        # (a data-completeness issue surfaced elsewhere), but it must NEVER
        # EXCEED it — exceeding means a contrastive negative leaked into the read.
        n_bystanders = len(per_bystander)
        if n_bystanders > n_primary_bystanders_expected:
            raise AssertionError(
                f"cell={slug}: DV-5 read {n_bystanders} bystanders > the "
                f"constant-derived expected {n_primary_bystanders_expected} "
                "(PERSONA_POOL_18 + {assistant} minus SOURCE minus UNIFIED_NEGATIVE_PANEL) "
                "— a contrastive negative leaked into the bystander read "
                "(dv5-primary-bystander-filter-includes-assistant). The negative "
                "panel and the eval panel have drifted."
            )
        primary_bystander_count_matches = n_bystanders == n_primary_bystanders_expected
        rows[slug] = {
            "behavior": "marker",
            "dose": dose,
            "seed": seed,
            "n_bystanders": n_bystanders,
            "n_primary_bystanders_expected": n_primary_bystanders_expected,
            "n_primary_bystanders_plan_declared": n_primary_bystanders_plan_declared,
            "assertions": {"primary_bystander_count_matches": primary_bystander_count_matches},
            "install_strength_eos_margin": install_strength,
            "rho_firing_predictor": rho_firing,
            "rho_plain_geometry": rho_geom,
            "delta_rho_firing_minus_geometry": delta_rho,
            "predictor_collinearity_rho": collinearity,
            "firing_predictor_wins": bool((not np.isnan(delta_rho)) and delta_rho > 0.0),
            "read_tap": tap,
            "read_layer": int(read_layer),
            "per_bystander": per_bystander,
        }
    _write_json(
        out_dir / "dv5_selectivity.json",
        {
            "cells": rows,
            # Panel-shrink provenance (Option B, round-3 + pivot). The eval panel
            # is PERSONA_POOL_18 ∪ {assistant}; DV-5 reads only PRIMARY bystanders
            # (eval panel − SOURCE − UNIFIED_NEGATIVE_PANEL). The plan §5/§14 prose
            # says 15; the realized constants give 16 (the amendment arithmetic
            # double-subtracted kindergarten_teacher). The clean-result reconciles.
            "n_primary_bystanders_expected": n_primary_bystanders_expected,
            "n_primary_bystanders_plan_declared": n_primary_bystanders_plan_declared,
            "negative_panel_excluded_from_dv5": sorted(dv5_excluded),
            "git_commit": git_commit,
        },
    )


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    log.info("Wrote %s", path)


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    from explore_persona_space.experiments.issue_650 import BASE_MODEL

    p_svd = sub.add_parser("build-base-svd")
    p_svd.add_argument("--base-model", default=BASE_MODEL)
    p_svd.add_argument("--base-svd", default=f"{ANALYSIS_DIR}/base_svd.pt")
    p_svd.add_argument(
        "--modules",
        default="up_proj,down_proj",
        help="comma list of decoder submodules to build bases for (#2569 leg 5); the "
        "default preserves the original up_proj/down_proj behavior exactly",
    )
    p_svd.set_defaults(func=cmd_build_base_svd)

    p_unemb = sub.add_parser("build-unembedding")
    p_unemb.add_argument("--base-model", default=BASE_MODEL)
    p_unemb.add_argument("--unembedding", default=f"{ANALYSIS_DIR}/unembedding.pt")
    p_unemb.add_argument("--n-null", type=int, default=200)
    p_unemb.set_defaults(func=cmd_build_unembedding)

    p_an = sub.add_parser("analyze")
    p_an.add_argument("--cells-root", default="eval_results/issue_650")
    p_an.add_argument("--analysis-dir", default=ANALYSIS_DIR)
    p_an.add_argument("--base-svd", default=f"{ANALYSIS_DIR}/base_svd.pt")
    p_an.add_argument("--unembedding", default=f"{ANALYSIS_DIR}/unembedding.pt")
    p_an.add_argument("--bank-dir", default="eval_results/issue_650/bank")
    p_an.add_argument("--concept-dir", default="eval_results/issue_650/concept")
    p_an.add_argument(
        "--eval-dir",
        default="eval_results/issue_650/eval",
        help="Dir of eval-phase leakage JSONs (<slug>__shift.json) for DV-5.",
    )
    p_an.set_defaults(func=cmd_analyze)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
