#!/usr/bin/env python3
"""Issue #634 Phase 2: joint geometry of behavior vectors with #594 contexts.

VM-side CPU analysis (no GPU). Loads #594's ``(50, 28, 3584)`` context-vector
bank + the Phase-1 ``(275, 28, 3584)`` behavior-vector bank (both from HF) and,
per layer, runs:

- the co-embeddability gate (variance ratio + median-norm ratio behavior/context);
- a joint PCA + UMAP-grid embedding (+ t-SNE at the best layer) on the
  global-mean-centered stacked bank;
- **H1**: each Panel-B behavior's nearest of the 50 contexts (centered cosine),
  matched-family rate vs a B=1000 shuffled-``fam_map`` permutation null, with
  the max-over-layers FWER summary (``max_over_layers_summary``);
- **H2**: LOO k-NN family purity (k=4) over the 275 behaviors labeled by
  role-axis family, vs a permutation null;
- **H3**: own-region fraction = frac of behaviors whose top-4 joint-space
  neighbors are behaviors (not contexts);
- the ``beh_pc_residual`` control (remove behavior-cloud PC1, re-run H1);
- the cross-space family-centroid CKA + Procrustes fallback (matched-N), with
  the preflight shape assert the plan §5/§11 mandates.

All metric primitives + the metrics/figure helpers are REUSED from
``issue594_analyze_context_geometry.py`` (``center``, ``cosine_dist``,
``knn_neighbor_order``, ``knn_purity``, ``per_family_purity``, ``linear_cka``,
``max_over_layers_summary``, ``savefig_paper``) — no reimplementation.

Outputs: ``eval_results/issue_634/{joint_geometry_metrics.json,
per_layer_nn_purity.json, coembeddability_gate.json, cross_space_alignment.json,
panelB_nn_table.json}`` + ``figures/issue_634/*.png``.

Usage::

    uv run --extra viz python scripts/issue634_joint_geometry.py \\
        --behavior-from-hf issue634_behavior_geometry/analysis_tensors \\
        --context-from-hf issue594_context_geometry/analysis_tensors

    # synthetic tiny-bank smoke (no HF, no GPU):
    uv run --extra viz python scripts/issue634_joint_geometry.py --smoke

NOTE: ``--extra viz`` is REQUIRED — umap-learn lives in the optional ``viz``
dependency group, not the default set (same as the #594 analysis command).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue594_analyze_context_geometry import (  # noqa: E402
    center,
    cosine_dist,
    jsonable,
    knn_neighbor_order,
    knn_purity,
    linear_cka,
    max_over_layers_summary,
    quartile_layers,
)
from issue594_common import (  # noqa: E402
    BATTERY_EXPECTED_TOTAL,
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    FAMILY_EXPECTED_COUNTS,
    HF_DATA_REPO,
)

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

load_dotenv()

logger = logging.getLogger("issue634_joint")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KNN_K = 4
UMAP_GRID = [(5, 0.1), (5, 0.5), (15, 0.1), (15, 0.5), (30, 0.1), (30, 0.5)]
TSNE_PERPLEXITIES = [5, 15, 30]
SEED = 42

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_634"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_634"

DEFAULT_BEHAVIOR_PREFIX = "issue634_behavior_geometry/analysis_tensors"
DEFAULT_CONTEXT_PREFIX = "issue594_context_geometry/analysis_tensors"

# The frozen Panel-B map (Phase 0) uses LONG #594-family names; the #594 context
# tensor stores SHORT family labels. This alias table translates the frozen map's
# family value -> the context tensor's family label so H1's matched-family test
# resolves. The mapping follows plan v2 §4's YAML comments (e.g. worked_example
# = the icl worked-example context family; instruction_reword = rephrase).
# ``co_embed_only`` covers the smoke path where the frozen map already uses the
# short names. Asserted against the context tensor's realized family set on load.
FAMILY_NAME_ALIAS = {
    "persona": "persona",
    "behavior_instruction": "behavior",
    "output_format": "format",
    "instruction_reword": "rephrase",
    "worked_example": "icl",
    "bare_default": "default",
    "wildchat_prefix": "wildchat",
    # identity passthroughs (short names already match the context tensor):
    "behavior": "behavior",
    "format": "format",
    "rephrase": "rephrase",
    "icl": "icl",
    "default": "default",
    "wildchat": "wildchat",
}
NULL_ANCHOR_FAMILY_SHORT = "default"  # bare_default -> the null-anchor, not tested

# Co-embeddability scale-mismatch flag: families differing in median norm or
# variance by more than this factor route H1 to the cross-space fallback.
COEMBED_RATIO_MAX = 3.0


# ── Data loading ─────────────────────────────────────────────────────────────


def download_from_hf(prefix: str, cache_dir: Path, repo: str = HF_DATA_REPO) -> Path:
    """Fetch analysis tensors from an HF dataset repo (per-file, never snapshot)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(repo, repo_type="dataset") if f.startswith(prefix + "/")]
    if not files:
        raise RuntimeError(f"no files under {prefix}/ on {repo}")
    for f in files:
        hf_hub_download(repo, f, repo_type="dataset", local_dir=str(cache_dir))
    logger.info("Downloaded %d files from %s/%s", len(files), repo, prefix)
    return cache_dir / prefix


def _assert_context_bank_coverage(blob: dict, tensors_dir: Path) -> str | None:
    """Fail-loud pre-index coverage guard for the #594 context bank (BLOCKER fix).

    The bank is the entire substrate for every Phase-2 output: a STALE/WRONG #594
    bank with the right ``(28, 3584)`` dims but a wrong ``n_ctx``, a mis-ordered /
    different-probe-pool family labeling, or a different bank revision passes every
    DOWNSTREAM guard (the joint-stack ``ctx.shape`` ndim read, ``resolve_panel_b``'s
    present-family check, the CKA matched-N preflight) and is analyzed SILENTLY,
    producing a plausible-but-false figure. The plan pre-registered both guards
    this closes — §4 line 219 "manifest ``probe_pool_hash`` asserted on load" and
    A2 line 281 "Phase 2 ``download_from_hf`` + shape assert" — so this is a
    plan-adherence fix, not robustness beyond the contract. Concern
    ``issue634-context-bank-coverage``.

    Asserts (all fail loud, not warn): (1) the required tensor/index keys exist;
    (2) the tensor is exactly ``(50, 28, 3584)`` = ``(BATTERY_EXPECTED_TOTAL,
    EXPECTED_LAYERS, EXPECTED_HIDDEN)``; (3) the realized family set equals #594's
    canonical 7-family set AND the per-family counts equal ``FAMILY_EXPECTED_COUNTS``
    (this pins identity, not just dims); (4) ``instance_ids`` length matches n_ctx.
    Returns the bank's ``probe_pool_hash`` (cross-checked against a co-uploaded
    manifest sidecar when present; see ``_check_probe_pool_hash``).

    Identity note: #594's family-set + exact per-family counts ARE the content
    fingerprint here (the 50-instance battery is fixed; #594's own
    ``validate_battery`` enforces the same ``FAMILY_EXPECTED_COUNTS``). The
    ``probe_pool_hash`` pins the probe pool the means were taken over; together
    they make a wrong/stale-bank substitution fail at the load boundary.
    """
    required = {"tensor", "instance_ids", "families"}
    missing = required - set(blob)
    if missing:
        raise RuntimeError(
            f"#594 context bank missing required keys {sorted(missing)} "
            f"(have {sorted(blob)}) in {tensors_dir / 'context_vectors_mean.pt'}"
        )
    tensor = blob["tensor"]
    expected_shape = (BATTERY_EXPECTED_TOTAL, EXPECTED_LAYERS, EXPECTED_HIDDEN)
    if tuple(tensor.shape) != expected_shape:
        raise RuntimeError(
            f"#594 context bank tensor shape {tuple(tensor.shape)} != {expected_shape} "
            f"(BATTERY_EXPECTED_TOTAL, EXPECTED_LAYERS, EXPECTED_HIDDEN)"
        )
    families = list(blob["families"])
    ids = list(blob["instance_ids"])
    if len(ids) != BATTERY_EXPECTED_TOTAL or len(families) != BATTERY_EXPECTED_TOTAL:
        raise RuntimeError(
            f"#594 context bank index length mismatch: ids={len(ids)} families={len(families)} "
            f"!= n_ctx={BATTERY_EXPECTED_TOTAL}"
        )
    realized_counts: dict[str, int] = {}
    for fam in families:
        realized_counts[fam] = realized_counts.get(fam, 0) + 1
    if set(realized_counts) != set(FAMILY_EXPECTED_COUNTS):
        raise RuntimeError(
            f"#594 context bank family SET {sorted(realized_counts)} != #594 expected "
            f"{sorted(FAMILY_EXPECTED_COUNTS)} — stale/wrong bank or different probe genre"
        )
    if realized_counts != FAMILY_EXPECTED_COUNTS:
        raise RuntimeError(
            f"#594 context bank per-family COUNTS {realized_counts} != #594 expected "
            f"{FAMILY_EXPECTED_COUNTS} — bank identity does not match the registered battery"
        )
    return _check_probe_pool_hash(blob, tensors_dir)


def _check_probe_pool_hash(blob: dict, tensors_dir: Path) -> str | None:
    """Cross-check the bank's probe_pool_hash against its manifest sidecar.

    #594's ``context_vectors_mean.pt`` blob carries ``probe_pool_hash`` and the
    co-uploaded ``extraction_manifest.json`` carries the SAME field. When both are
    present they MUST agree (catches a tensor swapped under a stale manifest, and
    vice versa). When the manifest is absent (downloaded subset, older bank) the
    blob hash is used alone and the missing sidecar is logged as a known limitation
    rather than failing the run — the keys/shape/family-set asserts above already
    catch a wrong bank; the hash is the extra probe-pool pin when available.
    """
    blob_hash = blob.get("probe_pool_hash")
    manifest_path = tensors_dir / "extraction_manifest.json"
    manifest_hash = None
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest_hash = json.load(f).get("probe_pool_hash")
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(f"#594 manifest {manifest_path} unreadable: {e}") from e
    if blob_hash is not None and manifest_hash is not None and blob_hash != manifest_hash:
        raise RuntimeError(
            f"#594 probe_pool_hash mismatch: bank tensor {blob_hash!r} != manifest "
            f"{manifest_hash!r} ({manifest_path}) — tensor/manifest out of sync"
        )
    if blob_hash is None:
        logger.warning(
            "#594 context bank has no probe_pool_hash key (keys/shape/family asserts "
            "still enforced); manifest sidecar present=%s",
            manifest_path.exists(),
        )
    else:
        logger.info(
            "#594 context bank probe_pool_hash=%s (manifest cross-check=%s)",
            blob_hash[:16],
            "match" if manifest_hash == blob_hash else "sidecar-absent",
        )
    return blob_hash


def load_context_bank(tensors_dir: Path, expect_full_bank: bool = True) -> dict:
    """Load #594's context_vectors_mean.pt -> (N, L, H) fp32 + ids + families.

    ``expect_full_bank`` (default True for the production HF load) runs the
    fail-loud coverage guard ``_assert_context_bank_coverage`` (required keys,
    ``(50, 28, 3584)`` shape, #594's exact family set + per-family counts,
    probe_pool_hash). The synthetic ``--smoke`` path builds a deliberately tiny
    bank (n_ctx=10, 3 families) and is exercised in-memory without this loader, so
    the full-bank contract never applies to it.
    """
    blob = torch.load(tensors_dir / "context_vectors_mean.pt", weights_only=True)
    probe_pool_hash = None
    if expect_full_bank:
        probe_pool_hash = _assert_context_bank_coverage(blob, tensors_dir)
    mean = blob["tensor"].float().numpy()
    return {
        "mean": mean,
        "ids": list(blob["instance_ids"]),
        "families": list(blob["families"]),
        "probe_pool_hash": probe_pool_hash,
    }


def load_behavior_bank(tensors_dir: Path) -> dict:
    """Load Phase-1 behavior_vectors_mean.pt -> (N, L, H) fp32 + role ids + families."""
    blob = torch.load(tensors_dir / "behavior_vectors_mean.pt", weights_only=True)
    mean = blob["tensor"].float().numpy()
    return {
        "mean": mean,
        "ids": list(blob["instance_ids"]),
        "families": list(blob["families"]),
        "seed": blob.get("seed"),
        "n_questions": blob.get("n_questions"),
        "sampled_question_indices_hash": blob.get("sampled_question_indices_hash"),
    }


def load_family_map(path: Path) -> dict[str, str]:
    """Load the frozen Panel-B map; tolerate the wrapped payload or a bare map."""
    with open(path) as f:
        payload = json.load(f)
    return payload.get("map", payload)


def resolve_panel_b(
    fam_map: dict[str, str], behavior_ids: list[str], ctx_family_set: set[str]
) -> dict[str, str]:
    """Translate the frozen map to context-family labels; keep only present roles.

    Returns role -> context-family (short label), restricted to: roles actually
    present in the behavior bank, with a non-anchor translated family that
    exists in the context tensor. The null-anchor (bare_default) is excluded
    from H1's tested set. Fails loud on an unknown family name.
    """
    panel_b: dict[str, str] = {}
    present = set(behavior_ids)
    for role, fam_long in fam_map.items():
        if role not in present:
            logger.warning("Panel-B role %s absent from behavior bank; skipping", role)
            continue
        if fam_long not in FAMILY_NAME_ALIAS:
            raise ValueError(f"frozen map family {fam_long!r} (role {role}) not in alias table")
        fam_short = FAMILY_NAME_ALIAS[fam_long]
        if fam_short == NULL_ANCHOR_FAMILY_SHORT:
            continue  # null-anchor, not a tested family
        if fam_short not in ctx_family_set:
            raise ValueError(
                f"translated family {fam_short!r} (role {role}, from {fam_long!r}) "
                f"not in context tensor families {sorted(ctx_family_set)}"
            )
        panel_b[role] = fam_short
    return panel_b


# ── Co-embeddability gate ────────────────────────────────────────────────────


def coembeddability_gate(beh_layer: np.ndarray, ctx_layer: np.ndarray) -> dict:
    """Per-layer variance + median-norm ratio (behavior/context).

    A layer fails (routes H1 to the fallback) when either ratio is outside
    [1/COEMBED_RATIO_MAX, COEMBED_RATIO_MAX]. Diagnostic only — does not raise.
    """
    var_b = float(np.var(beh_layer))
    var_c = float(np.var(ctx_layer))
    norm_b = float(np.median(np.linalg.norm(beh_layer, axis=1)))
    norm_c = float(np.median(np.linalg.norm(ctx_layer, axis=1)))
    var_ratio = var_b / var_c if var_c > 0 else float("inf")
    scale_ratio = norm_b / norm_c if norm_c > 0 else float("inf")
    ok = (
        1.0 / COEMBED_RATIO_MAX <= var_ratio <= COEMBED_RATIO_MAX
        and 1.0 / COEMBED_RATIO_MAX <= scale_ratio <= COEMBED_RATIO_MAX
    )
    return {
        "var_behavior": var_b,
        "var_context": var_c,
        "var_ratio_beh_over_ctx": var_ratio,
        "median_norm_behavior": norm_b,
        "median_norm_context": norm_c,
        "scale_ratio_beh_over_ctx": scale_ratio,
        "joint_centered_ok": bool(ok),
    }


# ── H1: behavior -> nearest context family ───────────────────────────────────


def nearest_context_family(
    beh_layer: np.ndarray,
    ctx_layer: np.ndarray,
    ctx_families: list[str],
    panel_b_idx: list[int],
) -> list[str]:
    """For each Panel-B behavior, the family of its nearest of the 50 contexts.

    Centered cosine: both banks are global-mean-centered on the JOINT stack
    (50 ctx + N_b beh) so the centering matches the joint embedding's. Returns
    one family label per Panel-B behavior, in panel_b_idx order.
    """
    n_ctx = ctx_layer.shape[0]
    joint = np.vstack([ctx_layer, beh_layer])
    joint_c = center(joint)
    ctx_c = joint_c[:n_ctx]
    beh_c = joint_c[n_ctx:]
    ctx_n = ctx_c / np.clip(np.linalg.norm(ctx_c, axis=1, keepdims=True), 1e-12, None)
    out: list[str] = []
    for bi in panel_b_idx:
        v = beh_c[bi]
        v = v / max(float(np.linalg.norm(v)), 1e-12)
        sims = ctx_n @ v  # cosine to each context
        out.append(ctx_families[int(np.argmax(sims))])
    return out


def h1_matched_rate(nn_fams: list[str], expected_fams: list[str]) -> float:
    """Matched-family rate = frac of Panel-B behaviors whose NN family matches."""
    if not nn_fams:
        return float("nan")
    return float(np.mean([nn == exp for nn, exp in zip(nn_fams, expected_fams, strict=True)]))


def h1_perm_null(
    nn_fams: list[str], panel_b_roles: list[str], expected_fams: list[str], perms: list[dict]
) -> np.ndarray:
    """Matched rate under each shuffled role->family permutation.

    The geometry (the nearest-context family per behavior) is FIXED; only the
    role->expected-family labels are shuffled. Each perm is a dict role->family
    over the SAME tested family multiset (label permutation).
    """
    rates = np.empty(len(perms))
    for b, perm in enumerate(perms):
        shuffled_expected = [perm[r] for r in panel_b_roles]
        rates[b] = h1_matched_rate(nn_fams, shuffled_expected)
    return rates


def make_label_perms(panel_b: dict[str, str], n_perms: int, rng: np.random.Generator) -> list[dict]:
    """B label permutations of the role->family assignment (shared across layers).

    Permutes the family VALUES across the roles (keeps the family multiset), so
    the null is "random role->family labels" exactly as plan §5 specifies.
    """
    roles = list(panel_b.keys())
    fams = list(panel_b.values())
    perms = []
    for _ in range(n_perms):
        shuffled = list(fams)
        rng.shuffle(shuffled)
        perms.append(dict(zip(roles, shuffled, strict=True)))
    return perms


# ── H1 soft k-NN family-match sensitivity (free-analysis follow-up) ───────────
#
# The strict H1 above asks "is the SINGLE nearest context vector in the matched
# family?" (k=1). This sensitivity check relaxes the criterion to a k-NN majority
# vote over the SAME panel/tensors/null, testing whether the H1 null is an
# artifact of nearest-only. For each k the "match" rule is:
#   k=1          : the 1 nearest is in the matched family (== strict H1).
#   k=2 majority : BOTH of the 2 nearest are in the matched family (unanimous,
#                  strict majority for an even k).
#   k=3 majority : >= 2 of the 3 nearest are in the matched family.
#   k=4 majority : >= 3 of the 4 nearest are in the matched family (strict
#                  majority for an even k).
# The rule generalizes to floor(k/2)+1 matches for any k (strict majority), which
# at k=1 reduces to "the single nearest matches" — so the k=1 row reproduces the
# strict H1 exactly (asserted as a parity guard in run_soft_knn_sensitivity).
SOFT_KNN_KS = [1, 2, 3, 4]


def _majority_threshold(k: int) -> int:
    """Strict-majority match threshold for k neighbors = floor(k/2) + 1.

    k=1->1 (the single nearest), k=2->2 (both), k=3->2, k=4->3. For even k this
    is a strict majority (> k/2), matching the brief's k=2 'both' and k=4 '>=3'.
    """
    return k // 2 + 1


def nearest_k_context_families(
    beh_layer: np.ndarray,
    ctx_layer: np.ndarray,
    ctx_families: list[str],
    panel_b_idx: list[int],
    k_max: int,
) -> list[list[str]]:
    """Top-``k_max`` nearest-context families per Panel-B behavior (one layer).

    Same joint-global-mean-centering + centered-cosine geometry as
    ``nearest_context_family`` (so the k=1 entry is identical to it); returns, per
    Panel-B behavior in ``panel_b_idx`` order, the families of its ``k_max`` nearest
    contexts ordered nearest-first. ``k_max`` is clamped to ``n_ctx``.
    """
    n_ctx = ctx_layer.shape[0]
    k_eff = min(k_max, n_ctx)
    joint = np.vstack([ctx_layer, beh_layer])
    joint_c = center(joint)
    ctx_c = joint_c[:n_ctx]
    beh_c = joint_c[n_ctx:]
    ctx_n = ctx_c / np.clip(np.linalg.norm(ctx_c, axis=1, keepdims=True), 1e-12, None)
    ctx_fams_arr = np.asarray(ctx_families)
    out: list[list[str]] = []
    for bi in panel_b_idx:
        v = beh_c[bi]
        v = v / max(float(np.linalg.norm(v)), 1e-12)
        sims = ctx_n @ v  # cosine to each context
        # argpartition for the top-k, then sort those k descending by similarity
        # (so the k=1 head matches nearest_context_family's argmax exactly).
        top = np.argpartition(-sims, k_eff - 1)[:k_eff]
        top = top[np.argsort(-sims[top])]
        out.append([str(f) for f in ctx_fams_arr[top]])
    return out


def soft_knn_matched_rate(nn_k_fams: list[list[str]], expected_fams: list[str], k: int) -> float:
    """Majority-vote matched rate over Panel-B at neighbor count ``k``.

    A behavior matches when >= ``_majority_threshold(k)`` of its k nearest
    contexts are in its expected family. ``nn_k_fams`` rows must each hold >= k
    families (the k_max top list); only the first k are consulted.
    """
    if not nn_k_fams:
        return float("nan")
    thr = _majority_threshold(k)
    hits = 0
    for fams_k, exp in zip(nn_k_fams, expected_fams, strict=True):
        n_in_family = sum(1 for f in fams_k[:k] if f == exp)
        if n_in_family >= thr:
            hits += 1
    return float(hits / len(nn_k_fams))


def soft_knn_perm_null(
    nn_k_fams: list[list[str]],
    panel_b_roles: list[str],
    perms: list[dict],
    k: int,
) -> np.ndarray:
    """Majority-vote matched rate under each shuffled role->family permutation.

    Geometry (the k nearest-context families per behavior) is FIXED; only the
    role->expected-family labels are shuffled, exactly as ``h1_perm_null`` does for
    the strict k=1 statistic. Returns one rate per permutation.
    """
    rates = np.empty(len(perms))
    for b, perm in enumerate(perms):
        shuffled_expected = [perm[r] for r in panel_b_roles]
        rates[b] = soft_knn_matched_rate(nn_k_fams, shuffled_expected, k)
    return rates


def _soft_knn_key(k: int) -> str:
    """JSON key for neighbor count ``k`` (k=1 strict, k>=2 majority-vote)."""
    return "k=1" if k == 1 else f"k={k}_majority"


def run_soft_knn_sensitivity(
    beh: np.ndarray,
    ctx: np.ndarray,
    ctx_families: list[str],
    panel_b: dict[str, str],
    panel_b_idx: list[int],
    panel_b_roles: list[str],
    expected_fams: list[str],
    n_perms: int,
    seed: int,
    expect_full_bank: bool,
) -> dict:
    """Compute the soft-k-NN family-match sensitivity over all layers and k values.

    For each k in ``SOFT_KNN_KS`` and each layer: the majority-vote matched rate
    (over the 26 Panel-B roles) + a B-perm shuffled-label null over the SAME
    role->family labels (``make_label_perms``), summarized with
    ``max_over_layers_summary`` (max-over-layers rate, argmax layer, null 95th pct,
    FWER p). The k=1 row is the strict nearest-only H1 and is asserted to match the
    canonical ``nearest_context_family`` geometry within float tolerance.

    Returns a dict ready to serialize (``per_k`` keyed by ``_soft_knn_key``).
    """
    n_layers = ctx.shape[1]
    k_max = max(SOFT_KNN_KS)
    rng = np.random.default_rng(seed)
    # ONE shared set of label permutations across layers AND k values, matching
    # the strict-H1 null construction (perms are role->family relabelings, k- and
    # layer-independent), so the four k rows share an identical null draw.
    label_perms = make_label_perms(panel_b, n_perms, rng)

    # Per-layer top-k nearest-context families (computed once, reused for all k).
    obs_rates: dict[int, np.ndarray] = {k: np.empty(n_layers) for k in SOFT_KNN_KS}
    null_rates: dict[int, np.ndarray] = {k: np.empty((n_perms, n_layers)) for k in SOFT_KNN_KS}
    parity_max_abs_diff = 0.0
    for li in range(n_layers):
        nn_k = nearest_k_context_families(
            beh[:, li, :], ctx[:, li, :], ctx_families, panel_b_idx, k_max
        )
        # Parity guard: the k=1 nearest family must equal the canonical strict-H1
        # nearest-context family for every Panel-B behavior at this layer.
        canon = nearest_context_family(beh[:, li, :], ctx[:, li, :], ctx_families, panel_b_idx)
        for got_row, canon_fam in zip(nn_k, canon, strict=True):
            if got_row[0] != canon_fam:
                raise AssertionError(
                    f"soft-knn k=1 parity break at layer {li}: top-1 {got_row[0]!r} != "
                    f"canonical nearest {canon_fam!r}"
                )
        for k in SOFT_KNN_KS:
            obs_rates[k][li] = soft_knn_matched_rate(nn_k, expected_fams, k)
            null_rates[k][:, li] = soft_knn_perm_null(nn_k, panel_b_roles, label_perms, k)
        # k=1 observed rate must equal the canonical h1_matched_rate too.
        canon_rate = h1_matched_rate(canon, expected_fams)
        parity_max_abs_diff = max(parity_max_abs_diff, abs(canon_rate - obs_rates[1][li]))

    per_k: dict[str, dict] = {}
    for k in SOFT_KNN_KS:
        summ = max_over_layers_summary(obs_rates[k].tolist(), null_rates[k])
        per_k[_soft_knn_key(k)] = {
            "k": k,
            "majority_threshold": _majority_threshold(k),
            "per_layer_rate": obs_rates[k].tolist(),
            "max_layer": int(summ["argmax_layer"]),
            "max_rate": float(summ["observed_max"]),
            "null_max_p95": float(summ["null_max_p95"]),
            "p_fwer": float(summ["p_fwer"]),
            "passes": bool(summ["passes"]),
            "null_mean_per_layer": summ["null_mean_per_layer"],
            "null_p95_per_layer": summ["null_p95_per_layer"],
        }

    any_pass = any(per_k[_soft_knn_key(k)]["passes"] for k in SOFT_KNN_KS)
    headline = (
        "at least one k=1..4 majority criterion EXCEEDS the permutation 95th "
        "percentile (null lifted for that k)"
        if any_pass
        else "all k=1..4 majority criteria remain inside or below the permutation "
        "95th percentile (null not lifted)"
    )
    return {
        "n_perms": int(n_perms),
        "seed": int(seed),
        "panel_b_n_roles": len(panel_b_roles),
        "panel_b_expected_families": expected_fams,
        "ks_tested": SOFT_KNN_KS,
        "majority_rule": (
            "match when >= floor(k/2)+1 of the k nearest contexts are in the "
            "expected family (k=1 strict nearest-only == H1; k=2 both; k=3 >=2 of 3; "
            "k=4 >=3 of 4)"
        ),
        "k1_parity_max_abs_diff_vs_strict_h1": float(parity_max_abs_diff),
        "per_k": per_k,
        "any_k_passes": bool(any_pass),
        "headline_summary": headline,
        "metadata": reproducibility_metadata(
            {"script": "issue634_joint_geometry", "mode": "soft_knn_sensitivity"}
        ),
    }


def fig_soft_knn_sensitivity(soft: dict, n_layers: int):
    """Per-layer soft-k-NN matched rate vs the permutation null, one panel per k."""
    layers = np.arange(n_layers)
    fig, axes = plt.subplots(1, len(SOFT_KNN_KS), figsize=(4 * len(SOFT_KNN_KS), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes.flat, SOFT_KNN_KS, strict=True):
        d = soft["per_k"][_soft_knn_key(k)]
        ax.fill_between(
            layers,
            d["null_mean_per_layer"],
            d["null_p95_per_layer"],
            color="0.85",
            label="permutation null (mean to 95%)",
        )
        ax.plot(
            layers,
            d["per_layer_rate"],
            color="#1f77b4",
            lw=2,
            marker="o",
            ms=3,
            label="observed matched rate",
        )
        ax.axvline(d["max_layer"], color="#d62728", lw=1, ls=":", label="best layer")
        title = "k=1 (nearest-only)" if k == 1 else f"k={k} majority (>={d['majority_threshold']})"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("decoder layer")
    axes.flat[0].set_ylabel("matched-family rate (Panel B)")
    axes.flat[0].legend(fontsize=6, loc="best")
    fig.suptitle("H1 soft k-NN family-match sensitivity vs shuffled-label null", fontsize=11)
    savefig_paper(fig, "h1_soft_knn_sensitivity", dir=FIG_DIR)
    plt.close(fig)


# ── H3: own-region fraction (joint-space neighbor mix) ───────────────────────


def own_region_fraction(beh_layer: np.ndarray, ctx_layer: np.ndarray, k: int = KNN_K) -> float:
    """Frac of behaviors whose top-k JOINT-space neighbors are all behaviors.

    Joint bank = [ctx; beh], global-mean-centered, cosine distance. For each
    behavior, look at its k nearest joint neighbors (self excluded) and count
    the fraction that are behaviors; own_region = mean over behaviors of
    (k neighbors all behaviors).
    """
    n_ctx = ctx_layer.shape[0]
    joint = center(np.vstack([ctx_layer, beh_layer]))
    d = cosine_dist(joint, centering="none")  # already centered
    order = knn_neighbor_order(d)
    n_total = joint.shape[0]
    is_beh = np.arange(n_total) >= n_ctx
    behavior_rows = np.where(is_beh)[0]
    frac_beh_neighbors = []
    for i in behavior_rows:
        neigh = order[i, :k]
        frac_beh_neighbors.append(float(np.mean(is_beh[neigh])))
    return float(np.mean(frac_beh_neighbors))


# ── Cross-space family-centroid fallback (matched-N) ─────────────────────────


def family_centroids(
    bank: np.ndarray, families: list[str], min_per_family: int = 1
) -> tuple[np.ndarray, list[str]]:
    """Per-family centroid bank for one layer's (N, H) matrix.

    Returns ((K, H) centroid stack, family-name list) for families with
    >= min_per_family members. Order is sorted family name (stable).
    """
    fams = np.asarray(families)
    out_fams = sorted(
        {f for f in families if f is not None and (fams == f).sum() >= min_per_family}
    )
    cents = (
        np.stack([bank[fams == f].mean(axis=0) for f in out_fams])
        if out_fams
        else np.empty((0, bank.shape[1]))
    )
    return cents, out_fams


def _raw_centroid_basis(cb: np.ndarray, cc: np.ndarray) -> np.ndarray:
    """Orthonormal (d, H) basis spanning the RAW (un-centered) centroid rows.

    The union of cb's and cc's rows spans <= 2K dims, so an SVD of the (2K, H)
    raw stack yields a small basis; the (H, H) Procrustes SVD never appears.
    Because BOTH banks' rows lie entirely in this span, projecting onto it is an
    ISOMETRY for them (``cb @ basis.T @ basis == cb`` exactly), so every Frobenius
    norm of a row-space vector is preserved. NO centering — this spans the raw
    rows so the raw residual is exact, not the centered/projected variant.
    """
    stack = np.vstack([cb, cc])
    _u, s, vt = np.linalg.svd(stack, full_matrices=False)
    keep = int((s > 1e-9 * s.max()).sum()) if s.size and s.max() > 0 else 0
    return vt[:keep] if keep >= 1 else np.empty((0, stack.shape[1]))


def _procrustes_resid_raw_lowdim(cb: np.ndarray, cc: np.ndarray) -> float:
    """PREREGISTERED raw Procrustes residual ``||Cb·R - Cc||_F / ||Cc||_F``.

    Exactly the plan §5 step-4 statistic: orthogonal Procrustes with NO joint
    centering and the RAW ``||Cc||_F`` denominator. Computed in the low-rank
    basis spanning the raw centroid rows (``_raw_centroid_basis``) purely to dodge
    the (H, H) SVD hang ``orthogonal_procrustes`` would otherwise trigger on the
    raw 3584-dim banks. Since both banks' rows lie in that span, the projection is
    an isometry and the residual is IDENTICAL to the full-space raw residual:
    the optimal full-space orthogonal R need only rotate within the span (it is
    free on the orthogonal complement, where both banks are zero), and Frobenius
    norms are preserved by the isometric projection. Restores the preregistration
    the round-1 centered/projected variant drifted from (concern
    issue634-procrustes-objective-drift, option a).
    """
    from scipy.linalg import orthogonal_procrustes

    basis = _raw_centroid_basis(cb, cc)
    if basis.shape[0] < 1:
        return 0.0
    cb_d = cb @ basis.T  # (K, d) — isometric image of the RAW rows (no centering)
    cc_d = cc @ basis.T  # (K, d)
    r, _scale = orthogonal_procrustes(cb_d, cc_d)  # SVD is (d, d), d <= 2K
    return float(np.linalg.norm(cb_d @ r - cc_d) / max(float(np.linalg.norm(cc_d)), 1e-12))


def _procrustes_resid_centered_lowdim(cb: np.ndarray, cc: np.ndarray) -> float:
    """Centered-Procrustes DIAGNOSTIC residual (NOT the preregistered statistic).

    Removes the shared centroid translation before alignment (Procrustes is a
    pure rotation; a shared translation otherwise inflates the residual) and uses
    the centered ``||Cc - mu||_F`` denominator. Kept ALONGSIDE the preregistered
    raw residual (``_procrustes_resid_raw_lowdim``) as a translation-free shape
    diagnostic; emitted under a clearly-distinct JSON key so it is never narrated
    as the preregistered residual. Same low-rank-basis trick to avoid the (H, H)
    SVD, here over the CENTERED stack.
    """
    from scipy.linalg import orthogonal_procrustes

    stack = np.vstack([cb, cc])
    mu = stack.mean(axis=0, keepdims=True)
    stack_c = stack - mu
    _u, s, vt = np.linalg.svd(stack_c, full_matrices=False)
    keep = int((s > 1e-9 * s.max()).sum()) if s.size and s.max() > 0 else 0
    if keep < 1:
        return 0.0
    basis = vt[:keep]  # (d, H)
    cb_d = (cb - mu) @ basis.T  # (K, d)
    cc_d = (cc - mu) @ basis.T  # (K, d)
    r, _scale = orthogonal_procrustes(cb_d, cc_d)  # SVD is (d, d), d <= 2K-1
    return float(np.linalg.norm(cb_d @ r - cc_d) / max(float(np.linalg.norm(cc_d)), 1e-12))


def cross_space_alignment(
    beh_bank: np.ndarray,
    ctx_bank: np.ndarray,
    panel_b: dict[str, str],
    behavior_ids: list[str],
    ctx_families: list[str],
    n_layers: int,
) -> dict:
    """Family-centroid linear CKA + orthogonal Procrustes, per layer (matched-N).

    Aggregates the 50 contexts to per-family centroids and the Panel-B
    behaviors to per-family centroids (>=2 Panel-B roles per family), intersects
    to families in BOTH, and runs linear_cka + orthogonal_procrustes on the
    matched-N centroid banks. The preflight shape assert (plan §5/§11) enforces
    Cc.shape[0] == Cb.shape[0] in code.

    The Procrustes rotation is computed in the LOW-DIMENSIONAL subspace the K
    centroids span (an SVD of the (2K, H) centroid stack), NOT the raw H=3584-dim
    space: ``orthogonal_procrustes(A, B)`` SVDs ``B.T @ A`` which is (H, H) — an
    SVD of a 3584x3584 matrix per layer pegs every core for minutes (real hang
    observed in the synthetic smoke). The K centroid rows of both banks span the
    same <= 2K-dim subspace, so projecting onto it is an ISOMETRY for them and
    the residual is EXACT. CKA is unaffected — its Gram form is already (K, K).

    Two residuals are emitted per layer: ``procrustes_resid_per_layer`` is the
    PREREGISTERED raw residual ``||Cb·R - Cc||_F / ||Cc||_F`` (no centering — the
    headline statistic, ``_procrustes_resid_raw_lowdim``); and
    ``procrustes_resid_centered_per_layer`` is a translation-free centered-Procrustes
    DIAGNOSTIC (``_procrustes_resid_centered_lowdim``), kept alongside but never
    narrated as the preregistered residual (concern issue634-procrustes-objective-drift).
    """

    # Behaviors -> short context-family labels via the resolved Panel-B map.
    beh_fam_short = [panel_b.get(r) for r in behavior_ids]  # None outside Panel B
    cka_per_layer: list[float] = []
    proc_resid_per_layer: list[float] = []
    proc_resid_centered_per_layer: list[float] = []
    shared_families: list[str] = []
    for li in range(n_layers):
        cb_all, fam_b = family_centroids(beh_bank[:, li, :], beh_fam_short, min_per_family=2)
        cc_all, fam_c = family_centroids(ctx_bank[:, li, :], ctx_families, min_per_family=1)
        shared = sorted(set(fam_b) & set(fam_c))
        if li == 0:
            shared_families = shared
        if len(shared) < 2:
            cka_per_layer.append(float("nan"))
            proc_resid_per_layer.append(float("nan"))
            proc_resid_centered_per_layer.append(float("nan"))
            continue
        cb = np.stack([cb_all[fam_b.index(f)] for f in shared])
        cc = np.stack([cc_all[fam_c.index(f)] for f in shared])
        # PREFLIGHT SHAPE ASSERT (plan §5/§11): matched N required for linear_cka.
        assert cc.shape[0] == cb.shape[0], (
            "cross-space CKA requires matched N — aggregate to family centroids first "
            f"(Cc={cc.shape}, Cb={cb.shape}, layer={li})"
        )
        cka_per_layer.append(linear_cka(cc, cb))
        proc_resid_per_layer.append(_procrustes_resid_raw_lowdim(cb, cc))
        proc_resid_centered_per_layer.append(_procrustes_resid_centered_lowdim(cb, cc))
    return {
        "shared_families_layer0": shared_families,
        "n_shared_families": len(shared_families),
        "cka_per_layer": cka_per_layer,
        "procrustes_resid_per_layer": proc_resid_per_layer,
        "procrustes_resid_is_preregistered_raw": True,
        "procrustes_resid_centered_per_layer": proc_resid_centered_per_layer,
        "procrustes_resid_centered_note": (
            "translation-free centered-Procrustes DIAGNOSTIC, NOT the preregistered "
            "raw residual (which is procrustes_resid_per_layer)"
        ),
        "note": (
            "family-centroid linear CKA + orthogonal Procrustes on matched-N "
            "(families present in BOTH banks, >=2 Panel-B roles per behavior family); "
            "procrustes_resid_per_layer is the preregistered raw "
            "||Cb·R - Cc||_F / ||Cc||_F"
        ),
    }


# ── Figures ──────────────────────────────────────────────────────────────────


def _ctx_palette(ctx_family_order: list[str]) -> dict[str, str]:
    pal = paper_palette(len(ctx_family_order))
    return dict(zip(ctx_family_order, pal, strict=True))


def _joint_scatter(ax, ctx_emb, ctx_families, beh_emb, panel_b_mask, beh_ids, ctx_family_order):
    """Contexts colored by family (circles); behaviors as grey x; Panel-B labeled."""
    colors = _ctx_palette(ctx_family_order)
    for fam in ctx_family_order:
        m = np.asarray(ctx_families) == fam
        if m.any():
            ax.scatter(ctx_emb[m, 0], ctx_emb[m, 1], s=26, color=colors[fam], marker="o", label=fam)
    ax.scatter(
        beh_emb[~panel_b_mask, 0],
        beh_emb[~panel_b_mask, 1],
        s=8,
        color="0.6",
        marker="x",
        label="behavior (non-PanelB)",
    )
    ax.scatter(
        beh_emb[panel_b_mask, 0],
        beh_emb[panel_b_mask, 1],
        s=22,
        color="0.1",
        marker="^",
        label="behavior (PanelB)",
    )
    for i in np.where(panel_b_mask)[0]:
        ax.annotate(beh_ids[i], (beh_emb[i, 0], beh_emb[i, 1]), fontsize=3.5, alpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_joint_embedding(
    ctx_bank, ctx_families, beh_bank, beh_ids, panel_b_mask, qlayers, ctx_family_order, umap_module
):
    """Hero: PCA (top) + UMAP n=15/d=0.1 (bottom) at the quartile layers."""
    n_ctx = ctx_bank.shape[0]
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(2, len(qlayers), figsize=(4 * len(qlayers), 8))
    axes = np.atleast_2d(axes)
    for col, li in enumerate(qlayers):
        joint = center(np.vstack([ctx_bank[:, li, :], beh_bank[:, li, :]]))
        pca_emb = PCA(n_components=2, random_state=SEED).fit_transform(joint)
        _joint_scatter(
            axes[0, col],
            pca_emb[:n_ctx],
            ctx_families,
            pca_emb[n_ctx:],
            panel_b_mask,
            beh_ids,
            ctx_family_order,
        )
        axes[0, col].set_title(f"PCA — layer {li}")
        um = umap_module.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED)
        um_emb = um.fit_transform(joint)
        _joint_scatter(
            axes[1, col],
            um_emb[:n_ctx],
            ctx_families,
            um_emb[n_ctx:],
            panel_b_mask,
            beh_ids,
            ctx_family_order,
        )
        axes[1, col].set_title(f"UMAP (n=15, d=0.1, seed 42) — layer {li}")
    axes[0, 0].legend(fontsize=5, loc="best")
    label = "_".join(f"L{li}" for li in qlayers)
    savefig_paper(fig, f"joint_embedding_pca_umap_{label}", dir=FIG_DIR)
    plt.close(fig)


def fig_nn_rate(nn_rate, null_summary, n_layers):
    """Hero: matched-family NN rate vs layer with permutation-null band."""
    layers = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(7, 4))
    nullmean = null_summary["null_mean_per_layer"]
    null95 = null_summary["null_p95_per_layer"]
    ax.fill_between(layers, nullmean, null95, color="0.85", label="permutation null (mean to 95%)")
    ax.plot(
        layers, nn_rate, color="#1f77b4", lw=2, marker="o", ms=3, label="matched-family NN rate"
    )
    ax.axvline(
        null_summary["argmax_layer"], color="#d62728", lw=1, ls=":", label="best layer (max rate)"
    )
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("matched-family NN rate (Panel B)")
    ax.set_title("H1: behavior -> nearest-context family match vs depth")
    ax.legend(fontsize=7)
    savefig_paper(fig, "nn_rate_vs_layer", dir=FIG_DIR)
    plt.close(fig)


def fig_purity(panelB_labeled_purity, null_summary, n_layers):
    """Panel-B-labeled-subset k-NN family purity vs layer with permutation null.

    H2 read over the Panel-B-LABELED subset (NOT all 275 roles — no 275-role
    role-axis label source exists); the title/label name the underpowered
    labeled-subset denominator (concern issue634-h2-panelb-only).
    """
    layers = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(
        layers,
        null_summary["null_mean_per_layer"],
        null_summary["null_p95_per_layer"],
        color="0.85",
        label="permutation null (mean to 95%)",
    )
    ax.plot(
        layers,
        panelB_labeled_purity,
        color="#2ca02c",
        lw=2,
        marker="o",
        ms=3,
        label="Panel-B-labeled purity",
    )
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("k-NN family purity (k=4)")
    ax.set_title("H2: Panel-B-labeled-subset family purity vs depth (not all 275 roles)")
    ax.legend(fontsize=7)
    savefig_paper(fig, "panelB_labeled_purity_vs_layer", dir=FIG_DIR)
    plt.close(fig)


def fig_tsne_joint(
    ctx_bank, ctx_families, beh_bank, beh_ids, panel_b_mask, best_layer, ctx_family_order
):
    """t-SNE of the joint bank at the best layer (perplexity grid)."""
    from sklearn.manifold import TSNE

    n_ctx = ctx_bank.shape[0]
    joint = center(np.vstack([ctx_bank[:, best_layer, :], beh_bank[:, best_layer, :]]))
    n_samples = joint.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, perp in zip(axes.flat, TSNE_PERPLEXITIES, strict=True):
        # sklearn requires perplexity < n_samples; clamp for tiny banks (the
        # synthetic smoke has 30 samples). On the real 325-point joint bank the
        # full {5,15,30} grid is unaffected.
        eff_perp = min(perp, n_samples - 1)
        ts = TSNE(perplexity=eff_perp, metric="cosine", init="random", random_state=SEED)
        emb = ts.fit_transform(joint)
        _joint_scatter(
            ax, emb[:n_ctx], ctx_families, emb[n_ctx:], panel_b_mask, beh_ids, ctx_family_order
        )
        ax.set_title(f"t-SNE perplexity={eff_perp}, seed 42 — layer {best_layer}", fontsize=8)
    savefig_paper(fig, "tsne_joint_best_layer", dir=FIG_DIR)
    plt.close(fig)


def fig_coembed_gate(gate_per_layer, n_layers):
    layers = np.arange(n_layers)
    var_r = [g["var_ratio_beh_over_ctx"] for g in gate_per_layer]
    scale_r = [g["scale_ratio_beh_over_ctx"] for g in gate_per_layer]
    fig, ax = plt.subplots(figsize=(7, 4))
    pal = paper_palette(2)
    ax.plot(layers, var_r, color=pal[0], lw=2, marker="o", ms=3, label="variance ratio")
    ax.plot(
        layers, scale_r, color=pal[1], lw=1.5, ls="--", marker="s", ms=3, label="median-norm ratio"
    )
    ax.axhline(COEMBED_RATIO_MAX, color="0.4", lw=1, ls=":")
    ax.axhline(1.0 / COEMBED_RATIO_MAX, color="0.4", lw=1, ls=":")
    ax.axhline(1.0, color="0.7", lw=1)
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("behavior / context ratio")
    ax.set_title(
        f"Co-embeddability gate (pass band [1/{COEMBED_RATIO_MAX:.0f}, {COEMBED_RATIO_MAX:.0f}])"
    )
    ax.legend(fontsize=7)
    savefig_paper(fig, "coembeddability_gate", dir=FIG_DIR)
    plt.close(fig)


def fig_cross_space(xspace, n_layers):
    layers = np.arange(n_layers)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pal = paper_palette(2)
    axes[0].plot(layers, xspace["cka_per_layer"], color=pal[0], lw=2, marker="o", ms=3)
    axes[0].set_xlabel("decoder layer")
    axes[0].set_ylabel("family-centroid linear CKA")
    axes[0].set_title("Cross-space CKA (matched-N family centroids)")
    axes[0].set_ylim(0, 1)
    axes[1].plot(layers, xspace["procrustes_resid_per_layer"], color=pal[1], lw=2, marker="s", ms=3)
    axes[1].set_xlabel("decoder layer")
    axes[1].set_ylabel("Procrustes residual ||Cb·R - Cc||/||Cc||")
    axes[1].set_title("Cross-space Procrustes alignment residual")
    savefig_paper(fig, "cross_space_alignment", dir=FIG_DIR)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def make_synthetic_banks(seed: int = SEED) -> tuple[dict, dict, dict]:
    """Tiny synthetic banks for --smoke (no HF, no GPU).

    Context (10, 28, 3584) over 3 short context-families; behavior (20, 28, 3584)
    over the SAME short families (so resolve_panel_b + the centroid fallback both
    exercise unequal-N aggregation). Returns (ctx, beh, fam_map) where fam_map
    uses short family names directly (passthrough alias).
    """
    rng = np.random.default_rng(seed)
    n_layers, hidden = 28, 3584
    ctx_fams = ["persona", "behavior", "format"]
    ctx_ids = [f"ctx_{f}_{i}" for f in ctx_fams for i in range(3)] + ["ctx_default_0"]
    ctx_families = [i.split("_")[1] for i in ctx_ids]
    n_ctx = len(ctx_ids)
    ctx = rng.standard_normal((n_ctx, n_layers, hidden)).astype(np.float32)
    beh_fams = ["persona", "persona", "behavior", "behavior", "format", "format"]
    beh_ids = [f"beh_{f}_{i}" for i, f in enumerate(beh_fams)] + [
        f"beh_extra_{i}" for i in range(14)
    ]
    beh = rng.standard_normal((len(beh_ids), n_layers, hidden)).astype(np.float32)
    fam_map = {bid: f for bid, f in zip(beh_ids[:6], beh_fams, strict=True)}
    fam_map["beh_extra_0"] = "default"  # null-anchor exercised (dropped from Panel B)
    ctx_bank = {"mean": ctx, "ids": ctx_ids, "families": ctx_families}
    beh_bank = {
        "mean": beh,
        "ids": beh_ids,
        "families": [fam_map.get(b) for b in beh_ids],
        "seed": seed,
        "n_questions": 2,
        "sampled_question_indices_hash": "synthetic",
    }
    return ctx_bank, beh_bank, fam_map


def main() -> int:
    global EVAL_DIR, FIG_DIR
    parser = argparse.ArgumentParser(description="Issue #634 Phase 2 joint geometry analysis.")
    parser.add_argument("--behavior-from-hf", default=DEFAULT_BEHAVIOR_PREFIX)
    parser.add_argument("--context-from-hf", default=DEFAULT_CONTEXT_PREFIX)
    parser.add_argument(
        "--behavior-dir", type=Path, default=None, help="local behavior tensors dir"
    )
    parser.add_argument("--context-dir", type=Path, default=None, help="local context tensors dir")
    parser.add_argument("--hf-repo", default=HF_DATA_REPO)
    parser.add_argument(
        "--family-map",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue634" / "behavior_family_map.json",
    )
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--n-perms", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the full pipeline on a tiny synthetic bank (no HF, no GPU)",
    )
    parser.add_argument(
        "--soft-knn-sensitivity",
        action="store_true",
        help=(
            "run ONLY the H1 soft-k-NN family-match sensitivity check (k=1..4 "
            "majority vote over the same panel/tensors/null) and write "
            "h1_soft_knn_sensitivity.json + figure; skips the rest of Phase 2"
        ),
    )
    args = parser.parse_args()

    EVAL_DIR = args.eval_dir
    FIG_DIR = args.fig_dir
    set_paper_style("blog")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # The synthetic --smoke bank is a tiny (10-context, 3-family) stand-in: the
    # full-bank coverage guard ((50, 28, 3584), #594's 7-family set) and the
    # H2 labeled-subset assert apply ONLY to the real HF load.
    expect_full_bank = not args.smoke
    if args.smoke:
        logger.info("SMOKE: synthetic banks (no HF, no GPU); full-bank coverage guard SKIPPED")
        ctx_bank, beh_bank, fam_map = make_synthetic_banks(args.seed)
    else:
        ctx_dir = args.context_dir or download_from_hf(
            args.context_from_hf, EVAL_DIR / "hf_context_cache", repo=args.hf_repo
        )
        beh_dir = args.behavior_dir or download_from_hf(
            args.behavior_from_hf, EVAL_DIR / "hf_behavior_cache", repo=args.hf_repo
        )
        ctx_bank = load_context_bank(ctx_dir, expect_full_bank=expect_full_bank)
        beh_bank = load_behavior_bank(beh_dir)
        fam_map = load_family_map(args.family_map)

    ctx = ctx_bank["mean"]  # (50, L, H)
    beh = beh_bank["mean"]  # (275, L, H)
    ctx_families = ctx_bank["families"]
    beh_ids = beh_bank["ids"]
    n_ctx, n_layers, hidden = ctx.shape
    assert beh.shape[1] == n_layers and beh.shape[2] == hidden, (beh.shape, ctx.shape)
    ctx_family_set = set(ctx_families)
    ctx_family_order = sorted(ctx_family_set)
    logger.info(
        "Loaded banks: ctx %s (%d families), beh %s",
        ctx.shape,
        len(ctx_family_set),
        beh.shape,
    )

    panel_b = resolve_panel_b(fam_map, beh_ids, ctx_family_set)
    panel_b_roles = list(panel_b.keys())
    panel_b_idx = [beh_ids.index(r) for r in panel_b_roles]
    expected_fams = [panel_b[r] for r in panel_b_roles]
    panel_b_mask = np.zeros(len(beh_ids), dtype=bool)
    panel_b_mask[panel_b_idx] = True
    n_tested_families = len(set(expected_fams))
    meets_floor = len(panel_b_roles) >= 12 and all(
        sum(1 for f in expected_fams if f == fam) >= 2 for fam in set(expected_fams)
    )
    logger.info(
        "Panel B: %d roles / %d tested families, meets_floor=%s",
        len(panel_b_roles),
        n_tested_families,
        meets_floor,
    )

    # ── Free-analysis follow-up: soft-k-NN family-match sensitivity ───────────
    # Runs ONLY the relaxed-criterion H1 (k=1..4 majority vote) over the SAME
    # resolved Panel B / banks / shuffled-label null, then exits — the rest of
    # Phase 2 (H2/H3, embeddings, cross-space, hero figures) is skipped.
    if args.soft_knn_sensitivity:
        logger.info("SOFT-kNN: H1 sensitivity (k=%s) only; skipping rest of Phase 2", SOFT_KNN_KS)
        soft = run_soft_knn_sensitivity(
            beh,
            ctx,
            ctx_families,
            panel_b,
            panel_b_idx,
            panel_b_roles,
            expected_fams,
            args.n_perms,
            args.seed,
            expect_full_bank,
        )
        out_path = EVAL_DIR / "h1_soft_knn_sensitivity.json"
        with open(out_path, "w") as f:
            json.dump(jsonable(soft), f, indent=2)
        logger.info("Wrote %s", out_path)
        fig_soft_knn_sensitivity(soft, n_layers)
        for k in SOFT_KNN_KS:
            d = soft["per_k"][_soft_knn_key(k)]
            logger.info(
                "  %s: max_rate=%.3f @ layer %d, null95=%.3f, p_fwer=%.3f, passes=%s",
                _soft_knn_key(k),
                d["max_rate"],
                d["max_layer"],
                d["null_max_p95"],
                d["p_fwer"],
                d["passes"],
            )
        logger.info(
            "SOFT-kNN done. %s (k=1 parity max|Δ|=%.2e vs strict H1)",
            soft["headline_summary"],
            soft["k1_parity_max_abs_diff_vs_strict_h1"],
        )
        return 0

    # Behavior labels for the H2 read. The ONLY family labels available are the
    # 27-role frozen Panel-B map (extraction writes families.append(
    # family_map.get(role)) = None for every non-Panel-B role); there is NO
    # 275-role role-type/axis labeling source. So the H2 read here is purity over
    # the Panel-B-LABELED SUBSET, NOT the planned all-275-role behavior-alone
    # purity. This is RE-LABELED as such throughout (concern issue634-h2-panelb-only,
    # option (b)): the JSON keys carry the `panelB_labeled` prefix and an explicit
    # `h2_is_full_275_role_purity: false` + denominator, so the analyzer surfaces
    # the underpowered ~26-tested-role denominator as a scope caveat and no
    # downstream consumer can read it as the planned 275-role H2.
    beh_fam_labels = np.asarray([fam for fam in beh_bank["families"]])
    has_fam = np.array([f is not None for f in beh_fam_labels])

    rng = np.random.default_rng(args.seed)
    label_perms = make_label_perms(panel_b, args.n_perms, rng)

    # ── Per-layer loop ───────────────────────────────────────────────────────
    gate_per_layer: list[dict] = []
    nn_rate = np.empty(n_layers)
    nn_rate_resid = np.empty(n_layers)
    null_nn = np.empty((args.n_perms, n_layers))
    # Residualized H1 control gets its OWN permutation null computed from the
    # PC1-removed geometry — summarizing it against the raw-H1 null (the round-1
    # bug) would falsely support/reject the control (concern
    # issue634-residual-null-mismatch).
    null_nn_resid = np.empty((args.n_perms, n_layers))
    panelB_labeled_purity = np.empty(n_layers)
    null_panelB_pur = np.empty((args.n_perms, n_layers))
    own_region = np.empty(n_layers)
    panelB_nn_table: dict[str, list[str]] = {r: [] for r in panel_b_roles}

    # H2 purity over the Panel-B-labeled subset (role-axis = the frozen-map
    # family); permutation null shuffles those labels. Built once (labels fixed
    # across layers).
    fam_idx = np.where(has_fam)[0]
    fam_labels_sub = beh_fam_labels[fam_idx]
    pur_perms = np.stack([rng.permutation(len(fam_idx)) for _ in range(args.n_perms)])
    n_total_behaviors = int(beh.shape[0])
    n_labeled_behaviors = len(fam_idx)
    # Production guard: this read is over a LABELED SUBSET, never all 275. The
    # synthetic --smoke bank labels all behaviors, so the assert is only enforced
    # for the real (full-bank) run, where it pins that H2 is NOT mis-presented as
    # the planned 275-role purity.
    if expect_full_bank:
        assert n_labeled_behaviors < n_total_behaviors, (
            "H2 read is the Panel-B-labeled subset; expected fewer labeled "
            f"behaviors than the full bank, got {n_labeled_behaviors} == "
            f"{n_total_behaviors} (a 275-role label source would change this DV's "
            "denominator — re-check the relabel)"
        )

    for li in range(n_layers):
        beh_li = beh[:, li, :]
        ctx_li = ctx[:, li, :]
        gate_per_layer.append(coembeddability_gate(beh_li, ctx_li))

        # H1
        nn_fams = nearest_context_family(beh_li, ctx_li, ctx_families, panel_b_idx)
        for r, nn in zip(panel_b_roles, nn_fams, strict=True):
            panelB_nn_table[r].append(nn)
        nn_rate[li] = h1_matched_rate(nn_fams, expected_fams)
        null_nn[:, li] = h1_perm_null(nn_fams, panel_b_roles, expected_fams, label_perms)

        # H1 control: remove behavior-cloud PC1, re-run — WITH its own null over
        # the residualized geometry (the residualized nearest-context families
        # feed h1_perm_null, NOT the raw nn_fams).
        from sklearn.decomposition import PCA

        beh_c = center(beh_li)
        pc1 = PCA(n_components=1, random_state=SEED).fit(beh_c).components_[0]
        beh_resid = beh_c - np.outer(beh_c @ pc1, pc1)
        nn_fams_r = nearest_context_family(beh_resid, ctx_li, ctx_families, panel_b_idx)
        nn_rate_resid[li] = h1_matched_rate(nn_fams_r, expected_fams)
        null_nn_resid[:, li] = h1_perm_null(nn_fams_r, panel_b_roles, expected_fams, label_perms)

        # H2 behavior-alone purity over the Panel-B-LABELED subset (re-labeled DV)
        d_beh = cosine_dist(beh_li[fam_idx], centering="global_mean")
        order_beh = knn_neighbor_order(d_beh)
        panelB_labeled_purity[li] = knn_purity(order_beh, fam_labels_sub, k=KNN_K)
        null_panelB_pur[:, li] = np.array(
            [knn_purity(order_beh, fam_labels_sub[p], k=KNN_K) for p in pur_perms]
        )

        # H3 own-region fraction
        own_region[li] = own_region_fraction(beh_li, ctx_li, k=KNN_K)
        if li % 7 == 0:
            logger.info("layer %d/%d done (nn_rate=%.3f)", li + 1, n_layers, nn_rate[li])

    nn_summary = max_over_layers_summary(nn_rate.tolist(), null_nn)
    nn_resid_summary = max_over_layers_summary(nn_rate_resid.tolist(), null_nn_resid)
    panelB_labeled_purity_summary = max_over_layers_summary(
        panelB_labeled_purity.tolist(), null_panelB_pur
    )
    best_layer = nn_summary["argmax_layer"]

    # ── Cross-space fallback (always computed; the headline read on gate fail) ─
    logger.info("cross-space family-centroid CKA + Procrustes")
    xspace = cross_space_alignment(beh, ctx, panel_b, beh_ids, ctx_families, n_layers)

    gate_any_fail = any(not g["joint_centered_ok"] for g in gate_per_layer)

    # ── Assemble JSONs (checkpoint each before figures) ──────────────────────
    coembed_json = {
        "ratio_pass_band": [1.0 / COEMBED_RATIO_MAX, COEMBED_RATIO_MAX],
        "per_layer": gate_per_layer,
        "any_layer_fails_gate": bool(gate_any_fail),
        "n_layers_fail": int(sum(1 for g in gate_per_layer if not g["joint_centered_ok"])),
    }
    with open(EVAL_DIR / "coembeddability_gate.json", "w") as f:
        json.dump(jsonable(coembed_json), f, indent=2)

    per_layer_nn = {
        "knn_k": KNN_K,
        "n_perms": args.n_perms,
        "seed": args.seed,
        "panel_b_roles": panel_b_roles,
        "panel_b_expected_families": expected_fams,
        "nn_rate_per_layer": nn_rate.tolist(),
        "nn_rate_residualized_per_layer": nn_rate_resid.tolist(),
        # H2 read: purity over the Panel-B-LABELED subset, NOT all 275 roles.
        "panelB_labeled_purity_per_layer": panelB_labeled_purity.tolist(),
        "own_region_fraction_per_layer": own_region.tolist(),
        "nn_summary": nn_summary,
        # Residualized H1 control summarized against its OWN residualized null.
        "nn_residualized_summary": nn_resid_summary,
        "nn_residualized_null_is_residualized": True,
        "panelB_labeled_purity_summary": panelB_labeled_purity_summary,
        # H2 denominator disclosure (concern issue634-h2-panelb-only, option b):
        "h2_is_full_275_role_purity": False,
        "h2_denominator": "panel_b_labeled_subset",
        "h2_n_labeled_behaviors": n_labeled_behaviors,
        "h2_n_total_behaviors": n_total_behaviors,
        "h2_note": (
            "behavior-alone k-NN family purity over the Panel-B-LABELED subset "
            f"({n_labeled_behaviors} of {n_total_behaviors} roles); NOT the planned "
            "275-role role-axis purity — no 275-role role-type labeling source "
            "exists, so the underpowered labeled-subset denominator is carried as a "
            "clean-result scope caveat."
        ),
    }
    with open(EVAL_DIR / "per_layer_nn_purity.json", "w") as f:
        json.dump(jsonable(per_layer_nn), f, indent=2)

    with open(EVAL_DIR / "panelB_nn_table.json", "w") as f:
        json.dump(
            jsonable({"panel_b_expected": panel_b, "nearest_family_per_layer": panelB_nn_table}),
            f,
            indent=2,
        )

    xspace_out = dict(xspace)
    with open(EVAL_DIR / "cross_space_alignment.json", "w") as f:
        json.dump(jsonable(xspace_out), f, indent=2)

    # H1 floor precondition + verdict
    if not meets_floor:
        h1_verdict = "UNDERPOWERED — Panel B below the pre-registered floor"
    elif nn_summary["passes"]:
        h1_verdict = "H1 SUPPORTED — matched-family NN rate > null 95th pct at best layer"
    else:
        h1_verdict = "H1 NOT SUPPORTED — NN rate indistinguishable from shuffled-label null"

    metrics = {
        "n_context": int(n_ctx),
        "n_behavior": int(beh.shape[0]),
        "n_layers": int(n_layers),
        "hidden": int(hidden),
        "knn_k": KNN_K,
        "n_perms": args.n_perms,
        "seed": args.seed,
        "behavior_seed": beh_bank.get("seed"),
        "behavior_n_questions": beh_bank.get("n_questions"),
        "behavior_sampled_question_indices_hash": beh_bank.get("sampled_question_indices_hash"),
        "panel_b": panel_b,
        "panel_b_n_roles": len(panel_b_roles),
        "panel_b_n_tested_families": n_tested_families,
        "panel_b_meets_floor": bool(meets_floor),
        "best_layer_by_nn_rate": int(best_layer),
        "coembeddability_gate_any_fail": bool(gate_any_fail),
        "context_probe_pool_hash": ctx_bank.get("probe_pool_hash"),
        "h1_nn_summary": nn_summary,
        "h1_residualized_summary": nn_resid_summary,
        "h1_residualized_null_is_residualized": True,
        "h1_verdict": h1_verdict,
        "h2_panelB_labeled_purity_summary": panelB_labeled_purity_summary,
        "h2_is_full_275_role_purity": False,
        "h2_denominator": "panel_b_labeled_subset",
        "h2_n_labeled_behaviors": n_labeled_behaviors,
        "h2_n_total_behaviors": n_total_behaviors,
        "h3_own_region_fraction_per_layer": own_region.tolist(),
        "h3_own_region_fraction_best_layer": float(own_region[best_layer]),
        "cross_space_alignment": xspace,
        "quartile_layers": quartile_layers(n_layers),
        "family_name_alias": FAMILY_NAME_ALIAS,
        "metadata": reproducibility_metadata({"script": "issue634_joint_geometry"}),
    }
    with open(EVAL_DIR / "joint_geometry_metrics.json", "w") as f:
        json.dump(jsonable(metrics), f, indent=2)
    logger.info("Wrote metrics JSONs to %s", EVAL_DIR)

    # ── Figures ──────────────────────────────────────────────────────────────
    logger.info("figures")
    import umap

    qlayers = quartile_layers(n_layers)
    fig_joint_embedding(
        ctx, ctx_families, beh, beh_ids, panel_b_mask, qlayers, ctx_family_order, umap
    )
    fig_nn_rate(nn_rate.tolist(), nn_summary, n_layers)
    fig_purity(panelB_labeled_purity.tolist(), panelB_labeled_purity_summary, n_layers)
    fig_tsne_joint(ctx, ctx_families, beh, beh_ids, panel_b_mask, best_layer, ctx_family_order)
    fig_coembed_gate(gate_per_layer, n_layers)
    fig_cross_space(xspace, n_layers)

    logger.info(
        "Done. best_layer=%d, nn_rate_max=%.3f (null95 %.3f), %s; gate_fail=%s; "
        "H2 panelB-labeled purity_max=%.3f (over %d of %d roles)",
        best_layer,
        nn_summary["observed_max"],
        nn_summary["null_max_p95"],
        h1_verdict,
        gate_any_fail,
        panelB_labeled_purity_summary["observed_max"],
        n_labeled_behaviors,
        n_total_behaviors,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
