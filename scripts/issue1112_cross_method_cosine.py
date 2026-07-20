"""#1112 free-analysis follow-up: CROSS-METHOD (full-FT vs LoRA) debiased
direction-cosine CIs over the persisted capture tensors.

The parent #1112 geometry pass reported the cross-method mean-shift-direction
cosine ONLY as raw point estimates (``cross_cell_diffs.H1_method_ftneg_vs_loraneg``
in ``geometry_per_cell.json``; ``cos_mu_s5_s3`` in ``geometry_lr_matched.json``)
and never ran the issue's own paired half-draw + attenuation-correction
machinery on the cross-method pairs. A raw cosine of 0.95 between two
mean-shift directions is uninterpretable without the attenuation floor: each
cell's OWN split-half self-cosine caps how high any cross-cell cosine can read,
so a "direction difference" claim needs the corrected read
``cross / geomean(same-cell split-half refs)``.

This script reuses the within-method debiased machinery verbatim
(``issue1112_debiased_cosine``: paired subsample-WITHOUT-replacement half-draws,
same row indices in both cells per draw, same-cell split-half attenuation
references at the same m, corrected = cross / sqrt(ref_a*ref_b)) and the #1112
geometry staging + loaders (``issue1112_geometry`` / ``experiments.issue_1112``)
— no re-implemented math, no re-download of the whole prefix. 0 GPU-h, CPU-only,
seconds of compute (every draw battery is one masked GEMM per (cell, half)).

Cross-method pairs (all at the SELECTED dose):
  H1x       full-FT+neg (s3_fullft_neg)   vs LoRA+neg (s1_lora_neg)          [matched install; lr-confounded]
  H1x_pos   full-FT+pos (s4_fullft_pos)   vs LoRA+pos (s2_lora_pos)          [install-mismatched: 0.615 vs 0.79]
  H1x_lrm   full-FT+neg (s3_fullft_neg)   vs lr-matched LoRA+neg@5e-6 (s5)   [matched install; LR-CONFOUND-FREE headline]
  marker    full-FT marker (m2_fullft_band8) vs LoRA marker (m1_lora_band8) [~4x install-mismatched; low-signal]

Arms: response (own-text), context (own-text), response (shared-text, from
capture_tf teacher-forced over shared base generations). Prefix arm is reported
as raw point cosine only — half-draw debiasing is degenerate there (6
structurally-unique rows). Registered layers 14 (sycophancy) / 25 (marker) are
required; full 28-layer profiles are produced for the response + context arms.

Usage:
    uv run python scripts/issue1112_cross_method_cosine.py \
        [--stage-dir data/issue_1112/hf_dl/cross_method] \
        [--out eval_results/issue_1112/geometry/cross_method_cosine.json] \
        [--fig figures/issue_1112/hero_cross_method_cosine.png] \
        [--draws 2000] [--seed 1112]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps land BEFORE numpy/torch import — load_dotenv setdefaults the
# OMP/MKL/OPENBLAS/NUMEXPR caps on the shared VM and the BLAS/torch pools freeze
# at import time.
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Sibling scripts live next to this one; make them importable under `uv run`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.experiments.issue_1112 import (  # noqa: E402
    MARKER_READ_LAYER,
    PRIMARY_LAYER,
)
from explore_persona_space.experiments.issue_1112 import geometry as geo  # noqa: E402
from explore_persona_space.experiments.issue_653.spectral import cosine  # noqa: E402

# Reuse the within-method debiased machinery (paths, math) verbatim.
from issue1112_debiased_cosine import (  # noqa: E402
    COS_CUTOFF,
    QUANTILES,
    analyze_pair,
    half_partition_masks,
)

# Reuse the #1112 geometry staging + store-reorder helpers verbatim.
from issue1112_geometry import _fetch_one, _reorder_store, _store_keys  # noqa: E402

logger = logging.getLogger("issue1112.cross_method_cosine")

# ── Pinned data-repo revisions (task body Repro footer; brief) ────────────────
OWN_REV = "e016910195b7ab846c83b87ec43140c36c51e35f"  # capture/ own-text (== PARENT_STORE_REV)
TF_REV = "97773142f252e90c85bd2024edf83b7e40496f87"  # capture_tf/ shared-text
LRM_REV = "d7080b974d001bec8f1afdbe945a2fd2ebd11f63"  # lr-matched round capture
DATA_REPO = "superkaiba1/explore-persona-space-data"
DATA_PREFIX = "issue1112_geometry2x2"

BASE_SYCO = "base_sycophancy"
BASE_MARKER = "base_marker"

# Sanity-gate references (geometry_per_cell.json H1 raw full-cloud cos_mu).
SANITY = {
    ("response", PRIMARY_LAYER): 0.9521929165407169,
    ("context", PRIMARY_LAYER): 0.8860145544875438,
}
SANITY_TOL = 1e-3

# Cross-method pairs. Each: (label, cell_a, rev_a, cell_b, rev_b, base_cell,
# registered_layer, arms, install_note). Arms are ("response","context") for
# own-text; a separate shared-text response arm is added when both cells have a
# capture_tf store.
PAIRS = (
    {
        "label": "H1x_ftneg_vs_loraneg",
        "cell_a": "s3_fullft_neg",
        "rev_a": OWN_REV,
        "cell_b": "s1_lora_neg",
        "rev_b": OWN_REV,
        "base_cell": BASE_SYCO,
        "registered_layer": PRIMARY_LAYER,
        "own_arms": ("response", "context"),
        "shared_text": True,
        "install_note": (
            "matched install (both +negatives) BUT lr-confounded: s1_lora_neg lr=1e-5, "
            "s3_fullft_neg lr=5e-6 — see H1x_lrm for the lr-confound-free comparator"
        ),
    },
    {
        "label": "H1x_pos_ftpos_vs_lorapos",
        "cell_a": "s4_fullft_pos",
        "rev_a": OWN_REV,
        "cell_b": "s2_lora_pos",
        "rev_b": OWN_REV,
        "base_cell": BASE_SYCO,
        "registered_layer": PRIMARY_LAYER,
        "own_arms": ("response", "context"),
        "shared_text": True,
        "install_note": "install-MISMATCHED (positives-only; judged 0.615 vs 0.79)",
    },
    {
        "label": "H1x_lrm_ftneg_vs_lora_lr5e6",
        "cell_a": "s3_fullft_neg",
        "rev_a": OWN_REV,
        "cell_b": "s5_lora_neg_lr5e6",
        "rev_b": LRM_REV,
        "base_cell": BASE_SYCO,
        "registered_layer": PRIMARY_LAYER,
        "own_arms": ("response", "context"),
        "shared_text": False,  # no capture_tf store for the lr-matched cell
        "install_note": (
            "matched install AND lr-confound-free (both at lr=5e-6) — the headline "
            "cross-method direction-similarity read"
        ),
    },
    {
        "label": "marker_ft_vs_lora",
        "cell_a": "m2_fullft_band8",
        "rev_a": OWN_REV,
        "cell_b": "m1_lora_band8",
        "rev_b": OWN_REV,
        "base_cell": BASE_MARKER,
        "registered_layer": MARKER_READ_LAYER,
        "own_arms": ("response",),
        "shared_text": False,
        "install_note": (
            "~4x install-mismatched AND low-signal: the LoRA marker split-half self-cosine "
            "was 0.311 — the attenuation floor is deep, read the corrected cosine with care"
        ),
    },
)

# Capture-tf response-arm cells available on the Hub (issue1112_geometry.TF_SYCO_CELLS
# ∩ the sycophancy 2x2). s5_lora_neg_lr5e6 + marker cells have no tf capture.
TF_CELLS = ("s1_lora_neg", "s2_lora_pos", "s3_fullft_neg", "s4_fullft_pos")


def _own_path(root: Path, cell: str) -> Path:
    dose = "base" if cell in (BASE_SYCO, BASE_MARKER) else "selected"
    return root / "capture" / cell / dose / "pooled.pt"


def _tf_path(root: Path, cell: str) -> Path:
    return root / "capture_tf" / cell / "selected" / "pooled.pt"


def stage(root: Path) -> None:
    """Per-file scoped download of exactly the stores the cross-method reads
    need (bases + own-text cells @ pinned revs + shared-text tf cells). Reuses
    ``issue1112_geometry._fetch_one`` (bounded retry, skips already-staged)."""
    # bases + own-text cells (dose/rev per cell)
    own_specs: list[tuple[str, str]] = [
        (BASE_SYCO, OWN_REV),
        (BASE_MARKER, OWN_REV),
    ]
    for p in PAIRS:
        own_specs.append((p["cell_a"], p["rev_a"]))
        own_specs.append((p["cell_b"], p["rev_b"]))
    seen: set[tuple[str, str]] = set()
    for cell, rev in own_specs:
        if (cell, rev) in seen:
            continue
        seen.add((cell, rev))
        dose = "base" if cell in (BASE_SYCO, BASE_MARKER) else "selected"
        _fetch_one(
            f"{DATA_PREFIX}/analysis_tensors/capture/{cell}/{dose}/pooled.pt",
            _own_path(root, cell),
            rev,
        )
    for cell in TF_CELLS:
        _fetch_one(
            f"{DATA_PREFIX}/analysis_tensors/capture_tf/{cell}/selected/pooled.pt",
            _tf_path(root, cell),
            TF_REV,
        )
    logger.info("[stage] cross-method stores staged under %s", root)


def _question_idx(base: dict) -> np.ndarray:
    return np.asarray([q for _, q in _store_keys(base)])


def _question_aligned_ok(q: np.ndarray) -> bool:
    """question_aligned needs an even unique-question count and equal rows per
    question (so both halves stay exact); else the scheme is skipped."""
    qs, counts = np.unique(q, return_counts=True)
    return len(qs) % 2 == 0 and len(set(counts.tolist())) == 1


def _masks_for(base: dict, n_draws: int, seed: int) -> dict[str, np.ndarray | None]:
    n = len(base["row_meta"])
    assert n % 2 == 0, f"odd cloud size {n} — half-draw split undefined"
    q = _question_idx(base)
    out: dict[str, np.ndarray | None] = {
        "row_random": half_partition_masks(n, n_draws, seed),
    }
    out["question_aligned"] = (
        half_partition_masks(n, n_draws, seed + 1, question_idx=q)
        if _question_aligned_ok(q)
        else None
    )
    return out


def _strip_draws(entry: dict, *, keep_draws: bool) -> dict:
    """analyze_pair carries the full per-draw arrays under 'draws'. Dropped by
    default (keep_draws=False) — the summaries carry the distribution and the
    draws reproduce from seed + n_draws + code — to bound the JSON."""
    if keep_draws:
        return entry
    return {k: v for k, v in entry.items() if k != "draws"}


def _arm_profile(
    cloud_fn,
    layers: list[int],
    masks_by_scheme: dict[str, np.ndarray | None],
    registered_layer: int,
) -> dict:
    """Per-layer debiased read for one (pair, arm). row_random is the PRIMARY
    scheme (all layers); question_aligned is a registered-layer-only companion.

    ``cloud_fn(layer) -> (cloud_a, cloud_b)`` returns the two paired Δx clouds.
    """
    schemes: dict[str, dict] = {}
    for scheme, masks in masks_by_scheme.items():
        if masks is None:
            schemes[scheme] = {"skipped": "panel does not support question-aligned halves"}
            continue
        # row_random: every layer. question_aligned: registered layer only.
        layer_set = layers if scheme == "row_random" else [registered_layer]
        per_layer: dict[str, dict] = {}
        for layer in layer_set:
            cloud_a, cloud_b = cloud_fn(layer)
            entry = analyze_pair(cloud_a, cloud_b, masks)
            # Per-draw arrays omitted (the summaries carry mean/std/quantiles/frac_below_cutoff
            # and the draws reproduce from seed + n_draws + code) — keeps the JSON lean.
            per_layer[str(layer)] = _strip_draws(entry, keep_draws=False)
        schemes[scheme] = {"per_layer": per_layer}
    return schemes


def _raw_prefix(cloud_fn, layers: list[int]) -> dict:
    """Raw full-cloud mean-shift cosine per layer for the degenerate prefix arm
    (6 structurally-unique rows; half-draw debiasing not meaningful)."""
    return {
        str(layer): {"point_cos_full_cloud": cosine(*(c.mean(axis=0) for c in cloud_fn(layer)))}
        for layer in layers
    }


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run(root: Path, *, n_draws: int, seed: int) -> dict:
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    # Bases + shared masks (one mask set per base, shared across all its pairs +
    # every cross/ref statistic of a draw => every comparison is PAIRED).
    base_syco = geo.load_store(_own_path(root, BASE_SYCO))
    base_marker = geo.load_store(_own_path(root, BASE_MARKER))
    syco_keys, marker_keys = _store_keys(base_syco), _store_keys(base_marker)
    layers = sorted(next(iter(base_syco["arms"].values())).keys())
    masks_syco = _masks_for(base_syco, n_draws, seed)
    masks_marker = _masks_for(base_marker, n_draws, seed)

    # Reordered own-text + tf stores, cached (s3_fullft_neg is reused across
    # H1x and H1x_lrm).
    own_cache: dict[str, dict] = {}
    tf_cache: dict[str, dict] = {}

    def own(cell: str, base_keys: list) -> dict:
        if cell not in own_cache:
            own_cache[cell] = _reorder_store(geo.load_store(_own_path(root, cell)), base_keys)
        return own_cache[cell]

    def tf(cell: str) -> dict:
        if cell not in tf_cache:
            store = _reorder_store(geo.load_store(_tf_path(root, cell)), syco_keys)
            cond = store.get("metadata", {}).get("conditioning")
            assert cond == "tf_shared_base", (cell, cond)
            tf_cache[cell] = store
        return tf_cache[cell]

    # ── Sanity gate: recompute H1 raw full-cloud cos_mu, assert vs the parent ─
    sa = own("s3_fullft_neg", syco_keys)
    sb = own("s1_lora_neg", syco_keys)
    sanity: dict[str, dict] = {}
    for (arm, layer), ref in SANITY.items():
        got = cosine(
            geo.delta_cloud(sa, base_syco, arm, layer).mean(axis=0),
            geo.delta_cloud(sb, base_syco, arm, layer).mean(axis=0),
        )
        ok = abs(got - ref) <= SANITY_TOL
        sanity[f"H1_{arm}_L{layer}"] = {"recomputed": got, "reference": ref, "ok": ok}
        if not ok:
            raise AssertionError(
                f"SANITY GATE FAILED: H1 {arm} L{layer} cos_mu {got} != {ref} (tol {SANITY_TOL}) "
                "— tensor orientation / staging bug; refusing to run the draw battery"
            )
    logger.info("[sanity] H1 raw cos_mu matches parent at L%d response + context", PRIMARY_LAYER)

    pairs_out: dict[str, dict] = {}
    prefix_raw: dict[str, dict] = {}
    for p in PAIRS:
        base = base_marker if p["base_cell"] == BASE_MARKER else base_syco
        base_keys = marker_keys if p["base_cell"] == BASE_MARKER else syco_keys
        masks = masks_marker if p["base_cell"] == BASE_MARKER else masks_syco
        store_a = own(p["cell_a"], base_keys)
        store_b = own(p["cell_b"], base_keys)
        arms_out: dict[str, dict] = {}
        for arm in p["own_arms"]:

            def _cloud(layer: int, _a=store_a, _b=store_b, _arm=arm):
                return (
                    geo.delta_cloud(_a, base, _arm, layer),
                    geo.delta_cloud(_b, base, _arm, layer),
                )

            arms_out[f"{arm}_own"] = {
                "capture": "own-text (each model's own greedy generations)",
                "n_rows": int(geo.delta_cloud(store_a, base, arm, p["registered_layer"]).shape[0]),
                "n_structural_unique_rows": geo.structural_unique_rows(store_a, arm),
                "m": len(base_keys) // 2,
                "registered_layer": p["registered_layer"],
                "schemes": _arm_profile(_cloud, layers, masks, p["registered_layer"]),
            }
        if p["shared_text"]:
            tf_a, tf_b = tf(p["cell_a"]), tf(p["cell_b"])

            def _cloud_tf(layer: int, _a=tf_a, _b=tf_b):
                return (
                    geo.delta_cloud(_a, base_syco, "response", layer),
                    geo.delta_cloud(_b, base_syco, "response", layer),
                )

            arms_out["response_shared"] = {
                "capture": "shared-text (teacher-forced over shared base greedy generations)",
                "n_rows": int(geo.delta_cloud(tf_a, base_syco, "response", 0).shape[0]),
                "n_structural_unique_rows": geo.structural_unique_rows(tf_a, "response"),
                "m": len(syco_keys) // 2,
                "registered_layer": p["registered_layer"],
                "schemes": _arm_profile(_cloud_tf, layers, masks_syco, p["registered_layer"]),
            }
        # Prefix arm — raw point cosine only (degenerate: 6 structurally-unique rows).
        if p["base_cell"] == BASE_SYCO:

            def _cloud_prefix(layer: int, _a=store_a, _b=store_b):
                return (
                    geo.delta_cloud(_a, base, "prefix", layer),
                    geo.delta_cloud(_b, base, "prefix", layer),
                )

            prefix_raw[p["label"]] = {
                "n_structural_unique_rows": geo.structural_unique_rows(store_a, "prefix"),
                "note": "half-draw debiasing degenerate (prefix depends only on context)",
                "raw_by_layer": _raw_prefix(_cloud_prefix, layers),
            }
        pairs_out[p["label"]] = {
            "cell_a": p["cell_a"],
            "cell_b": p["cell_b"],
            "base_cell": p["base_cell"],
            "dose": "selected",
            "registered_layer": p["registered_layer"],
            "install_note": p["install_note"],
            "arms": arms_out,
        }
        logger.info("[cross-method] %s done", p["label"])

    return {
        "schema_version": 1,
        "followup_label": "cross-method-debiased-cosine",
        "description": (
            "Cross-method (full-FT vs LoRA) mean-shift-direction cosine with paired "
            "subsample-WITHOUT-replacement half-draw CIs (m = n/2, same row indices in both "
            "cells per draw) + same-cell split-half attenuation references at the same m; "
            "corrected = cross / sqrt(ref_a*ref_b). Reuses the within-method #1112 debiased "
            "machinery on the cross-method cell pairs the parent reported only as raw points."
        ),
        "notes": (
            "cross uses the SAME half-A rows in both cells, so row-level noise correlated across "
            "cells (shared base-row subtraction) attenuates cross LESS than the independent-noise "
            "same-cell reference — a cross distribution BELOW the reference is conservative "
            "evidence of a genuine direction difference. corrected = cross/sqrt(ref_a*ref_b) "
            "assumes independent noise across cells and is a plug-in, not a headline. row_random "
            "is the PRIMARY scheme (full 28-layer profiles); question_aligned is a "
            "registered-layer-only companion (sycophancy panels only). Prefix arm: raw point "
            "cosine only — degenerate under half-draws (6 structurally-unique rows)."
        ),
        "primary_scheme": "row_random",
        "companion_scheme": "question_aligned (registered layer only; sycophancy panels)",
        "n_draws": n_draws,
        "seed_row_random": seed,
        "seed_question_aligned": seed + 1,
        "cutoff": COS_CUTOFF,
        "quantiles": list(QUANTILES),
        "registered_layers": {"sycophancy": PRIMARY_LAYER, "marker": MARKER_READ_LAYER},
        "capture": {
            "data_repo": DATA_REPO,
            "data_prefix": DATA_PREFIX,
            "own_text_rev": OWN_REV,
            "shared_text_rev": TF_REV,
            "lr_matched_rev": LRM_REV,
            "stage_root": str(root),
            "arm_provenance": {
                "response_own": "each model's own greedy generations (own-text response arm)",
                "context_own": "identical prompt tokens up to and incl. the user query",
                "response_shared": "teacher-forced over shared base greedy generations (capture_tf)",
                "prefix_own": "system+persona prefix tokens (context-only; 6 unique rows)",
            },
        },
        "install_context": {
            "matched_pairs": ["H1x_ftneg_vs_loraneg", "H1x_lrm_ftneg_vs_lora_lr5e6"],
            "H1x_lr_confound": "s1_lora_neg lr=1e-5 vs s3_fullft_neg lr=5e-6 (H1x_lrm removes it)",
            "H1x_pos_mismatch": "positives-only judged 0.615 vs 0.79",
            "marker_mismatch": "~4x install-mismatched; LoRA marker split-half self-cosine 0.311",
        },
        "sanity_gate": sanity,
        "pairs": pairs_out,
        "prefix_arm_raw": prefix_raw,
        "metadata": {
            "git_commit": _git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "reference_json": "eval_results/issue_1112/geometry/geometry_per_cell.json",
        },
    }


def make_figure(payload: dict, fig_path: Path) -> None:
    """Per-layer corrected cross-method cosine profiles (response own-text,
    context own-text, shared-text response), registered layer 14 marked, with
    the raw (uncorrected) values as faint companions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    role = {
        "H1x_ftneg_vs_loraneg": ("primary", "full-FT+neg vs LoRA+neg"),
        "H1x_pos_ftpos_vs_lorapos": ("baseline", "full-FT+pos vs LoRA+pos"),
        "H1x_lrm_ftneg_vs_lora_lr5e6": ("control", "full-FT+neg vs LoRA+neg (lr-matched 5e-6)"),
    }
    panels = [
        ("response_own", "Response arm (own-text)"),
        ("context_own", "Context arm (own-text)"),
        ("response_shared", "Response arm (shared-text)"),
    ]
    reg = payload["registered_layers"]["sycophancy"]

    def _series(pair_key: str, arm: str):
        arms = payload["pairs"].get(pair_key, {}).get("arms", {})
        pl = arms.get(arm, {}).get("schemes", {}).get("row_random", {}).get("per_layer")
        if not pl:
            return None
        layers = sorted(int(k) for k in pl)
        corr = [pl[str(li)]["summary"]["corrected"]["mean"] for li in layers]
        raw = [pl[str(li)]["summary"]["cross"]["mean"] for li in layers]
        return layers, corr, raw

    # No sharey: the context panel carries negative (anti-correlated) values —
    # the positives-only pair reaches ~-0.31 at L14 and ~-0.89 at L3-9 — so it
    # needs an extended y-range covering the data plus a zero reference line;
    # the response/shared panels stay [0,1].
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for ax, (arm, title) in zip(axes, panels, strict=True):
        for pair_key, (r, label) in role.items():
            s = _series(pair_key, arm)
            if s is None:
                continue
            layers, corr, raw = s
            color = paper_palette_role(r)
            ax.plot(layers, corr, color=color, lw=1.8, label=label, zorder=3)
            ax.plot(layers, raw, color=color, lw=1.0, ls=":", alpha=0.45, zorder=2)
        ax.axvline(reg, color=paper_palette_role("neutral"), lw=0.8, ls="--", alpha=0.7, zorder=1)
        if arm == "context_own":
            ax.axhline(0.0, color=paper_palette_role("neutral"), lw=0.8, alpha=0.7, zorder=1)
            ax.set_ylim(-0.95, 1.02)
        else:
            ax.set_ylim(0.0, 1.02)
        ax.set_title(title)
        ax.set_xlabel("Layer")
    axes[0].set_ylabel("Cross-method direction cosine")
    axes[0].legend(loc="lower left", fontsize=7, frameon=False)
    fig.tight_layout()
    savefig_paper(fig, fig_path.stem, dir=fig_path.parent, formats=("png",))
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage-dir", type=Path, default=Path("data/issue_1112/hf_dl/cross_method"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("eval_results/issue_1112/geometry/cross_method_cosine.json"),
    )
    p.add_argument(
        "--fig", type=Path, default=Path("figures/issue_1112/hero_cross_method_cosine.png")
    )
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1112)
    p.add_argument("--skip-stage", action="store_true", help="reuse already-staged tensors")
    args = p.parse_args(argv)

    t0 = time.time()
    if not args.skip_stage:
        stage(args.stage_dir)
    payload = run(args.stage_dir, n_draws=args.draws, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    logger.info("[cross-method] wrote %s (%.1fs)", args.out, time.time() - t0)
    make_figure(payload, args.fig)
    logger.info("[cross-method] wrote %s", args.fig)
    print(f"wrote {args.out} and {args.fig} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
