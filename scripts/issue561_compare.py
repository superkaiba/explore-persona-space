#!/usr/bin/env python3
"""#561 VM-side comparison: positive-only marker arm vs the persisted #551 tensors.

Loads the 9 NEW positive-only cells (3 seeds x 3 text flavors, produced by
``scripts/run_issue561_posonly.sh``) plus the 18 persisted #551 comparison
cells (contrastive marker + EM, private data repo at the pinned revision)
and computes, per cell:

- weighted SVD summary: top-direction share ``s_1/sum(s)``, U1, per-persona
  cos-to-U1, singular values, ``||M||_F`` (critique #15.6 magnitude read);
- unit-norm re-read (``issue551_unitnorm_reread.unit_normalize_columns``):
  unit-norm top share, U1 rotation ``|cos(U1_w, U1_u)|``, per-persona
  unit-norm cos, aligned-set membership at |cos| >= 0.5 under both reads;
- sign-flip (GATING) + row-shuffle (descriptive) nulls, 1,000 reps,
  null seed = cell seed;
- norm-vs-alignment Spearman rho + one-sided positive permutation p
  (10,000 draws) under BOTH reads (weighted matches #551 control C;
  unit-norm matches the #551 re-read);
- split-half reliability from ``delta_v_per_q``: even/odd halves + 50 random
  10/10 splits (rng seed 42);
- whole-response (mean-over-response) read: SVD top share + mean cos.

Cross-arm: per (variant, seed) ``|cos(U1_posonly, U1_em)|`` and
``|cos(U1_posonly, U1_marker_contrastive)|`` + an empirical random-pair
floor (1,000 gaussian pairs in R^H) — critique #15.6 direction-identity
diagnostic vs the #552 generic-direction finding.

Consistency cross-check (#551 weighted-consistency precedent): recomputed
weighted cos-to-U1 for the #551 same-text cells must match the persisted
``eval_results/issue_551/controls/norm_alignment.json`` within 0.001 —
refuses to compare reads built from different matrix conventions.

Headline summary evaluates the plan §3 pre-registered clauses MECHANICALLY
(per-seed top-share >= 0.46 in >= 2/3 seeds, mean >= 0.50, sign-flip pass,
unit-norm survival; falsify: mean <= 0.40 with marker-like unit-norm
behavior) and routes every other outcome to "indeterminate" (reconciler
binding rule #15.1). The analyzer owns the final verdict; threshold calls
within the realized extraction-smoke drift are measurement-limited (#15.3).

Zero GPU. Run from the repo root (VM, CPU)::

    uv run python scripts/issue561_compare.py \\
        --new-shifts-dir eval_results/issue_561/shifts \\
        --out eval_results/issue_561/comparison

Smoke (subset, local #551 tensors)::

    uv run python scripts/issue561_compare.py \\
        --new-shifts-dir /tmp/i561_smoke/new_shifts --new-cells marker_seed42 \\
        --variants same --issue551-local-dir /tmp/i561_smoke/i551_shifts \\
        --issue551-cells same_marker_seed42 same_em_seed42 \\
        --out /tmp/i561_smoke/out --figures-dir /tmp/i561_smoke/figs
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from issue551_controls import N_PERM, SOURCE_PERSONA, _git_commit, _one_sided_perm_p
from issue551_unitnorm_reread import unit_normalize_columns

from explore_persona_space.analysis.svd_direction_constancy import (
    assemble_M,
    cosine,
    row_shuffle_null,
    sign_flip_null,
    svd_summary,
)

logger = logging.getLogger(__name__)

VARIANTS = ("same", "base", "on_policy")
I551_ARMS = ("marker", "em")
SEEDS = (42, 137, 256)
N_NULL_REPS = 1000
ALIGNED_COS_THRESHOLD = 0.5
WEIGHTED_CONSISTENCY_TOL = 1e-3

# Plan §3 pre-registered thresholds (do NOT change without a plan amendment).
PER_SEED_TOP_SHARE = 0.46
MEAN_TOP_SHARE_CONFIRM = 0.50
MEAN_TOP_SHARE_FALSIFY = 0.40
UNITNORM_TOP_SHARE_MIN = 0.40
UNITNORM_TOP_SHARE_MARKER_LIKE = 0.32
ROTATION_MIN = 0.95
MEMBERSHIP_MIN = 12  # of 14 personas at |cos| > 0.5


@dataclass(frozen=True)
class CompareCell:
    """One analysis cell across the three arms (posonly is the NEW arm)."""

    arm: str  # "posonly" | "marker" | "em"
    variant: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.arm}:{self.variant}_seed{self.seed}"

    @property
    def file_stem(self) -> str:
        # New-arm tensors are written by the same dispatcher under the arm
        # name "marker"; the #551 tensors carry their own arm in the stem.
        file_arm = "marker" if self.arm == "posonly" else self.arm
        return f"{self.variant}_{file_arm}_seed{self.seed}"


def _load_shifts(shifts_dir: Path, stem: str) -> dict[str, dict[str, torch.Tensor]]:
    path = shifts_dir / f"{stem}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing shift tensor {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["shifts"]


def _download_i551(repo: str, prefix: str, revision: str, stems: list[str], dest: Path) -> Path:
    """Per-file `hf_hub_download` of the requested #551 tensors at the PIN."""
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        for suffix in (".pt", ".manifest.json"):
            fname = f"{stem}{suffix}"
            target = dest / fname
            if target.exists() and target.stat().st_size > 0:
                logger.info("[skip] %s already downloaded", fname)
                continue
            local = hf_hub_download(
                repo_id=repo,
                filename=f"{prefix}/{fname}",
                repo_type="dataset",
                revision=revision,
            )
            shutil.copy2(local, target)
            logger.info("[downloaded] %s (%.2f MB)", fname, target.stat().st_size / 1e6)
    return dest


def _split_half_reliability(
    per_q: torch.Tensor, *, n_random: int, rng: np.random.Generator
) -> dict | None:
    """Even/odd + random 50/50 question-split reliability (cosine of half-means)."""
    n = int(per_q.shape[0])
    if n < 2:
        return None
    even = per_q[0::2].mean(dim=0).numpy()
    odd = per_q[1::2].mean(dim=0).numpy()
    half = n // 2
    vals: list[float] = []
    for _ in range(n_random):
        perm = torch.as_tensor(rng.permutation(n))
        a = per_q[perm[:half]].mean(dim=0).numpy()
        b = per_q[perm[half : 2 * half]].mean(dim=0).numpy()
        vals.append(cosine(a, b))
    return {
        "even_odd_cosine": float(cosine(even, odd)),
        "random_splits_mean": float(np.mean(vals)),
        "random_splits_sd": float(np.std(vals)),
        "n_random_splits": n_random,
        "n_questions": n,
    }


def analyze_cell(
    cell: CompareCell,
    shifts: dict[str, dict[str, torch.Tensor]],
    *,
    n_random_splits: int,
    split_rng_seed: int,
) -> tuple[dict, dict]:
    """Full per-cell read. Returns (json_entry, in_memory_extras{U1_w, M})."""
    M, personas = assemble_M(shifts)  # sorted persona order (deterministic)
    svd_w = svd_summary(M)
    svd_u = svd_summary(unit_normalize_columns(M))
    rotation = abs(cosine(svd_u["U1"], svd_w["U1"]))
    norms = np.linalg.norm(M, axis=0)
    cos_w = {p: float(svd_w["cos_to_U1"][i]) for i, p in enumerate(personas)}
    cos_u = {p: float(svd_u["cos_to_U1"][i]) for i, p in enumerate(personas)}

    row_null = row_shuffle_null(M, n_reps=N_NULL_REPS, seed=cell.seed)
    sign_null = sign_flip_null(M, n_reps=N_NULL_REPS, seed=cell.seed)

    abs_cos_w = np.array([abs(cos_w[p]) for p in personas])
    abs_cos_u = np.array([abs(cos_u[p]) for p in personas])
    rho_w, p_w = _one_sided_perm_p(norms, np.array([cos_w[p] for p in personas]), seed=cell.seed)
    rho_u, p_u = _one_sided_perm_p(norms, abs_cos_u, seed=cell.seed)

    # Whole-response (mean-over-response) read.
    m_mr, _ = assemble_M(shifts, persona_order=personas, use_mean_resp=True)
    svd_mr = svd_summary(m_mr)
    sign_null_mr = sign_flip_null(m_mr, n_reps=N_NULL_REPS, seed=cell.seed)

    # Split-half reliability per persona (rng seed fixed across cells).
    rng = np.random.default_rng(split_rng_seed)
    reliability = {
        p: _split_half_reliability(shifts[p]["delta_v_per_q"], n_random=n_random_splits, rng=rng)
        if "delta_v_per_q" in shifts[p]
        else None
        for p in personas
    }

    entry = {
        "arm": cell.arm,
        "variant": cell.variant,
        "seed": cell.seed,
        "persona_order": personas,
        "M_shape": list(svd_w["M_shape"]),
        "frobenius_norm": float(np.sqrt(np.sum(np.asarray(svd_w["s"], dtype=np.float64) ** 2))),
        "singular_values": [float(s) for s in svd_w["s"]],
        "weighted": {
            "s_top1_frac": float(svd_w["s_top1_frac"]),
            "mean_cos_to_U1": float(np.mean(svd_w["cos_to_U1"])),
            "median_cos_to_U1": float(np.median(svd_w["cos_to_U1"])),
            "cos_to_U1": cos_w,
            "n_aligned_abs_cos_ge_0p5": int(np.sum(abs_cos_w >= ALIGNED_COS_THRESHOLD)),
            "aligned_personas": sorted(
                p for p in personas if abs(cos_w[p]) >= ALIGNED_COS_THRESHOLD
            ),
            "U1": [float(x) for x in svd_w["U1"]],
        },
        "unitnorm": {
            "s_top1_frac": float(svd_u["s_top1_frac"]),
            "cos_to_U1": cos_u,
            "n_aligned_abs_cos_ge_0p5": int(np.sum(abs_cos_u >= ALIGNED_COS_THRESHOLD)),
            "aligned_personas": sorted(
                p for p in personas if abs(cos_u[p]) >= ALIGNED_COS_THRESHOLD
            ),
            "abs_cos_U1_unitnorm_vs_weighted": float(rotation),
        },
        "nulls": {
            "sign_flip_p95": float(sign_null["p95"]),
            "sign_flip_p99": float(sign_null["p99"]),
            "row_shuffle_p95": float(row_null["p95"]),
            "row_shuffle_p99": float(row_null["p99"]),
            "n_reps": N_NULL_REPS,
            "passes_sign_flip_p95": bool(svd_w["s_top1_frac"] > sign_null["p95"]),
            "passes_row_shuffle_p95": bool(svd_w["s_top1_frac"] > row_null["p95"]),
        },
        "norm_vs_alignment": {
            "weighted": {"spearman_rho": rho_w, "p_one_sided_positive": p_w, "n_perm": N_PERM},
            "unitnorm_abs": {
                "spearman_rho": rho_u,
                "p_one_sided_positive": p_u,
                "n_perm": N_PERM,
            },
            "norms": {p: float(norms[i]) for i, p in enumerate(personas)},
        },
        "mean_resp": {
            "s_top1_frac": float(svd_mr["s_top1_frac"]),
            "mean_cos_to_U1": float(np.mean(svd_mr["cos_to_U1"])),
            "passes_sign_flip_p95": bool(svd_mr["s_top1_frac"] > sign_null_mr["p95"]),
            "sign_flip_p95": float(sign_null_mr["p95"]),
        },
        "split_half_reliability": reliability,
    }
    extras = {"U1_w": np.asarray(svd_w["U1"], dtype=np.float64), "personas": personas}
    logger.info(
        "[cell %s] top_w=%.4f top_u=%.4f rot=%.4f sign_p95=%.4f mean_cos=%.4f",
        cell.name,
        entry["weighted"]["s_top1_frac"],
        entry["unitnorm"]["s_top1_frac"],
        rotation,
        entry["nulls"]["sign_flip_p95"],
        entry["weighted"]["mean_cos_to_U1"],
    )
    return entry, extras


def _check_i551_consistency(per_cell: dict[str, dict], norm_align: dict) -> dict[str, float]:
    """Recomputed weighted cos-to-U1 vs the persisted #551 norm_alignment.json.

    Applies to the #551 same-text cells present in this run AND in the
    persisted JSON; max |delta| > 0.001 raises (different matrix convention).
    """
    out: dict[str, float] = {}
    stored_cells = norm_align.get("per_cell", {})
    for _name, entry in per_cell.items():
        if entry["arm"] == "posonly" or entry["variant"] != "same":
            continue
        stored_name = f"{entry['variant']}_{entry['arm']}_seed{entry['seed']}"
        stored = stored_cells.get(stored_name)
        if stored is None:
            logger.warning("[consistency] %s not in norm_alignment.json — skipped", stored_name)
            continue
        deltas = [
            abs(entry["weighted"]["cos_to_U1"][p] - float(stored["cos_to_U1"][p]))
            for p in entry["persona_order"]
        ]
        max_d = max(deltas)
        if max_d > WEIGHTED_CONSISTENCY_TOL:
            raise ValueError(
                f"{stored_name}: recomputed weighted cos-to-U1 disagrees with the persisted "
                f"#551 norm_alignment.json (max |delta|={max_d:.2e} > "
                f"{WEIGHTED_CONSISTENCY_TOL}); refusing to compare reads built from "
                f"different conventions."
            )
        out[stored_name] = float(max_d)
        logger.info("[consistency %s] max|delta|=%.2e OK", stored_name, max_d)
    return out


def _random_pair_floor(h: int, n_pairs: int = 1000, seed: int = 0) -> dict:
    """Empirical |cos| floor for independent random directions in R^h."""
    rng = np.random.default_rng(seed)
    vals = [abs(cosine(rng.standard_normal(h), rng.standard_normal(h))) for _ in range(n_pairs)]
    return {
        "mean_abs_cos": float(np.mean(vals)),
        "p95_abs_cos": float(np.percentile(vals, 95)),
        "h": int(h),
        "n_pairs": n_pairs,
    }


def _cross_arm_u1(extras: dict[str, dict]) -> dict:
    """|cos(U1_posonly, U1_{em,marker})| per (variant, seed) + random floor."""
    out: dict[str, dict] = {}
    h = None
    for name, ex in extras.items():
        if not name.startswith("posonly:"):
            continue
        suffix = name.split(":", 1)[1]  # "{variant}_seed{S}"
        h = ex["U1_w"].shape[0]
        row: dict[str, float] = {}
        for other_arm in I551_ARMS:
            other = extras.get(f"{other_arm}:{suffix}")
            if other is None:
                continue
            row[f"abs_cos_U1_vs_{other_arm}"] = abs(cosine(ex["U1_w"], other["U1_w"]))
        if row:
            out[suffix] = row
    payload: dict = {"per_cell": out}
    if h is not None:
        payload["random_pair_floor"] = _random_pair_floor(h)
    return payload


def _headline(per_cell: dict[str, dict]) -> dict:
    """Mechanical plan §3 clause evaluation over the same-text cells."""
    new_same = sorted(
        (v for v in per_cell.values() if v["arm"] == "posonly" and v["variant"] == "same"),
        key=lambda v: v["seed"],
    )
    bands = {}
    for arm in I551_ARMS:
        vals = [
            v["weighted"]["s_top1_frac"]
            for v in per_cell.values()
            if v["arm"] == arm and v["variant"] == "same"
        ]
        if vals:
            bands[arm] = {"min": float(min(vals)), "max": float(max(vals)), "n": len(vals)}
    if len(new_same) < 3:
        return {
            "evaluated": False,
            "reason": f"only {len(new_same)}/3 posonly same-text cells present (smoke subset?)",
            "i551_same_text_weighted_bands": bands,
        }

    tops = [v["weighted"]["s_top1_frac"] for v in new_same]
    mean_top = float(np.mean(tops))
    per_seed = {
        f"seed{v['seed']}": {
            "s_top1_frac_weighted": v["weighted"]["s_top1_frac"],
            "ge_0p46": v["weighted"]["s_top1_frac"] >= PER_SEED_TOP_SHARE,
            "passes_sign_flip_p95": v["nulls"]["passes_sign_flip_p95"],
            "s_top1_frac_unitnorm": v["unitnorm"]["s_top1_frac"],
            "unitnorm_ge_0p40": v["unitnorm"]["s_top1_frac"] >= UNITNORM_TOP_SHARE_MIN,
            "rotation": v["unitnorm"]["abs_cos_U1_unitnorm_vs_weighted"],
            "rotation_ge_0p95": (v["unitnorm"]["abs_cos_U1_unitnorm_vs_weighted"] >= ROTATION_MIN),
            "n_aligned_unitnorm": v["unitnorm"]["n_aligned_abs_cos_ge_0p5"],
            "membership_ge_12of14": (v["unitnorm"]["n_aligned_abs_cos_ge_0p5"] >= MEMBERSHIP_MIN),
        }
        for v in new_same
    }
    n_ge = sum(1 for v in per_seed.values() if v["ge_0p46"])
    all_sign = all(v["passes_sign_flip_p95"] for v in per_seed.values())
    n_unit_ok = sum(1 for v in per_seed.values() if v["unitnorm_ge_0p40"])
    n_rot_ok = sum(1 for v in per_seed.values() if v["rotation_ge_0p95"])
    n_member_ok = sum(1 for v in per_seed.values() if v["membership_ge_12of14"])
    unit_tops = [v["s_top1_frac_unitnorm"] for v in per_seed.values()]

    confirm = (
        n_ge >= 2
        and mean_top >= MEAN_TOP_SHARE_CONFIRM
        and all_sign
        and n_unit_ok >= 2
        and n_rot_ok >= 2
        and n_member_ok >= 2
    )
    marker_like_unitnorm = (n_rot_ok <= 1) or (
        float(np.mean(unit_tops)) <= UNITNORM_TOP_SHARE_MARKER_LIKE
    )
    falsify = (mean_top <= MEAN_TOP_SHARE_FALSIFY) and marker_like_unitnorm
    verdict = "confirm" if confirm else ("falsify" if falsify else "indeterminate")
    return {
        "evaluated": True,
        "mechanical_verdict": verdict,
        "note": (
            "Mechanical clause evaluation only — the analyzer owns the verdict. "
            "Non-confirm/non-falsify routes to indeterminate (reconciler binding "
            "#15.1); threshold calls within the realized extraction-smoke drift "
            "are measurement-limited (#15.3); a mean in [0.50, 0.5235] is "
            "'EM-like, at the lower margin of the EM band' (#15.4)."
        ),
        "mean_s_top1_frac_weighted": mean_top,
        "mean_s_top1_frac_unitnorm": float(np.mean(unit_tops)),
        "n_seeds_ge_0p46": n_ge,
        "all_pass_sign_flip_p95": all_sign,
        "n_seeds_unitnorm_ge_0p40": n_unit_ok,
        "n_seeds_rotation_ge_0p95": n_rot_ok,
        "n_seeds_membership_ge_12of14": n_member_ok,
        "per_seed": per_seed,
        "i551_same_text_weighted_bands": bands,
        "thresholds": {
            "per_seed_top_share": PER_SEED_TOP_SHARE,
            "mean_top_share_confirm": MEAN_TOP_SHARE_CONFIRM,
            "mean_top_share_falsify": MEAN_TOP_SHARE_FALSIFY,
            "unitnorm_top_share_min": UNITNORM_TOP_SHARE_MIN,
            "unitnorm_top_share_marker_like": UNITNORM_TOP_SHARE_MARKER_LIKE,
            "rotation_min": ROTATION_MIN,
            "membership_min": MEMBERSHIP_MIN,
        },
    }


def make_figures(per_cell: dict[str, dict], figures_dir: Path) -> None:
    """Hero three-arm band plot + profile/norm scatters (guarded on coverage)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    colors = paper_palette(4)
    arm_color = {"em": colors[0], "marker": colors[1], "posonly": colors[2]}
    arm_label = {
        "marker": "contrastive marker (#551)",
        "posonly": "positive-only marker (#561)",
        "em": "EM insecure-code (#551)",
    }
    arm_order = ["marker", "posonly", "em"]
    same = [v for v in per_cell.values() if v["variant"] == "same"]
    if not same:
        logger.warning("[figures] no same-text cells — skipping all figures")
        return

    # ── Hero: three-arm top-share bands, weighted vs unit-norm panels ──
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, read in zip(axes, ("weighted", "unitnorm"), strict=True):
        for i, arm in enumerate(arm_order):
            cells = sorted((v for v in same if v["arm"] == arm), key=lambda v: v["seed"])
            if not cells:
                continue
            xs = [i + (j - 1) * 0.12 for j in range(len(cells))]
            ys = [v[read]["s_top1_frac"] for v in cells]
            ax.scatter(xs, ys, s=46, color=arm_color[arm], zorder=3)
            for x, v in zip(xs, cells, strict=True):
                ax.annotate(
                    str(v["seed"]),
                    (x, v[read]["s_top1_frac"]),
                    fontsize=6.5,
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                )
                if read == "weighted":
                    ax.hlines(
                        v["nulls"]["sign_flip_p95"],
                        x - 0.07,
                        x + 0.07,
                        color="black",
                        lw=1.1,
                    )
        ax.set_xticks(range(len(arm_order)))
        ax.set_xticklabels([arm_label[a].replace(" (", "\n(") for a in arm_order], fontsize=7.5)
        ax.set_title(
            "norm-weighted SVD" if read == "weighted" else "unit-norm columns",
            fontsize=10,
        )
    axes[0].set_ylabel("top-direction share of spectrum")
    fig.suptitle(
        "Same-text shift-spectrum concentration by arm (ticks = per-cell sign-flip null p95)",
        fontsize=10,
    )
    fig.tight_layout()
    savefig_paper(fig, "three_arm_top_share_bands", dir=figures_dir)
    plt.close(fig)

    # ── Per-persona |cos| profile: posonly vs contrastive, per seed ─────
    pairs = []
    for v in same:
        if v["arm"] != "posonly":
            continue
        ref = next(
            (w for w in same if w["arm"] == "marker" and w["seed"] == v["seed"]),
            None,
        )
        if ref is not None:
            pairs.append((v, ref))
    if pairs:
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        ax.plot([0, 1], [0, 1], color="lightgray", lw=1.0, zorder=0)
        markers = {42: "o", 137: "^", 256: "s"}
        for v, ref in pairs:
            personas = v["persona_order"]
            ax.scatter(
                [abs(ref["weighted"]["cos_to_U1"][p]) for p in personas],
                [abs(v["weighted"]["cos_to_U1"][p]) for p in personas],
                s=26,
                color=arm_color["posonly"],
                marker=markers.get(v["seed"], "o"),
                alpha=0.75,
                label=f"seed {v['seed']}",
            )
        ax.set_xlabel("|cos to top direction| — contrastive marker (#551)")
        ax.set_ylabel("|cos to top direction| — positive-only marker (#561)")
        ax.set_title("Per-persona alignment: positive-only vs contrastive (same-text)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        savefig_paper(fig, "posonly_vs_contrastive_profiles", dir=figures_dir)
        plt.close(fig)

    # ── New arm: shift norm vs |cos| (weighted) ─────────────────────────
    new_same = [v for v in same if v["arm"] == "posonly"]
    if new_same:
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        for i, v in enumerate(sorted(new_same, key=lambda v: v["seed"])):
            personas = v["persona_order"]
            norms = v["norm_vs_alignment"]["norms"]
            ax.scatter(
                [norms[p] for p in personas],
                [abs(v["weighted"]["cos_to_U1"][p]) for p in personas],
                s=20,
                color=paper_palette(4)[i],
                alpha=0.8,
                label=f"seed {v['seed']}",
            )
        ax.set_xlabel("per-persona shift norm")
        ax.set_ylabel("|cosine to top direction|")
        ax.set_title("Positive-only marker arm: shift size vs alignment (same-text)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        savefig_paper(fig, "posonly_norm_vs_alignment", dir=figures_dir)
        plt.close(fig)

    logger.info("[figures] written to %s", figures_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="#561 comparison: positive-only marker vs persisted #551 tensors (CPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--new-shifts-dir", default="eval_results/issue_561/shifts")
    parser.add_argument(
        "--new-cells",
        nargs="+",
        default=["marker_seed42", "marker_seed137", "marker_seed256"],
        help="New-arm cell specs (file arm is 'marker'; analyzed as arm 'posonly').",
    )
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--issue551-repo", default="superkaiba1/explore-persona-space-data-private")
    parser.add_argument(
        "--issue551-prefix", default="issue551_shift_reextract/analysis_tensors/shifts"
    )
    parser.add_argument("--issue551-revision", default="08419ee885e962cb29c841d34041db419dbbc72c")
    parser.add_argument(
        "--issue551-local-dir",
        default=None,
        help="Read the #551 .pt files from this dir instead of downloading (smoke).",
    )
    parser.add_argument(
        "--issue551-cells",
        nargs="+",
        default=None,
        help=(
            "Subset of #551 cell stems (e.g. 'same_marker_seed42'). Default = all "
            "18 (2 arms x 3 seeds x the requested --variants)."
        ),
    )
    parser.add_argument(
        "--norm-align-json",
        default="eval_results/issue_551/controls/norm_alignment.json",
        help="Persisted #551 weighted read for the 0.001 consistency cross-check.",
    )
    parser.add_argument("--out", default="eval_results/issue_561/comparison")
    parser.add_argument("--figures-dir", default="figures/issue_561")
    parser.add_argument("--n-random-splits", type=int, default=50)
    parser.add_argument("--split-rng-seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    # `uv run python` does NOT auto-load .env; the private repo needs HF_TOKEN.
    load_dotenv()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)

    # Build the cell lists.
    new_cells: list[CompareCell] = []
    for spec in args.new_cells:
        arm, _, rest = spec.partition("_seed")
        assert arm == "marker", f"--new-cells spec {spec!r} must look like 'marker_seed42'"
        for variant in args.variants:
            new_cells.append(CompareCell(arm="posonly", variant=variant, seed=int(rest)))

    if args.issue551_cells is not None:
        i551_cells = []
        for stem in args.issue551_cells:
            variant, _, rest = stem.partition("_")
            arm, _, seed_s = rest.partition("_seed")
            assert variant in VARIANTS and arm in I551_ARMS, f"bad #551 stem {stem!r}"
            i551_cells.append(CompareCell(arm=arm, variant=variant, seed=int(seed_s)))
    else:
        i551_cells = [
            CompareCell(arm=a, variant=v, seed=s)
            for v in args.variants
            for a in I551_ARMS
            for s in SEEDS
        ]

    # Resolve tensor sources.
    new_dir = Path(args.new_shifts_dir)
    if args.issue551_local_dir:
        i551_dir = Path(args.issue551_local_dir)
        logger.info("[load] local #551 shifts dir %s", i551_dir)
    else:
        i551_dir = _download_i551(
            args.issue551_repo,
            args.issue551_prefix,
            args.issue551_revision,
            [c.file_stem for c in i551_cells],
            out_dir / "i551_shifts_downloaded",
        )

    meta = {
        "issue": 561,
        "analysis": "posonly_vs_persisted_551_comparison",
        "git_commit": _git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "new_shifts_dir": str(new_dir),
        "i551_source": (
            str(args.issue551_local_dir)
            if args.issue551_local_dir
            else f"hf://{args.issue551_repo}/{args.issue551_prefix}@{args.issue551_revision}"
        ),
        "n_null_reps": N_NULL_REPS,
        "n_perm": N_PERM,
        "n_random_splits": args.n_random_splits,
        "split_rng_seed": args.split_rng_seed,
        "aligned_cos_threshold": ALIGNED_COS_THRESHOLD,
        "weighted_consistency_tol": WEIGHTED_CONSISTENCY_TOL,
        "source_persona": SOURCE_PERSONA,
        "top_share_definition": "s_1 / sum(s) (matches svd_summary + the persisted JSONs)",
    }

    per_cell: dict[str, dict] = {}
    extras: dict[str, dict] = {}
    for cell in [*new_cells, *i551_cells]:
        src = new_dir if cell.arm == "posonly" else i551_dir
        shifts = _load_shifts(src, cell.file_stem)
        entry, ex = analyze_cell(
            cell,
            shifts,
            n_random_splits=args.n_random_splits,
            split_rng_seed=args.split_rng_seed,
        )
        per_cell[cell.name] = entry
        extras[cell.name] = ex

    # Consistency cross-check vs the persisted #551 controls (tol 0.001).
    norm_align_path = Path(args.norm_align_json)
    consistency: dict[str, float] = {}
    if norm_align_path.exists():
        with norm_align_path.open() as f:
            norm_align = json.load(f)
        consistency = _check_i551_consistency(per_cell, norm_align)
    else:
        logger.warning(
            "[consistency] %s not found — #551-side consistency cross-check SKIPPED "
            "(fine for smoke; production must run it)",
            norm_align_path,
        )

    # Checkpoint per phase: per-cell JSON FIRST, then summary, figures last.
    with (out_dir / "comparison_per_cell.json").open("w") as f:
        json.dump({"meta": meta, "per_cell": per_cell}, f, indent=2)
    logger.info("[wrote] %s", out_dir / "comparison_per_cell.json")

    summary = {
        "meta": meta,
        "headline": _headline(per_cell),
        "cross_arm_U1": _cross_arm_u1(extras),
        "i551_weighted_consistency_max_abs_delta": consistency,
        "per_cell_top_share": {
            name: {
                "arm": v["arm"],
                "variant": v["variant"],
                "seed": v["seed"],
                "s_top1_frac_weighted": v["weighted"]["s_top1_frac"],
                "s_top1_frac_unitnorm": v["unitnorm"]["s_top1_frac"],
                "rotation": v["unitnorm"]["abs_cos_U1_unitnorm_vs_weighted"],
                "mean_cos_to_U1": v["weighted"]["mean_cos_to_U1"],
                "frobenius_norm": v["frobenius_norm"],
                "passes_sign_flip_p95": v["nulls"]["passes_sign_flip_p95"],
            }
            for name, v in per_cell.items()
        },
    }
    with (out_dir / "comparison_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("[wrote] %s", out_dir / "comparison_summary.json")

    make_figures(per_cell, figures_dir)

    hl = summary["headline"]
    logger.info(
        "[phase=done] cells=%d mechanical_verdict=%s",
        len(per_cell),
        hl.get("mechanical_verdict", "n/a (not evaluated)"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
