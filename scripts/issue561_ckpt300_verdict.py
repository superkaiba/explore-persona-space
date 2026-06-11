#!/usr/bin/env python3
"""#561 follow-up `exposure-matched-ckpt300` — VM-side verdict of record (CPU).

Joins the NEW checkpoint-300 per-cell comparison JSON (produced by running
``scripts/issue561_compare.py`` UNCHANGED over the ckpt-300 shifts dir) with
the parent's committed step-600 per-cell JSON
(``eval_results/issue_561/comparison/comparison_per_cell.json`` @ git
``5f3ec9569``) and evaluates the plan-v2 §3 pre-registered clauses
MECHANICALLY over the same-text ckpt-300 cells:

- **collapse** (exposure-consistent): mean over ALL 3 seeds + >=2/3 seeds
  at/below the contrastive same-text band MAX on BOTH reads (weighted AND
  unit-norm) — reachable only when all 3 cells clear their sign-flip nulls;
- **persist** (real negatives effect): mean over ALL 3 seeds + >=2/3 seeds
  at/above the positive-only step-600 band MIN on BOTH reads — same
  all-cells-gated precondition;
- anything else -> **indeterminate** (parent reconciler binding #15.1;
  boundary calls inside the narrow gap zones are measurement-limited #15.3),
  INCLUDING any same-text cell failing its sign-flip null p95 (plan v2 §3
  line 56 lists a failing-null cell as an indeterminate trigger in its own
  right — concern `verdict-gated-mean`, round 2).

CRITICAL (plan v2 §14 fact-check amendment): the collapse/persist bands are
loaded from the UNROUNDED per-cell JSON values at runtime (e.g. posonly-600
same-text weighted min 0.364949, contrastive weighted band
[0.311322, 0.347698] / unit-norm [0.242471, 0.283958]) — never hardcoded
rounded prose values, to avoid hairline misclassification of boundary cells.

Verdict gating precondition (inherited, §3 lines 56/58): each same-text
ckpt-300 cell must clear its own sign-flip null p95 (weighted read). ANY
failing cell routes the verdict to indeterminate (line 56) with the named
exclusion; the failed cell's band placement never counts (line 58), so the
clause counts reported in that branch are descriptive only.

COLLAPSE narration (plan v2 Phase-3 critique amendment): the elevation
"develops in the second half of training (steps 300->600)"; within a fixed
arm, per-positive exposure, optimizer steps, and cosine-schedule position
are perfectly collinear, so exposure-matching is CONSISTENT WITH, not
demonstrated by, the collapse — never "exposure caused it".

Secondary (descriptive, no gates): |cos(U1_ckpt300, U1_posonly600)| per
(variant, seed) from the persisted U1 vectors (+ vs contrastive/EM and an
empirical random-pair floor); aligned-set membership delta 300 vs 600;
per-flavor band triangulation; step-300 bystander emission pattern from the
parent's committed periodic-eval JSONs.

Zero GPU, zero torch, zero tensor downloads — pure linear algebra over the
persisted JSONs. Run from the repo root (VM, CPU)::

    uv run python scripts/issue561_ckpt300_verdict.py \\
        --new-per-cell \
            eval_results/issue_561/exposure-matched-ckpt300/comparison/comparison_per_cell.json \\
        --parent-per-cell eval_results/issue_561/comparison/comparison_per_cell.json \\
        --out eval_results/issue_561/exposure-matched-ckpt300/comparison \\
        --figures-dir figures/issue_561/exposure-matched-ckpt300
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

VARIANTS = ("same", "base", "on_policy")
# Reader-facing flavor names for every rendered figure label (clean-result
# Lens 2/3: no project-internal slugs in tick labels / panel titles).
FLAVOR_LABELS = {
    "same": "trained-model text",
    "base": "base-model text",
    "on_policy": "own response",
}
SEEDS = (42, 137, 256)
SOURCE_PERSONA = "medical_doctor"  # matches issue551_controls.SOURCE_PERSONA
# Convention-drift guard: the #551 marker/em cells appear in BOTH JSONs
# (re-analyzed by the same deterministic analyze_cell over the same pinned
# tensors with the same null seeds), so their headline scalars must agree.
CROSS_CHECK_TOL = 1e-6
MIN_SEEDS_PER_CLAUSE = 2  # ">= 2/3 seeds" from plan v2 §3
# Plan v2 §4.0.3 pins the parent step-600 per-cell JSON (the band source) to
# this commit; the working-tree copy must parse-match the pinned blob so a
# later re-run cannot silently shift the collapse/persist bands.
PARENT_PIN_COMMIT = "5f3ec95695231fd530e69209e238c2840172d3b3"

NARRATION = {
    "collapse": (
        "The same-text elevation develops in the second half of training (steps 300->600): "
        "it is absent at the contrastive arm's matched per-positive exposure (24 epochs per "
        "positive), so the parent's named per-positive-exposure confound is closed. Within a "
        "fixed arm, per-positive exposure, raw optimizer steps, and cosine-schedule position "
        "are perfectly collinear, so exposure-matching is CONSISTENT WITH, not demonstrated "
        "by, the collapse. The parent headline is restated exposure-clean."
    ),
    "persist": (
        "Negatives exert a small genuine dispersive force (removing them concentrates the "
        "shift spectrum), far short of the EM gap — and this read matches row-visit exposure "
        "but NOT integrated LR (ckpt-300's 24 epochs accumulate at ~1.6x the integrated LR of "
        "the contrastive arm's full cosine schedule; plan v2 §12.9 scope caveat). The "
        "indeterminate hybrid signature gains a non-exposure explanation."
    ),
    "indeterminate": (
        "No clean collapse or persist: narrate descriptively against the bands per the "
        "parent's reconciler binding #15.1. The gap zones between the bands are narrow, so a "
        "boundary outcome is measurement-limited (#15.3), not a failure."
    ),
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors (asserts matching shape)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape and a.ndim == 1, (a.shape, b.shape)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _load_per_cell(path: Path) -> dict[str, dict]:
    with path.open() as f:
        payload = json.load(f)
    per_cell = payload.get("per_cell")
    if not per_cell:
        raise ValueError(f"{path}: no 'per_cell' key — not a comparison_per_cell.json?")
    required = {"arm", "variant", "seed", "persona_order", "weighted", "unitnorm", "nulls"}
    for name, v in per_cell.items():
        missing = required - set(v)
        if missing:
            raise ValueError(f"{path}: cell {name} missing keys {sorted(missing)}")
    return per_cell


def _cells(per_cell: dict[str, dict], arm: str, variant: str | None = None) -> list[dict]:
    out = [
        v
        for v in per_cell.values()
        if v["arm"] == arm and (variant is None or v["variant"] == variant)
    ]
    return sorted(out, key=lambda v: (v["variant"], v["seed"]))


def _band(cells: list[dict], read: str) -> dict:
    """Unrounded min/max band of `s_top1_frac` under `read` over `cells`."""
    assert cells, "empty cell list for band computation"
    vals = {f"seed{c['seed']}": float(c[read]["s_top1_frac"]) for c in cells}
    return {
        "min": min(vals.values()),
        "max": max(vals.values()),
        "n": len(vals),
        "values": vals,
    }


def _assert_persona_order(joined: list[dict]) -> list[str]:
    """All joined cells must share one persona_order — fail loud, never reorder."""
    orders = {tuple(v["persona_order"]) for v in joined}
    if len(orders) != 1:
        raise ValueError(
            f"persona_order mismatch across joined cells: {sorted(orders)} — refusing to "
            f"merge reads built over different panels."
        )
    return list(next(iter(orders)))


def _cross_check_shared_cells(
    new_pc: dict[str, dict], parent_pc: dict[str, dict]
) -> dict[str, float]:
    """#551 marker/em cells present in both JSONs must agree to CROSS_CHECK_TOL."""
    out: dict[str, float] = {}
    for name, v in new_pc.items():
        if v["arm"] not in ("marker", "em"):
            continue
        pv = parent_pc.get(name)
        if pv is None:
            continue
        d = max(
            abs(v["weighted"]["s_top1_frac"] - pv["weighted"]["s_top1_frac"]),
            abs(v["unitnorm"]["s_top1_frac"] - pv["unitnorm"]["s_top1_frac"]),
        )
        if d > CROSS_CHECK_TOL:
            raise ValueError(
                f"{name}: re-analyzed #551 cell disagrees with the parent's committed value "
                f"(max |delta|={d:.2e} > {CROSS_CHECK_TOL}) — analysis-convention drift; "
                f"refusing to place ckpt-300 cells against bands from a different convention."
            )
        out[name] = float(d)
    return out


def _assert_parent_pinned(parent_path: Path, pin: str) -> None:
    """Working-tree parent per-cell JSON must parse-match the pinned git blob.

    Plan v2 §4.0.3 pins the step-600 bands to git ``5f3ec9569``; a drifted
    working-tree copy would silently shift the collapse/persist bands on a
    later re-run. Pass ``--parent-pin ''`` to skip deliberately (ad-hoc
    fixture parent outside the repo). Raises on any mismatch / unresolvable
    pin — never degrades silently.
    """
    if not pin:
        logger.warning("[pin] parent per-cell pin check SKIPPED (--parent-pin '')")
        return
    repo_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    )
    try:
        rel = parent_path.resolve().relative_to(repo_root)
    except ValueError as e:
        raise ValueError(
            f"--parent-per-cell {parent_path} lies outside the git repo {repo_root}, so it "
            f"cannot be verified against pin {pin}. Pass --parent-pin '' to skip deliberately."
        ) from e
    blob = subprocess.check_output(["git", "show", f"{pin}:{rel.as_posix()}"], cwd=repo_root)
    if json.loads(blob) != json.loads(parent_path.read_text()):
        raise ValueError(
            f"{parent_path} differs from the pinned blob {pin}:{rel.as_posix()} — the "
            f"collapse/persist bands would silently shift; refusing to evaluate. Restore "
            f"the committed copy or re-pin deliberately via --parent-pin."
        )
    logger.info("[pin] parent per-cell JSON matches pinned blob %s:%s", pin[:12], rel.as_posix())


def _evaluate_clauses(
    new_same: list[dict],
    contrastive_w: dict,
    contrastive_u: dict,
    posonly600_w: dict,
    posonly600_u: dict,
) -> dict:
    """Mechanical plan-v2 §3 clause evaluation over the same-text ckpt-300 cells.

    Registered-text conformance (concern ``verdict-gated-mean``, round 2): §3
    line 56 lists "a same-text cell failing its sign-flip p95" as an
    INDETERMINATE trigger, so ANY gate-failing cell routes the verdict to
    indeterminate with the named exclusion (line 58: its band placement never
    counts; the counts recorded in that branch are descriptive). The
    determinate collapse/persist path therefore exists ONLY when all 3 cells
    pass their nulls, where the line-52/54 mean clauses are evaluated on the
    ALL-3-seed means (identical to the gated means in that branch by
    construction).
    """
    if len(new_same) < 3:
        return {
            "evaluated": False,
            "reason": f"only {len(new_same)}/3 ckpt-300 same-text cells present (smoke subset?)",
        }

    gated = [v for v in new_same if v["nulls"]["passes_sign_flip_p95"]]
    excluded = sorted(
        f"seed{v['seed']}" for v in new_same if not v["nulls"]["passes_sign_flip_p95"]
    )
    per_seed = {
        f"seed{v['seed']}": {
            "s_top1_frac_weighted": float(v["weighted"]["s_top1_frac"]),
            "s_top1_frac_unitnorm": float(v["unitnorm"]["s_top1_frac"]),
            "passes_sign_flip_p95": bool(v["nulls"]["passes_sign_flip_p95"]),
            "sign_flip_p95": float(v["nulls"]["sign_flip_p95"]),
            "le_contrastive_max_weighted": (v["weighted"]["s_top1_frac"] <= contrastive_w["max"]),
            "le_contrastive_max_unitnorm": (v["unitnorm"]["s_top1_frac"] <= contrastive_u["max"]),
            "ge_posonly600_min_weighted": (v["weighted"]["s_top1_frac"] >= posonly600_w["min"]),
            "ge_posonly600_min_unitnorm": (v["unitnorm"]["s_top1_frac"] >= posonly600_u["min"]),
        }
        for v in new_same
    }

    w_gated = [float(v["weighted"]["s_top1_frac"]) for v in gated]
    u_gated = [float(v["unitnorm"]["s_top1_frac"]) for v in gated]
    means = {
        "weighted_all": float(np.mean([v["weighted"]["s_top1_frac"] for v in new_same])),
        "unitnorm_all": float(np.mean([v["unitnorm"]["s_top1_frac"] for v in new_same])),
        "weighted_gated": float(np.mean(w_gated)) if w_gated else None,
        "unitnorm_gated": float(np.mean(u_gated)) if u_gated else None,
    }

    # Band-placement counts over gate-passing cells only (§3 line 58: a cell
    # below its chance floor never counts toward a clause).
    n_collapse_w = sum(1 for x in w_gated if x <= contrastive_w["max"])
    n_collapse_u = sum(1 for x in u_gated if x <= contrastive_u["max"])
    n_persist_w = sum(1 for x in w_gated if x >= posonly600_w["min"])
    n_persist_u = sum(1 for x in u_gated if x >= posonly600_u["min"])
    clause_counts = {
        "n_le_contrastive_max_weighted": n_collapse_w,
        "n_le_contrastive_max_unitnorm": n_collapse_u,
        "n_ge_posonly600_min_weighted": n_persist_w,
        "n_ge_posonly600_min_unitnorm": n_persist_u,
        "min_seeds_per_clause": MIN_SEEDS_PER_CLAUSE,
        "n_gated": len(gated),
    }
    gap_zones = {
        "weighted": [contrastive_w["max"], posonly600_w["min"]],
        "unitnorm": [contrastive_u["max"], posonly600_u["min"]],
    }

    if excluded:
        # §3 line 56 (literal): ANY same-text cell failing its sign-flip null
        # p95 is an indeterminate trigger in its own right — no determinate
        # verdict is reachable with a below-chance cell, regardless of where
        # the surviving cells sit. The counts above stay descriptive here.
        return {
            "evaluated": True,
            "verdict": "indeterminate",
            "reason": (
                f"{len(excluded)}/3 same-text cells fail their sign-flip null p95 "
                f"(excluded: {excluded}) — plan v2 §3 registers a failing-null same-text "
                f"cell as an indeterminate trigger; the failed cells' band placements do "
                f"not count and the clause counts are descriptive only."
            ),
            "gating_excluded": excluded,
            "n_gated": len(gated),
            "means": means,
            "clause_counts": clause_counts,
            "gap_zones": gap_zones,
            "per_seed": per_seed,
            "narration": NARRATION["indeterminate"],
        }

    # All 3 cells pass their nulls -> determinate path. The §3 line-52/54 mean
    # clauses read "mean over 3 seeds", so they are evaluated on the ALL-cell
    # means (== gated means here, zero exclusions) with counts over all 3 cells.
    collapse = (
        means["weighted_all"] <= contrastive_w["max"]
        and means["unitnorm_all"] <= contrastive_u["max"]
        and n_collapse_w >= MIN_SEEDS_PER_CLAUSE
        and n_collapse_u >= MIN_SEEDS_PER_CLAUSE
    )
    persist = (
        means["weighted_all"] >= posonly600_w["min"]
        and means["unitnorm_all"] >= posonly600_u["min"]
        and n_persist_w >= MIN_SEEDS_PER_CLAUSE
        and n_persist_u >= MIN_SEEDS_PER_CLAUSE
    )
    # The bands are disjoint (contrastive max < posonly-600 min on both reads,
    # asserted by the caller), so collapse and persist cannot both fire.
    assert not (collapse and persist), "collapse and persist both fired — bands not disjoint?"
    verdict = "collapse" if collapse else ("persist" if persist else "indeterminate")

    return {
        "evaluated": True,
        "verdict": verdict,
        "narration": NARRATION[verdict],
        "gating_excluded": excluded,
        "n_gated": len(gated),
        "means": means,
        "clause_counts": clause_counts,
        "gap_zones": gap_zones,
        "per_seed": per_seed,
        "note": (
            "Mechanical clause evaluation only — the analyzer owns the verdict prose. "
            "Determinate verdicts require all 3 same-text cells to pass their sign-flip "
            "nulls (§3 line 56); mean clauses are over ALL 3 seeds (lines 52/54) and "
            "both all-cell and gated means are recorded. COLLAPSE narration per the "
            "plan-v2 Phase-3 amendment (collinearity caveat); PERSIST carries the §12.9 "
            "integrated-LR caveat in the same sentence."
        ),
    }


def _u1(v: dict) -> np.ndarray:
    u1 = v["weighted"].get("U1")
    if not u1:
        raise ValueError(
            f"cell {v['arm']}:{v['variant']}_seed{v['seed']} carries no weighted.U1 vector"
        )
    return np.asarray(u1, dtype=np.float64)


def _random_pair_floor(h: int, n_pairs: int = 1000, seed: int = 0) -> dict:
    """Empirical |cos| floor for independent random directions in R^h."""
    rng = np.random.default_rng(seed)
    vals = [abs(_cosine(rng.standard_normal(h), rng.standard_normal(h))) for _ in range(n_pairs)]
    return {
        "mean_abs_cos": float(np.mean(vals)),
        "p95_abs_cos": float(np.percentile(vals, 95)),
        "h": int(h),
        "n_pairs": n_pairs,
    }


def _secondary_reads(new_pc: dict[str, dict], parent_pc: dict[str, dict]) -> dict:
    """Descriptive reads (no gates): U1 identity, membership delta, flavors."""

    def parent_cell(arm: str, variant: str, seed: int) -> dict | None:
        return parent_pc.get(f"{arm}:{variant}_seed{seed}")

    u1_identity: dict[str, dict] = {}
    membership: dict[str, dict] = {}
    h = None
    for v in _cells(new_pc, "posonly"):
        key = f"{v['variant']}_seed{v['seed']}"
        p600 = parent_cell("posonly", v["variant"], v["seed"])
        if p600 is None:
            logger.warning("[secondary] no parent posonly cell for %s — skipped", key)
            continue
        u1_new = _u1(v)
        h = u1_new.shape[0]
        row = {"abs_cos_U1_vs_posonly600": abs(_cosine(u1_new, _u1(p600)))}
        for other_arm in ("marker", "em"):
            other = parent_cell(other_arm, v["variant"], v["seed"])
            if other is not None:
                row[f"abs_cos_U1_vs_{other_arm}"] = abs(_cosine(u1_new, _u1(other)))
        u1_identity[key] = row
        membership[key] = {
            "weighted_300": int(v["weighted"]["n_aligned_abs_cos_ge_0p5"]),
            "weighted_600": int(p600["weighted"]["n_aligned_abs_cos_ge_0p5"]),
            "unitnorm_300": int(v["unitnorm"]["n_aligned_abs_cos_ge_0p5"]),
            "unitnorm_600": int(p600["unitnorm"]["n_aligned_abs_cos_ge_0p5"]),
        }
        membership[key]["delta_weighted"] = (
            membership[key]["weighted_300"] - membership[key]["weighted_600"]
        )
        membership[key]["delta_unitnorm"] = (
            membership[key]["unitnorm_300"] - membership[key]["unitnorm_600"]
        )

    flavor_triangulation: dict[str, dict] = {}
    for variant in VARIANTS:
        entry: dict = {}
        for arm_key, pc, arm in (
            ("ckpt300", new_pc, "posonly"),
            ("posonly600", parent_pc, "posonly"),
            ("contrastive", parent_pc, "marker"),
            ("em", parent_pc, "em"),
        ):
            cells = _cells(pc, arm, variant)
            if cells:
                entry[arm_key] = {
                    "weighted": _band(cells, "weighted"),
                    "unitnorm": _band(cells, "unitnorm"),
                }
        if entry:
            flavor_triangulation[variant] = entry

    out: dict = {
        "u1_identity": u1_identity,
        "membership_300_vs_600": membership,
        "flavor_triangulation": flavor_triangulation,
    }
    if h is not None:
        out["random_pair_floor"] = _random_pair_floor(h)
    return out


def _bystander_emission_step300(periodic_dir: Path) -> dict:
    """Step-300 emission pattern from the parent's committed periodic-eval JSONs."""
    out: dict[str, dict] = {}
    for seed in SEEDS:
        p = periodic_dir / f"marker_seed{seed}" / "periodic_eval" / "leakage_marker_step_300.json"
        if not p.exists():
            out[f"seed{seed}"] = {"found": False, "path": str(p)}
            continue
        d = json.loads(p.read_text())
        m = d["metrics_by_persona"]
        out[f"seed{seed}"] = {
            "found": True,
            "path": str(p),
            "source_log_p_marker_delta": float(m[SOURCE_PERSONA]["log_p_marker_delta"]),
            "source_emit_rate": float(m[SOURCE_PERSONA]["emit_rate"]),
            "emit_rate_by_persona": {k: float(m[k]["emit_rate"]) for k in sorted(m)},
        }
    return out


def make_figures(
    new_pc: dict[str, dict],
    parent_pc: dict[str, dict],
    secondary: dict,
    figures_dir: Path,
) -> None:
    """Hero four-arm band plot + 300-vs-600 exploratory panels."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    colors = paper_palette(4)
    arms = [
        ("contrastive\nmarker", _cells(parent_pc, "marker", "same"), colors[1]),
        (
            "positive-only, step 300\n(matched row-visit exposure)",
            _cells(new_pc, "posonly", "same"),
            colors[3],
        ),
        (
            "positive-only, step 600\n(full training)",
            _cells(parent_pc, "posonly", "same"),
            colors[2],
        ),
        ("EM\ninsecure-code", _cells(parent_pc, "em", "same"), colors[0]),
    ]
    if not any(cells for _, cells, _ in arms):
        logger.warning("[figures] no same-text cells anywhere — skipping all figures")
        return

    # ── Hero: four-arm same-text top-share bands, weighted vs unit-norm ──
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, read in zip(axes, ("weighted", "unitnorm"), strict=True):
        for i, (_label, cells, color) in enumerate(arms):
            for j, v in enumerate(sorted(cells, key=lambda v: v["seed"])):
                x = i + (j - 1) * 0.12
                y = v[read]["s_top1_frac"]
                ax.scatter([x], [y], s=46, color=color, zorder=3)
                ax.annotate(
                    str(v["seed"]),
                    (x, y),
                    fontsize=6.5,
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                )
                if read == "weighted":
                    ax.hlines(
                        v["nulls"]["sign_flip_p95"], x - 0.07, x + 0.07, color="black", lw=1.1
                    )
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a[0] for a in arms], fontsize=7.5)
        ax.set_title(
            "norm-weighted SVD" if read == "weighted" else "unit-norm columns", fontsize=10
        )
    axes[0].set_ylabel("top-direction share of spectrum")
    fig.suptitle(
        "Shift-spectrum concentration by arm, trained-model text "
        "(ticks = per-cell sign-flip null p95)",
        fontsize=10,
    )
    fig.tight_layout()
    savefig_paper(fig, "four_arm_top_share_bands", dir=figures_dir)
    plt.close(fig)

    # ── Paired 300 -> 600 top-share trajectories, per flavor ─────────────
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(11.5, 4.0), sharey=True)
    seed_markers = {42: "o", 137: "^", 256: "s"}
    for ax, variant in zip(axes, VARIANTS, strict=True):
        for v in _cells(new_pc, "posonly", variant):
            p600 = parent_pc.get(f"posonly:{variant}_seed{v['seed']}")
            if p600 is None:
                continue
            ys = [v["weighted"]["s_top1_frac"], p600["weighted"]["s_top1_frac"]]
            ax.plot(
                [0, 1],
                ys,
                marker=seed_markers.get(v["seed"], "o"),
                color=colors[3],
                alpha=0.85,
                label=f"seed {v['seed']}",
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["step 300", "step 600"])
        ax.set_title(FLAVOR_LABELS.get(variant, variant), fontsize=9)
    axes[0].set_ylabel("weighted top-direction share")
    axes[0].legend(fontsize=7)
    fig.suptitle("Positive-only arm: top-share trajectory 300 -> 600 by text flavor", fontsize=10)
    fig.tight_layout()
    savefig_paper(fig, "ckpt300_vs_600_top_share", dir=figures_dir)
    plt.close(fig)

    # ── U1 direction identity bars per (variant, seed) ──────────────────
    def _cell_key_sort(key: str) -> tuple[int, int]:
        variant, seed = key.rsplit("_seed", 1)
        return (VARIANTS.index(variant), int(seed))

    def _cell_key_label(key: str) -> str:
        variant, seed = key.rsplit("_seed", 1)
        return f"{FLAVOR_LABELS.get(variant, variant)}, seed {seed}"

    u1_identity = secondary.get("u1_identity", {})
    if u1_identity:
        fig, ax = plt.subplots(figsize=(8.4, 4.0))
        keys = sorted(u1_identity, key=_cell_key_sort)
        xs = np.arange(len(keys))
        ax.bar(
            xs,
            [u1_identity[k]["abs_cos_U1_vs_posonly600"] for k in keys],
            width=0.6,
            color=colors[3],
        )
        floor = secondary.get("random_pair_floor", {})
        if floor:
            ax.axhline(
                floor["p95_abs_cos"],
                color="gray",
                ls="--",
                lw=1.0,
                label=f"random-pair |cos| p95 (h={floor['h']})",
            )
            ax.legend(fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([_cell_key_label(k) for k in keys], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("|cos(top direction at step 300, step 600)|")
        ax.set_title("Top-direction identity across the 300 -> 600 step range", fontsize=10)
        fig.tight_layout()
        savefig_paper(fig, "ckpt300_u1_identity", dir=figures_dir)
        plt.close(fig)

    # ── Aligned-set membership 300 vs 600 ────────────────────────────────
    membership = secondary.get("membership_300_vs_600", {})
    if membership:
        fig, ax = plt.subplots(figsize=(8.4, 4.0))
        keys = sorted(membership, key=_cell_key_sort)
        xs = np.arange(len(keys))
        w = 0.35
        ax.bar(
            xs - w / 2,
            [membership[k]["unitnorm_300"] for k in keys],
            width=w,
            color=colors[3],
            label="step 300",
        )
        ax.bar(
            xs + w / 2,
            [membership[k]["unitnorm_600"] for k in keys],
            width=w,
            color=colors[2],
            label="step 600",
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([_cell_key_label(k) for k in keys], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("n personas with |cos| >= 0.5 (unit-norm)")
        ax.set_title("Aligned-set membership: step 300 vs step 600", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        savefig_paper(fig, "ckpt300_membership_300_vs_600", dir=figures_dir)
        plt.close(fig)

    logger.info("[figures] written to %s", figures_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="#561 exposure-matched-ckpt300 verdict of record (CPU, JSON-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--new-per-cell",
        default="eval_results/issue_561/exposure-matched-ckpt300/comparison/comparison_per_cell.json",
        help="comparison_per_cell.json produced by issue561_compare.py over the ckpt-300 shifts.",
    )
    parser.add_argument(
        "--parent-per-cell",
        default="eval_results/issue_561/comparison/comparison_per_cell.json",
        help="The parent's committed step-600 comparison_per_cell.json (@ git 5f3ec9569).",
    )
    parser.add_argument(
        "--parent-pin",
        default=PARENT_PIN_COMMIT,
        help=(
            "Git commit the parent per-cell JSON must parse-match (band-drift guard). "
            "Pass '' to skip deliberately (ad-hoc fixture parent)."
        ),
    )
    parser.add_argument(
        "--out", default="eval_results/issue_561/exposure-matched-ckpt300/comparison"
    )
    parser.add_argument("--figures-dir", default="figures/issue_561/exposure-matched-ckpt300")
    parser.add_argument(
        "--periodic-eval-dir",
        default="eval_results/issue_561",
        help="Parent eval dir holding marker_seed{S}/periodic_eval/leakage_marker_step_300.json.",
    )
    parser.add_argument(
        "--skip-figures", action="store_true", help="JSON verdict only (figure-free smoke)."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    new_path, parent_path = Path(args.new_per_cell), Path(args.parent_per_cell)
    # Band-drift guard BEFORE any evaluation: the parent JSON is the band source.
    _assert_parent_pinned(parent_path, args.parent_pin)
    new_pc = _load_per_cell(new_path)
    parent_pc = _load_per_cell(parent_path)

    # Hard assert: identical persona_order across every joined cell.
    persona_order = _assert_persona_order([*new_pc.values(), *parent_pc.values()])

    # Convention-drift guard over the shared #551 cells.
    cross_check = _cross_check_shared_cells(new_pc, parent_pc)

    # Bands from the UNROUNDED parent per-cell values (plan v2 §14 amendment).
    contrastive_same = _cells(parent_pc, "marker", "same")
    posonly600_same = _cells(parent_pc, "posonly", "same")
    if len(contrastive_same) != 3 or len(posonly600_same) != 3:
        raise ValueError(
            f"parent JSON must carry 3 contrastive + 3 posonly same-text cells; got "
            f"{len(contrastive_same)} + {len(posonly600_same)}"
        )
    bands = {
        "contrastive_same_weighted": _band(contrastive_same, "weighted"),
        "contrastive_same_unitnorm": _band(contrastive_same, "unitnorm"),
        "posonly600_same_weighted": _band(posonly600_same, "weighted"),
        "posonly600_same_unitnorm": _band(posonly600_same, "unitnorm"),
        "em_same_weighted": _band(_cells(parent_pc, "em", "same"), "weighted"),
        "source": f"{parent_path} (runtime-loaded, unrounded)",
    }
    # The §3 clauses presuppose disjoint bands (gap zones exist on both reads).
    for w_key, u_key in (("contrastive_same_weighted", "contrastive_same_unitnorm"),):
        if not (
            bands[w_key]["max"] < bands["posonly600_same_weighted"]["min"]
            and bands[u_key]["max"] < bands["posonly600_same_unitnorm"]["min"]
        ):
            raise ValueError(
                "contrastive and posonly-600 same-text bands overlap — the pre-registered "
                "collapse/persist clauses are ill-posed against this parent JSON."
            )

    new_same = _cells(new_pc, "posonly", "same")
    headline = _evaluate_clauses(
        new_same,
        bands["contrastive_same_weighted"],
        bands["contrastive_same_unitnorm"],
        bands["posonly600_same_weighted"],
        bands["posonly600_same_unitnorm"],
    )

    secondary = _secondary_reads(new_pc, parent_pc)
    secondary["bystander_emission_step300"] = _bystander_emission_step300(
        Path(args.periodic_eval_dir)
    )

    verdict_payload = {
        "meta": {
            "issue": 561,
            "followup_label": "exposure-matched-ckpt300",
            "analysis": "ckpt300_verdict_of_record",
            "git_commit": _git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "new_per_cell": str(new_path),
            "parent_per_cell": str(parent_path),
            "parent_pin": args.parent_pin or None,
            "cross_check_tol": CROSS_CHECK_TOL,
            "n_shared_cells_cross_checked": len(cross_check),
            "persona_order": persona_order,
            "note": (
                "The compare script's own comparison_summary.json headline evaluates the "
                "PARENT's §3 clauses (is ckpt-300 EM-like?) — context only. THIS file is "
                "the follow-up's verdict of record (plan v2 §14a)."
            ),
        },
        "bands": bands,
        "headline": headline,
        "secondary": secondary,
        "shared_cell_cross_check_max_abs_delta": cross_check,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ckpt300_verdict.json"
    with out_path.open("w") as f:
        json.dump(verdict_payload, f, indent=2)
    logger.info("[wrote] %s", out_path)

    if not args.skip_figures:
        make_figures(new_pc, parent_pc, secondary, Path(args.figures_dir))

    logger.info(
        "[phase=done] verdict=%s (evaluated=%s)",
        headline.get("verdict", "n/a"),
        headline.get("evaluated"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
