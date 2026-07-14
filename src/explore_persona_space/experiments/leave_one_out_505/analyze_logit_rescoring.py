# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Greek Δ/β + Qwen marker " ※" intentional
"""Task #505 follow-up ``logit-space-rescoring`` — Phase D (VM-side, CPU) analysis.

Re-runs the #505 expanded-covariate regressions at the frac-1.0 slice in THREE
readouts, all from the four-float HF capture (``slot_stats/`` written by
``dispatch_logit_rescoring`` Phases A-B):

  dv_delta_logp    Δ-leakage in log-prob space (PRIMARY, behavioral):
                   mean_q Δlog P(drop-j) − mean_q Δlog P(full-set), where
                   Δlog P = logp_g − logp_b per probe.
  dv_delta_margin  Δ-leakage in EOS-margin space (SECONDARY, mechanistic):
                   per-probe Δ(z_marker − z_eos) trained − base, then the
                   same drop-minus-full contrast. Gauge-invariant (cancels
                   common-mode logit shifts) and non-saturating.
  dv_delta_logz    the per-condition normalizer-drift DIAGNOSTIC:
                   per-probe ΔlogZ = logZ_g − logZ_b, drop-minus-full. This
                   answers the followup's motivating question directly —
                   does logZ drift differ between drop-j and full-set cells
                   (i.e. could a condition-dependent normalizer bias the
                   log-prob comparison invisibly)?

Where log-prob and logit reads diverge, the cell is saturated and the LOGIT
read is primary (``.claude/rules/marker-leakage-measurement.md`` § Analysis
contract) — the saturation report + cross-space agreement table localize
exactly that.

Reuses the round-1 rigs verbatim: ``fit_per_arm_models`` / ``fit_pooled_model``
from ``analyze_expanded`` (same per-arm HC2 OLS + pooled arm/seed-FE OLS with
bystander-clustered SEs) and the round-1 ``geometry_predictors.json`` as-is
(per the followup spec — no centroid recomputation).

Run on the VM after syncing the pod outputs:

    uv run python -m \\
      explore_persona_space.experiments.leave_one_out_505.analyze_logit_rescoring
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy import below — on the shared
# VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS, and the
# BLAS pools freeze at import time (the first-party imports below pull numpy
# transitively too).
load_dotenv()

import numpy as np  # noqa: E402

from explore_persona_space.experiments.leave_one_out_505 import (  # noqa: E402
    HEADLINE_LAYER,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.leave_one_out_505.analyze_expanded import (  # noqa: E402
    BASELINE_PREDICTORS as _R1_BASELINE_PREDICTORS,
)
from explore_persona_space.experiments.leave_one_out_505.analyze_expanded import (  # noqa: E402
    EXPANDED_PREDICTORS as _R1_EXPANDED_PREDICTORS,
)
from explore_persona_space.experiments.leave_one_out_505.analyze_expanded import (  # noqa: E402
    FULL_SET_SLUG,
    fit_per_arm_models,
    fit_pooled_model,
)
from explore_persona_space.experiments.leave_one_out_505.analyze_expanded import (  # noqa: E402
    PER_ARM_EXPANDED as _R1_PER_ARM_EXPANDED,
)
from explore_persona_space.experiments.leave_one_out_505.analyze_expanded import (  # noqa: E402
    PER_ARM_ORIGINAL as _R1_PER_ARM_ORIGINAL,
)
from explore_persona_space.experiments.leave_one_out_505.logit_rescoring import (  # noqa: E402
    SCHEMA_VERSION,
    TARGET_FRAC,
    repro_block,
    stored_records_at_frac,
)

log = logging.getLogger("issue_505.analyze_logit_rescoring")

DEFAULT_RESCORING_ROOT = Path("eval_results/issue_505/logit-space-rescoring")
DEFAULT_PANEL_GATE = Path("eval_results/issue_505/panel_coverage.json")
DEFAULT_GEOMETRY_JSON = Path(
    "eval_results/issue_505/expanded-predictor-reanalysis/geometry_predictors.json"
)
DEFAULT_SWEEP_DIR = Path("eval_results/issue_505/sweep")
DEFAULT_FIG_DIR = Path("figures/issue_505/logit-space-rescoring")

# Per-probe trained − base delta per readout, from a slot_stats leaf.
READOUT_LEAF_DELTA = {
    "dv_delta_logp": lambda leaf: leaf["delta_logp"],
    "dv_delta_margin": lambda leaf: leaf["delta_margin"],
    "dv_delta_logz": lambda leaf: leaf["delta_logz"],
}
# Per-probe BASE-side level in the matching space (for the base-prior covariate).
READOUT_LEAF_BASE = {
    "dv_delta_logp": lambda leaf: leaf["logp_b"],
    "dv_delta_margin": lambda leaf: leaf["z_marker_b"] - leaf["z_eos_b"],
    "dv_delta_logz": lambda leaf: leaf["logZ_b"],
}
READOUTS = tuple(READOUT_LEAF_DELTA)

# Predictor tuples are derived from the round-1 rig's tuples
# (``analyze_expanded.PER_ARM_ORIGINAL`` etc.) with ONE rename: the source-
# implant covariate is called ``delta_source_dg`` in the round-1 log-prob-
# only frame, but lives in three readout-matched spaces here, so the column
# in this rig's frame is named ``delta_source_shift`` (built in
# ``_attach_predictors`` per readout). Importing the round-1 tuples + renaming
# in one place keeps the predictor menus in sync — adding a covariate to the
# round-1 rig surfaces here as a missing-column raise rather than as silent
# divergence between the two analyses.
_RENAME_R1_TO_LR = {"delta_source_dg": "delta_source_shift"}


def _remap(predictors: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_RENAME_R1_TO_LR.get(p, p) for p in predictors)


PER_ARM_ORIGINAL_LR = _remap(_R1_PER_ARM_ORIGINAL)
PER_ARM_EXPANDED_LR = _remap(_R1_PER_ARM_EXPANDED)
POOLED_BASELINE_LR = _remap(_R1_BASELINE_PREDICTORS)
POOLED_EXPANDED_LR = _remap(_R1_EXPANDED_PREDICTORS)


# ── Loading ──────────────────────────────────────────────────────────────────


def load_slot_stats(stats_dir: Path) -> dict[tuple[str, int], dict]:
    """Load every ``slot_stats/<cell>_seed<S>.json``; fail loud on an empty dir."""
    files = sorted(stats_dir.glob("*_seed*.json"))
    if not files:
        raise FileNotFoundError(
            f"no slot_stats files under {stats_dir} — run dispatch_logit_rescoring (Phases A-B) "
            "and sync the pod outputs before Phase D."
        )
    out: dict[tuple[str, int], dict] = {}
    for f in files:
        payload = json.loads(f.read_text())
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{f}: schema_version={payload.get('schema_version')!r} != {SCHEMA_VERSION!r}"
            )
        out[(payload["cell"], int(payload["seed"]))] = payload
    log.info("[load] %d slot_stats cells from %s", len(out), stats_dir)
    return out


def _per_q_delta(payload: dict, persona: str, readout: str) -> dict[str, float]:
    leaf_fn = READOUT_LEAF_DELTA[readout]
    per_q = payload["slot_stats"].get(persona)
    if per_q is None:
        raise KeyError(f"{payload['cell']}_seed{payload['seed']}: persona {persona!r} missing")
    return {q: float(leaf_fn(leaf)) for q, leaf in per_q.items()}


# ── Frame ────────────────────────────────────────────────────────────────────


def build_logit_frame(
    stats: dict[tuple[str, int], dict],
    *,
    panel: list[str],
    non_default_negatives: list[str],
    source: str = SOURCE_PERSONA,
) -> dict:
    """Long-format (b, j_i, seed) frame with the three readout DVs.

    Mirrors round 1's ``compute_delta_leakage_table`` contrast (drop − full
    over the common-question intersection, mean-pooled over questions) at the
    frac-1.0 slice. ``delta_source_shift`` and ``base_prior_b`` are computed
    per readout in the MATCHING space (the source-implant covariate and the
    base-prior covariate must live in the same units as the DV they enter
    alongside).
    """
    seeds = sorted({seed for (_slug, seed) in stats})
    avail_personas = set(next(iter(stats.values()))["slot_stats"].keys())
    use_panel = [b for b in panel if b in avail_personas]
    if not use_panel:
        raise RuntimeError(
            "no overlap between the gate panel and the captured personas — wrong panel gate "
            "artifact or wrong slot_stats dir."
        )
    arms = [
        (j_idx, j_i)
        for j_idx, j_i in enumerate(non_default_negatives)
        if any((f"c505_drop_j{j_idx}", s) in stats for s in seeds)
    ]
    if not arms:
        raise RuntimeError("no drop-arm slot_stats present — nothing to contrast against full-set.")

    rows: list[dict[str, Any]] = []
    source_shift: dict[str, dict[tuple[str, int], float]] = {r: {} for r in READOUTS}
    for seed in seeds:
        if (FULL_SET_SLUG, seed) not in stats:
            raise RuntimeError(
                f"full-set slot_stats missing for seed {seed} — the drop-minus-full contrast "
                "is undefined; re-run/resume the dispatcher for that cell."
            )
        full = stats[(FULL_SET_SLUG, seed)]
        for j_idx, j_i in arms:
            slug = f"c505_drop_j{j_idx}"
            if (slug, seed) not in stats:
                log.warning("[frame] %s seed %d absent — skipping arm-seed", slug, seed)
                continue
            drop = stats[(slug, seed)]
            for readout in READOUTS:
                src_full = _per_q_delta(full, source, readout)
                src_drop = _per_q_delta(drop, source, readout)
                src_common = sorted(set(src_full) & set(src_drop))
                source_shift[readout][(j_i, seed)] = float(
                    np.mean([src_drop[q] for q in src_common])
                    - np.mean([src_full[q] for q in src_common])
                )
            for b in use_panel:
                row: dict[str, Any] = {"b": b, "j_i": j_i, "j_idx": j_idx, "seed": seed}
                for readout in READOUTS:
                    d_full = _per_q_delta(full, b, readout)
                    d_drop = _per_q_delta(drop, b, readout)
                    common = sorted(set(d_full) & set(d_drop))
                    if not common:
                        raise RuntimeError(f"no common questions for (b={b}, j={j_i}, seed={seed})")
                    row[readout] = float(
                        np.mean([d_drop[q] for q in common]) - np.mean([d_full[q] for q in common])
                    )
                    row["n_q"] = len(common)
                rows.append(row)

    # Per-bystander base prior per readout space, pooled over the in-design
    # cells (full-set + present drop arms) × seeds × questions.
    in_design = [FULL_SET_SLUG] + [f"c505_drop_j{j_idx}" for j_idx, _ in arms]
    base_prior: dict[str, dict[str, float]] = {r: {} for r in READOUTS}
    for readout in READOUTS:
        base_fn = READOUT_LEAF_BASE[readout]
        for b in use_panel:
            vals: list[float] = []
            for (slug, _seed), payload in stats.items():
                if slug not in in_design:
                    continue
                per_q = payload["slot_stats"].get(b, {})
                vals.extend(float(base_fn(leaf)) for leaf in per_q.values())
            base_prior[readout][b] = float(np.mean(vals))

    return {
        "rows": rows,
        "seeds": seeds,
        "panel": use_panel,
        "arms": [j_i for _idx, j_i in arms],
        "source_shift": {
            r: {f"{j}|{s}": v for (j, s), v in source_shift[r].items()} for r in READOUTS
        },
        "base_prior": base_prior,
        "frac": TARGET_FRAC,
        "n_rows": len(rows),
        "dv_construction": {
            "dv_delta_logp": (
                "mean_q Δlog P(drop-j) − mean_q Δlog P(full-set); Δlog P = logp_g − logp_b "
                "(HF four-float capture) at the post-response slot, frac 1.0"
            ),
            "dv_delta_margin": (
                "same contrast on the EOS margin: Δ(z_marker − z_eos) trained − base per probe"
            ),
            "dv_delta_logz": (
                "same contrast on the normalizer: ΔlogZ = logZ_g − logZ_b per probe "
                "(condition-dependent normalizer-drift diagnostic)"
            ),
            "delta_source_shift": "source-persona analogue of each DV, per (arm, seed)",
            "base_prior_b": (
                "per-bystander mean of the base-side level in the matching space over the "
                "in-design cells × seeds × questions"
            ),
        },
    }


def _attach_predictors(frame: dict, geom_layer: dict, *, readout: str):
    """pandas frame for one (layer, readout): DV + geometry + matched-space covariates."""
    import pandas as pd

    rows = frame["rows"]
    df = pd.DataFrame(
        [{"b": r["b"], "j_i": r["j_i"], "seed": r["seed"], readout: r[readout]} for r in rows]
    )
    df["cos_b_j"] = [geom_layer["cos_b_j"][r["b"]][r["j_i"]] for r in rows]
    df["cos_b_source"] = [geom_layer["cos_b_source"][r["b"]] for r in rows]
    df["shadow_angle"] = [geom_layer["shadow_angle"][r["b"]][r["j_i"]] for r in rows]
    df["d_nearest_remaining"] = [geom_layer["d_nearest_remaining"][r["b"]][r["j_i"]] for r in rows]
    df["base_prior_b"] = [frame["base_prior"][readout][r["b"]] for r in rows]
    df["delta_source_shift"] = [
        frame["source_shift"][readout][f"{r['j_i']}|{r['seed']}"] for r in rows
    ]
    return df


# ── Saturation report + cross-space agreement ───────────────────────────────


def build_saturation_report(
    stats: dict[tuple[str, int], dict], *, sweep_dir: Path, panel: list[str], source: str
) -> dict:
    """Stored frac-1.0 argmax/emission shares per cell + the new capture's shares.

    The stored shares are the spec's mandated lead read ("report bystander
    saturation share at frac 1.0 first"); the new HF-side argmax shares and
    the mean ΔlogZ per cell localize where the log-prob read is compressed.
    """
    per_cell: dict[str, dict[str, Any]] = {}
    for (slug, seed), payload in sorted(stats.items()):
        key = f"{slug}_seed{seed}"
        entry: dict[str, Any] = {"cell": slug, "seed": seed}
        traj = sweep_dir / slug / f"seed_{seed}" / "trajectory.json"
        stored = stored_records_at_frac(traj, frac=TARGET_FRAC, source=source)
        by_leafs = [leaf for p, per_q in stored.items() if p != source for leaf in per_q.values()]
        src_leafs = list(stored.get(source, {}).values())
        entry["stored"] = {
            "bystander_argmax_share": float(np.mean([leaf["argmax_marker"] for leaf in by_leafs])),
            "bystander_mean_g_logp": float(np.mean([leaf["g_logp"] for leaf in by_leafs])),
            "bystander_near_ceiling_share": float(
                np.mean([leaf["g_logp"] > -0.1 for leaf in by_leafs])
            ),
            "source_argmax_share": (
                float(np.mean([leaf["argmax_marker"] for leaf in src_leafs])) if src_leafs else None
            ),
        }
        new_by = [
            leaf
            for p, per_q in payload["slot_stats"].items()
            if p != source and p in panel
            for leaf in per_q.values()
        ]
        new_src = list(payload["slot_stats"].get(source, {}).values())
        entry["new_capture"] = {
            "bystander_argmax_share_hf": float(
                np.mean([leaf["argmax_marker_g"] for leaf in new_by])
            ),
            "bystander_mean_logp_g": float(np.mean([leaf["logp_g"] for leaf in new_by])),
            "bystander_mean_delta_logz": float(np.mean([leaf["delta_logz"] for leaf in new_by])),
            "bystander_collapse_share": float(np.mean([leaf["r_collapsed"] for leaf in new_by])),
            "source_argmax_share_hf": (
                float(np.mean([leaf["argmax_marker_g"] for leaf in new_src])) if new_src else None
            ),
            "source_mean_delta_logz": (
                float(np.mean([leaf["delta_logz"] for leaf in new_src])) if new_src else None
            ),
        }
        per_cell[key] = entry
    return {"frac": TARGET_FRAC, "per_cell": per_cell}


def build_cross_space_agreement(
    stats: dict[tuple[str, int], dict], *, panel: list[str], source: str
) -> dict:
    """Per-cell Δlog P vs Δz_marker agreement — the saturation signature localizer.

    Off saturation ΔlogZ ≈ 0 so Δlog P ≈ Δz_marker (agreement ⇒ the log-prob
    result is faithful); where they diverge the cell is saturated and the
    logit read is primary. Divergence is a FINDING, not an error.
    """
    per_cell: dict[str, dict[str, Any]] = {}
    for (slug, seed), payload in sorted(stats.items()):
        d_logp: list[float] = []
        d_z: list[float] = []
        d_logz: list[float] = []
        for p, per_q in payload["slot_stats"].items():
            if p == source or p not in panel:
                continue
            for leaf in per_q.values():
                d_logp.append(float(leaf["delta_logp"]))
                d_z.append(float(leaf["delta_z_marker"]))
                d_logz.append(float(leaf["delta_logz"]))
        a, z = np.asarray(d_logp), np.asarray(d_z)
        corr = (
            float(np.corrcoef(a, z)[0, 1]) if a.size >= 2 and a.std() > 0 and z.std() > 0 else None
        )
        mean_dlogz = float(np.mean(d_logz))
        per_cell[f"{slug}_seed{seed}"] = {
            "n_probes": int(a.size),
            "corr_delta_logp_vs_delta_z_marker": corr,
            "mean_delta_logp": float(a.mean()),
            "mean_delta_z_marker": float(z.mean()),
            "mean_delta_logz": mean_dlogz,
            "mean_abs_disagreement": float(np.mean(np.abs(a - z))),
            "saturation_signature": bool(abs(mean_dlogz) > 0.5),
        }
    return {
        "per_cell": per_cell,
        "read": (
            "saturation_signature flags |mean ΔlogZ| > 0.5 nat — there Δlog P understates the "
            "push and the EOS-margin / z_marker readout is primary."
        ),
    }


# ── Figures ──────────────────────────────────────────────────────────────────

_READOUT_LABELS = {
    "dv_delta_logp": "Δ-leakage, log-prob space",
    "dv_delta_margin": "Δ-leakage, EOS-margin space",
    "dv_delta_logz": "ΔlogZ drift (diagnostic)",
}
_PREDICTOR_LABELS = {
    "cos_b_j": "cos(bystander, dropped negative)",
    "shadow_angle": "shadow angle (bystander; dropped negative)",
    "d_nearest_remaining": "distance to nearest remaining negative",
    "cos_b_source": "cos(bystander, source)",
    "base_prior_b": "base-side prior (matching space)",
    "delta_source_shift": "source implant shift (drop − full)",
}


def figure_pooled_three_readouts(pooled: dict, fig_dir: Path, *, layer: int) -> None:
    """Standardized pooled coefficients, every expanded predictor × three readouts."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")
    colors = paper_palette(len(READOUTS))
    preds = list(POOLED_EXPANDED_LR)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    markers = ("o", "D", "s")
    offsets = (0.22, 0.0, -0.22)
    for readout, color, mk, off in zip(READOUTS, colors, markers, offsets, strict=True):
        coefs = pooled[str(layer)][readout]["expanded_standardized"]["coefficients"]
        for y, name in enumerate(preds):
            c = coefs[name]
            ax.errorbar(
                c["beta"],
                y + off,
                xerr=[[c["beta"] - c["ci95_low"]], [c["ci95_high"] - c["beta"]]],
                fmt=mk,
                color=color,
                markersize=5.5,
                capsize=2.5,
                lw=1.4,
                label=_READOUT_LABELS[readout] if y == 0 else None,
            )
    ax.axvline(0.0, color="0.55", lw=0.9, ls="--", zorder=0)
    ax.set_yticks(range(len(preds)))
    ax.set_yticklabels([_PREDICTOR_LABELS[p] for p in preds])
    ax.invert_yaxis()
    ax.set_xlabel("standardized slope (DV units per predictor SD), bystander-clustered 95% CI")
    ax.legend(loc="best", fontsize=9)
    set_title_subtitle(
        ax,
        f"Expanded-covariate pooled slopes at the frac-1.0 slice, layer {layer}",
        "Three readouts from the same HF four-float capture; pooled OLS with arm/seed fixed "
        "effects and bystander-clustered SEs",
    )
    # ``set_paper_style`` enables ``figure.constrained_layout.use`` (paper_plots
    # convention); calling tight_layout() on top triggers an engine-clash warn.
    savefig_paper(fig, "pooled_three_readouts", dir=fig_dir)
    plt.close(fig)


def figure_cross_space_scatter(frame: dict, fig_dir: Path) -> None:
    """Row-level Δ-leakage: log-prob space vs EOS-margin space, colored by ΔlogZ drift."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    rows = frame["rows"]
    x = np.array([r["dv_delta_logp"] for r in rows])
    y = np.array([r["dv_delta_margin"] for r in rows])
    c = np.array([r["dv_delta_logz"] for r in rows])
    # constrained_layout (not tight_layout) — colorbar + tight_layout clash
    # raises "Colorbar layout of new layout engine not compatible" after the
    # colorbar is attached. Using constrained at construction sidesteps this.
    fig, ax = plt.subplots(figsize=(6.8, 5.6), layout="constrained")
    sc = ax.scatter(x, y, c=c, cmap="coolwarm", s=22, alpha=0.85, linewidths=0)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, color="0.6", lw=0.9, ls="--", zorder=0)
    fig.colorbar(sc, ax=ax, label="ΔlogZ drift (drop − full, nats)")
    ax.set_xlabel("Δ-leakage, log-prob space (nats)")
    ax.set_ylabel("Δ-leakage, EOS-margin space (logits)")
    ax.set_title(
        "Cross-space agreement per (bystander × arm × seed) row\n"
        "off-diagonal departures track the normalizer drift (saturation signature)",
        fontsize=11,
    )
    savefig_paper(fig, "cross_space_agreement_scatter", dir=fig_dir)
    plt.close(fig)


def figure_delta_logz_by_cell(stats: dict, fig_dir: Path, *, panel: list[str], source: str) -> None:
    """Mean bystander ΔlogZ per cell (the drop-vs-full normalizer-drift question)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    slugs = sorted({slug for slug, _ in stats})
    per_slug: dict[str, list[float]] = {s: [] for s in slugs}
    for (slug, _seed), payload in stats.items():
        vals = [
            float(leaf["delta_logz"])
            for p, per_q in payload["slot_stats"].items()
            if p != source and p in panel
            for leaf in per_q.values()
        ]
        per_slug[slug].append(float(np.mean(vals)))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    color = paper_palette(1)[0]
    xs = range(len(slugs))
    means = [float(np.mean(per_slug[s])) for s in slugs]
    ax.bar(xs, means, color=color, alpha=0.75, width=0.62)
    for i, s in enumerate(slugs):
        ax.scatter([i] * len(per_slug[s]), per_slug[s], color="0.25", s=14, zorder=3)
    ax.axhline(0.0, color="0.5", lw=0.9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([s.removeprefix("c505_") for s in slugs], rotation=30, ha="right")
    ax.set_ylabel("mean bystander ΔlogZ (trained − base, nats)")
    ax.set_title(
        "Normalizer drift per cell at frac 1.0 (bars = cell mean, dots = seeds)", fontsize=11
    )
    # ``set_paper_style`` already enables constrained_layout; tight_layout() warns.
    savefig_paper(fig, "delta_logz_by_cell", dir=fig_dir)
    plt.close(fig)


# ── End-to-end ───────────────────────────────────────────────────────────────


def run_analysis(
    *,
    rescoring_root: Path,
    panel_gate_path: Path,
    geometry_json_path: Path,
    sweep_dir: Path,
    out_dir: Path,
    fig_dir: Path,
) -> dict:
    """Frame → regressions (3 readouts × layers) → saturation + agreement → figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "rescoring_root": str(rescoring_root),
        "panel_gate": str(panel_gate_path),
        "geometry_json": str(geometry_json_path),
        "sweep_dir": str(sweep_dir),
    }
    repro = repro_block(inputs)

    panel_payload = json.loads(panel_gate_path.read_text())
    if not panel_payload.get("gate_passed"):
        raise RuntimeError("panel coverage gate did not pass; refusing to analyze.")
    panel = list(panel_payload["panel"])
    non_default = list(panel_payload["non_default_negatives"])
    geometry = json.loads(geometry_json_path.read_text())
    layers = [int(layer) for layer in geometry["layers"]]

    stats = load_slot_stats(rescoring_root / "slot_stats")
    faith_path = rescoring_root / "faithfulness.json"
    faithfulness = json.loads(faith_path.read_text()) if faith_path.exists() else None
    if faithfulness is None:
        log.warning("[analysis] faithfulness.json absent — interpretation gate input missing.")

    # Phase D1: frame.
    frame = build_logit_frame(stats, panel=panel, non_default_negatives=non_default)
    (out_dir / "logit_frame.json").write_text(
        json.dumps({**frame, "reproducibility": repro}, indent=2)
    )
    log.info("[analysis] frame: %d rows × %d readouts", frame["n_rows"], len(READOUTS))

    # Phase D2: per-arm + pooled regressions per layer × readout.
    min_per_arm = len(PER_ARM_EXPANDED_LR) + 2
    n_per_arm = len(frame["panel"]) * len(frame["seeds"])
    if n_per_arm < min_per_arm:
        raise RuntimeError(
            f"per-arm OLS needs ≥{min_per_arm} rows per arm; have {n_per_arm} "
            "(panel × seeds too small — widen --personas/--seeds)."
        )
    per_arm_out: dict[str, Any] = {}
    pooled_out: dict[str, Any] = {}
    for layer in layers:
        geom_layer = geometry["per_layer"][str(layer)]
        per_arm_out[str(layer)] = {}
        pooled_out[str(layer)] = {}
        for readout in READOUTS:
            df = _attach_predictors(frame, geom_layer, readout=readout)
            per_arm_out[str(layer)][readout] = {
                "original_covariates": fit_per_arm_models(
                    df, dv=readout, predictors=PER_ARM_ORIGINAL_LR
                ),
                "expanded_covariates": fit_per_arm_models(
                    df, dv=readout, predictors=PER_ARM_EXPANDED_LR
                ),
            }
            pooled_out[str(layer)][readout] = {
                "baseline": fit_pooled_model(df, dv=readout, predictors=POOLED_BASELINE_LR),
                "expanded": fit_pooled_model(df, dv=readout, predictors=POOLED_EXPANDED_LR),
                "expanded_standardized": fit_pooled_model(
                    df, dv=readout, predictors=POOLED_EXPANDED_LR, standardize=True
                ),
            }
    (out_dir / "per_arm_logit_ols.json").write_text(
        json.dumps({"per_layer": per_arm_out, "reproducibility": repro}, indent=2)
    )
    (out_dir / "pooled_logit_ols.json").write_text(
        json.dumps({"per_layer": pooled_out, "reproducibility": repro}, indent=2)
    )
    log.info("[analysis] regressions done over layers %s", layers)

    # Phase D3: saturation report (stored shares FIRST, per spec) + agreement.
    saturation = build_saturation_report(
        stats, sweep_dir=sweep_dir, panel=frame["panel"], source=SOURCE_PERSONA
    )
    (out_dir / "saturation_report.json").write_text(
        json.dumps({**saturation, "reproducibility": repro}, indent=2)
    )
    agreement = build_cross_space_agreement(stats, panel=frame["panel"], source=SOURCE_PERSONA)
    (out_dir / "cross_space_agreement.json").write_text(
        json.dumps({**agreement, "reproducibility": repro}, indent=2)
    )

    # Phase D4: headline comparison across readouts at the headline layer.
    hl = str(HEADLINE_LAYER) if str(HEADLINE_LAYER) in pooled_out else str(layers[0])
    headline = {
        "headline_layer": int(hl),
        "frac": TARGET_FRAC,
        "pooled_cos_b_j_by_readout": {
            readout: {
                "baseline": pooled_out[hl][readout]["baseline"]["coefficients"]["cos_b_j"],
                "expanded": pooled_out[hl][readout]["expanded"]["coefficients"]["cos_b_j"],
                "expanded_standardized": pooled_out[hl][readout]["expanded_standardized"][
                    "coefficients"
                ]["cos_b_j"],
            }
            for readout in READOUTS
        },
        "per_arm_sign_agreement_by_readout": {
            readout: per_arm_out[hl][readout]["expanded_covariates"]["sign_agreement_cos_b_j"]
            for readout in READOUTS
        },
        "faithfulness_summary": (
            {
                cell: {
                    "hf_vs_stored_g_mae": v["hf_vs_stored"]["g"]["mae"],
                    "hf_vs_stored_g_spearman": v["hf_vs_stored"]["g"]["spearman_rho"],
                    **(
                        {
                            "vllm_vs_stored_g_mae": v["vllm_vs_stored"]["g"]["mae"],
                            "vllm_vs_stored_g_spearman": v["vllm_vs_stored"]["g"]["spearman_rho"],
                        }
                        if "vllm_vs_stored" in v
                        else {}
                    ),
                }
                for cell, v in faithfulness["per_cell"].items()
            }
            if faithfulness
            else None
        ),
        "n_saturation_flagged_cells": int(
            sum(1 for v in agreement["per_cell"].values() if v["saturation_signature"])
        ),
        "reproducibility": repro,
    }
    (out_dir / "headline_logit_comparison.json").write_text(json.dumps(headline, indent=2))

    # Phase D5: figures.
    figure_pooled_three_readouts(pooled_out, fig_dir, layer=int(hl))
    figure_cross_space_scatter(frame, fig_dir)
    figure_delta_logz_by_cell(stats, fig_dir, panel=frame["panel"], source=SOURCE_PERSONA)
    log.info("[analysis] figures under %s", fig_dir)
    return {"out_dir": str(out_dir), "n_rows": frame["n_rows"], "headline": headline}


def cli_main(argv: list[str] | None = None) -> int:
    """argparse entrypoint for the VM-side Phase D run."""
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    p = argparse.ArgumentParser(description="#505 logit-space-rescoring Phase D analysis")
    p.add_argument("--rescoring-root", default=str(DEFAULT_RESCORING_ROOT))
    p.add_argument("--panel-gate", default=str(DEFAULT_PANEL_GATE))
    p.add_argument("--geometry-json", default=str(DEFAULT_GEOMETRY_JSON))
    p.add_argument("--sweep-dir", default=str(DEFAULT_SWEEP_DIR))
    p.add_argument("--out-dir", default=None, help="default: <rescoring-root>/analysis")
    p.add_argument("--fig-dir", default=str(DEFAULT_FIG_DIR))
    args = p.parse_args(argv)
    root = Path(args.rescoring_root)
    result = run_analysis(
        rescoring_root=root,
        panel_gate_path=Path(args.panel_gate),
        geometry_json_path=Path(args.geometry_json),
        sweep_dir=Path(args.sweep_dir),
        out_dir=Path(args.out_dir) if args.out_dir else root / "analysis",
        fig_dir=Path(args.fig_dir),
    )
    log.info("[analysis-done] %s", result["out_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(cli_main(sys.argv[1:]))
