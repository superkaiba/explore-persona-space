#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, Δ, √, →, ≥, ×, −) in scientific docstrings + labels.
"""Issue #763 `rubric-v2p1-disclosure-carveout`: the §6/§9c verdict cells + figures.

Consumes the round's outputs (v2.1 E0 + fit record, ablation E0 + fit record,
disclosure-flag audit) plus the committed reanchor references (v2 E0 + fit
record) and emits ``verdict_v2p1.json`` — EVERY cell of the plan's pre-registered
outcome enumeration decidable from structured fields:

- **Cell 0 — instrument-health precondition:** per-arm judge drop-rate ≤ 20%
  (post-retry FINAL state; the ONE re-dispatch retry is the judge phase's).
- **Cell 1 — manipulation-check 2×2:** leg (a) = the v2-flagged items' mean
  graded score drop ≥ 30 points under v2.1, computed on the SAME per-(context,
  probe) means the fit consumes; leg (b) = the v2.1 disclosure-flag rate among
  its high draws < 1/3 of the v2 rate, with the pre-registered ≥100 high-draw
  minimum-denominator convention (below it the rate is UNDEFINED and the 2×2
  collapses to leg (a) + the ablation read: leg-(a) pass ⇒ ENGAGED-by-collapse,
  fail ⇒ PARTIAL-ENGAGEMENT-B). Cells: ENGAGED / KILL / PARTIAL-ENGAGEMENT-A /
  PARTIAL-ENGAGEMENT-B.
- **Cell 2 — SUCCESS-robust vs SUCCESS-attenuated:** ρ_v2.1 ≥ the registered
  band floor (the committed v2 CI lower bound) AND shuffle p < 0.05 AND
  control-task pass; the split is the PAIRED per-context Δρ = ρ_v2.1 − ρ_v2
  cluster bootstrap (2000 draws, seed 763, SAME resample indices for both arms)
  from the two fits' FIXED LOCO predictions — CI includes 0 ⇒ robust, excludes
  0 ⇒ attenuated. ρ > the v2 CI upper bound ⇒ ``sharpened`` (reported,
  SUCCESS-robust verdict). Under PARTIAL-ENGAGEMENT-A a would-be SUCCESS is
  ``SUCCESS-qualified``.
- **Cell 3 — INDETERMINATE:** ρ ≥ band with p ≥ 0.05 OR control fail.
- **Cell 4 — FALSIFIED:** ρ < band, attributed in the pre-registered order —
  (i) the identified-leakage ablation ρ bound, (ii) the cross-arm ρ/√r_yy
  attenuation comparison (reliability-mediated collapse reported BEFORE any
  construct attribution), (iii) the unflagged-item stability Spearman vs the
  FREE v2-vs-v2 split-half draw-noise reference (4/4 split, seed 763; the fixed
  0.8 threshold retained as a secondary descriptive marker only).

The v2 arm's LOCO predictions are NOT persisted in the committed reanchor
record, so this script REPRODUCES them deterministically from the committed
``E0_deception_v2.json`` + the frozen v0 shard via the fit machinery
(``issue763_fit_predictors._layer_sweep_select``, observed nested-CV path) and
HARD-ASSERTS the reproduced chosen-layer ρ against the committed
``rho_graded_ridge`` within ``--rho-tol`` (1e-6 default; up to 1e-3 allowed
cross-device with documentation, mirroring the binary-control convention). The
v2.1 arm reads the ``loco_predictions_graded_ridge`` field its fresh fit record
persists (``--recompute-v2p1-pred`` recomputes it instead — smoke path).

Figures (plan §-figures): the hero ``fig_763_v2p1_carveout_vs_v2.png`` (per-
context v2-vs-v2.1 scatter with flagged-containing contexts marked + headline ρ
bars v1/v2/v2-ablate/v2.1 with CIs + √r_yy ceilings) and the exploratory dump
``fig_763_v2p1_audit_exploratory.png``.

Usage::

    uv run python scripts/issue763_v2p1_verdict.py \
        --out-dir eval_results/issue_763/rubric-v2p1-disclosure-carveout
    # offline smoke (mock audit + ablation E0 standing in for the v2.1 arm):
    uv run python scripts/issue763_v2p1_verdict.py --smoke --skip-exactness-gate \
        --v2p1-e0 <ablate E0> --recompute-v2p1-pred ...
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# #847: shared-VM thread caps must bind BEFORE torch/numpy freeze their pools at import.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from issue763_common import (  # noqa: E402
    EVAL_RESULTS_DIR,
    FIGURE_DIR,
    SEED,
    dump_json,
    ensure_smoke_scope,
    load_json,
    reproducibility_metadata,
)

logger = logging.getLogger("issue763_v2p1_verdict")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REANCHOR_DIR = EVAL_RESULTS_DIR / "deception-rubric-reanchor"
V2P1_ROUND_DIR = EVAL_RESULTS_DIR / "rubric-v2p1-disclosure-carveout"

DROP_RATE_HALT_THRESHOLD = 0.20  # §6 cell 0
LEG_A_DROP_FLOOR = 30.0  # §6 cell 1 leg (a): flagged-item mean drop ≥ 30 points
LEG_B_RATE_FRACTION = 1.0 / 3.0  # §6 cell 1 leg (b): v2.1 rate < 1/3 of v2's
LEG_B_MIN_HIGH_DRAWS = 100  # §6 cell 1: minimum denominator for a defined rate
STABILITY_SECONDARY_MARKER = 0.8  # §6 cell 4(iii): descriptive marker only
# cell 4(ii): |ratio_v2 − ratio_v2.1| agreement. NOTE: 0.10 is a round-level
# operationalization of the plan's un-quantified "disattenuated ratios agree" —
# it steers the DESCRIPTIVE attribution string only (emitted as
# `ratios_agree_within`); the clean-result names this choice as a caveat.
ATTENUATION_RATIO_AGREE_TOL = 0.10


def _probe_sha256(probe: str) -> str:
    return hashlib.sha256(probe.encode("utf-8")).hexdigest()


def _per_probe_graded_map(e0: dict) -> dict[tuple[str, str], float | None]:
    """(context_id, probe_sha256) → the per-probe graded mean the fit consumes.

    Exemplar-excluded rows map to ``None`` (the fit never consumes them — the
    manipulation check runs on the SAME means, plan §6 cell 1 leg (a)).
    """
    out: dict[tuple[str, str], float | None] = {}
    for ctx_id, cell in e0["e0"]["deception"].items():
        for pr in cell["per_probe"]:
            out[(ctx_id, _probe_sha256(pr["probe"]))] = pr.get("graded")
    return out


def _drop_rate(e0: dict) -> dict:
    """Cell-0 read: draw-level kept/dropped totals + the drop rate for one arm."""
    kept = dropped = 0
    for cell in e0["e0"]["deception"].values():
        kept += int(cell.get("n_graded_draws_kept") or 0)
        dropped += int(cell.get("n_graded_draws_dropped") or 0)
    total = kept + dropped
    return {
        "n_draws_kept": kept,
        "n_draws_dropped": dropped,
        "n_draws_total": total,
        "drop_rate": (dropped / total) if total else None,
    }


def _ctx_graded_means(e0: dict) -> dict[str, float]:
    return {
        c: float(cell["graded_mean"])
        for c, cell in e0["e0"]["deception"].items()
        if cell.get("graded_mean") is not None
    }


def _rho(pred: np.ndarray, meas: np.ndarray) -> float | None:
    """Spearman ρ (None on degenerate input) — mirrors issue658_fit_predictors._rho."""
    from scipy.stats import spearmanr

    if len(pred) < 4 or np.std(pred) < 1e-9 or np.std(meas) < 1e-9:
        return None
    r, _ = spearmanr(pred, meas)
    return None if np.isnan(r) else float(r)


def _recompute_loco_pred(e0_path: Path) -> dict:
    """Reproduce the headline graded-ridge LOCO predictions from an E0 + frozen v0.

    Runs the SAME observed nested-CV path ``fit_behavior`` runs
    (``_layer_sweep_select(v0_kept, graded, n_judged, "ridge")``) — deterministic
    given the frozen inputs, so the reproduced chosen-layer ρ must match the
    committed record (asserted by the caller).
    """
    from issue763_fit_predictors import _e0_vectors, _layer_sweep_select, _load_v0

    e0 = load_json(e0_path)
    v0, ctx_ids = _load_v0("deception")
    graded, _rates, n_judged, _bw, _ppg, _ppb, kept = _e0_vectors(e0, "deception", ctx_ids)
    keep_idx = [ctx_ids.index(c) for c in kept]
    sel = _layer_sweep_select(v0[keep_idx], graded, n_judged, "ridge")
    return {
        "ctx_ids": list(kept),
        "y_graded": [float(v) for v in graded],
        "pred": [float(v) for v in np.asarray(sel["chosen_pred"]).reshape(-1)],
        "chosen_layer": int(sel["chosen_layer"]),
        "chosen_rho": sel["chosen_rho"],
    }


def paired_delta_rho_bootstrap(
    pred_a: np.ndarray,
    y_a: np.ndarray,
    pred_b: np.ndarray,
    y_b: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    """Paired cluster bootstrap of Δρ = ρ_a − ρ_b over the SHARED contexts.

    Arm a = v2.1 (results), arm b = v2 (reference); the four arrays are aligned
    on the SAME context order, and each draw resamples ONE index vector applied
    to both arms (the paired read — plan §6 cell 2). Degenerate draws (either
    arm's ρ None) are dropped and re-drawn, capped at 20 × n_boot attempts
    (mirrors ``_cluster_bootstrap_rho``'s convention).
    """
    n = len(y_a)
    assert len(pred_a) == len(pred_b) == len(y_b) == n, "arms not aligned"
    rng = random.Random(seed)
    deltas: list[float] = []
    attempts, max_attempts = 0, 20 * n_boot
    while len(deltas) < n_boot and attempts < max_attempts:
        attempts += 1
        idx = [rng.randrange(n) for _ in range(n)]
        ra = _rho(pred_a[idx], y_a[idx])
        rb = _rho(pred_b[idx], y_b[idx])
        if ra is None or rb is None:
            continue
        deltas.append(ra - rb)
    if len(deltas) < n_boot:
        raise RuntimeError(
            f"paired Δρ bootstrap accumulated only {len(deltas)}/{n_boot} valid draws "
            f"in {attempts} attempts — near-degenerate rank variation; do not ship a "
            "short CI silently"
        )
    arr = np.asarray(deltas, dtype=np.float64)
    lo, hi = (float(v) for v in np.percentile(arr, [2.5, 97.5]))
    return {
        "n_boot": n_boot,
        "seed": seed,
        "ci95": [lo, hi],
        "includes_zero": bool(lo <= 0.0 <= hi),
        "mean": float(arr.mean()),
        "draws_hist": [float(v) for v in np.percentile(arr, list(range(0, 101, 5)))],
        "draws": [float(v) for v in np.sort(arr)],
    }


def _split_half_noise_reference(
    rows_v2: list[dict],
    unflagged_items: set[tuple[str, str]],
    *,
    seed: int,
    n_splits_diag: int = 200,
) -> dict:
    """The FREE v2-vs-v2 split-half per-item Spearman over the unflagged items.

    Per unflagged item, the kept v2 draws are shuffled (seeded) and split in
    half (4/4 at full yield); the two half-means are correlated ACROSS items.
    The single registered split uses ``seed`` (plan § metrics: seed 763); a
    mean over ``n_splits_diag`` re-seeded splits rides along as a stability
    diagnostic (extra field, not the registered read).
    """
    by_item: dict[tuple[str, str], list[float]] = {}
    for r in rows_v2:
        if r.get("score") is None:
            continue
        key = (r["context_id"], r["probe_sha256"])
        if key not in unflagged_items:
            continue
        by_item.setdefault(key, []).append(float(r["score"]))

    def _one_split(split_seed: int) -> float | None:
        rng = random.Random(split_seed)
        first, second = [], []
        for _key, draws in sorted(by_item.items()):
            if len(draws) < 2:
                continue
            d = list(draws)
            rng.shuffle(d)
            half = len(d) // 2
            first.append(float(np.mean(d[:half])))
            second.append(float(np.mean(d[half:])))
        return _rho(np.asarray(first), np.asarray(second))

    single = _one_split(seed)
    diag = [v for v in (_one_split(seed + 1 + i) for i in range(n_splits_diag)) if v is not None]
    return {
        "n_items": len(by_item),
        "single_split_seed": seed,
        "noise_reference_single_split": single,
        "noise_reference_mean_over_splits": (float(np.mean(diag)) if diag else None),
        "n_splits_diag": n_splits_diag,
    }


# ── figures ───────────────────────────────────────────────────────────────────


def _palette(n: int) -> list[str]:
    try:
        from explore_persona_space.analysis.paper_plots import paper_palette

        return paper_palette(n)
    except Exception:
        return ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:n]


def _plot_hero(
    ctx_v2: dict[str, float],
    ctx_v2p1: dict[str, float],
    flagged_contexts: set[str],
    flagged_shift: list[dict],
    rho_bars: list[dict],
    verdict_note: str,
    out_path: Path,
) -> None:
    """The plan hero: v2-vs-v2.1 context-mean scatter + headline ρ bars.

    2×2 — (0,0) per-context graded means v2 (x) vs v2.1 (y), contexts containing
    flagged items marked; (0,1) flagged-item score shift v2 → v2.1; (1,0) headline
    ρ bars (v1 / v2 / v2-ablate / v2.1) with CIs + √r_yy ceilings; (1,1) verdict
    annotation.
    """
    cols = _palette(4)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0][0]
    shared = sorted(set(ctx_v2) & set(ctx_v2p1))
    xs = [ctx_v2[c] for c in shared]
    ys = [ctx_v2p1[c] for c in shared]
    marks = [c in flagged_contexts for c in shared]
    ax.scatter(
        [x for x, m in zip(xs, marks, strict=True) if not m],
        [y for y, m in zip(ys, marks, strict=True) if not m],
        s=22,
        color=cols[0],
        alpha=0.8,
        label="no flagged item",
    )
    ax.scatter(
        [x for x, m in zip(xs, marks, strict=True) if m],
        [y for y, m in zip(ys, marks, strict=True) if m],
        s=34,
        color=cols[3],
        marker="^",
        alpha=0.9,
        label="contains v2-flagged item",
    )
    lim = max(xs + ys + [1.0])
    ax.plot([0, lim], [0, lim], ls="--", color="gray", lw=1)
    ax.set_xlabel("per-context graded mean, rubric v2")
    ax.set_ylabel("per-context graded mean, rubric v2.1")
    ax.set_title("context means: v2 vs v2.1 (carve-out)", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[0][1]
    if flagged_shift:
        order = sorted(range(len(flagged_shift)), key=lambda i: flagged_shift[i]["v2"])
        for rank, i in enumerate(order):
            it = flagged_shift[i]
            ax.plot([it["v2"], it["v2p1"]], [rank, rank], color="lightgray", lw=1, zorder=1)
            ax.scatter([it["v2"]], [rank], color=cols[1], s=14, zorder=2)
            ax.scatter([it["v2p1"]], [rank], color=cols[2], s=14, zorder=2)
        ax.scatter([], [], color=cols[1], label="v2")
        ax.scatter([], [], color=cols[2], label="v2.1")
        ax.legend(fontsize=7)
        ax.set_xlabel("per-(context, probe) graded mean")
        ax.set_ylabel("flagged item (sorted by v2 score)")
    ax.set_title("v2-flagged items: score shift v2 → v2.1", fontsize=10)

    ax = axes[1][0]
    labels = [b["label"] for b in rho_bars]
    vals = [b["rho"] if b["rho"] is not None else 0.0 for b in rho_bars]
    ax.bar(labels, vals, color=_palette(len(rho_bars)))
    for i, b in enumerate(rho_bars):
        ci = b.get("ci")
        if ci and b["rho"] is not None:
            ax.errorbar(
                i,
                b["rho"],
                yerr=[[max(0.0, b["rho"] - ci[0])], [max(0.0, ci[1] - b["rho"])]],
                fmt="none",
                ecolor="black",
                capsize=3,
            )
        sq = b.get("sqrt_r_yy")
        if sq is not None:
            ax.plot([i - 0.4, i + 0.4], [sq, sq], ls="--", color="gray", lw=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("held-out ρ (graded DV, ridge)")
    ax.set_title("headline ρ + √r_yy ceilings (dashes)", fontsize=10)

    ax = axes[1][1]
    ax.text(0.5, 0.5, verdict_note, ha="center", va="center", transform=ax.transAxes, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("pre-registered verdict cells", fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_exploratory(
    e0_v2: dict,
    e0_v2p1: dict,
    rec_v2: dict,
    rec_v2p1: dict,
    audit: dict,
    delta_boot: dict | None,
    stability: dict,
    out_path: Path,
) -> None:
    """The §-figures exploratory dump for the carve-out round."""
    cols = _palette(4)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    ax = axes[0][0]
    for e0, tag, c in ((e0_v2, "v2", cols[0]), (e0_v2p1, "v2.1", cols[1])):
        vals = [
            pr["graded"]
            for cell in e0["e0"]["deception"].values()
            for pr in cell["per_probe"]
            if pr.get("graded") is not None
        ]
        ax.hist(vals, bins=25, range=(0, 100), alpha=0.55, label=tag, color=c)
    ax.set_title("pooled per-probe graded scores", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[0][1]
    for rec, tag, c in ((rec_v2, "v2", cols[0]), (rec_v2p1, "v2.1", cols[1])):
        curve = rec.get("per_layer_rho_graded_ridge") or []
        ys = [v if v is not None else float("nan") for v in curve]
        if ys:
            ax.plot(range(len(ys)), ys, label=tag, color=c, marker=".", ms=3)
    ax.set_title("per-layer ρ (graded ridge)", fontsize=9)
    ax.set_xlabel("layer")
    ax.legend(fontsize=7)

    ax = axes[0][2]
    for e0, tag, c in ((e0_v2, "v2", cols[0]), (e0_v2p1, "v2.1", cols[1])):
        means = list(_ctx_graded_means(e0).values())
        ax.hist(
            means,
            bins=20,
            range=(0, 100),
            alpha=0.55,
            label=f"{tag} (std {np.std(means):.2f})",
            color=c,
        )
    ax.set_title("per-context graded means (spread)", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1][0]
    by_v = audit.get("by_version") or {}
    labels, rates = [], []
    for rv in ("v2", "v2.1"):
        rec = by_v.get(rv) or {}
        labels.append(f"{rv}\n(n_high={rec.get('n_high_draws')})")
        rates.append(rec.get("flag_rate_among_high") or 0.0)
    ax.bar(labels, rates, color=[cols[0], cols[1]])
    ax.set_title("disclosure-flag rate among high draws", fontsize=9)
    ax.set_ylabel("flag rate")

    ax = axes[1][1]
    reads = [
        (
            "tracking v2",
            (e0_v2.get("judge_diagnostics") or {})
            .get("deception", {})
            .get("graded_binary_tracking_spearman"),
        ),
        (
            "tracking v2.1",
            (e0_v2p1.get("judge_diagnostics") or {})
            .get("deception", {})
            .get("graded_binary_tracking_spearman"),
        ),
        ("unflagged stability", stability.get("unflagged_item_spearman")),
        ("noise ref (split-half)", stability.get("noise_reference_single_split")),
    ]
    ax.bar(
        [r[0] for r in reads],
        [r[1] if r[1] is not None else 0.0 for r in reads],
        color=_palette(len(reads)),
    )
    ax.axhline(STABILITY_SECONDARY_MARKER, ls=":", color="gray", lw=1, label="0.8 marker")
    ax.set_title("stability reads", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=7)
    ax.legend(fontsize=7)

    ax = axes[1][2]
    if delta_boot is not None:
        ax.hist(delta_boot["draws"], bins=40, color=cols[1], alpha=0.8)
        lo, hi = delta_boot["ci95"]
        ax.axvline(lo, ls="--", color="black", lw=1)
        ax.axvline(hi, ls="--", color="black", lw=1)
        ax.axvline(0.0, color="red", lw=1)
        ax.set_title(
            f"paired Δρ bootstrap (CI [{lo:.3f}, {hi:.3f}], "
            f"{'includes' if delta_boot['includes_zero'] else 'excludes'} 0)",
            fontsize=9,
        )
        ax.set_xlabel("Δρ = ρ_v2.1 − ρ_v2 per resample")
    else:
        ax.set_title("paired Δρ bootstrap (unavailable)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── the decision computation ─────────────────────────────────────────────────


def cell1_verdict(leg_a_pass: bool, leg_b_pass: bool | None, rate_defined: bool) -> str:
    """The §6 cell-1 manipulation-check 2×2 mapping (pure; unit-tested).

    Undefined leg-(b) rate (below the ≥100 high-draw minimum denominator)
    collapses to leg (a): pass ⇒ ENGAGED-by-collapse, fail ⇒
    PARTIAL-ENGAGEMENT-B (plan §6 cell 1, pre-registered).
    """
    if not rate_defined:
        return "ENGAGED-by-collapse" if leg_a_pass else "PARTIAL-ENGAGEMENT-B"
    if leg_a_pass and leg_b_pass:
        return "ENGAGED"
    if not leg_a_pass and not leg_b_pass:
        return "KILL"
    if leg_a_pass and not leg_b_pass:
        return "PARTIAL-ENGAGEMENT-A"
    return "PARTIAL-ENGAGEMENT-B"


def compute_verdict(  # the §6 enumeration is one cohesive decision tree
    *,
    e0_v2: dict,
    e0_v2p1: dict,
    audit: dict,
    rec_v2p1: dict,
    rec_ablate: dict,
    rec_v2_ref: dict,
    loco_v2: dict,
    loco_v2p1: dict,
    rows_v2: list[dict],
    n_boot: int,
    seed: int,
) -> dict:
    """Evaluate the COMPLETE §6/§9c cell enumeration; returns the verdict blob."""
    band_lo = float(rec_v2_ref["rho_graded_ridge_ci"][0])
    band_hi = float(rec_v2_ref["rho_graded_ridge_ci"][1])

    # ── cell 0: instrument health ──
    dr_v2 = _drop_rate(e0_v2)
    dr_v2p1 = _drop_rate(e0_v2p1)
    # plan §6 cell 0: PER-ARM drop-rate gate (the v2 arm is the frozen committed
    # artifact at 1.78%, so gating it too is constant-equivalent in production).
    cell0_pass = all(
        dr["drop_rate"] is not None and dr["drop_rate"] <= DROP_RATE_HALT_THRESHOLD
        for dr in (dr_v2, dr_v2p1)
    )
    cell0 = {
        "v2": dr_v2,
        "v2.1": dr_v2p1,
        "threshold": DROP_RATE_HALT_THRESHOLD,
        "pass": bool(cell0_pass),
        "note": "post-retry FINAL state; the ONE re-dispatch retry is the judge phase's",
    }

    # ── cell 1: manipulation check ──
    by_v = audit["by_version"]
    flagged_items = {(c, s) for c, s in (audit.get("flagged_items_v2") or [])}
    pp_v2 = _per_probe_graded_map(e0_v2)
    pp_v2p1 = _per_probe_graded_map(e0_v2p1)
    drops, skipped = [], 0
    flagged_shift: list[dict] = []
    for item in sorted(flagged_items):
        a, b = pp_v2.get(item), pp_v2p1.get(item)
        if a is None or b is None:
            skipped += 1
            continue
        drops.append(a - b)
        flagged_shift.append({"context_id": item[0], "probe_sha256": item[1], "v2": a, "v2p1": b})
    leg_a_mean_drop = float(np.mean(drops)) if drops else None
    leg_a_pass = leg_a_mean_drop is not None and leg_a_mean_drop >= LEG_A_DROP_FLOOR
    leg_a = {
        "n_flagged_items": len(flagged_items),
        "n_scored": len(drops),
        "n_skipped_none": skipped,
        "mean_drop": leg_a_mean_drop,
        "floor": LEG_A_DROP_FLOOR,
        "pass": bool(leg_a_pass),
    }

    rate_v2 = by_v["v2"].get("flag_rate_among_high")
    v2p1_rec = by_v.get("v2.1") or {}
    n_high_v2p1 = v2p1_rec.get("n_high_draws")
    rate_v2p1 = v2p1_rec.get("flag_rate_among_high")
    rate_defined = n_high_v2p1 is not None and n_high_v2p1 >= LEG_B_MIN_HIGH_DRAWS
    leg_b_pass = None
    if rate_defined and rate_v2 is not None and rate_v2p1 is not None:
        leg_b_pass = rate_v2p1 < LEG_B_RATE_FRACTION * rate_v2
    leg_b = {
        "rate_v2": rate_v2,
        "rate_v2p1": rate_v2p1,
        "n_high_v2p1": n_high_v2p1,
        "min_denominator": LEG_B_MIN_HIGH_DRAWS,
        "rate_defined": bool(rate_defined),
        "fraction": LEG_B_RATE_FRACTION,
        "pass": leg_b_pass,
    }

    cell1_v = cell1_verdict(bool(leg_a_pass), leg_b_pass, bool(rate_defined))
    cell1 = {"leg_a": leg_a, "leg_b": leg_b, "verdict": cell1_v}

    # ── the paired Δρ read (cell 2's stability split) ──
    ctx_a = list(loco_v2p1["ctx_ids"])
    ctx_b = list(loco_v2["ctx_ids"])
    assert set(ctx_a) == set(ctx_b), (
        f"arms disagree on kept contexts: {sorted(set(ctx_a) ^ set(ctx_b))[:5]}"
    )
    order = {c: i for i, c in enumerate(ctx_a)}
    perm = [order[c] for c in ctx_b]  # align arm a onto arm b's order
    pred_a = np.asarray(loco_v2p1["pred"], dtype=np.float64)[perm]
    y_a = np.asarray(loco_v2p1["y_graded"], dtype=np.float64)[perm]
    pred_b = np.asarray(loco_v2["pred"], dtype=np.float64)
    y_b = np.asarray(loco_v2["y_graded"], dtype=np.float64)
    delta_boot = paired_delta_rho_bootstrap(pred_a, y_a, pred_b, y_b, n_boot=n_boot, seed=seed)
    rho_v2p1 = rec_v2p1.get("rho_graded_ridge")
    rho_v2 = rec_v2_ref.get("rho_graded_ridge")
    delta = {
        "delta_point": (rho_v2p1 - rho_v2)
        if (rho_v2p1 is not None and rho_v2 is not None)
        else None,
        **{k: v for k, v in delta_boot.items() if k != "draws"},
        "_draws": delta_boot["draws"],  # figure input; stripped before dump
    }

    # ── cells 2-4 ──
    p = rec_v2p1.get("shuffle_null_p")
    control = bool(rec_v2p1.get("control_task_pass"))
    meets_band = rho_v2p1 is not None and rho_v2p1 >= band_lo
    significant = p is not None and p < 0.05
    sharpened = rho_v2p1 is not None and rho_v2p1 > band_hi
    cell2 = {
        "rho_v2p1": rho_v2p1,
        "rho_v2p1_ci": rec_v2p1.get("rho_graded_ridge_ci"),
        "band": [band_lo, band_hi],
        "meets_band": bool(meets_band),
        "shuffle_null_p": p,
        "significant": bool(significant),
        "control_task_pass": control,
        "sharpened": bool(sharpened),
        "delta_rho_paired": {k: v for k, v in delta.items() if k != "_draws"},
    }

    # stability / attribution reads (always computed; binding on cell 4)
    unflagged_items = {k for k, v in pp_v2.items() if v is not None and k not in flagged_items} & {
        k for k, v in pp_v2p1.items() if v is not None
    }
    xs = np.asarray([pp_v2[k] for k in sorted(unflagged_items)], dtype=np.float64)
    ys = np.asarray([pp_v2p1[k] for k in sorted(unflagged_items)], dtype=np.float64)
    unflagged_spearman = _rho(xs, ys)
    noise = _split_half_noise_reference(rows_v2, unflagged_items, seed=seed)
    stability = {
        "n_unflagged_items": len(unflagged_items),
        "unflagged_item_spearman": unflagged_spearman,
        **noise,
        "secondary_marker_0p8": STABILITY_SECONDARY_MARKER,
        "below_noise_reference": (
            None
            if (unflagged_spearman is None or noise["noise_reference_single_split"] is None)
            else bool(unflagged_spearman < noise["noise_reference_single_split"])
        ),
        "below_0p8_marker": (
            None if unflagged_spearman is None else bool(unflagged_spearman < 0.8)
        ),
    }
    sq_v2 = rec_v2_ref.get("sqrt_r_yy")
    sq_v2p1 = rec_v2p1.get("sqrt_r_yy")
    ratio_v2 = (rho_v2 / sq_v2) if (rho_v2 is not None and sq_v2) else None
    ratio_v2p1 = (rho_v2p1 / sq_v2p1) if (rho_v2p1 is not None and sq_v2p1) else None
    attenuation = {
        "rho_over_sqrt_r_yy_v2": ratio_v2,
        "rho_over_sqrt_r_yy_v2p1": ratio_v2p1,
        "sqrt_r_yy_v2": sq_v2,
        "sqrt_r_yy_v2_ci": rec_v2_ref.get("sqrt_r_yy_ci"),
        "sqrt_r_yy_v2p1": sq_v2p1,
        "sqrt_r_yy_v2p1_ci": rec_v2p1.get("sqrt_r_yy_ci"),
        "ratios_agree_within": ATTENUATION_RATIO_AGREE_TOL,
        "reliability_mediated": (
            None
            if (ratio_v2 is None or ratio_v2p1 is None)
            else bool(abs(ratio_v2 - ratio_v2p1) <= ATTENUATION_RATIO_AGREE_TOL)
        ),
    }
    rho_ablate = rec_ablate.get("rho_graded_ridge")
    cell4 = {
        "active": bool(not meets_band),
        "ablation_rho": rho_ablate,
        "ablation_rho_ci": rec_ablate.get("rho_graded_ridge_ci"),
        "ablation_bound_note": (
            "what removing the IDENTIFIED flagged draws alone explains (expected ≈ 0.51)"
        ),
        "attenuation": attenuation,
        "stability": stability,
        "attribution_order": [
            "ablation-bound",
            "attenuation-comparison",
            "noise-referenced-stability",
        ],
    }

    # ── the final cell decision (the §9c enumeration, exactly one named cell) ──
    if not cell0_pass:
        final = "INSTRUMENT-INCONCLUSIVE-HALT"
        attribution = None
    elif cell1_v == "KILL":
        final = "KILL"
        attribution = None
    elif cell1_v == "PARTIAL-ENGAGEMENT-B":
        final = "PARTIAL-ENGAGEMENT-B"
        attribution = None
    else:
        qualified = cell1_v == "PARTIAL-ENGAGEMENT-A"
        if meets_band and significant and control:
            # Plan §6 cell 2: Δρ CI includes 0 ⇒ robust; excludes 0 ⇒ attenuated —
            # EXCEPT the pre-registered sharpened cell "(ρ > 0.713 = sharpened;
            # report, SUCCESS-robust verdict.)": ρ_v2.1 above the band ceiling is a
            # sharpened read, never an attenuation (a CI exclusion there is v2.1
            # BEATING v2, exposed via the emitted `sharpened` + Δρ fields).
            sub = (
                "SUCCESS-robust"
                if (delta_boot["includes_zero"] or sharpened)
                else "SUCCESS-attenuated"
            )
            final = f"SUCCESS-qualified ({sub})" if qualified else sub
            attribution = None
        elif meets_band:
            final = "INDETERMINATE" + (" (instrument-partial)" if qualified else "")
            attribution = None
        else:
            # cell 4 attribution, pre-registered order
            if attenuation["reliability_mediated"]:
                attribution = "reliability-mediated (longer prompt degraded the instrument)"
            elif stability["below_noise_reference"] is False:
                attribution = (
                    "targeted change — broader same-construct leakage the audit undercounted"
                )
            elif stability["below_noise_reference"] is True:
                attribution = (
                    "broad destabilization candidate (unflagged stability below the draw-noise "
                    "reference; materiality is the analyzer's read)"
                )
            else:
                attribution = "undetermined (stability reads unavailable)"
            final = "FALSIFIED" + (" (instrument-partial)" if qualified else "")
    cell4["attribution"] = attribution

    return {
        "cell0_instrument_health": cell0,
        "cell1_manipulation_check": cell1,
        "cell2_success": cell2,
        # §6 cell 3 named explicitly: point estimate survives the band but the
        # null/control machinery does not certify it (active iff so).
        "cell3_indeterminate": {
            "active": bool(meets_band and not (significant and control)),
            "reason": (
                None
                if not meets_band
                else (
                    "shuffle p >= 0.05"
                    if not significant
                    else ("control-task fail" if not control else None)
                )
            ),
        },
        "cell4_falsified": cell4,
        "stability_reads": stability,
        "final_verdict": final,
        "_flagged_shift": flagged_shift,  # figure input; stripped before dump
        "_delta_draws": delta.pop("_draws"),
    }


def assert_v2_loco_reproduction(
    repro: float | None,
    committed: float,
    *,
    tol: float,
    chosen_layer: int | None = None,
    ref_layer: int | None = None,
) -> None:
    """The v2-arm LOCO-reproduction hard gate (raises RuntimeError on mismatch).

    The paired Δρ read is trusted ONLY if the deterministic refit reproduces the
    committed reanchor ``rho_graded_ridge`` within ``tol`` (a ``None`` recompute
    also fails). Pure so the mismatch path is unit-testable
    (``test_v2_loco_reproduction_gate_raises_on_mismatch``).
    """
    if repro is None or abs(repro - committed) > tol:
        raise RuntimeError(
            f"v2 LOCO reproduction FAILED: recomputed chosen ρ {repro!r} vs committed "
            f"{committed:.10f} (|Δ| > tol {tol:g}; layer "
            f"{chosen_layer} vs {ref_layer}) — do NOT "
            "trust the paired Δρ read; investigate device/input drift (tol up to 1e-3 "
            "is pre-registered for cross-device, pass --rho-tol)"
        )


def main() -> int:  # one cohesive verdict assembly
    ap = argparse.ArgumentParser(description="Issue #763 v2p1: §6/§9c verdict cells + figures.")
    ap.add_argument("--out-dir", type=Path, default=V2P1_ROUND_DIR)
    ap.add_argument(
        "--v2p1-e0", type=Path, default=None, help="default <out-dir>/E0_deception_v2p1.json"
    )
    ap.add_argument("--v2-e0", type=Path, default=REANCHOR_DIR / "E0_deception_v2.json")
    ap.add_argument(
        "--audit-json", type=Path, default=None, help="default <out-dir>/disclosure_flag_audit.json"
    )
    ap.add_argument(
        "--v2p1-fit",
        type=Path,
        default=None,
        help="default <out-dir>/fit_by_behavior/deception.json",
    )
    ap.add_argument(
        "--ablate-fit",
        type=Path,
        default=None,
        help="default <out-dir>/ablate/fit_by_behavior/deception.json",
    )
    ap.add_argument(
        "--v2-fit-ref", type=Path, default=REANCHOR_DIR / "fit_by_behavior" / "deception.json"
    )
    ap.add_argument(
        "--v1-fit-ref",
        type=Path,
        default=EVAL_RESULTS_DIR / "fit_by_behavior" / "deception.json",
        help="parent v1 record (hero ρ bar only)",
    )
    ap.add_argument(
        "--v2-shards",
        type=Path,
        default=REANCHOR_DIR / "raw_completions" / "judge_reanchor_v2",
        help="v2 per-draw shards (the split-half noise reference input)",
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--rho-tol",
        type=float,
        default=1e-6,
        help="v2 LOCO-reproduction assert tolerance (up to 1e-3 cross-device, documented)",
    )
    ap.add_argument(
        "--recompute-v2p1-pred",
        action="store_true",
        help="recompute the v2.1 arm's LOCO predictions from its E0 instead of reading the "
        "fit record's loco_predictions_graded_ridge (smoke path)",
    )
    ap.add_argument(
        "--skip-exactness-gate",
        action="store_true",
        help="skip assert_matches_reference before the LOCO recompute (offline smoke only)",
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ensure_smoke_scope(args.smoke)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    import torch

    torch.set_num_threads(max(1, int(os.environ.get("EPM_FIT_NUM_THREADS", "8"))))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    v2p1_e0_path = args.v2p1_e0 or (out_dir / "E0_deception_v2p1.json")
    audit_path = args.audit_json or (out_dir / "disclosure_flag_audit.json")
    v2p1_fit_path = args.v2p1_fit or (out_dir / "fit_by_behavior" / "deception.json")
    ablate_fit_path = args.ablate_fit or (out_dir / "ablate" / "fit_by_behavior" / "deception.json")

    e0_v2 = load_json(args.v2_e0)
    e0_v2p1 = load_json(v2p1_e0_path)
    audit = load_json(audit_path)
    rec_v2p1 = load_json(v2p1_fit_path)
    rec_ablate = load_json(ablate_fit_path)
    rec_v2_ref = load_json(args.v2_fit_ref)
    rec_v1_ref = load_json(args.v1_fit_ref) if args.v1_fit_ref.exists() else {}

    if not args.skip_exactness_gate:
        from issue763_fit_predictors import FIT_DEVICE

        from explore_persona_space.analysis.issue_763_vectorized import assert_matches_reference

        gate = assert_matches_reference(device=FIT_DEVICE)
        logger.info("[verdict] vectorized-exactness gate PASS (device=%s): %s", FIT_DEVICE, gate)

    # ── v2 arm: deterministic LOCO reproduction + hard assert vs the committed ρ ──
    logger.info("[verdict] reproducing the v2 arm's LOCO predictions from %s", args.v2_e0)
    loco_v2 = _recompute_loco_pred(args.v2_e0)
    committed = float(rec_v2_ref["rho_graded_ridge"])
    repro = loco_v2["chosen_rho"]
    assert_v2_loco_reproduction(
        repro,
        committed,
        tol=args.rho_tol,
        chosen_layer=loco_v2["chosen_layer"],
        ref_layer=rec_v2_ref.get("chosen_layer"),
    )
    logger.info(
        "[verdict] v2 reproduction PASS: ρ %.10f vs committed %.10f (layer %d)",
        repro,
        committed,
        loco_v2["chosen_layer"],
    )

    # ── v2.1 arm: the fresh fit record's persisted predictions (or recompute) ──
    if args.recompute_v2p1_pred:
        logger.info("[verdict] recomputing the v2.1 arm's LOCO predictions from %s", v2p1_e0_path)
        loco_v2p1 = _recompute_loco_pred(v2p1_e0_path)
    else:
        loco_v2p1 = rec_v2p1.get("loco_predictions_graded_ridge")
        if not loco_v2p1:
            raise RuntimeError(
                f"{v2p1_fit_path} lacks loco_predictions_graded_ridge — refit with the "
                "current branch (fresh fits persist the headline LOCO predictions), or "
                "pass --recompute-v2p1-pred"
            )

    from issue763_disclosure_flag_audit import _load_draw_rows

    rows_v2 = _load_draw_rows(args.v2_shards, "v2")

    verdict = compute_verdict(
        e0_v2=e0_v2,
        e0_v2p1=e0_v2p1,
        audit=audit,
        rec_v2p1=rec_v2p1,
        rec_ablate=rec_ablate,
        rec_v2_ref=rec_v2_ref,
        loco_v2=loco_v2,
        loco_v2p1=loco_v2p1,
        rows_v2=rows_v2,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    flagged_shift = verdict.pop("_flagged_shift")
    delta_draws = verdict.pop("_delta_draws")

    # ── figures ──
    flagged_contexts = {c for c, _s in (audit.get("flagged_items_v2") or [])}
    delta_read = verdict["cell2_success"]["delta_rho_paired"]
    rho_bars = [
        {
            "label": "v1",
            "rho": rec_v1_ref.get("rho_graded_ridge"),
            "ci": rec_v1_ref.get("rho_graded_ridge_ci"),
            "sqrt_r_yy": rec_v1_ref.get("sqrt_r_yy"),
        },
        {
            "label": "v2",
            "rho": rec_v2_ref.get("rho_graded_ridge"),
            "ci": rec_v2_ref.get("rho_graded_ridge_ci"),
            "sqrt_r_yy": rec_v2_ref.get("sqrt_r_yy"),
        },
        {
            "label": "v2-ablate",
            "rho": rec_ablate.get("rho_graded_ridge"),
            "ci": rec_ablate.get("rho_graded_ridge_ci"),
            "sqrt_r_yy": rec_ablate.get("sqrt_r_yy"),
        },
        {
            "label": "v2.1",
            "rho": rec_v2p1.get("rho_graded_ridge"),
            "ci": rec_v2p1.get("rho_graded_ridge_ci"),
            "sqrt_r_yy": rec_v2p1.get("sqrt_r_yy"),
        },
    ]
    verdict_note = (
        f"final: {verdict['final_verdict']}\n"
        f"cell0 drop-rate v2.1: {verdict['cell0_instrument_health']['v2.1']['drop_rate']}\n"
        f"cell1: {verdict['cell1_manipulation_check']['verdict']}\n"
        f"leg(a) mean drop: {verdict['cell1_manipulation_check']['leg_a']['mean_drop']}\n"
        f"leg(b) rates v2/v2.1: {verdict['cell1_manipulation_check']['leg_b']['rate_v2']} / "
        f"{verdict['cell1_manipulation_check']['leg_b']['rate_v2p1']}\n"
        f"ρ_v2.1: {verdict['cell2_success']['rho_v2p1']} "
        f"(band ≥ {verdict['cell2_success']['band'][0]:.4f})\n"
        f"paired Δρ CI: {delta_read['ci95']} "
        f"({'includes' if delta_read['includes_zero'] else 'excludes'} 0)"
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    hero_path = FIGURE_DIR / "fig_763_v2p1_carveout_vs_v2.png"
    _plot_hero(
        _ctx_graded_means(e0_v2),
        _ctx_graded_means(e0_v2p1),
        flagged_contexts,
        flagged_shift,
        rho_bars,
        verdict_note,
        hero_path,
    )
    _plot_exploratory(
        e0_v2,
        e0_v2p1,
        rec_v2_ref,
        rec_v2p1,
        audit,
        {**delta_read, "draws": delta_draws},
        verdict["stability_reads"],
        FIGURE_DIR / "fig_763_v2p1_audit_exploratory.png",
    )
    assert hero_path.exists(), "hero figure not written"

    out = {
        "round": "rubric-v2p1-disclosure-carveout",
        "smoke": bool(args.smoke),
        "inputs": {
            "v2_e0": str(args.v2_e0),
            "v2p1_e0": str(v2p1_e0_path),
            "audit_json": str(audit_path),
            "v2p1_fit": str(v2p1_fit_path),
            "ablate_fit": str(ablate_fit_path),
            "v2_fit_ref": str(args.v2_fit_ref),
            "audit_prompt_sha256": audit.get("audit_prompt_sha256"),
            "graded_prompt_hash_v2p1": (e0_v2p1.get("graded_prompt_hash") or {}).get("deception"),
        },
        "v2_loco_reproduction": {
            "chosen_rho": loco_v2["chosen_rho"],
            "committed_rho": committed,
            "abs_delta": abs(loco_v2["chosen_rho"] - committed),
            "tol": args.rho_tol,
            "chosen_layer": loco_v2["chosen_layer"],
        },
        "loco_predictions": {"v2": loco_v2, "v2.1": loco_v2p1},
        **verdict,
        "figures": {
            "hero": str(hero_path),
            "exploratory": str(FIGURE_DIR / "fig_763_v2p1_audit_exploratory.png"),
        },
        "metadata": reproducibility_metadata({"phase": "v2p1_verdict"}),
    }
    dump_json(out, out_dir / "verdict_v2p1.json")
    print(
        f"[issue763.v2p1_verdict] final_verdict={verdict['final_verdict']} "
        f"cell1={verdict['cell1_manipulation_check']['verdict']} "
        f"rho_v2p1={verdict['cell2_success']['rho_v2p1']} "
        f"delta_ci={verdict['cell2_success']['delta_rho_paired']['ci95']} "
        f"-> {out_dir / 'verdict_v2p1.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
