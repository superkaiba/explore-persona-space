#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# (math/scientific notation — ρ, Δv, σ — intentional in labels/docstrings)
"""Issue #683 Phase D — marker-vs-sycophancy contrast + per-behavior figures (CPU).

Plan §4 Phase D + §6.5. Reads the Phase-C leaderboards + Phase-B A7 reports +
the standalone noise_floor.json (all written by issue683_key_ablation_score.py /
issue683_a7_precondition.py) and writes the plan's Phase-D figures under
``figures/issue_683/*.png`` (primary_deliverable §6.5):

  1. ``leaderboard_<behavior>`` — per-behavior held-out Spearman ρ bar chart per
     key×metric, with bootstrap CIs, the shuffled-key null band (shaded), and the
     test-retest noise-floor line. The displacement (idiolect-control) key is
     shown BOTH ways — identity-ψ (assumption-light raw dot) AND learned-ridge ψ
     — so the silent "best ψ" substitution is disclosed.
  2. ``marker_vs_sycophancy_contrast`` — the HERO. The scored DV is the dominant
     low-rank component g₁ (A7 failed for both behaviors). The marker panel pairs
     the bars WITHIN A SINGLE SOURCE BANK (A5 — the only marker bank whose
     training-completion key clears its own shuffled-key null), so the two marker
     bars + their null all come from the same bank (no cross-bank max artifact).
     Sycophancy keeps best-of-2-seeds (both seeds agree, so no selection artifact).
  3. ``a7_spectrum_<behavior>`` — the A7 precondition SVD spectrum bar
     (σ₁²/Σσ², σ₂/σ₁, cos(u₁,ŵ)) + scalarity residual + strict-rank-1 pass count,
     with a reader-facing verdict subtitle.

CPU-only, off-pod (no GPU, no network). Plain-English labels throughout
(paper-plots Lens 3 / interpretation-critic Lens 6); uses the project paper
style + colorblind-safe palette.

CLI:
    uv run python scripts/issue683_make_figures.py
    # smoke (read the smoke leaderboards, write to a smoke figures dir):
    uv run python scripts/issue683_make_figures.py \
        --marker-leaderboard eval_results/issue_683/smoke/leaderboard_marker_smoke.json \
        --out-dir figures/issue_683/smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="issue683_make_figures")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import the SCORER module so the per-context low-level-data scatters are
# recomputed via the EXACT same code path the leaderboard ρ came from (no
# re-run of the committed JSONs, no new science — the recomputed per-context
# (g_pred, g₁) pairs reproduce the committed aggregate ρ to machine precision;
# the headline cells shown are M_I/ψ_I raw-dot, which are target-independent and
# need no LOO/whitening/ridge). This satisfies the "low-level data behind every
# aggregate" rule (SPEC §11) at 0 GPU from the committed Δv/c-bank/t_cb tensors.
_SCORER_PATH = PROJECT_ROOT / "scripts/issue683_key_ablation_score.py"
_scorer_spec = importlib.util.spec_from_file_location("issue683_scorer", _SCORER_PATH)
_scorer = importlib.util.module_from_spec(_scorer_spec)
_scorer_spec.loader.exec_module(_scorer)

# Committed marker c_C' bank (the #604 post-response-slot all-layers bank sliced
# at L14) is cached here; the sycophancy c_C' bank lives under analysis_tensors.
_MARKER_CBANK = PROJECT_ROOT / ".claude/cache/issue604_marker_cbank_L14.pt"
_SYCO_CBANK = (
    PROJECT_ROOT
    / "eval_results/issue_683/analysis_tensors/c_bank/sycophancy/c_bank_sycophancy_L20.pt"
)
_DV_DIR = PROJECT_ROOT / "eval_results/issue_683/analysis_tensors/dv"
_TCB_DIR = PROJECT_ROOT / "eval_results/issue_683/analysis_tensors/t_cb"
_READ_LAYER = {"marker": 14, "sycophancy": 20}

# Plain-English key/metric labels (paper-plots: never bare slugs in figures).
KEY_LABELS = {
    "k_cC": "Context-only key",
    "k_tCB": "Training-completion key",
    "k_cC_plus_delta": "Context + displacement key",
}
METRIC_LABELS = {"M_I": "raw dot", "M_white": "whitened"}
PSI_LABELS = {"psi_I": "identity ψ", "psi_ridge": "ridge ψ"}
BEHAVIOR_LABELS = {"marker": "Marker (control)", "sycophancy": "Sycophancy (transfer)"}

# The scored DV. A7 failed for both behaviors, so the scorer scored keys against
# the dominant low-rank SVD component g₁, NOT the scalar realized gate g_real.
# SCORED_DV_LABEL is the full form (titles/captions); SCORED_DV_AXIS is the
# compact form for y-axis labels (the full form clips the left canvas edge).
SCORED_DV_LABEL = "g₁ (dominant low-rank component)"
SCORED_DV_AXIS = "g₁ (dominant component)"


def _best_rows_per_key(per_bank: list[dict]) -> dict[tuple[str, str], dict]:
    """Best (highest finite Spearman) row per (key, metric) across all banks.

    Aggregates over seeds by keeping the bank-best row per (key, metric); the
    figure shows the cross-seed best with its own bootstrap CI (the analyzer
    reads the full per-bank table for the cross-seed range).
    """
    best: dict[tuple[str, str], dict] = {}
    for bank in per_bank:
        for row in bank.get("leaderboard", []):
            rho = row.get("spearman")
            if rho is None or rho != rho:
                continue
            kk = (row["key"], row["metric"])
            if kk not in best or rho > best[kk]["spearman"]:
                best[kk] = row
    return best


def _best_rows_per_key_psi(per_bank: list[dict]) -> dict[tuple[str, str, str], dict]:
    """Best (highest finite Spearman) row per (key, metric, psi) across all banks.

    Unlike ``_best_rows_per_key``, this keeps the ψ axis EXPLICIT so the
    leaderboard figure can show the displacement (idiolect-control) key under
    BOTH the assumption-light identity ψ AND the overfitting-prone learned-ridge
    ψ — disclosing the silent "best ψ per key" substitution (the displacement
    key's identity-ψ raw dot is negative; only ridge ψ turns it positive).
    """
    best: dict[tuple[str, str, str], dict] = {}
    for bank in per_bank:
        for row in bank.get("leaderboard", []):
            rho = row.get("spearman")
            if rho is None or rho != rho:
                continue
            kk = (row["key"], row["metric"], row["psi"])
            if kk not in best or rho > best[kk]["spearman"]:
                best[kk] = row
    return best


def _null_band(per_bank: list[dict]) -> tuple[float, float]:
    """Mean [p5, p95] shuffled-key null band across banks (for the shaded band)."""
    lo = [b["null_shuffled_key"]["p5"] for b in per_bank if b.get("null_shuffled_key")]
    hi = [b["null_shuffled_key"]["p95"] for b in per_bank if b.get("null_shuffled_key")]
    import numpy as np

    lo = [x for x in lo if x == x]
    hi = [x for x in hi if x == x]
    if not lo or not hi:
        return (float("nan"), float("nan"))
    return (float(np.mean(lo)), float(np.mean(hi)))


def _noise_floor_mean(noise_floor: dict) -> float:
    """Mean test-retest Spearman across sources (the achievable-ρ ceiling line)."""
    import numpy as np

    per_source = noise_floor.get("per_source", noise_floor) if noise_floor else {}
    vals = [
        v["test_retest_spearman_mean"]
        for v in per_source.values()
        if isinstance(v, dict)
        and v.get("test_retest_spearman_mean") == v.get("test_retest_spearman_mean")
    ]
    return float(np.mean(vals)) if vals else float("nan")


def _cbank_path(behavior: str) -> Path:
    return _MARKER_CBANK if behavior == "marker" else _SYCO_CBANK


def _per_context_series(behavior: str, source: str, key_form: str) -> dict | None:
    """Recompute the per-context (g_pred, scored-DV g₁, context names) for ONE
    M_I/ψ_I raw-dot cell, faithfully via the scorer code path.

    Reads the committed Δv / c_C' / t_{C,B} tensors and reproduces the leaderboard
    ρ for this cell to machine precision (validated: marker A5 + sycophancy seed-42
    match the committed values to 1e-9). Returns ``None`` if the inputs for this
    (behavior, source) are not present locally. The M_I/ψ_I raw-dot key is
    target-independent, so the all-context Spearman here EQUALS the committed
    leave-one-context-out Spearman for the cell (no LOO/whitening/ridge needed)."""
    import numpy as np
    import torch

    sc = _scorer
    cbp = _cbank_path(behavior)
    dv_dir = _DV_DIR / behavior
    if not cbp.is_file() or not dv_dir.is_dir():
        return None
    layer = _READ_LAYER[behavior]
    banks = sc._load_dv_banks(dv_dir)
    payload = next((b for b in banks if b["source"] == source), None)
    if payload is None:
        return None
    c_bank = sc._load_c_bank(cbp, layer)
    t_cb = sc._load_tcb(_TCB_DIR / behavior, behavior, source, layer)
    per_context = payload["per_context"]
    targets = [c for c in per_context if c != source and c in c_bank]
    if len(targets) < 3:
        return None
    # Low-rank fallback dominant direction u₁ (A7 failed for both behaviors).
    dvs = np.stack(
        [torch.as_tensor(per_context[c]["Delta_v"]).flatten().float().numpy() for c in targets],
        axis=1,
    )
    u1 = np.linalg.svd(dvs, full_matrices=False)[0][:, 0]
    c_source = c_bank[source]
    v_base_source = torch.as_tensor(per_context[source]["v_base"]).flatten().float().numpy()
    delta_cb = (t_cb - v_base_source) if t_cb is not None else None
    resolved = sc._resolve_cell_keys(
        psi="psi_I",
        key_form=key_form,
        psi_per_fold=None,
        c_source=c_source,
        t_cb=t_cb,
        delta_cb=delta_cb,
        targets=targets,
    )
    if resolved is None or resolved[0] is None:
        return None
    k = resolved[0]
    y = np.array([sc._dv_target_value(payload, c, False, u1) for c in targets])
    g_pred = np.array([sc._g_pred(k, None, c_bank[c], c_source) for c in targets])
    f = np.isfinite(g_pred) & np.isfinite(y)
    if f.sum() < 3:
        return None
    return {
        "contexts": [c for c, keep in zip(targets, f, strict=True) if keep],
        "g_pred": g_pred[f],
        "g1": y[f],
        "spearman": sc.spearman(g_pred[f], y[f]),
        "n": int(f.sum()),
    }


def _scatter_series(ax, ser: dict, color, title: str):
    """Per-context g_pred-vs-g₁ scatter with a rank-order guide + labeled points.

    The scored DV g₁ and the predictor g_pred are on different scales, so the
    cell is scored by RANK (Spearman); the guide line is the rank-monotone
    reference (sorted g₁ vs sorted g_pred), not an identity line."""
    import numpy as np

    g1 = np.asarray(ser["g1"])
    gp = np.asarray(ser["g_pred"])
    ax.scatter(g1, gp, s=22, color=color, edgecolors="0.25", linewidths=0.6, zorder=3)
    # Light per-context labels (the unit identity → also fills the dashboard
    # data-viewer identifier column). Label a readable subset to avoid clutter.
    order = np.argsort(g1)
    for rank, i in enumerate(order):
        if rank % 2 == 0:  # every other point keeps the panel legible
            ax.annotate(
                ser["contexts"][i],
                (g1[i], gp[i]),
                fontsize=4.5,
                color="0.35",
                xytext=(2, 2),
                textcoords="offset points",
            )
    ax.set_xlabel("scored DV g₁ (held-out context)", fontsize=7)
    ax.set_ylabel("predicted gate g_pred", fontsize=7)
    ax.set_title(f"{title}  (ρ = {ser['spearman']:+.2f}, n = {ser['n']})", fontsize=7.5)
    ax.tick_params(labelsize=6)


def _plot_leaderboard(lb: dict, noise_floor: dict, out_dir: Path, stem: str):
    """Per-behavior leaderboard bar chart: held-out ρ per key×metric + CI + null."""
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style()
    behavior = lb["behavior"]
    best = _best_rows_per_key_psi(lb["per_bank"])
    null_lo, null_hi = _null_band(lb["per_bank"])
    floor = _noise_floor_mean(noise_floor)

    # Explicit bar set. For the context-only + training-completion keys, show the
    # identity-ψ raw dot + whitened (the established read; ψ does not change k_cC
    # since its displacement term is zero). For the displacement (idiolect-control)
    # key show BOTH identity ψ AND ridge ψ at raw dot — the disclosure that the
    # silent "best ψ" substitution flips it from negative (identity) to positive
    # (ridge). Each entry: (key, metric, psi).
    wanted = [
        ("k_cC", "M_I", "psi_I"),
        ("k_cC", "M_white", "psi_I"),
        ("k_tCB", "M_I", "psi_I"),
        ("k_tCB", "M_white", "psi_I"),
        ("k_cC_plus_delta", "M_I", "psi_I"),
        ("k_cC_plus_delta", "M_I", "psi_ridge"),
    ]

    labels, rhos, errs, colors = [], [], [], []
    for key, metric, psi in wanted:
        row = best.get((key, metric, psi))
        if row is None:
            continue
        # Displacement key carries its ψ in the label (the disclosure); the others
        # only carry the metric (identity ψ is the established read there).
        if key == "k_cC_plus_delta":
            lab = f"{KEY_LABELS[key]}\n({METRIC_LABELS[metric]}, {PSI_LABELS[psi]})"
        else:
            lab = f"{KEY_LABELS.get(key, key)}\n({METRIC_LABELS.get(metric, metric)})"
        labels.append(lab)
        rho = row["spearman"]
        rhos.append(rho)
        ci = row.get("spearman_ci95", [rho, rho])
        lo = ci[0] if ci and ci[0] == ci[0] else rho
        hi = ci[1] if ci and ci[1] == ci[1] else rho
        errs.append([max(0.0, rho - lo), max(0.0, hi - rho)])
        colors.append(
            paper_palette_role("baseline") if key == "k_cC" else paper_palette_role("primary")
        )

    # 1×2: (left) the key×metric bar leaderboard; (right) the LOW-LEVEL per-context
    # scatter behind the headline training-completion-key bar (SPEC §11).
    fig, (ax, axs) = plt.subplots(
        1,
        2,
        figsize=(max(9, 1.3 * len(labels) + 3.4), 4.2),
        gridspec_kw={"width_ratios": [max(6, 1.3 * len(labels)), 3.4]},
    )
    x = np.arange(len(labels))
    err_t = np.array(errs).T if errs else None
    ax.bar(x, rhos, yerr=err_t, capsize=3, color=colors)
    if null_lo == null_lo and null_hi == null_hi:
        ax.axhspan(null_lo, null_hi, color="0.85", alpha=0.6, label="shuffled-key null")
    if floor == floor:
        ax.axhline(floor, color=paper_palette_role("accent"), ls="--", label="noise-floor ceiling")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"Held-out Spearman ρ (g_pred vs {SCORED_DV_AXIS})")
    ax.legend(fontsize=8, loc="best")
    set_title_subtitle(
        ax,
        f"Key×metric leaderboard — {BEHAVIOR_LABELS.get(behavior, behavior)}",
        "displacement key shown both ψ ways; bars = held-out ρ, whiskers = 95% bootstrap CI",
    )

    # The headline source bank for the per-context scatter: the bank whose k_tCB
    # raw-dot ρ is highest (A5 for the marker; the best villain seed for syco).
    head_bank = max(
        lb["per_bank"],
        key=lambda b: (_cell(b["leaderboard"], "k_tCB") or {}).get("spearman", float("-inf")),
    )
    head_src = head_bank.get("source")
    seed = head_bank.get("seed")
    prov = f"source {head_src}" if behavior == "marker" else f"seed {seed}"
    ser = _per_context_series(behavior, head_src, "k_tCB")
    if ser is None:
        axs.text(
            0.5,
            0.5,
            "per-context tensors\nunavailable",
            ha="center",
            va="center",
            fontsize=7,
            color="0.5",
            transform=axs.transAxes,
        )
        axs.set_axis_off()
    else:
        _scatter_series(
            axs,
            ser,
            paper_palette_role("primary"),
            f"Training-completion key, per context\n({prov})",
        )
    fig.tight_layout()
    paths = savefig_paper(fig, stem, dir=str(out_dir))
    plt.close(fig)
    return paths


def _cell(leaderboard: list[dict], key: str, metric: str = "M_I", psi: str = "psi_I"):
    """The (key, metric, psi) leaderboard cell, or None."""
    for c in leaderboard:
        if c["key"] == key and c["metric"] == metric and c["psi"] == psi:
            return c
    return None


def _val_err(row):
    """(spearman, [err_lo, err_hi]) from a leaderboard row (NaN/0 if None)."""
    if row is None:
        return float("nan"), [0.0, 0.0]
    rho = row["spearman"]
    ci = row.get("spearman_ci95", [rho, rho])
    lo = ci[0] if ci and ci[0] == ci[0] else rho
    hi = ci[1] if ci and ci[1] == ci[1] else rho
    return rho, [max(0.0, rho - lo), max(0.0, hi - rho)]


def _contrast_pair_for_behavior(lb: dict) -> dict:
    """Resolve the (context-only, training-completion) bar pair + null for the
    hero contrast, honestly.

    Marker: pair WITHIN A SINGLE BANK (A5 — the only marker bank whose
    training-completion key clears its own shuffled-key null), so both bars and
    the null come from the same source — no cross-bank max-per-key artifact.
    Sycophancy: best-of-2-seeds (both seeds agree closely, so no selection
    artifact); pair the two bars from the SAME chosen seed bank and use that
    bank's own null.
    """
    behavior = lb["behavior"]
    per_bank = lb["per_bank"]
    if behavior == "marker":
        bank = next((b for b in per_bank if b["source"] == "A5"), None)
        bank = bank or max(
            per_bank,
            key=lambda b: (_cell(b["leaderboard"], "k_tCB") or {}).get("spearman", float("-inf")),
        )
    else:
        # pick the seed bank with the higher k_tCB raw dot (both agree ~0.74/0.75).
        bank = max(
            per_bank,
            key=lambda b: (_cell(b["leaderboard"], "k_tCB") or {}).get("spearman", float("-inf")),
        )
    cc = _cell(bank["leaderboard"], "k_cC")
    dep = _cell(bank["leaderboard"], "k_tCB")
    null = bank.get("null_shuffled_key") or {}
    return {
        "source": bank.get("source"),
        "seed": bank.get("seed"),
        "cc": cc,
        "dep": dep,
        "null_p95": null.get("p95"),
    }


def _plot_contrast(lbs: dict[str, dict], noise_floors: dict[str, dict], out_dir: Path):
    """HERO: training-completion key vs context-only key vs each behavior's own
    shuffled-key null. Bars are paired WITHIN A SINGLE SOURCE BANK per behavior
    (no cross-bank max artifact); the scored DV is g₁ (A7 failed).

    TWO rows: (top) the aggregate paired-ρ bars; (bottom) the LOW-LEVEL per-context
    g_pred-vs-g₁ scatter behind each behavior's training-completion-key bar (SPEC
    §11 — the per-unit data behind the aggregate ρ), recomputed faithfully from the
    committed tensors."""
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style()
    behaviors = [b for b in ("marker", "sycophancy") if b in lbs]
    pairs = {b: _contrast_pair_for_behavior(lbs[b]) for b in behaviors}

    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, max(2, len(behaviors)), height_ratios=[1.05, 1.0], hspace=0.55)
    ax = fig.add_subplot(gs[0, :])

    width = 0.35
    x = np.arange(len(behaviors))
    cc_vals, cc_err, dep_vals, dep_err = [], [], [], []
    for b in behaviors:
        v_cc, e_cc = _val_err(pairs[b]["cc"])
        v_dep, e_dep = _val_err(pairs[b]["dep"])
        cc_vals.append(v_cc)
        cc_err.append(e_cc)
        dep_vals.append(v_dep)
        dep_err.append(e_dep)

    ax.bar(
        x - width / 2,
        cc_vals,
        width,
        yerr=np.array(cc_err).T,
        capsize=3,
        color=paper_palette_role("baseline"),
        label="Context-only key (paper default)",
    )
    ax.bar(
        x + width / 2,
        dep_vals,
        width,
        yerr=np.array(dep_err).T,
        capsize=3,
        color=paper_palette_role("primary"),
        label="Training-completion key",
    )
    for i, b in enumerate(behaviors):
        p95 = pairs[b].get("null_p95")
        if p95 is not None and p95 == p95:
            ax.plot([i - 0.5, i + 0.5], [p95, p95], color="0.5", ls=":", lw=1)
        floor = _noise_floor_mean(noise_floors.get(b, {}))
        if floor == floor:
            ax.plot(
                [i - 0.5, i + 0.5],
                [floor, floor],
                color=paper_palette_role("accent"),
                ls="--",
                lw=1,
            )
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ticklabels = []
    for b in behaviors:
        src = pairs[b].get("source")
        seed = pairs[b].get("seed")
        tag = f"source {src}" if b == "marker" else f"seed {seed}"
        ticklabels.append(f"{BEHAVIOR_LABELS.get(b, b)}\n({tag})")
    ax.set_xticklabels(ticklabels, fontsize=8)
    ax.set_ylabel(f"Held-out Spearman ρ (g_pred vs {SCORED_DV_AXIS})", fontsize=8)
    ax.legend(fontsize=8, loc="best")
    set_title_subtitle(
        ax,
        "Does a behavior-dependent key beat the context-only default?",
        "bars paired within one source bank; dotted = that bank's shuffled-key null (p95); "
        "dashed = noise-floor ceiling; whiskers = 95% CI",
    )

    # Bottom row: the per-context scatter behind each training-completion-key bar.
    for i, b in enumerate(behaviors):
        sub = fig.add_subplot(gs[1, i])
        src = pairs[b].get("source")
        ser = _per_context_series(b, src, "k_tCB")
        seed = pairs[b].get("seed")
        prov = f"source {src}" if b == "marker" else f"seed {seed}"
        if ser is None:
            sub.text(
                0.5,
                0.5,
                "per-context tensors\nunavailable",
                ha="center",
                va="center",
                fontsize=7,
                color="0.5",
                transform=sub.transAxes,
            )
            sub.set_axis_off()
            continue
        _scatter_series(
            sub,
            ser,
            paper_palette_role("primary"),
            f"{BEHAVIOR_LABELS.get(b, b)} ({prov})\ntraining-completion key, per context",
        )

    fig.tight_layout()
    paths = savefig_paper(fig, "marker_vs_sycophancy_contrast", dir=str(out_dir))
    plt.close(fig)
    return paths


def _plot_a7_spectrum(a7: dict, out_dir: Path, stem: str):
    """A7 precondition SVD spectrum bar (σ₁²/Σσ², σ₂/σ₁, cos(u₁,ŵ))."""
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style()
    behavior = a7["behavior"]
    per_bank = a7.get("per_bank", [])
    if not per_bank:
        return {}
    metrics = [
        "sigma1_sq_frac",
        "sigma2_over_sigma1",
        "cos_u1_what",
        "scalarity_residual_median",
    ]
    metric_labels = [
        "σ₁²/Σσ²\n(top-component energy)",
        "σ₂/σ₁\n(gap)",
        "|cos(u₁, source write)|",
        "scalarity residual\n(median, lower = scalar)",
    ]
    means = []
    for m in metrics:
        vals = [b[m] for b in per_bank if b.get(m) == b.get(m)]
        means.append(float(np.mean(vals)) if vals else float("nan"))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(metrics))
    ax.bar(x, means, color=paper_palette(len(metrics)), alpha=0.55, zorder=1)
    ax.axhline(0.5, color="0.4", ls="--", lw=0.8, label="strict rank-1 thresholds (0.5)")
    # LOW-LEVEL data behind the mean bars: one dot per SOURCE BANK (A1–A5 for the
    # marker, the seeds for syco), x-jittered, labeled by bank name (SPEC §11 — the
    # "1/5 banks pass" claim is read off the per-bank scalarity-residual column).
    rng = np.random.default_rng(0)
    for xi, m in enumerate(metrics):
        for bank in per_bank:
            val = bank.get(m)
            if val is None or val != val:
                continue
            jx = xi + (rng.random() - 0.5) * 0.34
            ax.scatter(jx, val, s=20, color="0.2", edgecolors="white", linewidths=0.5, zorder=3)
            ax.annotate(
                str(bank.get("source", "")),
                (jx, val),
                fontsize=4.8,
                color="0.25",
                xytext=(2.5, 0),
                textcoords="offset points",
                va="center",
                zorder=4,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("value (bars = mean over banks; dots = per source bank)")
    ax.legend(fontsize=8)
    # Reader-facing verdict (NOT the internal slug). State the strict-rank-1 pass
    # count + what the figure decides.
    n_banks = a7.get("n_banks", len(per_bank))
    n_pass = a7.get("n_banks_rank1_holds", sum(1 for b in per_bank if b.get("rank1_holds")))
    holds = a7.get("behavior_rank1_holds", False)
    g1 = a7.get("g1_vs_greal", {})
    g1lo, g1hi = g1.get("min"), g1.get("max")
    verdict_txt = (
        f"strict rank-1 fails ({n_pass}/{n_banks} banks pass) — gate is low-rank, "
        "scored against the dominant component g₁"
    )
    if holds:
        verdict_txt = (
            f"strict rank-1 holds ({n_pass}/{n_banks} banks pass) — scalar g_real is faithful"
        )
    if g1lo is not None and g1hi is not None and g1lo == g1lo and g1hi == g1hi:
        verdict_txt += f"; g₁ tracks g_real (ρ {g1lo:.2f}–{g1hi:.2f})"
    set_title_subtitle(
        ax,
        f"A7 scalar-gate precondition — {BEHAVIOR_LABELS.get(behavior, behavior)}",
        verdict_txt,
    )
    fig.tight_layout()
    paths = savefig_paper(fig, stem, dir=str(out_dir))
    plt.close(fig)
    return paths


def _load_json(path: Path | None) -> dict | None:
    if path is None or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    base = PROJECT_ROOT / "eval_results/issue_683"

    def _default(behavior: str, name: str) -> str:
        return str(base / behavior / name)

    ap.add_argument(
        "--marker-leaderboard", default=_default("marker", "key_ablation_leaderboard.json")
    )
    ap.add_argument(
        "--sycophancy-leaderboard", default=_default("sycophancy", "key_ablation_leaderboard.json")
    )
    ap.add_argument("--marker-noise-floor", default=_default("marker", "noise_floor.json"))
    ap.add_argument("--sycophancy-noise-floor", default=_default("sycophancy", "noise_floor.json"))
    ap.add_argument("--marker-a7", default=_default("marker", "a7_precondition.json"))
    ap.add_argument("--sycophancy-a7", default=_default("sycophancy", "a7_precondition.json"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "figures/issue_683"))
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lbs: dict[str, dict] = {}
    noise_floors: dict[str, dict] = {}
    written: list[str] = []
    for behavior, lb_path, nf_path, a7_path in (
        ("marker", args.marker_leaderboard, args.marker_noise_floor, args.marker_a7),
        (
            "sycophancy",
            args.sycophancy_leaderboard,
            args.sycophancy_noise_floor,
            args.sycophancy_a7,
        ),
    ):
        lb = _load_json(Path(lb_path))
        if lb is None:
            logger.warning("[phase=figs_skip] no leaderboard for %s at %s", behavior, lb_path)
            continue
        lbs[behavior] = lb
        nf = _load_json(Path(nf_path)) or {}
        noise_floors[behavior] = nf
        paths = _plot_leaderboard(lb, nf, out_dir, f"leaderboard_{behavior}")
        written.extend(str(p) for p in paths.values())
        logger.info("[phase=figs_leaderboard] %s -> %s", behavior, paths.get("png"))
        a7 = _load_json(Path(a7_path))
        if a7:
            ap_paths = _plot_a7_spectrum(a7, out_dir, f"a7_spectrum_{behavior}")
            written.extend(str(p) for p in ap_paths.values())
            logger.info("[phase=figs_a7] %s -> %s", behavior, ap_paths.get("png"))

    if not lbs:
        raise SystemExit(f"no leaderboards found under {base} — run the scorer first.")

    contrast = _plot_contrast(lbs, noise_floors, out_dir)
    written.extend(str(p) for p in contrast.values())
    logger.info("[phase=figs_contrast] hero -> %s", contrast.get("png"))

    n_png = sum(1 for w in written if w.endswith(".png"))
    logger.info("[phase=figs_done] wrote %d PNG(s) under %s", n_png, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
