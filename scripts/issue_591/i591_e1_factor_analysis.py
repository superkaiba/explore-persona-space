#!/usr/bin/env python3
"""Task #591 e1 — cross-behavior flat-panel factor analysis (statistics + figures).

Consumes ``eval_results/issue_591/e1/cell_table.json`` (i591_e1_build_table.py)
and runs the registered inference suite (plan #591 v1 §6):

  - Univariate, CELL-level factors only (cosine, bystander base propensity,
    negative membership), per behavior: within-source permutation test on the
    leak label (B=10,000; seeds 5911-5913). ``self_delta`` is EXCLUDED here —
    panel-constant, so the within-source null is a point mass (p == 1.0 by
    construction; verified against i480_analyze.py::_stratified_permutation_p).
  - H2 (implant strength), source-level inference: exact permutation across
    the 6 sources per behavior (6! = 720 enumerated) on the 18-row panel
    table, plus a pooled 18-panel permutation stratified by behavior
    (B=10,000, seed 5915).
  - Joint Firth logistic (i591_firth.py, sex2-validated) per behavior and
    pooled with behavior fixed effects; ORs + profile-likelihood CIs +
    penalized-LR p-values (pooled CIs descriptive).
  - Classification: ONE pooled ROC/AUC over concatenated left-out-source
    predictions (mean per-fold AUC undefined: syco leak counts 15/6/0/0/0/0)
    + per-left-out-source contributions.
  - Sensitivities (run AFTER the primary fit, never replacing it):
    tau in {0.05, 0.15, 0.20}; EM survivor filter (<24 of 480); per-arm
    base-substrate cosine swapped in; base-rate>0.5 headroom exclusion (the
    H3 mechanical-coupling guard); suppression cells (delta <= -0.10) tabled.

Writes ``eval_results/issue_591/e1/factor_analysis.json`` + figures under
``figures/issue_591/``.

Smoke:

    uv run python scripts/issue_591/i591_e1_factor_analysis.py --perm-b 50 --skip-profile-ci
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "issue_591"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from i591_firth import firth_logistic  # noqa: E402

BEHAVIORS = ("sycophancy", "refusal", "em")
SOURCES = (
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
)
CELL_FACTORS = ("cos_to_source", "bystander_base_rate", "neg_member")
FACTOR_SEEDS = {"cos_to_source": 5911, "bystander_base_rate": 5912, "neg_member": 5913}
H2_POOLED_SEED = 5915
PERM_B = 10_000
EM_SURVIVOR_MIN = 24
OUT_ROOT_DEFAULT = REPO / "eval_results" / "issue_591"
FIG_DIR_DEFAULT = REPO / "figures" / "issue_591"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho (nan if either side is constant)."""
    from scipy.stats import spearmanr

    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _within_source_permutation(
    x: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    *,
    b: int,
    seed: int,
) -> dict:
    """House-standard within-source permutation (i480_analyze recipe).

    Shuffles ``y`` within each source stratum; two-sided on |Spearman rho|.
    """
    rng = np.random.default_rng(seed)
    obs = _spearman(x, y)
    if np.isnan(obs):
        return {"rho": None, "p_perm": None, "b": b, "note": "constant factor or label"}
    count = 0
    for _ in range(b):
        y_shuf = y.copy()
        for s in np.unique(strata):
            idx = np.where(strata == s)[0]
            y_shuf[idx] = y[rng.permutation(idx)]
        if abs(_spearman(x, y_shuf)) >= abs(obs):
            count += 1
    return {"rho": obs, "p_perm": (count + 1) / (b + 1), "b": b}


def _exact_6_permutation(x: np.ndarray, y: np.ndarray) -> dict:
    """Exact 6! = 720 permutation across sources (panel-level H2 read)."""
    assert len(x) == len(y) == 6, (len(x), len(y))
    obs = _spearman(x, y)
    if np.isnan(obs):
        return {"rho": None, "p_exact": None, "n_perms": 720, "note": "constant"}
    count = 0
    total = 0
    for perm in itertools.permutations(range(6)):
        total += 1
        if abs(_spearman(x[np.array(perm)], y)) >= abs(obs) - 1e-12:
            count += 1
    return {"rho": obs, "p_exact": count / total, "n_perms": total}


def _pooled_18_permutation(
    self_delta: np.ndarray,
    n_leak: np.ndarray,
    behavior: np.ndarray,
    *,
    b: int,
    seed: int,
) -> dict:
    """Pooled 18-panel permutation: permute self_delta within behavior strata."""
    rng = np.random.default_rng(seed)
    obs = _spearman(self_delta, n_leak)
    if np.isnan(obs):
        return {"rho": None, "p_perm": None, "b": b, "note": "constant"}
    count = 0
    for _ in range(b):
        x_shuf = self_delta.copy()
        for beh in np.unique(behavior):
            idx = np.where(behavior == beh)[0]
            x_shuf[idx] = self_delta[rng.permutation(idx)]
        if abs(_spearman(x_shuf, n_leak)) >= abs(obs):
            count += 1
    return {"rho": obs, "p_perm": (count + 1) / (b + 1), "b": b}


def _zscore(v: np.ndarray) -> np.ndarray:
    sd = np.std(v)
    if sd == 0:
        return np.zeros_like(v)
    return (v - np.mean(v)) / sd


def _rank_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney ROC AUC (rank-based; no sklearn dependency)."""
    from scipy.stats import rankdata

    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _design(
    cells: list[dict],
    *,
    cos_field: str = "cos_to_source",
    include_self_delta: bool = True,
    behavior_fe: bool = False,
    z_within_behavior: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Build (X, y, names, meta) for the Firth fits. Z-scores continuous factors.

    ``z_within_behavior`` z-scores cos/self_delta/base_rate within each
    behavior (the pooled-fit scale guard from plan §6 analyzer guidance);
    behavior FE dummies (reference = first sorted behavior) when requested.
    """
    has_self = include_self_delta and all(c["self_delta"] is not None for c in cells)
    behavior = np.array([c["behavior"] for c in cells])
    cols: dict[str, np.ndarray] = {
        "cos_to_source": np.array([c[cos_field] for c in cells], dtype=float),
        "bystander_base_rate": np.array([c["bystander_base_rate"] for c in cells], dtype=float),
    }
    if has_self:
        cols["self_delta"] = np.array([c["self_delta"] for c in cells], dtype=float)
    for name in list(cols):
        if z_within_behavior:
            z = np.empty(len(cells))
            for beh in np.unique(behavior):
                m = behavior == beh
                z[m] = _zscore(cols[name][m])
            cols[name] = z
        else:
            cols[name] = _zscore(cols[name])
    cols["neg_member"] = np.array([c["neg_member"] for c in cells], dtype=float)
    names = (
        ["cos_to_source"]
        + (["self_delta"] if has_self else [])
        + [
            "bystander_base_rate",
            "neg_member",
        ]
    )
    X = np.column_stack([cols[n] for n in names])
    if behavior_fe:
        uniq = sorted(set(behavior.tolist()))
        for beh in uniq[1:]:
            X = np.column_stack([X, (behavior == beh).astype(float)])
            names = [*names, f"behavior_{beh}"]
    y = np.array([c["leak"] for c in cells], dtype=float)
    meta = {"self_delta_included": has_self, "cos_field": cos_field, "n": len(cells)}
    return X, y, names, meta


def _firth_summary(
    cells: list[dict],
    *,
    profile_ci: bool,
    plr_pvalues: bool,
    cos_field: str = "cos_to_source",
    behavior_fe: bool = False,
    z_within_behavior: bool = False,
) -> dict:
    X, y, names, meta = _design(
        cells,
        cos_field=cos_field,
        behavior_fe=behavior_fe,
        z_within_behavior=z_within_behavior,
    )
    res = firth_logistic(X, y, names, profile_ci=profile_ci, plr_pvalues=plr_pvalues)
    out = res.to_dict()
    out["design"] = meta
    return out


def _loso_pooled_roc(cells: list[dict], *, z_within_behavior: bool = True) -> dict:
    """ONE pooled ROC over concatenated left-out-source predictions."""
    all_scores: list[float] = []
    all_labels: list[int] = []
    per_source: dict[str, dict] = {}
    for held in SOURCES:
        train = [c for c in cells if c["source"] != held]
        test = [c for c in cells if c["source"] == held]
        X_tr, y_tr, names, _ = _design(train, behavior_fe=True, z_within_behavior=z_within_behavior)
        # Fit WITHOUT profile CIs (speed); prediction needs beta only.
        res = firth_logistic(X_tr, y_tr, names, profile_ci=False, plr_pvalues=False)
        # Build the test design with the TRAIN fold's z-parameters: re-derive by
        # z-scoring train+test jointly per behavior would leak; instead z-score
        # test with train moments. For simplicity and rank-invariance of AUC,
        # z-scoring constants per fold do not change within-fold ranking, but
        # POOLED ranking mixes folds — so use train-fold moments explicitly.
        behavior_tr = np.array([c["behavior"] for c in train])
        moments: dict[tuple[str, str], tuple[float, float]] = {}
        for fname in ("cos_to_source", "self_delta", "bystander_base_rate"):
            for beh in np.unique(behavior_tr):
                vals = np.array(
                    [c[fname] for c in train if c["behavior"] == beh and c[fname] is not None],
                    dtype=float,
                )
                if vals.size:
                    moments[(fname, beh)] = (float(np.mean(vals)), float(np.std(vals)) or 1.0)
        has_self = "self_delta" in res.names

        def _x_row(c: dict, *, _res=res, _moments=moments, _has_self=has_self) -> list[float]:
            row = []
            fnames = ["cos_to_source"] + (["self_delta"] if _has_self else [])
            fnames.append("bystander_base_rate")
            for fname in fnames:
                mu, sd = _moments[(fname, c["behavior"])]
                row.append((float(c[fname]) - mu) / sd)
            row.append(float(c["neg_member"]))
            for nm in _res.names:
                if nm.startswith("behavior_"):
                    row.append(1.0 if nm == f"behavior_{c['behavior']}" else 0.0)
            return row

        from scipy.special import expit

        X_te = np.array([_x_row(c) for c in test])
        X_te = np.column_stack([np.ones(len(test)), X_te])
        scores = expit(X_te @ res.beta)
        labels = [int(c["leak"]) for c in test]
        all_scores.extend(scores.tolist())
        all_labels.extend(labels)
        per_source[held] = {
            "n_cells": len(test),
            "n_leak": int(sum(labels)),
            "mean_score_leak": (
                float(np.mean([s for s, lk in zip(scores, labels, strict=True) if lk == 1]))
                if sum(labels)
                else None
            ),
            "mean_score_no_leak": (
                float(np.mean([s for s, lk in zip(scores, labels, strict=True) if lk == 0]))
                if len(labels) - sum(labels)
                else None
            ),
        }
    auc = _rank_auc(np.array(all_labels), np.array(all_scores))
    return {"pooled_auc": auc, "per_left_out_source": per_source, "n_total": len(all_labels)}


def _univariate_suite(cells: list[dict], *, b: int) -> dict:
    """Per behavior: within-source permutation for the 3 cell-level factors."""
    out: dict[str, dict] = {}
    for beh in BEHAVIORS:
        sub = [c for c in cells if c["behavior"] == beh]
        strata = np.array([c["source"] for c in sub])
        y = np.array([c["leak"] for c in sub], dtype=float)
        delta = np.array([c["delta"] for c in sub], dtype=float)
        out[beh] = {}
        for factor in CELL_FACTORS:
            x = np.array([c[factor] for c in sub], dtype=float)
            res = _within_source_permutation(x, y, strata, b=b, seed=FACTOR_SEEDS[factor])
            # Continuous-delta rank analysis reported alongside (plan §6).
            res["rho_vs_delta"] = _spearman(x, delta)
            out[beh][factor] = res
    return out


def _h2_suite(panels: list[dict], *, b: int) -> dict:
    """Between-source H2 inference on the 18-row panel table."""
    out: dict = {"per_behavior": {}, "pooled": None, "note": None}
    usable = [p for p in panels if p["self_delta"] is not None]
    if len(usable) < len(panels):
        out["note"] = (
            "self_delta missing for some panels (self_rates.json absent / partial) — "
            "H2 computed on available behaviors only"
        )
    for beh in BEHAVIORS:
        sub = [p for p in usable if p["behavior"] == beh]
        if len(sub) != 6:
            out["per_behavior"][beh] = {"skipped": True, "n_panels": len(sub)}
            continue
        x = np.array([p["self_delta"] for p in sub], dtype=float)
        out["per_behavior"][beh] = {
            "vs_n_leak_cells": _exact_6_permutation(
                x, np.array([p["n_leak_cells"] for p in sub], dtype=float)
            ),
            "vs_panel_sd": _exact_6_permutation(
                x, np.array([p["panel_sd"] for p in sub], dtype=float)
            ),
        }
    pooled = [p for p in usable]
    if len(pooled) >= 12:
        out["pooled"] = _pooled_18_permutation(
            np.array([p["self_delta"] for p in pooled], dtype=float),
            np.array([p["n_leak_cells"] for p in pooled], dtype=float),
            np.array([p["behavior"] for p in pooled]),
            b=b,
            seed=H2_POOLED_SEED,
        )
        out["pooled"]["n_panels"] = len(pooled)
    return out


def _relabel_leak(cells: list[dict], tau: float) -> list[dict]:
    out = []
    for c in cells:
        c2 = dict(c)
        c2["leak"] = int(c["delta"] >= tau)
        out.append(c2)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _fig_leak_map(cells: list[dict], panels: list[dict], fig_dir: Path):
    """Hero: 3-panel 6x23 leak map colored by delta, negatives hatched."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    bystanders = sorted({c["bystander"] for c in cells})
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    titles = {
        "sycophancy": "Sycophancy (#411)",
        "refusal": "Refusal (#518)",
        "em": "Emergent misalignment (#518, survivor-rate DV)",
    }
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        grid = np.full((len(SOURCES), len(bystanders)), np.nan)
        for c in (c for c in cells if c["behavior"] == beh):
            i = SOURCES.index(c["source"])
            if c["bystander"] in bystanders:
                grid[i, bystanders.index(c["bystander"])] = c["delta"]
        vmax = max(0.2, np.nanmax(np.abs(grid)))
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for c in (c for c in cells if c["behavior"] == beh):
            i = SOURCES.index(c["source"])
            j = bystanders.index(c["bystander"])
            if c["leak"]:
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.6)
                )
            if c["neg_member"]:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, ec="grey", lw=0.8, hatch="///"
                    )
                )
        max_cos = {p["source"]: p["max_bystander_cos"] for p in panels if p["behavior"] == beh}
        ax.set_yticks(range(len(SOURCES)))
        ax.set_yticklabels(
            [
                f"{s.replace('_', ' ')} (max cos {max_cos.get(s, float('nan')):.3f})"
                for s in SOURCES
            ],
            fontsize=8,
        )
        ax.set_xticks(range(len(bystanders)))
        ax.set_xticklabels([b.replace("_", " ") for b in bystanders], rotation=90, fontsize=7)
        ax.set_title(titles[beh], fontsize=11)
        fig.colorbar(im, ax=ax, label="leakage delta (trained - base)", shrink=0.8)
    fig.suptitle(
        "Per-cell leakage delta by behavior - leak cells outlined, training negatives hatched",
        fontsize=12,
    )
    savefig_paper(fig, "e1_leak_map_hero", dir=fig_dir)
    plt.close(fig)


def _fig_forest(per_behavior_fits: dict, pooled_fit: dict, fig_dir: Path):
    """Forest plot of factor coefficients with CI-method-aware rendering.

    Reads the per-coefficient ``ci95_method_low/high`` provenance tags
    (#591 e5): a bound fit via the Wald fallback is drawn as a dashed CI
    with a black-edged diamond, tagged ``[Wald CI]`` in the row label and
    named in a legend; the axis label says "95% profile CI" only when every
    plotted bound is profile. JSONs without the tags render as all-profile.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8, 6))
    rows = []
    for beh, fit in {**per_behavior_fits, "pooled": pooled_fit}.items():
        if fit is None or "names" not in fit:
            continue
        n = len(fit["names"])
        meth_lo = fit.get("ci95_method_low") or ["profile"] * n
        meth_hi = fit.get("ci95_method_high") or ["profile"] * n
        for j, name in enumerate(fit["names"]):
            if name == "intercept" or name.startswith("behavior_"):
                continue
            lo = fit.get("ci95_low_coef", [None] * n)[j]
            hi = fit.get("ci95_high_coef", [None] * n)[j]
            fb = "wald-fallback" in (meth_lo[j], meth_hi[j])
            label = f"{beh}: {name.replace('_', ' ')}" + (" [Wald CI]" if fb else "")
            rows.append((label, fit["coef"][j], lo, hi, fb))
    ys = np.arange(len(rows))[::-1]
    any_fallback = any(r[4] for r in rows)
    for y, (label, coef, lo, hi, fb) in zip(ys, rows, strict=True):
        color = (
            paper_palette_role("primary") if "pooled" in label else paper_palette_role("neutral")
        )
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=color, lw=1.5, ls="--" if fb else "-")
        if fb:
            ax.plot(coef, y, "D", color=color, markeredgecolor="black")
        else:
            ax.plot(coef, y, "o", color=color)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    if any_fallback:
        ax.set_xlabel(
            "Firth log-odds coefficient (z-scored factor), "
            "95% CI (profile; dashed diamond = Wald fallback)"
        )
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="D",
                    ls="--",
                    color="grey",
                    markeredgecolor="black",
                    label="Wald-fallback CI (profile bound non-estimable)",
                )
            ],
            fontsize=7,
            loc="lower right",
        )
    else:
        ax.set_xlabel("Firth log-odds coefficient (z-scored factor), 95% profile CI")
    ax.set_title("Factor coefficients per behavior and pooled")
    savefig_paper(fig, "e1_factor_forest", dir=fig_dir)
    plt.close(fig)


def _fig_tau_sensitivity(tau_results: dict, fig_dir: Path):
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    taus = sorted(float(t) for t in tau_results)
    aucs = [tau_results[f"{t:.2f}"]["loso"]["pooled_auc"] for t in taus]
    ax.plot(taus, aucs, "o-", label="pooled left-out-source AUC")
    ax.axhline(0.75, color="grey", ls="--", lw=0.8, label="success bar 0.75")
    ax.set_xlabel("leak threshold tau (delta >= tau)")
    ax.set_ylabel("pooled AUC")
    ax.set_title("Classification quality vs leak threshold")
    ax.legend()
    savefig_paper(fig, "e1_tau_sensitivity", dir=fig_dir)
    plt.close(fig)


def _fig_delta_vs_base_rate(cells: list[dict], fig_dir: Path):
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        sub = [c for c in cells if c["behavior"] == beh]
        x = [c["bystander_base_rate"] for c in sub]
        y = [c["delta"] for c in sub]
        leak = [c["leak"] for c in sub]
        ax.scatter(
            [xv for xv, lk in zip(x, leak, strict=True) if not lk],
            [yv for yv, lk in zip(y, leak, strict=True) if not lk],
            s=14,
            color=paper_palette_role("neutral"),
            label="no leak",
        )
        ax.scatter(
            [xv for xv, lk in zip(x, leak, strict=True) if lk],
            [yv for yv, lk in zip(y, leak, strict=True) if lk],
            s=18,
            color=paper_palette_role("accent"),
            label="leak (delta >= 0.10)",
        )
        ax.axhline(0.10, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("bystander base rate")
        ax.set_title(beh)
    axes[0].set_ylabel("leakage delta (raw)")
    axes[0].legend(fontsize=8)
    savefig_paper(fig, "e1_delta_vs_base_rate_raw", dir=fig_dir)
    plt.close(fig)


def _fig_panel_table(panels: list[dict], fig_dir: Path):
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    markers = {"sycophancy": "o", "refusal": "s", "em": "^"}
    for beh in BEHAVIORS:
        sub = [p for p in panels if p["behavior"] == beh and p["self_delta"] is not None]
        if not sub:
            continue
        ax.scatter(
            [p["self_delta"] for p in sub],
            [p["n_leak_cells"] for p in sub],
            marker=markers[beh],
            label=beh,
            color=paper_palette_role(
                {"sycophancy": "primary", "refusal": "accent", "em": "control"}[beh]
            ),
        )
        for p in sub:
            ax.annotate(
                p["source"].replace("_", " "),
                (p["self_delta"], p["n_leak_cells"]),
                fontsize=6,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.set_xlabel("source self-implant delta (manipulation check)")
    ax.set_ylabel("leak cells on the panel (of 23)")
    ax.set_title("Implant strength vs panel leakage (18 panels)")
    ax.legend(fontsize=8)
    savefig_paper(fig, "e1_panel_self_delta_vs_leak", dir=fig_dir)
    plt.close(fig)


def _fig_suppression(cells: list[dict], fig_dir: Path):
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    counts = {
        beh: sum(1 for c in cells if c["behavior"] == beh and c["delta"] <= -0.10)
        for beh in BEHAVIORS
    }
    ax.bar(list(counts.keys()), list(counts.values()))
    ax.set_ylabel("suppression cells (delta <= -0.10)")
    ax.set_title("Suppression cells per behavior (tabled separately)")
    savefig_paper(fig, "e1_suppression_counts", dir=fig_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------


def run(
    out_root: Path,
    fig_dir: Path,
    *,
    perm_b: int,
    profile_ci: bool,
    cell_table: Path | None = None,
    out_subdir: str = "e1",
) -> dict:
    """Registered factor suite over a cell table.

    e5 override flags (plan v2 §3.7; defaults -> round-1 behavior unchanged):
    ``cell_table`` points at the corrected table (e5/cell_table.json),
    ``out_subdir`` redirects factor_analysis.json (e1 outputs untouched).
    """
    table_path = cell_table or (out_root / "e1" / "cell_table.json")
    table = json.loads(table_path.read_text())
    cells: list[dict] = table["cells"]
    panels: list[dict] = table["panels"]
    assert len(cells) == 414 and len(panels) == 18, (len(cells), len(panels))
    have_all_self = all(c["self_delta"] is not None for c in cells)

    results: dict = {"inputs": {"cell_table": str(table_path)}}

    # --- primary suite at tau = 0.10 ---
    results["univariate_within_source"] = _univariate_suite(cells, b=perm_b)
    results["h2_between_source"] = _h2_suite(panels, b=perm_b)
    per_beh_fits: dict[str, dict] = {}
    for beh in BEHAVIORS:
        sub = [c for c in cells if c["behavior"] == beh]
        per_beh_fits[beh] = _firth_summary(sub, profile_ci=profile_ci, plr_pvalues=True)
    pooled_fit = _firth_summary(
        cells,
        profile_ci=profile_ci,
        plr_pvalues=True,
        behavior_fe=True,
        z_within_behavior=True,
    )
    results["firth_per_behavior"] = per_beh_fits
    results["firth_pooled"] = pooled_fit
    results["loso"] = _loso_pooled_roc(cells)

    # neg_member x cosine collinearity bookkeeping (analyzer guidance).
    collin = {}
    for beh in BEHAVIORS:
        sub = [c for c in cells if c["behavior"] == beh]
        collin[beh] = _spearman(
            np.array([c["neg_member"] for c in sub], dtype=float),
            np.array([c["cos_to_source"] for c in sub], dtype=float),
        )
    collin["pooled"] = _spearman(
        np.array([c["neg_member"] for c in cells], dtype=float),
        np.array([c["cos_to_source"] for c in cells], dtype=float),
    )
    results["neg_member_cosine_spearman"] = collin

    # --- sensitivities (after the primary, never replacing it) ---
    sens: dict = {}
    tau_res = {}
    for tau in table["tau_sensitivity"]:
        cells_t = _relabel_leak(cells, tau)
        tau_res[f"{tau:.2f}"] = {
            "univariate": _univariate_suite(cells_t, b=perm_b),
            "firth_pooled": _firth_summary(
                cells_t,
                profile_ci=False,
                plr_pvalues=False,
                behavior_fe=True,
                z_within_behavior=True,
            ),
            "loso": _loso_pooled_roc(cells_t),
            "n_leak_per_behavior": {
                beh: int(sum(c["leak"] for c in cells_t if c["behavior"] == beh))
                for beh in BEHAVIORS
            },
        }
    # Include the primary tau in the curve for plotting continuity.
    tau_res[f"{table['tau_primary']:.2f}"] = {
        "loso": results["loso"],
        "n_leak_per_behavior": {
            beh: int(sum(c["leak"] for c in cells if c["behavior"] == beh)) for beh in BEHAVIORS
        },
    }
    sens["tau"] = tau_res

    em_cells = [c for c in cells if c["behavior"] == "em"]
    em_kept = [c for c in em_cells if c["n_rollouts_after_coherence_filter"] >= EM_SURVIVOR_MIN]
    em_dropped = [c for c in em_cells if c["n_rollouts_after_coherence_filter"] < EM_SURVIVOR_MIN]
    sens["em_survivor_filter"] = {
        "min_survivors": EM_SURVIVOR_MIN,
        "n_dropped": len(em_dropped),
        "dropped_leak_composition": {
            "leak": int(sum(c["leak"] for c in em_dropped)),
            "no_leak": int(sum(1 - c["leak"] for c in em_dropped)),
        },
        "firth_em_filtered": _firth_summary(em_kept, profile_ci=False, plr_pvalues=True)
        if len(em_kept) > 20
        else None,
        "univariate_em_filtered": {
            factor: _within_source_permutation(
                np.array([c[factor] for c in em_kept], dtype=float),
                np.array([c["leak"] for c in em_kept], dtype=float),
                np.array([c["source"] for c in em_kept]),
                b=perm_b,
                seed=FACTOR_SEEDS[factor],
            )
            for factor in CELL_FACTORS
        },
    }

    sens["arm_substrate_cosine"] = {
        beh: _firth_summary(
            [c for c in cells if c["behavior"] == beh],
            profile_ci=False,
            plr_pvalues=True,
            cos_field="cos_arm_substrate",
        )
        for beh in ("refusal", "em")
    }

    # H3 headroom guard: exclude base_rate > 0.5 cells (EM has the high-base cells).
    sens["headroom_base_rate_le_0p5"] = {}
    for beh in BEHAVIORS:
        sub = [c for c in cells if c["behavior"] == beh and c["bystander_base_rate"] <= 0.5]
        n_excl = sum(1 for c in cells if c["behavior"] == beh) - len(sub)
        sens["headroom_base_rate_le_0p5"][beh] = {
            "n_excluded": n_excl,
            "firth": _firth_summary(sub, profile_ci=False, plr_pvalues=True)
            if n_excl > 0
            else "no cells excluded — identical to primary fit",
        }

    suppression = [
        {k: c[k] for k in ("behavior", "source", "bystander", "delta")}
        for c in cells
        if c["delta"] <= -0.10
    ]
    sens["suppression_cells"] = suppression
    results["sensitivity"] = sens

    results["metadata"] = {
        "git_commit_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "perm_b": perm_b,
        "profile_ci": profile_ci,
        "self_delta_complete": have_all_self,
        "numpy_version": np.__version__,
        "seeds": {**FACTOR_SEEDS, "h2_pooled": H2_POOLED_SEED},
    }

    out_path = out_root / out_subdir / "factor_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[e1-factors] -> {out_path}")

    # --- figures ---
    fig_dir.mkdir(parents=True, exist_ok=True)
    _fig_leak_map(cells, panels, fig_dir)
    _fig_forest(per_beh_fits, pooled_fit, fig_dir)
    _fig_tau_sensitivity(sens["tau"], fig_dir)
    _fig_delta_vs_base_rate(cells, fig_dir)
    _fig_panel_table(panels, fig_dir)
    _fig_suppression(cells, fig_dir)
    print(f"[e1-factors] figures -> {fig_dir}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="#591 e1 factor analysis (permutation + Firth + LOSO ROC + figures).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR_DEFAULT)
    parser.add_argument(
        "--perm-b", type=int, default=PERM_B, help="Permutation count (smoke: e.g. 50)."
    )
    parser.add_argument(
        "--skip-profile-ci",
        action="store_true",
        help="Skip profile-likelihood CIs (smoke; production computes them).",
    )
    parser.add_argument(
        "--cell-table",
        type=Path,
        default=None,
        help="e5 override: explicit cell-table path (default <out-root>/e1/cell_table.json).",
    )
    parser.add_argument(
        "--out-subdir",
        default="e1",
        help="Output subdir under --out-root for factor_analysis.json (e5 refit uses 'e5').",
    )
    args = parser.parse_args(argv)
    run(
        args.out_root,
        args.fig_dir,
        perm_b=args.perm_b,
        profile_ci=not args.skip_profile_ci,
        cell_table=args.cell_table,
        out_subdir=args.out_subdir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
