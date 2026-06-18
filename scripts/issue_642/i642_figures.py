#!/usr/bin/env python3
# Research notation (−, Δ, ρ) is intentional in labels + prose.
# ruff: noqa: RUF001, RUF002, RUF003
"""Task #642 — figures (3-arm decomposition heroes + exploratory dump, plan §6).

Reads ``analysis.json`` + the cmft stage-A trajectory per behavior and writes to
``figures/issue_642/`` via the paper-plots conventions.

Hero candidates (plan §6): (1) the DECOMPOSITION BAR — #606's measured gap
(+0.098) split into the measured Δ_rank (adapter-vs-dense bundle) + Δ_coverage
stack with CIs (the headline figure); (2) leakage-vs-strength overlay (x = s, y =
38-bystander-mean delta; LoRA orange, cmft green, FT blue; matched band shaded);
(3) per-persona profile scatters at s* (cmft-vs-LoRA and ft-vs-cmft, identity
line, twins/roster/negative-members marked, ρ annotated). Exploratory: stage-A
cmft trajectory; per-persona heatmaps (clean AND raw); Δ_rank/Δ_coverage-vs-s*
sweep curves.

Usage::

    uv run python scripts/issue_642/i642_figures.py --behavior sycophancy \
        --eval-root eval_results/issue_642 [--out-dir figures/issue_642]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "issue_642"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from i642_common import (  # noqa: E402
    ISSUE606_GAP,
    S_BAND,
    S_TARGET,
    SOURCE_PERSONA,
    SYCO_EXPECTED_NEGATIVES,
    TWIN_PROMPTS,
)

log = logging.getLogger("issue_642.figures")

ARM_COLORS = {"lora": "tab:orange", "cmft": "tab:green", "ft": "tab:blue"}
ARM_LABELS = {"lora": "LoRA", "cmft": "coverage-matched FT", "ft": "full FT"}
NEG_MEMBERS = {"sycophancy": set(SYCO_EXPECTED_NEGATIVES)}


def _save(fig, name: str, out_dir: Path) -> None:
    from explore_persona_space.analysis.paper_plots import savefig_paper

    savefig_paper(fig, name, dir=out_dir)
    plt.close(fig)
    log.info("figure -> %s/%s", out_dir, name)


def _arm_points(analysis: dict) -> dict[str, list[tuple[str, float, float]]]:
    """Per arm: (cell, s_stage_b, 38-bystander-mean clean delta) sorted by s."""
    s = analysis["s_stage_b"]
    tables = analysis["per_cell_tables"]
    bystanders = list(analysis["per_persona_at_target"]["lora"])
    out: dict[str, list[tuple[str, float, float]]] = {"lora": [], "cmft": [], "ft": []}
    for cell, s_val in s.items():
        arm = cell.split("_step")[0]
        if arm not in out:
            continue
        deltas = [
            tables[cell][p]["delta_clean"]
            for p in bystanders
            if tables[cell][p]["delta_clean"] is not None
        ]
        out[arm].append((cell, s_val, float(np.nanmean(deltas))))
    for arm in out:
        out[arm].sort(key=lambda t: t[1])
    return out


def fig_decomposition_bar(analysis: dict, behavior: str, out_dir: Path) -> None:
    """Hero candidate 1: the #606 gap split into Δ_rank + Δ_coverage (stacked,
    with CIs) — the headline figure."""
    h = analysis["headline"]
    dr = h["delta_rank"]["gap_plugin"]
    dc = h["delta_coverage"]["gap_plugin"]
    dr_ci = h["delta_rank"]["gap_ci95"]
    dc_ci = h["delta_coverage"]["gap_ci95"]
    recon = h["additive_identity"]["reconstructed_gap_plugin"]
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    # Stacked bar for the measured decomposition.
    ax.bar(
        ["measured\ndecomposition"], [dr], color=ARM_COLORS["cmft"], label="Δ_rank (cmft − LoRA)"
    )
    ax.bar(
        ["measured\ndecomposition"],
        [dc],
        bottom=[dr],
        color=ARM_COLORS["ft"],
        label="Δ_coverage (FT − cmft)",
    )
    # Reference bar for #606's measured total gap.
    ax.bar(["#606 gap\n(FT − LoRA)"], [ISSUE606_GAP], color="tab:grey", alpha=0.6)
    # CIs drawn at the top of each component segment (matches the bar's
    # cumulative-effect interpretation; midpoint placement can fall outside the
    # CI when the value is small relative to the CI half-width).
    ax.errorbar(
        0,
        dr,
        yerr=[[max(0.0, dr - dr_ci[0])], [max(0.0, dr_ci[1] - dr)]],
        fmt="none",
        ecolor="black",
        capsize=3,
        alpha=0.7,
    )
    ax.errorbar(
        0,
        dr + dc,
        yerr=[[max(0.0, dc - (dc_ci[0]))], [max(0.0, dc_ci[1] - dc)]],
        fmt="none",
        ecolor="black",
        capsize=3,
        alpha=0.7,
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel("bystander-mean leakage gap at s*=0.50")
    ax.set_title(
        f"{behavior}: #606 gap decomposition (verdict {h['verdict']})\n"
        f"Δ_rank={dr:+.3f} + Δ_coverage={dc:+.3f} = {recon:+.3f} (target +{ISSUE606_GAP})"
    )
    ax.legend(fontsize=8)
    _save(fig, f"{behavior}_decomposition_bar_hero", out_dir)


def fig_leakage_vs_strength(analysis: dict, behavior: str, out_dir: Path) -> None:
    """Hero candidate 2: 38-bystander-mean leakage vs implant strength, 3 arms."""
    pts = _arm_points(analysis)
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for arm in ("lora", "cmft", "ft"):
        rows = pts[arm]
        if not rows:
            continue
        xs = [0.0, *(r[1] for r in rows)]
        ys = [0.0, *(r[2] for r in rows)]
        ax.plot(xs, ys, "o-", color=ARM_COLORS[arm], label=ARM_LABELS[arm])
    ax.axvspan(S_BAND[0], S_BAND[1], alpha=0.10, color="grey")
    ax.axvline(S_TARGET, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("source-implant strength s (source-self rate delta)")
    ax.set_ylabel("bystander-mean leakage delta")
    ax.set_title(f"{behavior}: leakage vs implant strength (3 arms)")
    ax.legend()
    _save(fig, f"{behavior}_leakage_vs_strength_hero", out_dir)


def _profile_scatter(
    analysis: dict, behavior: str, out_dir: Path, arm_hi: str, arm_lo: str
) -> None:
    """Hero candidate 3: per-persona arm_hi vs arm_lo at s* (identity line)."""
    per = analysis["per_persona_at_target"]
    contrast_key = "delta_rank" if (arm_hi, arm_lo) == ("cmft", "lora") else "delta_coverage"
    rho = analysis["headline"][contrast_key]["profile_spearman_rho"]
    negs = NEG_MEMBERS.get(behavior, set())
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    groups = {"twin": [], "roster": [], "negative-member": []}
    for p in per[arm_lo]:
        kind = "negative-member" if p in negs else ("twin" if p in TWIN_PROMPTS else "roster")
        groups[kind].append(p)
    styles = {
        "roster": {"marker": "o", "color": "tab:grey"},
        "twin": {"marker": "^", "color": "tab:green"},
        "negative-member": {"marker": "s", "color": "tab:red"},
    }
    for kind, members in groups.items():
        if not members:
            continue
        ax.scatter(
            [per[arm_lo][p] for p in members],
            [per[arm_hi][p] for p in members],
            s=28,
            label=kind,
            **styles[kind],
        )
    all_vals = list(per[arm_lo].values()) + list(per[arm_hi].values())
    lim = [min(all_vals) - 0.05, max(all_vals) + 0.05]
    ax.plot(lim, lim, color="black", lw=0.8, ls=":")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"{ARM_LABELS[arm_lo]} per-persona leakage delta at s*")
    ax.set_ylabel(f"{ARM_LABELS[arm_hi]} per-persona leakage delta at s*")
    ax.set_title(
        f"{behavior}: {ARM_LABELS[arm_hi]} vs {ARM_LABELS[arm_lo]} (Spearman ρ = {rho:.2f})"
    )
    ax.legend(fontsize=8)
    _save(fig, f"{behavior}_profile_{arm_hi}_vs_{arm_lo}_hero", out_dir)


def fig_trajectories(trajectory: dict, behavior: str, out_dir: Path) -> None:
    """Exploratory: stage-A cmft s(step) (this run's only trained trajectory)."""
    cells = trajectory["cells"]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    rows = sorted(
        ((rec["step"], rec.get("s")) for c, rec in cells.items() if rec.get("arm") == "cmft"),
        key=lambda t: t[0],
    )
    rows = [(st, s) for st, s in rows if s is not None]
    if rows:
        ax.plot(
            [r[0] for r in rows],
            [r[1] for r in rows],
            "o-",
            color=ARM_COLORS["cmft"],
            label="coverage-matched FT (stage-A)",
        )
    ax.axhspan(S_BAND[0], S_BAND[1], alpha=0.10, color="grey")
    ax.axhline(S_TARGET, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("source-implant strength s")
    ax.set_title(f"{behavior}: cmft stage-A install trajectory")
    ax.legend(fontsize=8)
    _save(fig, f"{behavior}_cmft_stage_a_trajectory", out_dir)


def fig_heatmaps(analysis: dict, behavior: str, out_dir: Path) -> None:
    """Exploratory: per-persona delta heatmaps, CLEAN and RAW side by side."""
    tables = analysis["per_cell_tables"]
    s = analysis["s_stage_b"]
    cells = sorted(s, key=lambda c: s[c])
    personas = sorted(p for p in next(iter(tables.values())) if p != SOURCE_PERSONA)
    negs = NEG_MEMBERS.get(behavior, set())
    fig, axes = plt.subplots(1, 2, figsize=(16, 0.32 * len(personas) + 2), sharey=True)
    for ax, key, title in zip(axes, ("delta_clean", "delta_raw"), ("clean", "raw"), strict=True):
        mat = np.array([[tables[c][p][key] for c in cells] for p in personas], dtype=float)
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([f"{c}\n(s={s[c]:.2f})" for c in cells], fontsize=6, rotation=45)
        ax.set_title(f"{behavior}: per-persona delta ({title})")
        fig.colorbar(im, ax=ax, shrink=0.6)
    labels = [
        f"{p} (neg)" if p in negs else (f"{p} (twin)" if p in TWIN_PROMPTS else p) for p in personas
    ]
    axes[0].set_yticks(range(len(personas)))
    axes[0].set_yticklabels(labels, fontsize=6)
    _save(fig, f"{behavior}_per_persona_heatmap", out_dir)


def fig_sweep(analysis: dict, behavior: str, out_dir: Path) -> None:
    """Exploratory: Δ_rank + Δ_coverage vs matched-strength target."""
    sweep = analysis["sweep"]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for key, color, label in (
        ("delta_rank", ARM_COLORS["cmft"], "Δ_rank (cmft − LoRA)"),
        ("delta_coverage", ARM_COLORS["ft"], "Δ_coverage (FT − cmft)"),
    ):
        rows = sweep[key]
        xs = [r["target"] for r in rows]
        ys = [r["gap_plugin"] for r in rows]
        lo = [max(0.0, r["gap_plugin"] - r["gap_ci"][0]) for r in rows]
        hi = [max(0.0, r["gap_ci"][1] - r["gap_plugin"]) for r in rows]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=color, capsize=3, label=label)
        for r in rows:
            if not r["in_range_both_arms"]:
                ax.plot(r["target"], r["gap_plugin"], "x", color="red", ms=9)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axhspan(-0.04, 0.04, alpha=0.10, color="green")
    ax.axvline(S_TARGET, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("matched-strength target s*")
    ax.set_ylabel("contrast (bystander mean)")
    ax.set_title(f"{behavior}: Δ_rank / Δ_coverage vs s* (x = out of range)")
    ax.legend(fontsize=8)
    _save(fig, f"{behavior}_decomposition_vs_target_sweep", out_dir)


def _pretty_persona(p: str) -> str:
    label = p.replace("_", " ").capitalize()
    return (
        label.replace("Ai ", "AI ")
        if label.startswith("Ai ")
        else ("AI" if label == "Ai" else label)
    )


def fig_endpoint_comparison(analysis: dict, behavior: str, out_dir: Path) -> None:
    """Exploratory: endpoint (near-ceiling) descriptive 3-arm per-persona bars."""
    tables = analysis["per_cell_tables"]
    s = analysis["s_stage_b"]
    endpoints = {}
    for arm in ("lora", "cmft", "ft"):
        arm_cells = [c for c in s if c.startswith(f"{arm}_")]
        if arm_cells:
            endpoints[arm] = max(arm_cells, key=lambda c: int(c.split("step")[-1]))
    if len(endpoints) < 2:
        return
    personas = sorted(p for p in tables[next(iter(endpoints.values()))] if p != SOURCE_PERSONA)
    fig, ax = plt.subplots(figsize=(max(8, 0.25 * len(personas)), 4.2))
    x = np.arange(len(personas))
    n = len(endpoints)
    for i, (arm, cell) in enumerate(endpoints.items()):
        ys = [tables[cell][p]["delta_clean"] for p in personas]
        ax.bar(
            x + (i - (n - 1) / 2) * 0.27,
            ys,
            width=0.27,
            color=ARM_COLORS[arm],
            label=f"{ARM_LABELS[arm]} (endpoint, step {cell.split('step')[-1]})",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_persona(p) for p in personas], fontsize=5, rotation=90)
    ax.set_ylabel("leakage delta (clean)")
    ax.set_title(f"{behavior}: endpoint cells, descriptive (near ceiling — not the headline)")
    ax.legend(fontsize=8)
    _save(fig, f"{behavior}_endpoint_descriptive", out_dir)


V9_ARM_COLORS = {"loraRefOP": "tab:orange", "cmftRefOP": "tab:green"}
V9_ARM_LABELS = {"loraRefOP": "LoRA (refusal)", "cmftRefOP": "coverage-matched FT (refusal)"}


def fig_v9_cross_behavior_hero(analysis: dict, out_dir: Path) -> None:
    """v9 HERO (plan §6): the cross-behavior adapter-vs-dense bar — refusal
    Δ_rank_matched (this round) beside round 4's sycophancy +0.063, both at
    s*=0.50 with bootstrap CIs and the ±0.04 band, so the generality claim reads
    off one figure."""
    h = analysis["headline"]
    dr = h["delta_rank_matched"]
    refusal_point = dr.get("gap_plugin", 0.0)
    refusal_ci = dr.get("gap_ci95", [0.0, 0.0])
    syco_point = h["round4_syco_delta_rank"]
    thr = h["separation_threshold"]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    labels = ["refusal\n(this round)", "sycophancy\n(round 4)"]
    points = [refusal_point, syco_point]
    colors = ["tab:green", "tab:grey"]
    ax.bar(labels, points, color=colors, alpha=0.85)
    # CI on the refusal bar only (round-4's CI is the prior result; point shown).
    ax.errorbar(
        0,
        refusal_point,
        yerr=[[max(0.0, refusal_point - refusal_ci[0])], [max(0.0, refusal_ci[1] - refusal_point)]],
        fmt="none",
        ecolor="black",
        capsize=4,
        alpha=0.8,
    )
    ax.axhspan(-thr, thr, color="grey", alpha=0.15, label=f"±{thr} separation band")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel("adapter-vs-dense bystander-leakage gap at s*=0.50")
    ax.set_title(
        f"Does the adapter-vs-dense leakage gap hold across behaviors?\n"
        f"verdict {h['verdict']}: refusal Δ_rank={refusal_point:+.3f} vs sycophancy +{syco_point}"
    )
    ax.legend(fontsize=8)
    _save(fig, "refusal_cross_behavior_adapter_vs_dense_hero", out_dir)


def fig_v9_leakage_vs_strength(analysis: dict, out_dir: Path) -> None:
    """v9 exploratory: the two-arm refusal leakage-vs-strength overlay (x=s,
    y=29-bystander-mean refusal delta, matched band shaded)."""
    s = analysis["s_stage_b"]
    tables = analysis["per_cell_tables"]
    profile = analysis.get("profile", {})
    bystanders = list(profile.get("per_persona_at_target", {}).get("loraRefOP", {})) or [
        p for p in next(iter(tables.values())) if p != "villain"
    ]
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for arm in ("loraRefOP", "cmftRefOP"):
        pts = []
        for cell, s_val in s.items():
            if cell.split("_step")[0] != arm:
                continue
            deltas = [
                tables[cell][p]["delta_clean"]
                for p in bystanders
                if tables[cell][p]["delta_clean"] is not None
            ]
            pts.append((s_val, float(np.nanmean(deltas))))
        pts.sort()
        if not pts:
            continue
        xs = [0.0, *(t[0] for t in pts)]
        ys = [0.0, *(t[1] for t in pts)]
        ax.plot(xs, ys, marker="o", color=V9_ARM_COLORS[arm], label=V9_ARM_LABELS[arm])
    ax.axvspan(S_BAND[0], S_BAND[1], color="grey", alpha=0.12, label="matched band")
    ax.axvline(S_TARGET, color="black", ls="--", lw=0.8)
    ax.set_xlabel("source-self install strength s (refusal)")
    ax.set_ylabel("29-bystander-mean refusal leakage")
    ax.set_title("refusal: bystander leakage vs install strength (2 arms)")
    ax.legend(fontsize=8)
    _save(fig, "refusal_leakage_vs_strength", out_dir)


def fig_v9_profile_scatter(analysis: dict, out_dir: Path) -> None:
    """v9 exploratory: per-persona profile scatter at s*, cmftRefOP vs loraRefOP,
    identity line, ρ annotated."""
    profile = analysis.get("profile", {})
    pps = profile.get("per_persona_at_target", {})
    if "cmftRefOP" not in pps or "loraRefOP" not in pps:
        log.warning("v9 profile scatter: per-persona reads missing — skipping")
        return
    personas = sorted(pps["loraRefOP"])
    xs = [pps["loraRefOP"][p] for p in personas]
    ys = [pps["cmftRefOP"][p] for p in personas]
    rho = profile.get("rho", float("nan"))
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(xs, ys, color="tab:green", alpha=0.8)
    lim = [min([*xs, *ys, 0.0]), max([*xs, *ys, 0.01])]
    ax.plot(lim, lim, color="black", ls="--", lw=0.8, label="identity")
    ax.set_xlabel("LoRA refusal leakage per bystander (s*)")
    ax.set_ylabel("cmft refusal leakage per bystander (s*)")
    ax.set_title(f"refusal per-persona profile: cmft vs LoRA (ρ={rho:.2f})")
    ax.legend(fontsize=8)
    _save(fig, "refusal_profile_cmftRefOP_vs_loraRefOP", out_dir)


def fig_v9_trajectories(trajectory: dict, out_dir: Path, *, name: str) -> None:
    """v9 exploratory: per-arm s(step) source-self refusal trajectories overlaid
    (showing where each arm brackets s* at the chosen LR)."""
    cells = trajectory["cells"]
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for arm in ("loraRefOP", "cmftRefOP"):
        pts = sorted(
            (
                (rec["step"], rec.get("s", float("nan")))
                for rec in cells.values()
                if rec.get("arm") == arm
            ),
            key=lambda t: t[0],
        )
        if not pts:
            continue
        ax.plot(
            [t[0] for t in pts],
            [t[1] for t in pts],
            marker="o",
            color=V9_ARM_COLORS[arm],
            label=V9_ARM_LABELS[arm],
        )
    ax.axhspan(S_BAND[0], S_BAND[1], color="grey", alpha=0.12, label="matched band")
    ax.axhline(S_TARGET, color="black", ls="--", lw=0.8)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("source-self refusal install strength s")
    ax.set_title("refusal source-self install trajectories (2 arms)")
    ax.legend(fontsize=8)
    _save(fig, name, out_dir)


def fig_v9_heatmap(analysis: dict, out_dir: Path) -> None:
    """v9 exploratory: per-persona refusal-leakage heatmap (clean delta at s*)."""
    profile = analysis.get("profile", {})
    pps = profile.get("per_persona_at_target", {})
    if "cmftRefOP" not in pps or "loraRefOP" not in pps:
        return
    personas = sorted(pps["loraRefOP"])
    arms = ["loraRefOP", "cmftRefOP"]
    mat = np.array([[pps[a][p] for p in personas] for a in arms])
    fig, ax = plt.subplots(figsize=(max(6.0, 0.3 * len(personas)), 3.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([V9_ARM_LABELS[a] for a in arms])
    ax.set_xticks(range(len(personas)))
    ax.set_xticklabels(personas, rotation=90, fontsize=6)
    ax.set_title("refusal per-persona leakage at s*=0.50")
    fig.colorbar(im, ax=ax, fraction=0.025)
    _save(fig, "refusal_per_persona_heatmap", out_dir)


def fig_v9_sweep(analysis: dict, out_dir: Path) -> None:
    """v9 exploratory: Δ_rank_matched-vs-s* sweep curve."""
    sweep = analysis.get("sweep", {}).get("delta_rank_matched", [])
    if not sweep:
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(
        [r["s_target"] for r in sweep],
        [r["gap"] for r in sweep],
        marker="o",
        color="tab:green",
    )
    ax.axvline(S_TARGET, color="black", ls="--", lw=0.8, label="s*=0.50")
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xlabel("matched-strength target s*")
    ax.set_ylabel("Δ_rank_matched (cmft − LoRA)")
    ax.set_title("refusal Δ_rank_matched vs s*")
    ax.legend(fontsize=8)
    _save(fig, "refusal_delta_rank_vs_target_sweep", out_dir)


def fig_v9_install_pilot(pilot_gate: dict, out_dir: Path) -> None:
    """v9 exploratory: the install-pilot trajectory panel (the LR decision
    evidence — per-arm s by step at each piloted LR leg)."""
    legs = pilot_gate.get("legs", [pilot_gate])
    fig, axes = plt.subplots(
        1, max(1, len(legs)), figsize=(5.5 * max(1, len(legs)), 4.0), squeeze=False
    )
    for j, leg in enumerate(legs):
        ax = axes[0][j]
        lr = leg.get("lr", leg.get("pilot_lr", "?"))
        for arm, v in leg.get("arms", {}).items():
            sbs = v.get("s_by_step", {})
            steps = sorted(int(k) for k in sbs)
            ax.plot(
                steps,
                [sbs[str(s)] for s in steps],
                marker="o",
                color=V9_ARM_COLORS.get(arm, "tab:purple"),
                label=V9_ARM_LABELS.get(arm, arm),
            )
        ax.axhline(S_TARGET, color="black", ls="--", lw=0.8)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("source-self refusal s")
        ax.set_title(f"install-pilot leg (LR {lr})")
        ax.legend(fontsize=7)
    fig.suptitle(
        f"install-pilot LR decision (chosen matched LR = {pilot_gate.get('chosen_lr', '?')})"
    )
    _save(fig, "refusal_install_pilot_trajectory", out_dir)


def _figures_v9(eval_root: Path, out_dir: Path) -> None:
    behavior = "refusal"
    broot = eval_root / behavior
    # Minor 1 (round-1 review): analyzer writes analysis.json (plan §6.5 contract).
    analysis = json.loads((broot / "analysis.json").read_text())
    fig_v9_cross_behavior_hero(analysis, out_dir)
    fig_v9_leakage_vs_strength(analysis, out_dir)
    fig_v9_profile_scatter(analysis, out_dir)
    fig_v9_heatmap(analysis, out_dir)
    fig_v9_sweep(analysis, out_dir)
    traj_path = broot / "stage_a" / f"trajectory_{behavior}.json"
    if traj_path.exists():
        fig_v9_trajectories(
            json.loads(traj_path.read_text()), out_dir, name="refusal_source_self_trajectories"
        )
    pilot_path = broot / "stage_a_pilot" / "pilot_gate.json"
    if pilot_path.exists():
        fig_v9_install_pilot(json.loads(pilot_path.read_text()), out_dir)
    log.info("all v9 figures written to %s", out_dir)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=figures] %(message)s")
    p = argparse.ArgumentParser(description="#642 figures (decomposition heroes + exploratory).")
    p.add_argument("--behavior", choices=["sycophancy", "refusal"])
    p.add_argument("--eval-root", type=Path, default=REPO / "eval_results" / "issue_642")
    p.add_argument("--out-dir", type=Path, default=REPO / "figures" / "issue_642")
    p.add_argument(
        "--v9",
        action="store_true",
        help="v9 refusal figures (plan v10 §6): cross-behavior adapter-vs-dense hero + the "
        "exploratory dump (leakage-vs-strength, profile scatter, trajectories, heatmap, sweep, "
        "install-pilot panel). Reads analysis.json; writes to figures/issue_642/v9/.",
    )
    args = p.parse_args(argv)

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style("blog")

    if args.v9:
        out_dir = args.out_dir
        if out_dir == REPO / "figures" / "issue_642":
            out_dir = out_dir / "v9"
        out_dir.mkdir(parents=True, exist_ok=True)
        _figures_v9(args.eval_root, out_dir)
        return 0

    if not args.behavior:
        raise SystemExit("--behavior is required (unless --v9)")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    broot = args.eval_root / args.behavior
    analysis = json.loads((broot / "analysis.json").read_text())
    trajectory = json.loads((broot / "stage_a" / f"trajectory_{args.behavior}.json").read_text())

    fig_decomposition_bar(analysis, args.behavior, args.out_dir)
    fig_leakage_vs_strength(analysis, args.behavior, args.out_dir)
    _profile_scatter(analysis, args.behavior, args.out_dir, "cmft", "lora")
    _profile_scatter(analysis, args.behavior, args.out_dir, "ft", "cmft")
    fig_trajectories(trajectory, args.behavior, args.out_dir)
    fig_heatmaps(analysis, args.behavior, args.out_dir)
    fig_sweep(analysis, args.behavior, args.out_dir)
    fig_endpoint_comparison(analysis, args.behavior, args.out_dir)
    log.info("all figures written to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
