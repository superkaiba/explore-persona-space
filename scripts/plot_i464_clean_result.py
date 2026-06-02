"""Plots for issue #464 clean-result.

Issue #464 plan v2 §4.1 Phase 5 + §6.2 + §6.3.

Reads ``eval_results/issue_464/analysis.json`` (Phase 5 output) plus the
per-cell JSON dump (``eval_results/issue_464/cross_eval/per_cell/``) and
emits the canonical plots into ``figures/issue_464/``:

  hero.png                          - per-arm bars: own-persona elicitation
                                       + symmetric leakage (paired bootstrap CI annotations)
  matrix_<arm>.png                  - 5x2 heatmap (eval-encoding x marker) per arm
  leakage_by_eval_encoding.png      - exploratory: leakage by wrong-encoding type
  per_seed.png                      - per-seed elicitation vs leakage scatter
  raw_alongside_processed.png       - raw trained log P + ΔlogP side-by-side
  dynamic_range_check.png           - per-cell raw log-P histograms
  argmax_emission_per_cell.png      - emission-rate heatmap, legibility

Each figure is paired with a ``<name>.meta.json`` capturing the git
commit + input files (CLAUDE.md reproducibility metadata).

CLI:
    uv run python scripts/plot_i464_clean_result.py
    uv run python scripts/plot_i464_clean_result.py --no-show
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from explore_persona_space.experiments import i464_encodings as enc

load_dotenv()
matplotlib.use("Agg")  # headless

logger = logging.getLogger("i464.plot")

PER_CELL_DIR = Path("eval_results/issue_464/cross_eval/per_cell")
ANALYSIS_PATH = Path("eval_results/issue_464/analysis.json")
ONPOLICY_PATH = Path("eval_results/issue_464/onpolicy_validation.json")
FIG_DIR = Path("figures/issue_464")

SEEDS_DEFAULT = (42, 137, 1337)

# Per-arm color palette. Stable across plots so a reader recognizes
# system_plain / system_padded / role / role_nonsense / role_mismatch
# at a glance. role_nonsense (orange) and role_mismatch (purple) visually
# pair with role (blue) — the three role-family colors together make the
# "what does the role slot's reduction depend on?" comparison readable.
ARM_COLORS: dict[str, str] = {
    "system_plain": "#666",
    "system_padded": "#a44",
    "role": "#48a",
    "role_nonsense": "#e6a23c",
    "role_mismatch": "#8a4ab0",
}
ARM_COLORS_LIST: list[str] = [ARM_COLORS.get(a, "#888") for a in enc.ARMS]


def _git_commit_hash() -> str:
    """Return HEAD sha or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, env={**os.environ}
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _save(fig, name: str, sources: list[str]) -> None:
    """Save fig + sidecar meta.json."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / f"{name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    meta = {
        "name": name,
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "sources": sources,
    }
    (FIG_DIR / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %s + meta.json", fig_path)


def _load_per_cell(cell: str, e_eval: str, marker_persona: str) -> dict | None:
    """Read one per-cell JSON or return None if missing."""
    p = PER_CELL_DIR / f"{cell}__{e_eval}__marker_{marker_persona}.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    return json.loads(p.read_text())


def _own_persona_elicitation(arm: enc.Arm, seed: int) -> float:
    """Mean own-persona elicitation log-prob across the 2 personas under each arm's own encoding.

    Own-encoding mapping (mirror of phase5 ``_own_eval_encoding_for``):
      system_plain / system_padded → system_<persona>
      role                          → role_<persona>
      role_nonsense                 → role_nonsense_<persona>
      role_mismatch                 → role_mismatch_<persona>
    """
    cell = f"{arm}_seed{seed}"
    own_logps: list[float] = []
    for persona in enc.PERSONAS:
        if arm == "role":
            e = f"role_{persona}"
        elif arm == "role_nonsense":
            e = f"role_nonsense_{persona}"
        elif arm == "role_mismatch":
            e = f"role_mismatch_{persona}"
        else:
            e = f"system_{persona}"
        payload = _load_per_cell(cell, e, persona)
        if payload is not None:
            own_logps.append(payload["g_logprob"])
    return float(np.mean(own_logps)) if own_logps else float("nan")


def _l_arm_values(analysis: dict, arm: str) -> list[float]:
    """Return per-seed L_arm values as a list (handles both v1 list + v2 dict schemas).

    Round-2 Phase 5 (schema_version i464_phase5_v2) writes
    ``L_per_arm_per_seed`` as ``dict[arm] -> dict[int seed -> float]``.
    Round-1 v1 wrote it as ``dict[arm] -> list[float]``. Plot script
    handles both so an old analysis.json doesn't crash the new plot.
    """
    raw = analysis.get("L_per_arm_per_seed", {}).get(arm)
    if raw is None:
        return []
    if isinstance(raw, dict):
        # JSON round-trip converts int seeds to str; sort numerically.
        return [raw[k] for k in sorted(raw.keys(), key=lambda x: int(x))]
    if isinstance(raw, list):
        return list(raw)
    return []


def plot_hero(analysis: dict) -> None:
    """3 arms x {elicitation, symmetric-leakage} bars + per-seed error bars."""
    seeds = analysis["seeds"]
    arms = list(enc.ARMS)
    # Per-arm L values list (handles v1 list + v2 dict schemas).
    L_per_arm = {a: _l_arm_values(analysis, a) for a in arms}

    elicit_by_arm = {a: [_own_persona_elicitation(a, s) for s in seeds] for a in arms}  # type: ignore[arg-type]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(arms))

    ax[0].bar(
        x,
        [np.mean(elicit_by_arm[a]) for a in arms],
        yerr=[np.std(elicit_by_arm[a]) for a in arms],
        color=ARM_COLORS_LIST,
    )
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(arms, rotation=15)
    ax[0].set_ylabel("raw trained log P(marker | own encoding)")
    ax[0].set_title("Own-persona elicitation")
    for i, a in enumerate(arms):
        for v in elicit_by_arm[a]:
            ax[0].plot(i, v, "o", color="black", markersize=3, alpha=0.6)

    leak_means = [np.mean(L_per_arm[a]) if L_per_arm[a] else float("nan") for a in arms]
    leak_stds = [np.std(L_per_arm[a]) if L_per_arm[a] else 0 for a in arms]
    ax[1].bar(x, leak_means, yerr=leak_stds, color=ARM_COLORS_LIST)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(arms, rotation=15)
    ax[1].set_ylabel("symmetric leakage L_arm (nats; lower = less leakage)")
    ax[1].set_title("Symmetric leakage (headline)")
    for i, a in enumerate(arms):
        for v in L_per_arm[a]:
            ax[1].plot(i, v, "o", color="black", markersize=3, alpha=0.6)

    # Round-2 headline shape: { status, d_seed_plain: {mean, ci_lo_95, ...}, ... }
    # for "ok"/"partial"/"fail" status; for inconclusive/blocked the d_seed
    # entries are descriptive lists or absent. Render only when the full
    # bootstrap CI keys are present.
    h = analysis.get("headline") or {}
    if (
        isinstance(h.get("d_seed_plain"), dict)
        and "mean" in h["d_seed_plain"]
        and isinstance(h.get("d_seed_padded"), dict)
        and "mean" in h["d_seed_padded"]
    ):
        sub = (
            f"status={h.get('status', '?')}; "
            f"d_plain mean={h['d_seed_plain']['mean']:.2f} "
            f"CI=[{h['d_seed_plain']['ci_lo_95']:.2f}, {h['d_seed_plain']['ci_hi_95']:.2f}]; "
            f"d_padded mean={h['d_seed_padded']['mean']:.2f} "
            f"CI=[{h['d_seed_padded']['ci_lo_95']:.2f}, {h['d_seed_padded']['ci_hi_95']:.2f}]"
        )
    elif h.get("status"):
        sub = f"status={h['status']}: {h.get('reason', '')[:120]}"
    else:
        sub = ""
    if sub:
        fig.suptitle(sub, fontsize=9, y=1.02)
    _save(fig, "hero", [str(ANALYSIS_PATH)])


def plot_matrix_per_arm(analysis: dict) -> None:
    """5x2 heatmap (eval encoding x marker) per arm, averaged across 3 seeds."""
    seeds = analysis["seeds"]
    for arm in enc.ARMS:
        grid = np.full((len(enc.EVAL_ENCODINGS), len(enc.PERSONAS)), np.nan)
        for i, e_eval in enumerate(enc.EVAL_ENCODINGS):
            for j, persona in enumerate(enc.PERSONAS):
                vals: list[float] = []
                for seed in seeds:
                    cell = f"{arm}_seed{seed}"
                    p = _load_per_cell(cell, e_eval, persona)
                    if p is not None:
                        vals.append(p["g_logprob"])
                if vals:
                    grid[i, j] = float(np.mean(vals))
        fig, ax = plt.subplots(figsize=(4, 5))
        im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=-15, vmax=0)
        ax.set_yticks(np.arange(len(enc.EVAL_ENCODINGS)))
        ax.set_yticklabels(enc.EVAL_ENCODINGS)
        ax.set_xticks(np.arange(len(enc.PERSONAS)))
        ax.set_xticklabels([f"marker_{p}" for p in enc.PERSONAS])
        ax.set_title(f"raw trained log P — arm={arm} (mean over {len(seeds)} seeds)")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, label="log P")
        _save(fig, f"matrix_{arm}", [str(PER_CELL_DIR)])


def plot_per_seed_scatter(analysis: dict) -> None:
    """For each seed, scatter (elicitation, leakage) per arm.

    Round-5 fix (review blocker #4): when an arm is missing leakage cells
    (smoke runs with one cell only, or `--allow-partial` production
    runs that lose a cell mid-sweep), `elicits` and `leaks` arrays
    differ in length and `ax.scatter` raises ``ValueError: x and y
    must be the same size``. The defensive fix pairs elicit+leak per
    seed and only plots when BOTH are present for that seed. Arms
    with no usable (x, y) pairs are skipped (logged as a warning so
    the operator sees the missing-data signal).
    """
    seeds = analysis["seeds"]
    fig, ax = plt.subplots(figsize=(6, 5))
    arms = list(enc.ARMS)
    colors = ARM_COLORS
    skipped: list[str] = []
    for arm in arms:
        elicits = [_own_persona_elicitation(arm, s) for s in seeds]  # type: ignore[arg-type]
        leaks = _l_arm_values(analysis, arm)
        # Pair by index up to the shorter list; drop pairs with NaN elicit
        # (cell missing) so the scatter is honest about what's plotted.
        n_pairs = min(len(elicits), len(leaks))
        xs = [
            elicit
            for elicit, _leak in zip(elicits[:n_pairs], leaks[:n_pairs], strict=True)
            if not (isinstance(elicit, float) and np.isnan(elicit))
        ]
        ys = [
            leak
            for elicit, leak in zip(elicits[:n_pairs], leaks[:n_pairs], strict=True)
            if not (isinstance(elicit, float) and np.isnan(elicit))
        ]
        if not xs or not ys:
            skipped.append(arm)
            continue
        ax.scatter(xs, ys, s=60, color=colors[arm], label=arm)
    ax.set_xlabel("own-persona elicitation log P")
    ax.set_ylabel("symmetric leakage L_arm")
    title = "Per-seed: elicitation vs leakage by arm"
    if skipped:
        title += f"  (skipped arms with missing cells: {skipped})"
        logger.warning("plot_per_seed_scatter: skipped arms with missing cells: %s", skipped)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "per_seed", [str(ANALYSIS_PATH)])


def plot_raw_alongside_processed(analysis: dict) -> None:
    """For each arm x marker, plot raw log P AND ΔlogP side-by-side (9x2 mini)."""
    seeds = analysis["seeds"]
    arms = list(enc.ARMS)
    fig, axes = plt.subplots(2, len(arms), figsize=(4 * len(arms), 6), sharex=True)
    for col, arm in enumerate(arms):
        own_raw = [_own_persona_elicitation(arm, s) for s in seeds]  # type: ignore[arg-type]
        leak_raw = _l_arm_values(analysis, arm)
        axes[0, col].bar(
            [0, 1],
            [np.mean(own_raw), np.mean(leak_raw) if leak_raw else float("nan")],
            yerr=[np.std(own_raw), np.std(leak_raw) if leak_raw else 0],
        )
        axes[0, col].set_xticks([0, 1])
        axes[0, col].set_xticklabels(["own", "wrong"])
        axes[0, col].set_title(f"raw log P — {arm}")
        # Delta: use raw_per_cell from analysis to compute ΔlogP avg per cell.
        deltas: list[float] = []
        for seed in seeds:
            for persona in enc.PERSONAS:
                other = "villain" if persona == "pirate" else "pirate"
                for e_wrong in [f"system_{other}", f"role_{other}"]:
                    p = _load_per_cell(f"{arm}_seed{seed}", e_wrong, persona)
                    if p is not None:
                        deltas.append(p["delta_g"])
        if deltas:
            axes[1, col].hist(deltas, bins=10, color="#a44", alpha=0.7)
            axes[1, col].set_title(f"ΔlogP (wrong) — {arm}")
        axes[1, col].set_xlabel("Δ log P (trained - base)")
    _save(fig, "raw_alongside_processed", [str(ANALYSIS_PATH), str(PER_CELL_DIR)])


def plot_dynamic_range(analysis: dict) -> None:
    """Per-arm histogram of raw trained log P across the symmetric leakage cells."""
    fig, axes = plt.subplots(1, len(enc.ARMS), figsize=(4 * len(enc.ARMS), 4), sharex=True)
    for col, arm in enumerate(enc.ARMS):
        all_raw: list[float] = []
        for seed_vals in analysis["raw_per_cell"][arm].values():
            all_raw.extend(seed_vals)
        if all_raw:
            axes[col].hist(all_raw, bins=12, color="#48a", alpha=0.7)
            sd = analysis["dynamic_range_gate"]["per_arm"][arm]["sd"]
            axes[col].set_title(f"{arm} (sd={sd:.2f})")
            axes[col].set_xlabel("raw trained log P")
    _save(fig, "dynamic_range_check", [str(ANALYSIS_PATH)])


def plot_argmax_emission(analysis: dict) -> None:
    """Emission-rate heatmap (10 cells per LoRA) averaged across seeds."""
    seeds = analysis["seeds"]
    for arm in enc.ARMS:
        grid = np.full((len(enc.EVAL_ENCODINGS), len(enc.PERSONAS)), np.nan)
        for i, e_eval in enumerate(enc.EVAL_ENCODINGS):
            for j, persona in enumerate(enc.PERSONAS):
                vals: list[float] = []
                for seed in seeds:
                    p = _load_per_cell(f"{arm}_seed{seed}", e_eval, persona)
                    if p is not None:
                        vals.append(p["emission_recompute_rate"])
                if vals:
                    grid[i, j] = float(np.mean(vals))
        fig, ax = plt.subplots(figsize=(4, 5))
        im = ax.imshow(grid, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_yticks(np.arange(len(enc.EVAL_ENCODINGS)))
        ax.set_yticklabels(enc.EVAL_ENCODINGS)
        ax.set_xticks(np.arange(len(enc.PERSONAS)))
        ax.set_xticklabels([f"marker_{p}" for p in enc.PERSONAS])
        ax.set_title(f"argmax==marker fraction — arm={arm}")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, label="fraction")
        _save(fig, f"argmax_emission_{arm}", [str(PER_CELL_DIR)])


def plot_leakage_by_eval_encoding(analysis: dict) -> None:
    """Exploratory: leakage decomposed by wrong-encoding family + default."""
    seeds = analysis["seeds"]
    arms = list(enc.ARMS)
    encs = ["system_OTHER", "role_OTHER", "default_assistant"]
    grid = np.full((len(arms), len(encs)), np.nan)
    for i, arm in enumerate(arms):
        for j, ekey in enumerate(encs):
            vals: list[float] = []
            for seed in seeds:
                for persona in enc.PERSONAS:
                    other = "villain" if persona == "pirate" else "pirate"
                    if ekey == "system_OTHER":
                        e = f"system_{other}"
                    elif ekey == "role_OTHER":
                        e = f"role_{other}"
                    else:
                        e = "default_assistant"
                    p = _load_per_cell(f"{arm}_seed{seed}", e, persona)
                    if p is not None:
                        vals.append(p["g_logprob"])
            if vals:
                grid[i, j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(encs))
    n_arms = len(arms)
    w = 0.8 / max(n_arms, 1)
    colors = ARM_COLORS_LIST
    # Center the cluster around each x tick: offsets = (i - (n_arms-1)/2) * w
    for i, arm in enumerate(arms):
        ax.bar(x + (i - (n_arms - 1) / 2) * w, grid[i, :], width=w, color=colors[i], label=arm)
    ax.set_xticks(x)
    ax.set_xticklabels(encs)
    ax.set_ylabel("raw trained log P at slot (mean over seeds x personas)")
    ax.set_title("Leakage decomposed by wrong-encoding family (lower = less leakage)")
    ax.legend()
    _save(fig, "leakage_by_eval_encoding", [str(PER_CELL_DIR)])


def plot_role_nonsense_comparison(analysis: dict) -> None:
    """role_nonsense follow-up: visualise whether semantic name buys anything.

    Two panels:
      (1) Per-arm symmetric-leakage means for ALL 4 arms — shows where
          role_nonsense sits between role (low) and system_plain (high).
      (2) Paired per-seed deltas:
            d_seed_role_nonsense_vs_plain = L_system_plain - L_role_nonsense
              (>0: nonsense role-slot alone reduces leakage vs system_plain)
            d_seed_role_nonsense_vs_role  = L_role - L_role_nonsense
              (≈0: semantic name adds nothing on top of the slot
              >0: semantics buys further reduction
              <0: semantics hurts)

    Skipped if role_nonsense_descriptive is missing or empty in the
    analysis (e.g. a partial run that didn't include the 4th arm).
    """
    rn = analysis.get("role_nonsense_descriptive") or {}
    d_vs_plain = rn.get("d_seed_role_nonsense_vs_plain") or []
    d_vs_role = rn.get("d_seed_role_nonsense_vs_role") or []
    if not d_vs_plain:
        logger.warning(
            "role_nonsense_comparison.png skipped — role_nonsense_descriptive "
            "absent / empty (no paired seeds for the follow-up arm yet)"
        )
        return

    arms = list(enc.ARMS)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    # Panel 1: per-arm symmetric leakage means.
    L_means = [
        float(np.mean(_l_arm_values(analysis, a))) if _l_arm_values(analysis, a) else float("nan")
        for a in arms
    ]
    L_stds = [
        float(np.std(_l_arm_values(analysis, a))) if _l_arm_values(analysis, a) else 0.0
        for a in arms
    ]
    x = np.arange(len(arms))
    ax[0].bar(x, L_means, yerr=L_stds, color=ARM_COLORS_LIST)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(arms, rotation=15)
    ax[0].set_ylabel("symmetric leakage L_arm (nats; lower = less leakage)")
    ax[0].set_title("Per-arm leakage incl. role_nonsense")
    for i, a in enumerate(arms):
        for v in _l_arm_values(analysis, a):
            ax[0].plot(i, v, "o", color="black", markersize=3, alpha=0.6)

    # Panel 2: paired per-seed deltas (descriptive — no PASS gate).
    rn_seeds = rn.get("complete_seeds") or list(range(len(d_vs_plain)))
    width = 0.35
    xs = np.arange(len(rn_seeds))
    ax[1].bar(
        xs - width / 2,
        d_vs_plain,
        width,
        label="vs system_plain",
        color=ARM_COLORS["role_nonsense"],
    )
    ax[1].bar(xs + width / 2, d_vs_role, width, label="vs role", color=ARM_COLORS["role"])
    ax[1].axhline(0, color="black", linewidth=0.5)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"seed {s}" for s in rn_seeds])
    ax[1].set_ylabel("paired Δ vs role_nonsense (nats)")
    ax[1].set_title(
        "Does the role SLOT alone suffice?\n"
        "(Δvs_plain >0: slot helps; Δvs_role ≈0: semantics adds nothing)"
    )
    ax[1].legend(fontsize=8)

    sub = f"mean Δ_vs_plain = {np.mean(d_vs_plain):.2f}; mean Δ_vs_role = {np.mean(d_vs_role):.2f}"
    fig.suptitle(sub, fontsize=9, y=1.02)
    _save(fig, "role_nonsense_comparison", [str(ANALYSIS_PATH)])


def plot_role_mismatch_comparison(analysis: dict) -> None:
    """role_mismatch follow-up: visualise whether real-but-mismatched meaning matters.

    Two panels:
      (1) Per-arm symmetric-leakage means across the three role-family
          arms + system_plain reference — shows where role_mismatch sits
          relative to role (matched meaning), role_nonsense (no meaning),
          and system_plain (no role slot).
      (2) Paired per-seed deltas:
            d_seed_role_mismatch_vs_plain    = L_system_plain  - L_role_mismatch
              (>0: role_mismatch leaks less than system_plain — slot helps)
            d_seed_role_mismatch_vs_role     = L_role          - L_role_mismatch
              (≈0: matched semantics doesn't matter — slot does the work
              <0: matched semantics genuinely helps vs mismatched
              >0: mismatch helps further)
            d_seed_role_mismatch_vs_nonsense = L_role_nonsense - L_role_mismatch
              (≈0: meaningfulness adds nothing on top of the slot
              >0: real meaning (even mismatched) reduces leakage vs gibberish
              <0: real-but-wrong meaning HURTS vs gibberish)

    Skipped if role_mismatch_descriptive is missing or empty in the
    analysis (e.g. a partial run that didn't include the 5th arm).
    """
    rm = analysis.get("role_mismatch_descriptive") or {}
    d_vs_plain = rm.get("d_seed_role_mismatch_vs_plain") or []
    d_vs_role = rm.get("d_seed_role_mismatch_vs_role") or []
    d_vs_nonsense = rm.get("d_seed_role_mismatch_vs_nonsense") or []
    if not d_vs_plain:
        logger.warning(
            "role_mismatch_comparison.png skipped — role_mismatch_descriptive "
            "absent / empty (no paired seeds for the follow-up arm yet)"
        )
        return

    arms = list(enc.ARMS)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    # Panel 1: per-arm symmetric leakage means.
    L_means = [
        float(np.mean(_l_arm_values(analysis, a))) if _l_arm_values(analysis, a) else float("nan")
        for a in arms
    ]
    L_stds = [
        float(np.std(_l_arm_values(analysis, a))) if _l_arm_values(analysis, a) else 0.0
        for a in arms
    ]
    x = np.arange(len(arms))
    ax[0].bar(x, L_means, yerr=L_stds, color=ARM_COLORS_LIST)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(arms, rotation=15)
    ax[0].set_ylabel("symmetric leakage L_arm (nats; lower = less leakage)")
    ax[0].set_title("Per-arm leakage incl. role_mismatch")
    for i, a in enumerate(arms):
        for v in _l_arm_values(analysis, a):
            ax[0].plot(i, v, "o", color="black", markersize=3, alpha=0.6)

    # Panel 2: paired per-seed deltas (descriptive — no PASS gate).
    rm_seeds = rm.get("complete_seeds") or list(range(len(d_vs_plain)))
    n_groups = 3 if d_vs_nonsense else 2
    width = 0.8 / n_groups
    xs = np.arange(len(rm_seeds))
    ax[1].bar(
        xs - width * (n_groups - 1) / 2,
        d_vs_plain,
        width,
        label="vs system_plain",
        color=ARM_COLORS["role_mismatch"],
    )
    ax[1].bar(
        xs - width * (n_groups - 1) / 2 + width,
        d_vs_role,
        width,
        label="vs role",
        color=ARM_COLORS["role"],
    )
    if d_vs_nonsense:
        ax[1].bar(
            xs - width * (n_groups - 1) / 2 + 2 * width,
            d_vs_nonsense,
            width,
            label="vs role_nonsense",
            color=ARM_COLORS["role_nonsense"],
        )
    ax[1].axhline(0, color="black", linewidth=0.5)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"seed {s}" for s in rm_seeds])
    ax[1].set_ylabel("paired Δ vs role_mismatch (nats)")
    ax[1].set_title(
        "Does the role name need to MATCH the trained persona?\n"
        "(Δvs_role ≈0: no; Δvs_role <0: matched semantics genuinely helps)"
    )
    ax[1].legend(fontsize=8)

    pieces = [
        f"mean Δ_vs_plain = {np.mean(d_vs_plain):.2f}",
        f"mean Δ_vs_role = {np.mean(d_vs_role):.2f}",
    ]
    if d_vs_nonsense:
        pieces.append(f"mean Δ_vs_nonsense = {np.mean(d_vs_nonsense):.2f}")
    fig.suptitle("; ".join(pieces), fontsize=9, y=1.02)
    _save(fig, "role_mismatch_comparison", [str(ANALYSIS_PATH)])


def plot_onpolicy_validation() -> None:
    """MF-B(2) on-policy validation visualization (review blocker #5).

    Reads ``eval_results/issue_464/onpolicy_validation.json`` (Phase 4.5
    output). One panel: per-arm mean normalized character edit-distance
    between the trained-greedy R and R_canon, with the 1.5x switch
    threshold annotated. If the file is absent (e.g. CPU smoke run that
    skipped Phase 4.5), log a warning and skip — never crash the plot
    pipeline.
    """
    if not ONPOLICY_PATH.exists() or ONPOLICY_PATH.stat().st_size == 0:
        logger.warning(
            "onpolicy_validation.png skipped — %s missing (Phase 4.5 not run yet?)",
            ONPOLICY_PATH,
        )
        return
    payload = json.loads(ONPOLICY_PATH.read_text())
    per_arm = payload.get("per_arm", {})
    arms = list(enc.ARMS)
    means = [(per_arm.get(a, {}) or {}).get("mean") or float("nan") for a in arms]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(arms))
    colors = ARM_COLORS_LIST
    ax.bar(x, means, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=15)
    ax.set_ylabel("mean normalized char edit-distance(R_trained, R_canon)")
    ax.set_title("Phase 4.5 on-policy validation (MF-B(2))")

    # Annotate 1.5x switch threshold relative to system_plain.
    sp_mean = per_arm.get("system_plain", {}).get("mean")
    if sp_mean and sp_mean > 0:
        threshold = sp_mean * payload.get("switch_threshold", 1.5)
        ax.axhline(
            threshold,
            linestyle="--",
            color="red",
            label=f"switch threshold = {payload.get('switch_threshold', 1.5)}x system_plain",
        )
        ax.legend()

    ratio = payload.get("role_over_system_plain_ratio")
    switch = payload.get("switch_headline_to_trained_R", False)
    sub = f"role/system_plain ratio = {ratio}; switch_headline = {switch}"
    fig.suptitle(sub, fontsize=9, y=1.02)
    _save(fig, "onpolicy_validation", [str(ONPOLICY_PATH)])


def plot_trajectory() -> None:  # noqa: C901 - two source paths (trajectory.json + per-step JSON glob) push branching to 16
    """MF-C marker-log-prob trajectory plot (review blocker #5).

    Source order:
      1. ``eval_results/issue_464/trajectory.json`` — analyst-exported
         WandB run history (one row per step, keys mirror the callback's
         ``marker_logp/{arm}/{persona}/{e_eval}`` shape). This is the
         canonical post-hoc path; the analyzer wrangles WandB's API in a
         small wrangling script and dumps the JSON.
      2. ``adapters/i464_<arm>_seed<seed>/_traj_adapter/_logprobs_step
         *.json`` — raw per-step JSONs the callback wrote during training.
         Used when no curated trajectory.json exists.
      3. If neither source is present, log a warning and skip — never
         crash the plot pipeline.

    Each panel = one arm. Line per (persona, eval_encoding) keyed in the
    callback's namespace.

    Round-3 fix (review blocker #2): the round-2 fallback globbed
    ``data/issue_464/*_traj_adapter``, but the MF-C callback's actual
    dump dir is `<output_dir>/_traj_adapter` where `output_dir =
    adapters/i464_<arm>_seed<seed>` (set in `scripts/i464_phase23_train
    .py:405` and forwarded to `train_lora` which adapter-dumps under
    `<output_dir>/_traj_adapter` in `src/explore_persona_space/train/
    sft.py:706-708`). The round-2 wrong glob meant trajectory.png was
    NEVER produced in a normal run; the round-3 glob is
    `adapters/i464_*_seed*/_traj_adapter`.
    """
    traj_path = Path("eval_results/issue_464/trajectory.json")
    series_by_arm: dict[str, dict[str, list[tuple[int, float]]]] = {}

    if traj_path.exists() and traj_path.stat().st_size > 0:
        payload = json.loads(traj_path.read_text())
        # Expected schema: {"steps": [int, ...],
        #                   "metrics": {"marker_logp/<arm>/<persona>/<e_eval>": [float, ...]}}
        steps = payload.get("steps", [])
        metrics = payload.get("metrics", {})
        for key, values in metrics.items():
            # Key shape: "marker_logp/<arm>/<persona>/<e_eval>"
            parts = key.split("/", 3)
            if len(parts) != 4:
                continue
            _, arm, persona, e_eval = parts
            series_key = f"{persona}/{e_eval}"
            series_by_arm.setdefault(arm, {}).setdefault(series_key, [])
            series_by_arm[arm][series_key].extend(zip(steps, values, strict=False))
    else:
        # Fall back to raw per-step JSONs the callback dumped during training.
        # Round-3: glob the REAL callback dump-dir layout (was the wrong
        # base path in round-2 — see docstring).
        adapter_dirs = sorted(Path("adapters").glob("i464_*_seed*/_traj_adapter"))
        for ad in adapter_dirs:
            # The arm is encoded in the dispatcher cell name; the dump dir
            # path itself doesn't carry it. The dispatcher uses
            # output_dir=adapters/i464_<arm>_seed<seed>, and the callback
            # writes the adapter under <output_dir>/_traj_adapter. So the
            # arm is parseable from the parent dir.
            parent = ad.parent.name  # e.g. "i464_system_plain_seed42"
            if not parent.startswith("i464_"):
                continue
            tail = parent[len("i464_") :]  # "system_plain_seed42"
            if "_seed" not in tail:
                continue
            arm = tail.rsplit("_seed", 1)[0]
            for step_json in sorted(ad.glob("_logprobs_step*.json")):
                try:
                    payload = json.loads(step_json.read_text())
                except json.JSONDecodeError:
                    continue
                # filename: _logprobs_step<N>.json
                stem = step_json.stem  # "_logprobs_step100"
                try:
                    step = int(stem.split("step")[-1])
                except ValueError:
                    continue
                for key, lp in payload.get("per_key_logp", {}).items():
                    # Callback wrote keys like "<arm>/<persona>/<e_eval>"
                    # but only its own arm — so we can re-key as
                    # "<persona>/<e_eval>".
                    parts = key.split("/", 2)
                    if len(parts) != 3:
                        continue
                    _, persona, e_eval = parts
                    series_key = f"{persona}/{e_eval}"
                    series_by_arm.setdefault(arm, {}).setdefault(series_key, []).append(
                        (step, float(lp))
                    )

    if not series_by_arm:
        logger.warning(
            "trajectory.png skipped — neither %s nor callback per-step JSONs "
            "are present (Phase 3 not run yet?)",
            traj_path,
        )
        return

    arms_present = [a for a in enc.ARMS if a in series_by_arm] or list(series_by_arm)
    fig, axes = plt.subplots(1, len(arms_present), figsize=(5 * len(arms_present), 4), sharey=True)
    if len(arms_present) == 1:
        axes = [axes]
    for ax, arm in zip(axes, arms_present, strict=True):
        for series_key, points in sorted(series_by_arm[arm].items()):
            points_sorted = sorted(points, key=lambda p: p[0])
            xs = [p[0] for p in points_sorted]
            ys = [p[1] for p in points_sorted]
            ax.plot(xs, ys, marker="o", markersize=3, label=series_key)
        ax.set_title(f"marker log-prob trajectory — {arm}")
        ax.set_xlabel("training step")
        ax.set_ylabel("log P(marker | encoding + R)")
        ax.legend(fontsize=7, loc="best")
    _save(
        fig,
        "trajectory",
        [str(traj_path) if traj_path.exists() else "adapters/i464_*_seed*/_traj_adapter/"],
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the plot script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args = ap.parse_args(argv)
    _ = args  # currently unused; kept for argparse symmetry

    if not ANALYSIS_PATH.exists():
        raise FileNotFoundError(
            f"{ANALYSIS_PATH} missing — run scripts/i464_phase5_analyze.py first."
        )
    analysis = json.loads(ANALYSIS_PATH.read_text())

    plot_hero(analysis)
    plot_matrix_per_arm(analysis)
    plot_per_seed_scatter(analysis)
    plot_raw_alongside_processed(analysis)
    plot_dynamic_range(analysis)
    plot_argmax_emission(analysis)
    plot_leakage_by_eval_encoding(analysis)
    plot_role_nonsense_comparison(analysis)  # role_nonsense follow-up arm
    plot_role_mismatch_comparison(analysis)  # role_mismatch follow-up arm
    plot_onpolicy_validation()  # blocker #5 — MF-B(2) visualization
    plot_trajectory()  # blocker #5 — MF-C visualization
    logger.info("All plots written to %s", FIG_DIR)


if __name__ == "__main__":
    main()
