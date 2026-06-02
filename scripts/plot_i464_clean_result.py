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
    """Mean own-persona elicitation log-prob across the 2 personas under each arm's own encoding."""
    cell = f"{arm}_seed{seed}"
    own_logps: list[float] = []
    for persona in enc.PERSONAS:
        e = f"role_{persona}" if arm == "role" else f"system_{persona}"
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
        color=["#666", "#a44", "#48a"],
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
    ax[1].bar(x, leak_means, yerr=leak_stds, color=["#666", "#a44", "#48a"])
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
    """For each seed, scatter (elicitation, leakage) per arm."""
    seeds = analysis["seeds"]
    fig, ax = plt.subplots(figsize=(6, 5))
    arms = list(enc.ARMS)
    colors = {"system_plain": "#666", "system_padded": "#a44", "role": "#48a"}
    for arm in arms:
        elicits = [_own_persona_elicitation(arm, s) for s in seeds]  # type: ignore[arg-type]
        leaks = _l_arm_values(analysis, arm)
        ax.scatter(elicits, leaks, s=60, color=colors[arm], label=arm)
    ax.set_xlabel("own-persona elicitation log P")
    ax.set_ylabel("symmetric leakage L_arm")
    ax.set_title("Per-seed: elicitation vs leakage by arm")
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

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(encs))
    w = 0.25
    colors = ["#666", "#a44", "#48a"]
    for i, arm in enumerate(arms):
        ax.bar(x + (i - 1) * w, grid[i, :], width=w, color=colors[i], label=arm)
    ax.set_xticks(x)
    ax.set_xticklabels(encs)
    ax.set_ylabel("raw trained log P at slot (mean over seeds x personas)")
    ax.set_title("Leakage decomposed by wrong-encoding family (lower = less leakage)")
    ax.legend()
    _save(fig, "leakage_by_eval_encoding", [str(PER_CELL_DIR)])


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
    colors = ["#666", "#a44", "#48a"]
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
    plot_onpolicy_validation()  # blocker #5 — MF-B(2) visualization
    plot_trajectory()  # blocker #5 — MF-C visualization
    logger.info("All plots written to %s", FIG_DIR)


if __name__ == "__main__":
    main()
