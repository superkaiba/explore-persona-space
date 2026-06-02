# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (※, ′, ×, →, —) in docstrings/comments matching the project house style.
"""Build the matched-contrast headline table + figures for task #466.

Plan §6 — analysis unit is the **4 matched pairwise contrasts** (2
behaviors × 2 slices). For each contrast we report:

  - matched DV  Δ_marker(B, slice) = logp(※|S′_B, slice) − logp(※|S, slice)
  - blind predictors  (averaged_js_union, end_of_system_prompt cosine L21)
                      — one value per pair, broadcast across slices
  - sighted predictors (slice JS, slice cosine (a)/(b) L21)
                      — split across slices by construction
  - artifact-control contrasts  (S, Always_*) at the same slice — separate
                                shape in the scatter, not a correlation cell
  - topic-suppression controls  (S, slice_X) trigger vs non-trigger drop

This script is descriptive, not inferential — N=4 contrasts make a ρ
degenerate. It emits a table CSV + the 9 figures plan §6 enumerates,
ready for the analyzer agent to narrow into the clean-result body.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logger = logging.getLogger("issue466_analyze")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EVAL_ROOT = PROJECT_ROOT / "eval_results" / "issue_466"


# ── Behavior ↔ slice name normalization ───────────────────────────────────
# Predictors use "trigger"/"nontrigger"; marker_logp uses "trigger_A"/
# "trigger_B"/"nontrigger". Map both to a canonical (behavior, slice_kind)
# tuple where slice_kind ∈ {trigger, nontrigger}.

BEHAVIORS = ("A_spanish_restaurants", "B_caps_sports")
SLICE_KINDS = ("nontrigger", "trigger")


def _marker_slice(behavior: str, slice_kind: str) -> str:
    if slice_kind == "nontrigger":
        return "nontrigger"
    if behavior == "A_spanish_restaurants":
        return "trigger_A"
    if behavior == "B_caps_sports":
        return "trigger_B"
    raise ValueError((behavior, slice_kind))


# ── Loaders ────────────────────────────────────────────────────────────────


def _load_predictor(behavior: str) -> dict[str, Any]:
    path = EVAL_ROOT / "predictors" / f"{behavior}.json"
    if not path.exists():
        raise FileNotFoundError(f"predictor JSON missing: {path}")
    with open(path) as f:
        return json.load(f)


def _load_marker_cell(persona: str, slice_name: str) -> dict[str, Any] | None:
    path = EVAL_ROOT / "onpolicy_endpos_logp" / f"{persona}_{slice_name}.json"
    if not path.exists():
        logger.warning("missing marker log-p file %s — cell will be None", path)
        return None
    with open(path) as f:
        return json.load(f)


def _load_gen_cell(persona: str, slice_name: str) -> dict[str, Any] | None:
    path = EVAL_ROOT / "onpolicy_gen" / f"{persona}_{slice_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── Build the matched-contrast table ──────────────────────────────────────


def build_matched_contrast_table(
    headline_layer: int,
) -> list[dict[str, Any]]:
    """Build the 4-row matched-contrast table plan §6 calls for.

    Each row: behavior, slice_kind, blind/sighted predictor values, the
    matched marker Δ, and the matched (S, Always_*) artifact-control Δ.
    """
    pair_keys = {
        "A_spanish_restaurants": ("S", "S_prime_A_spanish_restaurants", "always_A_spanish"),
        "B_caps_sports": ("S", "S_prime_B_caps_sports", "always_B_caps"),
    }
    rows: list[dict[str, Any]] = []
    for behavior in BEHAVIORS:
        pred = _load_predictor(behavior)
        s_name, sp_name, always_name = pair_keys[behavior]
        avg_js = pred["js"]["averaged_js_union"]
        cos_a0 = pred["cosine"]["extraction_a0_endofsystemprompt"]
        # JSON keys are strings.
        a0_layer_key = str(headline_layer)
        cos_a0_layer = cos_a0.get(a0_layer_key, cos_a0.get(headline_layer))
        cos_a_per_slice = pred["cosine"]["extraction_a_lastinputtoken_per_slice_per_layer"]
        cos_b_per_slice = pred["cosine"]["extraction_b_ownresponsemean_per_slice_per_layer"]

        for slice_kind in SLICE_KINDS:
            marker_slice = _marker_slice(behavior, slice_kind)
            s_marker = _load_marker_cell(s_name, marker_slice)
            sp_marker = _load_marker_cell(sp_name, marker_slice)
            always_marker = _load_marker_cell(always_name, marker_slice)
            slice_js = pred["js"]["slice_mean_js"].get(slice_kind)
            slice_cos_a = (
                cos_a_per_slice.get(slice_kind, {}).get(a0_layer_key) if cos_a_per_slice else None
            )
            slice_cos_b = (
                cos_b_per_slice.get(slice_kind, {}).get(a0_layer_key) if cos_b_per_slice else None
            )

            delta_marker = None
            if s_marker is not None and sp_marker is not None:
                delta_marker = sp_marker["mean_logp_trained"] - s_marker["mean_logp_trained"]
            delta_marker_normed = None
            if s_marker is not None and sp_marker is not None:
                # "trained − base" subtraction already done per-cell; the
                # matched contrast subtracts the two ``delta`` fields so the
                # response-prior nets out.
                delta_marker_normed = sp_marker["delta"] - s_marker["delta"]
            artifact_delta = None
            if always_marker is not None and s_marker is not None:
                artifact_delta = always_marker["mean_logp_trained"] - s_marker["mean_logp_trained"]
            artifact_delta_normed = None
            if always_marker is not None and s_marker is not None:
                artifact_delta_normed = always_marker["delta"] - s_marker["delta"]

            rows.append(
                {
                    "behavior": behavior,
                    "slice_kind": slice_kind,
                    # Blind predictors (broadcast across slices within a behavior)
                    "blind_avg_js": avg_js,
                    "blind_cos_a0_L21": cos_a0_layer,
                    # Sighted predictors
                    "sighted_slice_js": slice_js,
                    "sighted_slice_cos_a_L21": slice_cos_a,
                    "sighted_slice_cos_b_L21": slice_cos_b,
                    # Marker DVs
                    "delta_marker_logp_trained": delta_marker,
                    "delta_marker_logp_normed": delta_marker_normed,
                    # Artifact control (S, Always_*)
                    "artifact_delta_logp_trained": artifact_delta,
                    "artifact_delta_logp_normed": artifact_delta_normed,
                    # Free anchors
                    "S_marker": _summarize_marker(
                        s_marker, slice_kind, _load_gen_cell(s_name, marker_slice)
                    ),
                    "S_prime_marker": _summarize_marker(
                        sp_marker, slice_kind, _load_gen_cell(sp_name, marker_slice)
                    ),
                    "Always_marker": _summarize_marker(
                        always_marker, slice_kind, _load_gen_cell(always_name, marker_slice)
                    ),
                }
            )
    return rows


def _summarize_marker(
    marker: dict[str, Any] | None, slice_kind: str, gen: dict[str, Any] | None
) -> dict[str, Any] | None:
    if marker is None:
        return None
    return {
        "slice_kind": slice_kind,
        "mean_logp_trained": marker["mean_logp_trained"],
        "mean_logp_base": marker["mean_logp_base"],
        "delta": marker["delta"],
        "n_contexts": marker["n_contexts"],
        "emission_rate": (gen or {}).get("emission_rate"),
        "truncation_frac": (gen or {}).get("truncation_frac"),
    }


# ── Figures ────────────────────────────────────────────────────────────────


def _figdir(out_dir: Path) -> Path:
    p = out_dir / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _try_paper_style():
    try:
        from explore_persona_space.analysis.paper_plots import apply_paper_style  # type: ignore

        apply_paper_style()
    except Exception:
        # Plain matplotlib defaults are fine for the descriptive panels;
        # paper style is a nice-to-have, not a hard dep.
        pass


def figure_hero_a_scatter(rows: list[dict[str, Any]], out_path: Path) -> None:
    """hero_a_predictor_vs_marker_scatter.png — 4 panels (blind/sighted JS + cos).

    Per panel: x = predictor value, y = matched Δ_marker. Shape codes
    slice (○ nontrigger, ● trigger). Color codes behavior (A/B).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _try_paper_style()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panels = [
        ("blind_avg_js", "Averaged JS (BLIND)"),
        ("blind_cos_a0_L21", "End-of-system-prompt cos L21 (BLIND)"),
        ("sighted_slice_js", "Slice JS (SIGHTED)"),
        ("sighted_slice_cos_b_L21", "Slice cos own-response L21 (SIGHTED)"),
    ]
    colors = {"A_spanish_restaurants": "#2563EB", "B_caps_sports": "#DC2626"}
    for ax, (key, title) in zip(axes, panels, strict=True):
        for row in rows:
            x = row.get(key)
            y = row.get("delta_marker_logp_normed")
            if x is None or y is None:
                continue
            marker_shape = "o" if row["slice_kind"] == "nontrigger" else "*"
            facecolor = "none" if row["slice_kind"] == "nontrigger" else colors[row["behavior"]]
            ax.scatter(
                x,
                y,
                s=180,
                marker=marker_shape,
                edgecolor=colors[row["behavior"]],
                facecolor=facecolor,
                linewidth=2.0,
                label=f"{row['behavior']} {row['slice_kind']}",
            )
            # Also plot the artifact-control point (S, Always_*) at the
            # same x but with a triangle marker.
            if row.get("artifact_delta_logp_normed") is not None:
                ax.scatter(
                    x,
                    row["artifact_delta_logp_normed"],
                    s=140,
                    marker="^",
                    edgecolor=colors[row["behavior"]],
                    facecolor="white",
                    linewidth=1.5,
                )
        ax.set_xlabel(title)
        ax.set_ylabel("matched Δ_marker (logp_S' − logp_S)")
        ax.set_title(title)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Matched pairwise contrasts: blind vs sighted predictors vs marker Δ\n"
        "(○ nontrigger ★ trigger; ▲ artifact-control (S, Always_*))"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def figure_hero_b_bars(rows: list[dict[str, Any]], out_path: Path) -> None:
    """hero_b_marker_logp_bars.png — grouped bars per (persona, slice) cell."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _try_paper_style()
    # Flatten all 11 cells from rows + free-anchor S* summaries.
    cells: list[tuple[str, str, str, float | None]] = []
    seen = set()
    for row in rows:
        for key in ("S_marker", "S_prime_marker", "Always_marker"):
            payload = row.get(key)
            if not payload:
                continue
            persona = {
                "S_marker": "S",
                "S_prime_marker": row["behavior"]
                .replace("_spanish_restaurants", "_S′_A")
                .replace("_caps_sports", "_S′_B"),
                "Always_marker": row["behavior"]
                .replace("_spanish_restaurants", "_Always_A")
                .replace("_caps_sports", "_Always_B"),
            }[key]
            slice_label = f"{row['behavior'][:1]}-{row['slice_kind'][:4]}"
            unique = (persona, slice_label)
            if unique in seen:
                continue
            seen.add(unique)
            cells.append((persona, slice_label, key, payload.get("delta")))
    if not cells:
        logger.warning("no marker cells found for figure_hero_b_bars")
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    xs = list(range(len(cells)))
    ys = [c[3] if c[3] is not None else 0 for c in cells]
    color_for = {"S_marker": "#94A3B8", "S_prime_marker": "#2563EB", "Always_marker": "#F59E0B"}
    colors = [color_for[c[2]] for c in cells]
    ax.bar(xs, ys, color=colors)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c[0]}\n{c[1]}" for c in cells], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("marker logp Δ (trained − base)")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_title("Per-cell marker log-p delta (gray=S, blue=S′, orange=Always)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def figure_exp_js_per_position(out_path: Path) -> None:
    """exp_js_per_position.png — per-position JS trajectory per slice."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _try_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, behavior in zip(axes, BEHAVIORS, strict=True):
        try:
            pred = _load_predictor(behavior)
        except FileNotFoundError:
            continue
        traj = pred["js"]["per_position_traj"]
        for slice_kind in SLICE_KINDS:
            traces = traj.get(slice_kind, [])
            if not traces:
                continue
            # Pad with NaN to a common length, mean across responses.
            max_len = max(len(t) for t in traces)
            import numpy as np

            padded = np.full((len(traces), max_len), np.nan)
            for i, t in enumerate(traces):
                padded[i, : len(t)] = t
            mean = np.nanmean(padded, axis=0)
            ax.plot(mean, label=f"{slice_kind} (n={len(traces)})", linewidth=1.5)
            # Faded per-response traces for raw alongside processed.
            for t in traces[:20]:
                ax.plot(t, alpha=0.07, color="grey", linewidth=0.5)
        ax.set_xlabel("response position")
        ax.set_ylabel("JS (base 2)")
        ax.set_title(behavior)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


# ── Reproducibility metadata ───────────────────────────────────────────────


def _metadata() -> dict[str, Any]:
    git_commit = "unknown"
    try:
        # epm-lint: subprocess-env-inherit -- git rev-parse needs no credential env
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            git_commit = out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {
        "script": "issue466_analyze",
        "git_commit": git_commit,
        "ts_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headline-layer", type=int, default=21)
    parser.add_argument("--out-dir", type=Path, default=EVAL_ROOT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = build_matched_contrast_table(headline_layer=args.headline_layer)
    table_path = args.out_dir / "matched_contrast_table.json"
    with open(table_path, "w") as f:
        json.dump(
            {
                "headline_layer": args.headline_layer,
                "rows": table,
                "metadata": _metadata(),
            },
            f,
            indent=2,
        )
    logger.info("Wrote %s (%d rows)", table_path, len(table))

    # Figures — over-produce; analyzer picks the hero.
    figdir = _figdir(args.out_dir)
    figure_hero_a_scatter(table, figdir / "hero_a_predictor_vs_marker_scatter.png")
    figure_hero_b_bars(table, figdir / "hero_b_marker_logp_bars.png")
    figure_exp_js_per_position(figdir / "exp_js_per_position.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
