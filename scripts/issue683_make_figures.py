#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# (math/scientific notation — ρ, Δv, σ — intentional in labels/docstrings)
"""Issue #683 Phase D — marker-vs-sycophancy contrast + per-behavior figures (CPU).

Plan §4 Phase D + §6.5. Reads the Phase-C leaderboards + Phase-B A7 reports +
the standalone noise_floor.json (all written by issue683_key_ablation_score.py /
issue683_a7_precondition.py) and writes the plan's Phase-D figures under
``figures/issue_683/*.png`` (primary_deliverable §6.5):

  1. ``leaderboard_<behavior>`` — per-behavior held-out Spearman ρ bar chart per
     key×metric (best ψ per key), with bootstrap CIs, the shuffled-key null band
     (shaded), and the test-retest noise-floor line.
  2. ``marker_vs_sycophancy_contrast`` — the HERO: best behavior-dependent key
     vs k_cC vs the shuffled-key null, both behaviors side by side.
  3. ``a7_spectrum_<behavior>`` — the A7 precondition SVD spectrum bar
     (σ₁²/Σσ², σ₂/σ₁, cos(u₁,ŵ)) — the marker-vs-sycophancy precondition contrast.

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
import json
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="issue683_make_figures")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Plain-English key/metric labels (paper-plots: never bare slugs in figures).
KEY_LABELS = {
    "k_cC": "Context-only key",
    "k_tCB": "Training-completion key",
    "k_cC_plus_delta": "Context + displacement key",
}
METRIC_LABELS = {"M_I": "raw dot", "M_white": "whitened"}
BEHAVIOR_LABELS = {"marker": "Marker (control)", "sycophancy": "Sycophancy (transfer)"}


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
    best = _best_rows_per_key(lb["per_bank"])
    null_lo, null_hi = _null_band(lb["per_bank"])
    floor = _noise_floor_mean(noise_floor)

    labels, rhos, errs, colors = [], [], [], []
    for (key, metric), row in sorted(best.items()):
        labels.append(f"{KEY_LABELS.get(key, key)}\n({METRIC_LABELS.get(metric, metric)})")
        rho = row["spearman"]
        rhos.append(rho)
        ci = row.get("spearman_ci95", [rho, rho])
        lo = ci[0] if ci and ci[0] == ci[0] else rho
        hi = ci[1] if ci and ci[1] == ci[1] else rho
        errs.append([max(0.0, rho - lo), max(0.0, hi - rho)])
        colors.append(
            paper_palette_role("baseline") if key == "k_cC" else paper_palette_role("primary")
        )

    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(labels)), 4))
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
    ax.set_ylabel("Held-out Spearman ρ (g_pred vs g_real)")
    ax.legend(fontsize=8, loc="best")
    set_title_subtitle(
        ax,
        f"Key×metric leaderboard — {BEHAVIOR_LABELS.get(behavior, behavior)}",
        "best ψ per key; bars = held-out ρ, whiskers = 95% bootstrap CI",
    )
    fig.tight_layout()
    paths = savefig_paper(fig, stem, dir=str(out_dir))
    plt.close(fig)
    return paths


def _plot_contrast(lbs: dict[str, dict], noise_floors: dict[str, dict], out_dir: Path):
    """HERO: best behavior-dependent key vs k_cC vs null, both behaviors."""
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
    fig, ax = plt.subplots(figsize=(7, 4.2))
    width = 0.35
    x = np.arange(len(behaviors))

    cc_vals, cc_err, dep_vals, dep_err = [], [], [], []
    for b in behaviors:
        best = _best_rows_per_key(lbs[b]["per_bank"])
        cc = max(
            (r for (k, _m), r in best.items() if k == "k_cC"),
            key=lambda r: r["spearman"],
            default=None,
        )
        dep = max(
            (r for (k, _m), r in best.items() if k != "k_cC"),
            key=lambda r: r["spearman"],
            default=None,
        )

        def _val_err(row):
            if row is None:
                return float("nan"), [0.0, 0.0]
            rho = row["spearman"]
            ci = row.get("spearman_ci95", [rho, rho])
            lo = ci[0] if ci and ci[0] == ci[0] else rho
            hi = ci[1] if ci and ci[1] == ci[1] else rho
            return rho, [max(0.0, rho - lo), max(0.0, hi - rho)]

        v_cc, e_cc = _val_err(cc)
        v_dep, e_dep = _val_err(dep)
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
        label="Best behavior-dependent key",
    )
    # per-behavior null band + noise floor as short markers at each group.
    for i, b in enumerate(behaviors):
        nlo, nhi = _null_band(lbs[b]["per_bank"])
        if nlo == nlo and nhi == nhi:
            ax.plot([i - 0.5, i + 0.5], [nhi, nhi], color="0.5", ls=":", lw=1)
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
    ax.set_xticklabels([BEHAVIOR_LABELS.get(b, b) for b in behaviors])
    ax.set_ylabel("Held-out Spearman ρ (g_pred vs g_real)")
    ax.legend(fontsize=8, loc="best")
    set_title_subtitle(
        ax,
        "Does a behavior-dependent key beat the context-only default?",
        "dotted = shuffled-key null (upper); dashed = noise-floor ceiling; whiskers = 95% CI",
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
    metrics = ["sigma1_sq_frac", "sigma2_over_sigma1", "cos_u1_what"]
    metric_labels = ["σ₁²/Σσ² (top-component energy)", "σ₂/σ₁ (gap)", "|cos(u₁, source write)|"]
    means = []
    for m in metrics:
        vals = [b[m] for b in per_bank if b.get(m) == b.get(m)]
        means.append(float(np.mean(vals)) if vals else float("nan"))

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(metrics))
    ax.bar(x, means, color=paper_palette(len(metrics)))
    ax.axhline(0.5, color="0.4", ls="--", lw=0.8, label="rank-1 energy threshold (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("value (mean over banks)")
    ax.legend(fontsize=8)
    verdict = a7.get("verdict", "")
    set_title_subtitle(
        ax,
        f"A7 scalar-gate precondition — {BEHAVIOR_LABELS.get(behavior, behavior)}",
        f"verdict: {verdict}",
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
