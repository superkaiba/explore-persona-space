"""Figures for the #1415 answer-position-shift-profile follow-up (plan v8 §6).

VM-side (off-pod phase A3): reads the committed JSONs under
eval_results/issue_1415/answer_position_shift_profile/ and writes PNGs to
figures/issue_1415/. Hero = per-bin shift magnitude + alignment vs bin,
2x2 (arm x steer layer), matched vs cross lines, per-bin null p97.5
(alignment axis) + baseline split-half noise floor (magnitude axis); the
registered Delta and the width-matched Delta_width companion ride each
panel title. Plus the §6 exploratory dump (over-produced by design).

Usage: uv run python scripts/issue1415_position_profile_figures.py
       [--in-root <dir>] [--fig-root <dir>]   # overrides for smoke runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("prefix", "context")


def _nan(vals):
    return np.array([np.nan if v is None else v for v in vals], dtype=float)


def _cell_rows(profiles: list[dict], arm: str, layer: int, rnd: str = "primary") -> list[dict]:
    return [
        p for p in profiles if p["round"] == rnd and p["arm"] == arm and p["steer_layer"] == layer
    ]


def _bin_mat(rows: list[dict], key: str) -> np.ndarray:
    """(n_pairs, 13) matrix of a per-bin scalar (None -> NaN)."""
    return np.stack([_nan([b.get(key) for b in r["bins"]]) for r in rows])


def _delta_title(summary: dict, cell_label: str) -> str:
    for c in summary["cells"]:
        if c["cell"] == cell_label:
            ci = c.get("ci95") or [np.nan, np.nan]
            w = c.get("delta_width") or {}
            wm = w.get("delta_mean")
            wtxt = f"; width {wm:.2f}" if wm is not None else ""
            return f"Δ̄={c['delta_mean']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]{wtxt}"
    return ""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--in-root", default=str(ROOT / "eval_results/issue_1415/answer_position_shift_profile")
    )
    ap.add_argument("--fig-root", default=str(ROOT / "figures/issue_1415"))
    args = ap.parse_args(argv)
    in_root, fig_root = Path(args.in_root), Path(args.fig_root)
    fig_root.mkdir(parents=True, exist_ok=True)

    pp = json.load(open(in_root / "per_pair_profiles.json"))
    summary = json.load(open(in_root / "summary.json"))
    lengths = json.load(open(in_root / "answer_length_distributions.json"))
    bins = pp["bin_names"]
    profiles = pp["profiles"]
    x = np.arange(len(bins))
    layers = sorted({p["steer_layer"] for p in profiles if p["round"] == "primary"})
    set_paper_style("blog")
    C = paper_palette_blog(6)

    def _grid():
        fig, axes = plt.subplots(
            2, len(layers), figsize=(6.4 * len(layers), 9.2), constrained_layout=True
        )
        return fig, np.atleast_2d(axes).reshape(2, len(layers))

    def _xticks(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=60, ha="right", fontsize=8)

    # ── hero: magnitude + alignment vs bin, 2x2 (arm x layer) ─────────
    fig, axes = _grid()
    for i, arm in enumerate(ARMS):
        for j, layer in enumerate(layers):
            ax = axes[i, j]
            rows = _cell_rows(profiles, arm, layer)
            mag = _bin_mat(rows, "magnitude")
            ali = _bin_mat(rows, "alignment_disjoint")
            floor = _bin_mat(rows, "noise_floor")
            null = _bin_mat(rows, "null_p975")
            matched = np.array([r["pair_type"] == "matched" for r in rows])
            for sel, lab, c in ((matched, "matched", C[0]), (~matched, "cross", C[1])):
                ax.plot(x, np.nanmean(mag[sel], 0), "-o", color=c, ms=3, label=f"‖shift‖ {lab}")
            ax.plot(x, np.nanmean(floor, 0), ":", color="grey", label="noise floor")
            ax.set_ylabel("‖shift‖ (mean over pairs)")
            ax2 = ax.twinx()
            for sel, lab, c in ((matched, "matched", C[2]), (~matched, "cross", C[3])):
                ax2.plot(x, np.nanmean(ali[sel], 0), "--s", color=c, ms=3, label=f"cos {lab}")
            ax2.plot(x, np.nanmean(null, 0), ":", color="k", alpha=0.6, label="null p97.5")
            ax2.set_ylabel("cos(shift, target), disjoint")
            _xticks(ax)
            ax.set_title(f"{arm} · L{layer}  ({_delta_title(summary, f'primary/{arm}/L{layer}')})")
            if i == 0 and j == 0:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    savefig_paper(fig, "position_profile_hero", dir=fig_root)
    plt.close(fig)

    # ── per-pair small multiples (magnitude), one figure per (arm, layer) ──
    for arm in ARMS:
        for layer in layers:
            rows = sorted(_cell_rows(profiles, arm, layer), key=lambda r: r["pair_id"])
            n = len(rows)
            ncol = 7
            nrow = int(np.ceil(n / ncol))
            fig, axg = plt.subplots(
                nrow, ncol, figsize=(2.3 * ncol, 1.9 * nrow), constrained_layout=True
            )
            axg = np.atleast_2d(axg)
            for k, r in enumerate(rows):
                ax = axg[k // ncol, k % ncol]
                ax.plot(x, _nan([b.get("magnitude") for b in r["bins"]]), "-", color=C[0], lw=1)
                ax.plot(x, _nan([b.get("noise_floor") for b in r["bins"]]), ":", color="grey", lw=1)
                flags = [f for f, v in r["flags"].items() if v]
                ax.set_title(r["pair_id"] + (f" [{','.join(flags)}]" if flags else ""), fontsize=6)
                ax.set_xticks([])
            for k in range(n, nrow * ncol):
                axg[k // ncol, k % ncol].axis("off")
            fig.suptitle(f"per-pair ‖shift‖ profiles — {arm} · L{layer}", fontsize=10)
            savefig_paper(fig, f"position_profile_perpair_{arm}_L{layer}", dir=fig_root)
            plt.close(fig)

    # ── rep43/44 overlays on the parent L14 profile ────────────────────
    rep_rounds = sorted({p["round"] for p in profiles if p["round"].startswith("rep")})
    if rep_rounds:
        rep_layer = next(p["steer_layer"] for p in profiles if p["round"] == rep_rounds[0])
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
        for i, arm in enumerate(ARMS):
            ax = axes[i]
            base_rows = _cell_rows(profiles, arm, rep_layer)
            ax.plot(
                x,
                np.nanmean(_bin_mat(base_rows, "magnitude"), 0),
                "-o",
                color=C[0],
                ms=3,
                label=f"parent L{rep_layer}",
            )
            for k, rnd in enumerate(rep_rounds):
                rows = _cell_rows(profiles, arm, rep_layer, rnd=rnd)
                ax.plot(
                    x,
                    np.nanmean(_bin_mat(rows, "magnitude"), 0),
                    "--s",
                    color=C[2 + k],
                    ms=3,
                    label=rnd,
                )
            _xticks(ax)
            ax.set_title(f"{arm} · L{rep_layer} — fresh-sampling overlays")
            ax.set_ylabel("‖shift‖ (mean over pairs)")
            ax.legend(fontsize=8)
        savefig_paper(fig, "position_profile_rep_overlay", dir=fig_root)
        plt.close(fig)

    # ── answer-length histograms per condition ─────────────────────────
    conds: dict[str, list[int]] = {}
    call = {}
    for d in lengths["distributions"]:
        conds.setdefault(d["condition"], []).extend(d["token_counts"])
        if d["pair_id"] in lengths["callouts"].values():
            call.setdefault(d["pair_id"], []).extend(d["token_counts"])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4), constrained_layout=True)
    for k, (cond, vals) in enumerate(sorted(conds.items())):
        axes[0].hist(vals, bins=30, histtype="step", label=cond, color=C[k % 6])
    axes[0].set_xlabel("answer tokens")
    axes[0].set_title("answer-length distributions by condition")
    axes[0].legend(fontsize=8)
    for k, (pid, vals) in enumerate(sorted(call.items())):
        axes[1].hist(vals, bins=30, histtype="step", label=pid, color=C[k % 6])
    axes[1].set_xlabel("answer tokens")
    axes[1].set_title("call-out pairs (terse / formal), all conditions")
    axes[1].legend(fontsize=8)
    savefig_paper(fig, "position_profile_lengths", dir=fig_root)
    plt.close(fig)

    # ── first-token vs decile-profile comparison (per pair) ────────────
    dec_idx = [bins.index(f"dec{d}") for d in range(1, 11)]
    fig, axes = _grid()
    for i, arm in enumerate(ARMS):
        for j, layer in enumerate(layers):
            ax = axes[i, j]
            rows = _cell_rows(profiles, arm, layer)
            mag = _bin_mat(rows, "magnitude")
            first = mag[:, bins.index("first")]
            decm = np.nanmean(mag[:, dec_idx], 1)
            for r, fx, dy in zip(rows, first, decm, strict=True):
                c = C[0] if r["pair_type"] == "matched" else C[1]
                ax.scatter(fx, dy, s=14, color=c)
            lim = np.nanmax([np.nanmax(first), np.nanmax(decm)])
            ax.plot([0, lim], [0, lim], ":", color="grey", lw=1)
            ax.set_xlabel("first-token ‖shift‖")
            ax.set_ylabel("mean decile ‖shift‖")
            ax.set_title(f"{arm} · L{layer}")
    savefig_paper(fig, "position_profile_first_vs_deciles", dir=fig_root)
    plt.close(fig)

    # ── alignment-vs-magnitude shape comparison (max-normalized) ───────
    fig, axes = _grid()
    for i, arm in enumerate(ARMS):
        for j, layer in enumerate(layers):
            ax = axes[i, j]
            rows = _cell_rows(profiles, arm, layer)
            mag = np.nanmean(_bin_mat(rows, "magnitude"), 0)
            ali = np.nanmean(_bin_mat(rows, "alignment_disjoint"), 0)
            ax.plot(x, mag / np.nanmax(np.abs(mag)), "-o", color=C[0], ms=3, label="‖shift‖ / max")
            ax.plot(x, ali / np.nanmax(np.abs(ali)), "--s", color=C[2], ms=3, label="cos / max")
            _xticks(ax)
            ax.set_title(f"{arm} · L{layer} — shape comparison")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    savefig_paper(fig, "position_profile_shape_comparison", dir=fig_root)
    plt.close(fig)

    # ── disjoint-vs-shared convention dumbbells per bin ────────────────
    fig, axes = _grid()
    for i, arm in enumerate(ARMS):
        for j, layer in enumerate(layers):
            ax = axes[i, j]
            rows = _cell_rows(profiles, arm, layer)
            dis = np.nanmean(_bin_mat(rows, "alignment_disjoint"), 0)
            sha = np.nanmean(_bin_mat(rows, "alignment_shared"), 0)
            for xi, (d, s) in enumerate(zip(dis, sha, strict=True)):
                ax.plot([xi, xi], [d, s], "-", color="grey", lw=1)
            ax.scatter(x, dis, s=16, color=C[0], label="disjoint (primary)")
            ax.scatter(x, sha, s=16, color=C[1], label="shared (secondary)")
            _xticks(ax)
            ax.set_title(f"{arm} · L{layer} — convention dumbbells")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    savefig_paper(fig, "position_profile_convention_dumbbells", dir=fig_root)
    plt.close(fig)

    # ── target-magnitude profile + traversal fraction ──────────────────
    fig, axes = _grid()
    for i, arm in enumerate(ARMS):
        for j, layer in enumerate(layers):
            ax = axes[i, j]
            rows = _cell_rows(profiles, arm, layer)
            tmag = np.nanmean(_bin_mat(rows, "target_magnitude"), 0)
            frac = np.nanmean(_bin_mat(rows, "traversal_frac"), 0)
            ax.plot(x, tmag, "-o", color=C[0], ms=3, label="‖target‖")
            ax.set_ylabel("‖target‖")
            ax2 = ax.twinx()
            ax2.plot(x, frac, "--s", color=C[2], ms=3, label="‖shift‖/‖target‖ (exploratory)")
            ax2.set_ylabel("traversal fraction")
            _xticks(ax)
            ax.set_title(f"{arm} · L{layer}")
            if i == 0 and j == 0:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=8)
    savefig_paper(fig, "position_profile_target_magnitude", dir=fig_root)
    plt.close(fig)

    # ── Delta lattice: full vs sensitivity variants (clamped errorbars) ─
    prim = [c for c in summary["cells"] if c.get("registered_primary")]
    fig, ax = plt.subplots(figsize=(10.4, 4.8), constrained_layout=True)
    variants = [
        ("delta (28 pairs)", lambda c: c, C[0]),
        ("width-matched", lambda c: c.get("delta_width"), C[1]),
        ("excl. medical", lambda c: c.get("sensitivity_exclude_medical"), C[2]),
        (
            "matched-length",
            lambda c: (c.get("sensitivity_matched_length") or {}).get("stats"),
            C[3],
        ),
        ("noise-floor ref", lambda c: c.get("delta_floor"), C[4]),
    ]
    width = 0.16
    for k, (lab, get, col) in enumerate(variants):
        xs, vs, lo, hi = [], [], [], []
        for ci_, c in enumerate(prim):
            s = get(c)
            if not s or s.get("delta_mean") is None:
                continue
            xs.append(ci_ + (k - 2) * width)
            vs.append(s["delta_mean"])
            ci = s.get("ci95") or [s["delta_mean"], s["delta_mean"]]
            lo.append(max(0.0, s["delta_mean"] - ci[0]))  # NON-NEGATIVE offsets (gotcha)
            hi.append(max(0.0, ci[1] - s["delta_mean"]))
        if xs:
            ax.bar(xs, vs, width=width, color=col, label=lab, yerr=np.array([lo, hi]), capsize=2)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(range(len(prim)))
    ax.set_xticklabels([c["cell"] for c in prim], fontsize=8)
    ax.set_ylabel("Δ = log(mean EARLY ‖shift‖) − log(mean LATE ‖shift‖)")
    ax.legend(fontsize=8)
    savefig_paper(fig, "position_profile_delta_lattice", dir=fig_root)
    plt.close(fig)

    print(f"figures written under {fig_root}")


if __name__ == "__main__":
    main()
