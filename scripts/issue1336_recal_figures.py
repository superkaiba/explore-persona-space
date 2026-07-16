#!/usr/bin/env python
"""Issue #1336 — E1 recalibration-round figures (plan v9 §6).

Hero: (left) the read ladder — committed raw R^2 -> in-sample recalibrated ->
HELD-OUT recalibrated S_r, with B_r and bar_r drawn; (right) the gain spectrum
a_j vs dim variance, Llama vs Qwen overlaid, with the global a=1 reference.
Exploratory dump (over-produce, plan §6): per-layer held-out recal curves
(raw + in-sample companions), optimism gap, fold-mean-norm observed vs iid
reference bands, seed-0 vs seed-1 R^2, per-fold gain panels, a_j histograms,
duplicate-count bars, Qwen recal inertness, bootstrap distribution of S_r,
and (if E2 fired) the v5 read.

Reads the step JSONs + npz tensors written by scripts/issue1336_recal_verdict.py;
writes PNGs + meta.json to --out (production: figures/issue_1336/diagnosis/recal;
the dispatch smoke diverts to its scratch root).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style  # noqa: E402

CHAT = "rlvr_chat_lmsys5k"
NAT = "rlvr_naturalistic_lmsys5k"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--recal-dir", type=Path, default=Path("eval_results/issue_1336/diagnosis/recal")
    )
    ap.add_argument("--out", type=Path, default=Path("figures/issue_1336/diagnosis/recal"))
    ap.add_argument("--cells", default=f"{CHAT},{NAT}")
    return ap.parse_args()


def _load(path: Path) -> dict:
    assert path.exists(), f"figure input missing: {path}"
    return json.loads(path.read_text())


def _maybe(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _save(fig, out: Path, name: str, written: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    written.append(name)
    print(f"[recal-figs] wrote {path}")


def fig_hero(hr: dict, verdict: dict, ab_npz, qwen: dict | None, qwen_npz, out, written):
    c = paper_palette(4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    li = str(verdict["mechanism_account"]["layer"])
    ladder = [
        ("committed\nraw", verdict["mechanism_account"]["r2_v0_l29"]),
        ("in-sample\nrecal", hr["per_layer"][li]["insample_recal_r2"]),
        ("held-out\nrecal (S_r)", verdict["lattice_inputs"]["s_r"]),
    ]
    ax1.bar([x[0] for x in ladder], [x[1] for x in ladder], color=c[:3], width=0.6)
    ax1.axhline(verdict["lattice_inputs"]["b_r"], color=c[3], ls="--", lw=1.2, label="B_r (p97.5)")
    ax1.axhline(verdict["lattice_inputs"]["bar_r"], color="black", ls=":", lw=1.2, label="bar_r")
    ax1.axhline(0.0, color="gray", lw=0.8)
    ax1.set_ylabel("pooled R²")
    ax1.set_title("Read ladder (chat)")
    ax1.legend(frameon=False, fontsize=8)

    layer_int = int(li)
    a = ab_npz[f"a_l{layer_int}"].mean(axis=0)
    # Variance proxy from the gain-spec Spearman input is not persisted per dim;
    # rank the dims by |a-1| against the stored decile medians instead.
    gs = hr["gain_spectrum"][li]
    med = [m for m in gs["binned_median_a_by_var_decile"] if m is not None]
    ax2.hist(a, bins=60, color=c[0], alpha=0.75, label="Llama a_j (across-fold mean)")
    if qwen is not None and qwen_npz is not None:
        aq = qwen_npz["a"].mean(axis=0)
        ax2.hist(aq, bins=60, color=c[1], alpha=0.55, label="Qwen a_j")
    ax2.axvline(1.0, color="black", ls=":", lw=1.2)
    ax2.set_xlabel("per-dim gain a_j")
    ax2.set_ylabel("dims")
    ax2.set_title(f"Gain spectrum @L{li} (var-decile medians: {len(med)} bins)")
    ax2.legend(frameon=False, fontsize=8)
    _save(fig, out, "hero_recal_verdict.png", written)


def fig_perlayer(hrs: dict[str, dict], out, written):
    c = paper_palette(3)
    fig, axes = plt.subplots(1, len(hrs), figsize=(5.2 * len(hrs), 4.0), squeeze=False)
    for ax, (cell, hr) in zip(axes[0], hrs.items(), strict=True):
        layers = sorted(int(k) for k in hr["per_layer"])
        for key, label, col in (
            ("raw_r2", "raw", c[0]),
            ("insample_recal_r2", "in-sample recal", c[1]),
            ("heldout_recal_r2", "held-out recal", c[2]),
        ):
            ax.plot(
                layers,
                [hr["per_layer"][str(li)][key] for li in layers],
                "o-",
                color=col,
                label=label,
                ms=4,
            )
        ax.axhline(hr["recal_null"]["band_p975_layer_max"], color="gray", ls="--", lw=1.0)
        ax.set_title(cell)
        ax.set_xlabel("layer")
        ax.set_ylabel("pooled R²")
        ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "perlayer_recal_curves.png", written)


def fig_optimism(hrs: dict[str, dict], out, written):
    c = paper_palette(len(hrs))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for ci, (cell, hr) in enumerate(hrs.items()):
        layers = sorted(int(k) for k in hr["per_layer"])
        ax.plot(
            layers,
            [hr["per_layer"][str(li)]["optimism_gap"] for li in layers],
            "o-",
            color=c[ci],
            label=cell,
            ms=4,
        )
    ax.set_xlabel("layer")
    ax.set_ylabel("in-sample minus held-out recal R²")
    ax.set_title("Cross-fit optimism gap")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "optimism_gap.png", written)


def fig_fold_norms(fes: dict[str, dict], out, written):
    cells = list(fes)
    fig, axes = plt.subplots(1, len(cells), figsize=(5.6 * len(cells), 4.0), squeeze=False)
    c = paper_palette(2)
    for ax, cell in zip(axes[0], cells, strict=True):
        fe = fes[cell]
        layers = sorted(int(k) for k in fe["fold_mean_norms"])
        for si, stat in enumerate(("y", "resid")):
            obs = [fe["fold_mean_norms"][str(li)][stat]["observed_max"] for li in layers]
            ref = [fe["fold_mean_norms"][str(li)][stat]["ref_p975_max"] for li in layers]
            ax.plot(layers, obs, "o-", color=c[si], label=f"{stat} observed max", ms=4)
            ax.plot(layers, ref, "s--", color=c[si], alpha=0.6, label=f"{stat} iid p97.5", ms=4)
        ax.set_title(cell)
        ax.set_xlabel("layer")
        ax.set_ylabel("fold-mean shift ‖·‖₂")
        ax.legend(frameon=False, fontsize=7)
    _save(fig, out, "fold_mean_norms.png", written)


def fig_seed_refit(fes: dict[str, dict], out, written):
    c = paper_palette(2)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for ci, (cell, fe) in enumerate(fes.items()):
        sr = fe["seed_refit"]
        layers = sorted(int(k) for k in sr["r2_seed0_per_layer"])
        ax.plot(
            layers,
            [sr["r2_seed0_per_layer"][str(li)] for li in layers],
            "o-",
            color=c[ci],
            label=f"{cell} seed 0",
            ms=4,
        )
        ax.plot(
            layers,
            [sr["r2_seed1_per_layer"][str(li)] for li in layers],
            "s--",
            color=c[ci],
            alpha=0.6,
            label=f"{cell} seed 1",
            ms=4,
        )
    ax.set_xlabel("layer")
    ax.set_ylabel("raw pooled R²")
    ax.set_title("Fold-randomization refit (seed 0 vs seed 1)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "seed0_vs_seed1.png", written)


def fig_perfold_gain(hr: dict, out, written):
    layers = sorted(int(k) for k in hr["gain_spectrum"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    c = paper_palette(len(layers))
    width = 0.8 / max(len(layers), 1)
    for ix, li in enumerate(layers):
        pf = hr["gain_spectrum"][str(li)]["per_fold_mean_a"]
        xs = np.arange(len(pf)) + ix * width
        ax.bar(xs, pf, width=width, color=c[ix], label=f"L{li}")
    ax.axhline(1.0, color="black", ls=":", lw=1.0)
    ax.set_xlabel("fold")
    ax.set_ylabel("mean a_j (per fold)")
    ax.set_title("Per-fold gain (chat)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "perfold_gain_chat.png", written)


def fig_boot(hr: dict, verdict: dict, out, written):
    c = paper_palette(3)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    boot = np.asarray(hr["bootstrap"]["s_r_per_draw"], dtype=float)
    ax.hist(boot, bins=40, color=c[0], alpha=0.8)
    ax.axvline(verdict["lattice_inputs"]["s_r"], color=c[1], lw=1.4, label="S_r")
    ax.axvline(verdict["lattice_inputs"]["b_r"], color=c[2], ls="--", lw=1.2, label="B_r")
    ax.axvline(verdict["lattice_inputs"]["bar_r"], color="black", ls=":", lw=1.2, label="bar_r")
    ax.set_xlabel("per-resample layer-max recal R²")
    ax.set_ylabel("resamples")
    ax.set_title("Bootstrap distribution of S_r (chat)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "bootstrap_s_r.png", written)


def fig_qwen(qwen: dict, out, written):
    c = paper_palette(3)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    bars = [
        ("committed\nanchor", qwen["committed_anchor"]),
        ("raw refit", qwen["r2_raw_committed_grid"]),
        ("held-out\nrecal", qwen["s_qwen_recal"]),
    ]
    ax.bar([b[0] for b in bars], [b[1] for b in bars], color=c, width=0.6)
    lo = qwen["committed_anchor"] - qwen["v_gate"]["threshold"]
    hi = qwen["committed_anchor"] + qwen["v_gate"]["threshold"]
    ax.axhspan(lo, hi, color="gray", alpha=0.18)
    ax.set_ylabel("pooled R² (L19)")
    ax.set_title(f"Qwen validate-before-use (V {'PASS' if qwen['v_gate']['pass'] else 'FAIL'})")
    _save(fig, out, "qwen_inertness.png", written)


def fig_dup(fes: dict[str, dict], out, written):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    cells = list(fes)
    c = paper_palette(len(cells))
    metrics = ("n_dup_groups", "total_dup_pairs", "cross_fold_dup_pairs")
    width = 0.8 / len(cells)
    for ci, cell in enumerate(cells):
        tier = fes[cell]["dup_audit"]["tiers"]["normalized"]
        xs = np.arange(len(metrics)) + ci * width
        ax.bar(xs, [tier[m] for m in metrics], width=width, color=c[ci], label=cell)
    ax.set_xticks(np.arange(len(metrics)) + 0.4 - width / 2)
    ax.set_xticklabels(["dup groups", "dup pairs", "cross-fold pairs"])
    ax.set_ylabel("count (normalized-text tier)")
    ax.set_title("Near-duplicate audit (kept prompts)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "dup_audit.png", written)


def fig_v5(v5: dict, out, written):
    c = paper_palette(2)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    layers = sorted(int(k) for k in v5["per_layer"])
    ax.plot(
        layers,
        [v5["per_layer"][str(li)]["raw_r2"] for li in layers],
        "o-",
        color=c[0],
        label="raw (E1)",
        ms=4,
    )
    ax.plot(
        layers,
        [v5["per_layer"][str(li)]["heldout_recal_r2"] for li in layers],
        "s-",
        color=c[1],
        label=f"{v5['variant']} held-out recal",
        ms=4,
    )
    ax.axhline(v5["recal_null"]["band_p975_layer_max"], color="gray", ls="--", lw=1.0)
    ax.set_xlabel("layer")
    ax.set_ylabel("pooled R²")
    ax.set_title(f"E2 {v5['variant']} (chat)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "refit_v5_chat.png", written)


def main() -> int:
    args = parse_args()
    set_paper_style()
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    rd = args.recal_dir
    written: list[str] = []
    hrs = {c: _load(rd / f"heldout_recal_{c}.json") for c in cells}
    fes = {c: _load(rd / f"fold_exch_{c}.json") for c in cells}
    verdict = _load(rd / "recal_verdict.json")
    qwen = _maybe(rd / "qwen_recal_cal.json")
    chat = cells[0]
    ab_npz = np.load(rd / "tensors" / f"recal_ab_{chat}.npz")
    qwen_npz_path = rd / "tensors" / "qwen_recal_draws.npz"
    qwen_npz = np.load(qwen_npz_path) if qwen_npz_path.exists() else None

    fig_hero(hrs[chat], verdict, ab_npz, qwen, qwen_npz, args.out, written)
    fig_perlayer(hrs, args.out, written)
    fig_optimism(hrs, args.out, written)
    fig_fold_norms(fes, args.out, written)
    fig_seed_refit(fes, args.out, written)
    fig_perfold_gain(hrs[chat], args.out, written)
    fig_boot(hrs[chat], verdict, args.out, written)
    if qwen is not None:
        fig_qwen(qwen, args.out, written)
    fig_dup(fes, args.out, written)
    v5 = _maybe(rd / f"refit_v5_{chat}.json")
    if v5 is not None:
        fig_v5(v5, args.out, written)

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_REPO_ROOT
    ).stdout.strip()
    meta = {
        "git_commit": sha,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/issue1336_recal_figures.py",
        "recal_dir": str(rd),
        "figures": written,
        "routed_decision": verdict["routed_decision"],
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[recal-figs] {len(written)} figures + meta.json -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
