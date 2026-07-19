#!/usr/bin/env python
"""Issue #1345 slot-ablation figures (plan v10 §6).

Renders, from the slot-round eval JSONs + preds caches (missing inputs skip
with a log line — the smoke leg may not produce every arm):
  hero:        per-slot L19 within-story R^2 bars (anchor + 4 candidates),
               Bonferroni-4 CIs as NON-NEGATIVE error offsets, the landed
               chat matched-row ceiling band + fresh chat obs, per-slot
               shuffle-null p95 ticks
  sweep:       28-layer R^2 overlay — one curve per slot-store cell + the
               chat comparator + the max-over-slots shuffle-null p95
  deficit:     D_k per slot with 95% CIs + the per-draw-max D distribution
  scatter:     per-conversation squared-error, best & worst slot vs chat
               (points colored by chat-error decile)
  secondary:   transfer / reparam recovery-vs-null bars per slot @ L19
  exploratory: slot-position histograms

Outputs: <fig-dir>/*.png (one PNG per figure; captions live in the report).
"""

from __future__ import annotations

import argparse
import json
import sys
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
import issue1345_common as c  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import set_paper_style  # noqa: E402

L19 = 19
SLOT_LABEL = {
    "R_instruct_r4slot_anchor_context": "attribution colon\n(anchor, landed)",
    "R_instruct_r4slot_qend_context": "end of question",
    "R_instruct_r4slot_preattr_context": "pre-attribution",
    "R_instruct_r4slot_preans_context": "pre-answer",
    "R_instruct_r4slot_attrmean_context": "attribution mean-pool",
}
VERDICT_LABEL = {
    "qend": "end of question",
    "preattr": "pre-attribution",
    "preans": "pre-answer",
    "attrmean": "attribution mean-pool",
}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _err_offsets(v: float, lo: float, hi: float) -> tuple[float, float]:
    """Non-negative errorbar offsets from CI bounds (errorbar rule; #547/#1335)."""
    return max(0.0, v - lo), max(0.0, hi - v)


def _null_p95_l19(eval_dir: Path, cid: str) -> float | None:
    d = _load(eval_dir / f"nulls_{cid}.json")
    if not d or not d.get("null_matrix"):
        return None
    mat = np.asarray(d["null_matrix"], dtype=np.float64)
    return float(np.nanquantile(mat[:, L19], 0.95))


def fig_hero(eval_dir: Path, fig_dir: Path) -> None:
    lattice = _load(eval_dir / "slot_verdict_lattice.json")
    cids = [c.SLOT_ANCHOR_CELL] + [c.SLOT_VERDICT_CELLS[k] for k in c.SLOT_VERDICT_CELLS]
    cells = {cid: _load(eval_dir / f"cells_{cid}.json") for cid in cids}
    have = [cid for cid in cids if cells[cid] is not None]
    if not have:
        print("[slot-plots] SKIP hero: no slot cell JSONs", flush=True)
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs, vals = [], []
    for i, cid in enumerate(have):
        v = float(cells[cid]["r2_per_layer_obs"][L19])
        xs.append(i)
        vals.append(v)
        lo, hi = None, None
        if cid == c.SLOT_ANCHOR_CELL:
            boot = cells[cid].get("r2_bootstrap_ci_frozen_layers_conv", {}).get(str(L19))
            if boot:
                lo, hi = boot["ci_lo"], boot["ci_hi"]
        elif lattice and "battery" in lattice:
            key = next((k for k, cc in c.SLOT_VERDICT_CELLS.items() if cc == cid), None)
            slot = lattice["battery"]["per_slot"].get(key or "")
            if slot:
                lo, hi = slot["delta_ci_bonferroni4"]
        if lo is not None:
            e_lo, e_hi = _err_offsets(v, lo, hi)
            ax.errorbar(i, v, yerr=[[e_lo], [e_hi]], fmt="none", ecolor="black", capsize=3)
        p95 = _null_p95_l19(eval_dir, cid)
        if p95 is not None:
            ax.plot([i - 0.3, i + 0.3], [p95, p95], color="gray", lw=1.2, ls=":")
    ax.bar(xs, vals, color="#4477AA", width=0.62)
    # Chat matched-row ceiling: landed conv-bootstrap band + fresh obs line.
    landed = _load(Path(c.SLOT_REFIT_ANCHOR_FILES[c.SLOT_CHAT_MATCHED_CELL]))
    if landed:
        boot = landed.get("r2_bootstrap_ci_frozen_layers_conv", {}).get(str(L19))
        if boot:
            ax.axhspan(boot["ci_lo"], boot["ci_hi"], color="#CCBB44", alpha=0.35, lw=0)
    fresh_chat = _load(eval_dir / f"cells_{c.SLOT_CHAT_MATCHED_CELL}.json")
    if fresh_chat:
        ax.axhline(float(fresh_chat["r2_per_layer_obs"][L19]), color="#CCBB44", lw=1.5)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([SLOT_LABEL[cid] for cid in have], fontsize=8)
    ax.set_ylabel("within-story held-out $R^2$ (L19)")
    ax.set_title("Story context-slot ablation: per-slot map strength vs chat ceiling")
    fig.savefig(fig_dir / "slot_hero_l19_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[slot-plots] wrote {fig_dir / 'slot_hero_l19_bar.png'}", flush=True)


def fig_layer_sweep(eval_dir: Path, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    plotted = False
    all_cells = [*SLOT_LABEL, c.SLOT_PREFIX_CELL]
    label_map = {**SLOT_LABEL, c.SLOT_PREFIX_CELL: "prefix (held fixed)"}
    null_curves = []
    for cid in all_cells:
        d = _load(eval_dir / f"cells_{cid}.json")
        if d is None:
            continue
        ax.plot(d["r2_per_layer_obs"], lw=1.4, label=label_map[cid])
        nd = _load(eval_dir / f"nulls_{cid}.json")
        if nd and nd.get("null_matrix"):
            null_curves.append(
                np.nanquantile(np.asarray(nd["null_matrix"], dtype=np.float64), 0.95, axis=0)
            )
        plotted = True
    chat = _load(eval_dir / f"cells_{c.SLOT_CHAT_MATCHED_CELL}.json")
    if chat:
        ax.plot(chat["r2_per_layer_obs"], lw=2.0, color="black", label="chat matched (same rows)")
        plotted = True
    if null_curves:
        ax.plot(
            np.max(np.stack(null_curves), axis=0),
            lw=1.0,
            ls=":",
            color="gray",
            label="shuffle-null p95 (max over slots)",
        )
    if not plotted:
        plt.close(fig)
        print("[slot-plots] SKIP layer sweep: no cell JSONs", flush=True)
        return
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out $R^2$")
    ax.set_title("28-layer within-story $R^2$ per slot")
    ax.legend(fontsize=7)
    fig.savefig(fig_dir / "slot_layer_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[slot-plots] wrote {fig_dir / 'slot_layer_sweep.png'}", flush=True)


def fig_deficit(eval_dir: Path, preds_dir: Path, fig_dir: Path) -> None:
    lattice = _load(eval_dir / "slot_verdict_lattice.json")
    if not lattice or "battery" not in lattice:
        print("[slot-plots] SKIP deficit panel: no battery in lattice", flush=True)
        return
    bat = lattice["battery"]
    keys = list(bat["per_slot"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    ax = axes[0]
    for i, k in enumerate(keys):
        v = bat["per_slot"][k]["d_k_obs"]
        lo, hi = bat["per_slot"][k]["d_k_ci95"]
        e_lo, e_hi = _err_offsets(v, lo, hi)
        ax.errorbar(i, v, yerr=[[e_lo], [e_hi]], fmt="o", color="#4477AA", capsize=3)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([VERDICT_LABEL[k] for k in keys], fontsize=8)
    ax.set_ylabel("paired deficit $D_k$ vs chat (L19)")
    ax.set_title("Per-slot paired deficit (95% CI)")
    ax2 = axes[1]
    draws_path = preds_dir / "slot_verdict_draws.npz"
    if draws_path.exists():
        d = np.load(draws_path)["d"]
        ax2.hist(d[np.isfinite(d)], bins=40, color="#4477AA", alpha=0.85)
        ax2.axvline(0.0, color="black", lw=0.8)
        for b in bat["d_ci95"]:
            ax2.axvline(b, color="#EE6677", lw=1.0, ls="--")
        ax2.set_xlabel("per-draw max-over-slots deficit $D$")
        ax2.set_ylabel("bootstrap draws")
        ax2.set_title(f"D distribution (verdict: {lattice.get('verdict')})")
    fig.savefig(fig_dir / "slot_paired_deficit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[slot-plots] wrote {fig_dir / 'slot_paired_deficit.png'}", flush=True)


def _per_conv_err(npz_path: Path) -> dict[str, float] | None:
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=False)
    pred, true = d["pred"].astype(np.float64), d["true"].astype(np.float64)
    conv = np.asarray([str(x) for x in d["conv_ids"]])
    err = ((true - pred) ** 2).sum(axis=1)
    out: dict[str, float] = {}
    for cid, e in zip(conv, err, strict=True):
        out[cid] = out.get(cid, 0.0) + float(e)
    return out


def fig_scatter(eval_dir: Path, preds_dir: Path, fig_dir: Path) -> None:
    lattice = _load(eval_dir / "slot_verdict_lattice.json")
    if not lattice or "battery" not in lattice:
        print("[slot-plots] SKIP scatter: no battery in lattice", flush=True)
        return
    per_slot = lattice["battery"]["per_slot"]
    ranked = sorted(per_slot, key=lambda k: per_slot[k]["d_k_obs"])
    picks = [("worst", ranked[0]), ("best", ranked[-1])]
    chat_err = _per_conv_err(preds_dir / f"{c.SLOT_CHAT_MATCHED_CELL}_L{L19}.npz")
    if chat_err is None:
        print("[slot-plots] SKIP scatter: chat preds cache missing", flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), layout="constrained")
    for ax, (tag, key) in zip(axes, picks, strict=True):
        slot_err = _per_conv_err(preds_dir / f"{c.SLOT_VERDICT_CELLS[key]}_L{L19}.npz")
        if slot_err is None:
            continue
        common = sorted(set(chat_err) & set(slot_err))
        xc = np.asarray([chat_err[i] for i in common])
        ys = np.asarray([slot_err[i] for i in common])
        deciles = np.clip((np.argsort(np.argsort(xc)) * 10) // max(len(xc), 1), 0, 9)
        sc = ax.scatter(xc, ys, c=deciles, cmap="viridis", s=8, alpha=0.6)
        lim = [min(xc.min(), ys.min()), max(xc.max(), ys.max())]
        ax.plot(lim, lim, color="black", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("chat matched per-conversation squared error (L19)")
        ax.set_ylabel(f"{VERDICT_LABEL[key]} squared error")
        ax.set_title(f"{tag} slot: {VERDICT_LABEL[key]} (n={len(common)})")
        fig.colorbar(sc, ax=ax, label="chat-error decile (conv rank)")
    fig.savefig(fig_dir / "slot_per_conversation_scatter.png", dpi=200)
    plt.close(fig)
    print(f"[slot-plots] wrote {fig_dir / 'slot_per_conversation_scatter.png'}", flush=True)


def fig_secondary(eval_dir: Path, fig_dir: Path) -> None:
    rows = []
    for key in c.SLOT_VERDICT_CELLS:
        xfer = _load(eval_dir / f"cross_regime_transfer_r1_r4slot_{key}.json")
        rep = _load(eval_dir / f"reparam_recovery_r1_r4slot_{key}.json")
        if xfer or rep:
            rows.append((key, xfer, rep))
    if not rows:
        print("[slot-plots] SKIP secondary: no transfer/reparam JSONs", flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    width = 0.35
    for i, (_key, xfer, rep) in enumerate(rows):
        if xfer:
            axes[0].bar(
                i - width / 2,
                xfer["legs"]["r1_to_slot"]["r2_by_layer"][str(L19)],
                width,
                color="#4477AA",
                label="chat→slot" if i == 0 else None,
            )
            axes[0].bar(
                i + width / 2,
                xfer["legs"]["slot_to_r1"]["r2_by_layer"][str(L19)],
                width,
                color="#EE6677",
                label="slot→chat" if i == 0 else None,
            )
        if rep:
            axes[1].bar(
                i - width / 2,
                rep["recov"]["b2i"],
                width,
                color="#4477AA",
                label="recovered (slot center)" if i == 0 else None,
            )
            null = rep["reparam"][str(L19)]["matched_capacity_nulls"]["b2i"]["null_recovery_r2"]
            axes[1].plot([i - width, i + width], [null, null], color="gray", lw=1.2, ls=":")
    for ax, title, ylab in (
        (axes[0], "Cross-framing transfer @ L19", "transfer $R^2$"),
        (axes[1], "A·M·B reparam recovery @ L19 vs matched-capacity null", "recovery $R^2$"),
    ):
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([VERDICT_LABEL[r[0]] for r in rows], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7)
    fig.savefig(fig_dir / "slot_transfer_reparam_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[slot-plots] wrote {fig_dir / 'slot_transfer_reparam_bars.png'}", flush=True)


def fig_positions(turnstore_dir: Path, fig_dir: Path) -> None:
    stem = c.stem_for("instruct", "r4slot")
    diag = _load(turnstore_dir / f"{stem}_slot_diagnostics.json")
    if not diag:
        print("[slot-plots] SKIP positions: diagnostics missing", flush=True)
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name in c.SLOT_SINGLE_ORDER:
        ax.hist(diag["positions"][name], bins=40, histtype="step", lw=1.4, label=name)
    ax.set_xlabel("token index")
    ax.set_ylabel("stories")
    ax.set_title("Slot read positions per story (exploratory)")
    ax.legend(fontsize=7)
    fig.savefig(fig_dir / "slot_positions_hist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[slot-plots] wrote {fig_dir / 'slot_positions_hist.png'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=c.EVAL_DIR)
    ap.add_argument("--fig-dir", type=Path, default=c.FIG_DIR)
    ap.add_argument("--turnstore-dir", type=Path, default=c.TURNSTORE_DIR)
    ap.add_argument("--preds-dir", type=Path, default=c.PREDS_CACHE_DIR)
    args = ap.parse_args()

    assert c.HAS_SLOT_ABLATION, (
        f"issue1345_slot_plots requires EPM_I1345_VARIANT in {c.SLOT_ABLATION_VARIANTS} "
        f"(got {c.VARIANT!r})"
    )
    set_paper_style()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig_hero(args.out_dir, args.fig_dir)
    fig_layer_sweep(args.out_dir, args.fig_dir)
    fig_deficit(args.out_dir, args.preds_dir, args.fig_dir)
    fig_scatter(args.out_dir, args.preds_dir, args.fig_dir)
    fig_secondary(args.out_dir, args.fig_dir)
    fig_positions(args.turnstore_dir, args.fig_dir)
    print("[slot-plots] complete", flush=True)


if __name__ == "__main__":
    main()
