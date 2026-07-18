#!/usr/bin/env python
"""#1434 figures (plan §6 HERO pair; exploratory dumps ride the persisted
matrices via the analyzer's /paper-plots pass).

1. ``install_grid.png`` — per-context x per-lr selected-rung Tier-2 judged
   rates with Wilson 95% CIs, base bars alongside, the 0.60-0.85 band shaded.
2. ``rb_validity_scatter.png`` — judged delta vs r_B projection at the
   read-out layer (leakage grid), honest randnorm band + the |rho|<=1 ceiling
   annotated in the caption sidecar.

Errorbars are NON-NEGATIVE OFFSETS (np.maximum(0, .) element-wise — the
#547/#1335 xerr/yerr rule); tiny-n inverted quantile CIs clamp instead of
crashing savefig.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1434_cells as cells  # noqa: E402

PLAIN = {
    "ws-pers": "Persona (software engineer)",
    "ws-bare": "Bare assistant",
    "ws-conv": "WildChat prefix",
    "ws-icl": "ICL prefix",
}
PLAIN.update({f"ws-po-{k.removeprefix('ws-')}": v for k, v in list(PLAIN.items())})
LR_LABEL = {"lr1e5": "lr 1e-5", "lr3e5": "lr 3e-5", "lr1e4": "lr 1e-4"}
# Reader-facing names for READ contexts (leakage-grid eval contexts). Internal
# slugs in tick labels are a clean-result-critic Lens 3 FAIL (#1434 round 3).
READ_CTX_PLAIN = {
    "default": "Default assistant",
    "persona_software_engineer": "Persona (software engineer)",
    "neg_sp_police": "Police officer",
    "neg_sp_ph4": "Philosopher",
    "icl_prefix_writing_style": "Two-shot ICL prefix",
    "wildchat_prefix_real545": "WildChat prefix",
}


def _save(fig, out: Path) -> Path:
    """Save PNG + PDF + .meta.json sidecar via savefig_paper; returns the PNG path."""
    from explore_persona_space.analysis.paper_plots import savefig_paper

    paths = savefig_paper(fig, out.stem, dir=out.parent)
    plt.close(fig)
    return paths["png"]


def _err(rate: float, ci: list[float]) -> tuple[float, float]:
    """Non-negative (lo, hi) errorbar OFFSETS from a Wilson CI (#547/#1335)."""
    lo, hi = ci
    return (float(np.maximum(0.0, rate - lo)), float(np.maximum(0.0, hi - rate)))


def fig_install_grid(
    agg: dict,
    out_dir: Path,
    *,
    cell_keys: tuple[str, ...] = cells.CELL_KEYS,
    runs=cells.I1434_RUNS,
    fname: str = "install_grid.png",
    title_prefix: str = "writing_style",
) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    ax.axhspan(*agg["band"], color="tab:green", alpha=0.12, label="target band 0.60-0.85")
    xticks, xlabels = [], []
    x = 0.0
    for cell_key in cell_keys:
        entry = agg["tier2"].get(cell_key) or {}
        base = entry.get("base")
        group_x = []
        if base and base.get("rate") is not None:
            lo, hi = _err(base["rate"], base["wilson_95"])
            ax.bar(x, base["rate"], width=0.8, color="0.7", label=None)
            ax.errorbar(x, base["rate"], yerr=[[lo], [hi]], fmt="none", ecolor="k", capsize=3)
            xticks.append(x)
            xlabels.append(f"{PLAIN[cell_key]}\nbase")
            group_x.append(x)
            x += 1.0
        for run in runs:
            if run.cell_key != cell_key:
                continue
            tag = run.run_id.rsplit("-", 1)[-1]
            ladder = agg["ladders"].get(run.run_id) or {}
            rec = None
            va = (entry.get("verdict_arm") or {}).get("run_id")
            if va == run.run_id and entry.get("trained"):
                rec = entry["trained"]
            if rec and rec.get("rate") is not None:
                lo, hi = _err(rec["rate"], rec["wilson_95"])
                ax.bar(x, rec["rate"], width=0.8, color="tab:blue")
                ax.errorbar(x, rec["rate"], yerr=[[lo], [hi]], fmt="none", ecolor="k", capsize=3)
            elif ladder.get("selection"):
                # non-verdict arms: Tier-1 selected-rung rate (no fresh Tier-2)
                ax.bar(x, float(ladder["selection"]["rate"]), width=0.8, color="tab:cyan")
            xticks.append(x)
            xlabels.append(f"{PLAIN[cell_key]}\n{LR_LABEL.get(tag, tag)}")
            group_x.append(x)
            x += 1.0
        x += 0.8
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("judged casual-register rate (pv rubric)")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"{title_prefix} install grid — Tier-2 verdict-arm rates (blue), Tier-1 "
        "selected-rung rates (cyan), base (grey); Wilson 95% CIs"
    )
    return _save(fig, out_dir / fname)


def fig_rb_validity(val: dict, proj: dict, agg: dict, out_dir: Path) -> Path:
    grid = (val.get("grids") or {}).get("leakage") or {}
    arm = grid.get("response_shared") or {}
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    if arm.get("observed_abs_rho_per_layer"):
        layer = int(arm["max_layer"])
        xs, ys, labels = [], [], []
        for _cell_key, prec in (agg.get("panel") or {}).items():
            rid = prec["run_id"]
            pr = proj["states"].get(rid)
            if pr is None:
                continue
            for ctx_id, row in prec["contexts"].items():
                if row.get("delta") is None:  # all-dropped arm (None-propagated)
                    continue
                xs.append(float(pr["response_shared"]["projection"][layer]))
                ys.append(float(row["delta"]))
                labels.append(f"{rid}@{ctx_id}")
        ax.scatter(xs, ys, s=18, color="tab:blue")
        for xi, yi, lab in zip(xs, ys, labels, strict=True):
            ax.annotate(lab, (xi, yi), fontsize=4, alpha=0.6)
        band = (arm.get("null_within_class") or {}).get("max_selected_p97_5")
        title = f"r_B validity — leakage grid, layer {layer}: |rho|={arm['max_abs_rho']:.3f}" + (
            f"; honest randnorm max-selected p97.5={band:.3f}; ceiling |rho|<=1" if band else ""
        )
    else:
        ax.text(0.5, 0.5, f"insufficient cells: {grid}", ha="center", va="center")
        title = "r_B validity — insufficient cells"
    ax.set_xlabel("shift . r_B-hat (response-shared arm)")
    ax.set_ylabel("judged rate delta (trained - base)")
    ax.set_title(title, fontsize=9)
    out = out_dir / "rb_validity_scatter.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ── i1434po regime figures (amendment plan §6) ───────────────────────────────


def _ci_err(v: float, ci: list[float]) -> tuple[float, float]:
    """Non-negative errorbar offsets from a CI around v (#547/#1335 clamp)."""
    return (float(np.maximum(0.0, v - ci[0])), float(np.maximum(0.0, ci[1] - v)))


def fig_regime_hero(contrast: dict, out_dir: Path) -> Path:
    """HERO (plan §6): per training context, grouped pooled non-source leakage
    DELTA bars (trained - shared base; contrastive vs positive-only, Newcombe
    CIs) + the pooled trained-vs-trained D with its Newcombe CI. Dose-unmatched
    contexts are flagged in the tick label (caption territory — no overlays)."""
    ctxs = [c for c in contrast.get("contexts", {})]
    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(10, 7), layout="constrained", height_ratios=[2, 1], sharex=True
    )
    xs = np.arange(len(ctxs), dtype=float)
    labels = []
    for i, ck in enumerate(ctxs):
        entry = contrast["contexts"][ck]
        pooled = entry.get("pooled") or {}
        lab = PLAIN.get(ck, ck)
        if (entry.get("dose") or {}).get("dose_unmatched"):
            lab += "\n[dose-unmatched]"
        labels.append(lab)
        for off, key, color, name in (
            (-0.18, "delta_con_vs_base", "tab:blue", "contrastive"),
            (0.18, "delta_po_vs_base", "tab:orange", "positive-only"),
        ):
            rec = pooled.get(key)
            if not rec or rec.get("delta") is None:
                continue
            lo, hi = _ci_err(rec["delta"], rec["newcombe_95"])
            ax.bar(
                xs[i] + off, rec["delta"], width=0.34, color=color, label=name if i == 0 else None
            )
            ax.errorbar(
                xs[i] + off, rec["delta"], yerr=[[lo], [hi]], fmt="none", ecolor="k", capsize=3
            )
        if pooled.get("status") == "computed":
            lo, hi = _ci_err(pooled["D"], pooled["newcombe_95"])
            axd.errorbar(xs[i], pooled["D"], yerr=[[lo], [hi]], fmt="o", color="tab:red", capsize=4)
            axd.annotate(pooled.get("lattice", ""), (xs[i], pooled["D"]), fontsize=6, alpha=0.7)
    ax.axhline(0.0, color="0.4", lw=0.8)
    axd.axhline(0.0, color="0.4", lw=0.8)
    ax.set_ylabel("pooled non-source leakage delta\n(trained - base, judged rate)")
    axd.set_ylabel("D = positive-only - contrastive\n(trained-vs-trained)")
    axd.set_xticks(xs)
    axd.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title(
        "writing_style regime contrast — pooled non-source leakage per training context "
        "(band-matched verdict arms; Newcombe 95% CIs)",
        fontsize=10,
    )
    return _save(fig, out_dir / "po_regime_hero.png")


def fig_regime_cells(contrast: dict, out_dir: Path) -> Path:
    """Low-level companion (plan §6 item 1): the per-(training ctx x read ctx)
    D dot plot, point-labeled, Newcombe CIs as non-negative offsets."""
    rows = [c for c in contrast.get("cells", []) if c.get("status") == "computed"]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(rows) + 1)), layout="constrained")
    ys = np.arange(len(rows), dtype=float)
    for y, cell in zip(ys, rows, strict=True):
        lo, hi = _ci_err(cell["D"], cell["newcombe_95"])
        ax.errorbar(cell["D"], y, xerr=[[lo], [hi]], fmt="o", color="tab:purple", capsize=3)
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [
            f"{PLAIN.get(c['training_cell'], c['training_cell'])} @ "
            f"{READ_CTX_PLAIN.get(c['read_ctx'], c['read_ctx'])}"
            for c in rows
        ],
        fontsize=6,
    )
    ax.set_xlabel("D = positive-only - contrastive (judged rate, trained arms)")
    ax.set_title(
        f"per-cell regime contrast ({len(rows)} training x read cells; Newcombe 95%)",
        fontsize=10,
    )
    return _save(fig, out_dir / "po_regime_cells.png")


def fig_ladder_overlays(po_agg: dict, con_agg: dict, out_dir: Path) -> Path:
    """Exploratory (plan §6): per-context Tier-1 dose-ladder overlays,
    contrastive (solid) vs positive-only (dashed), one panel per context."""
    fig, axes = plt.subplots(1, len(cells.PO_CELL_KEYS), figsize=(16, 4), layout="constrained")
    colors = {"lr1e5": "tab:blue", "lr3e5": "tab:green", "lr1e4": "tab:red"}
    for ax, po_ck in zip(np.atleast_1d(axes), cells.PO_CELL_KEYS, strict=True):
        con_ck = cells.parent_cell_key(po_ck)
        for agg, style, prefix in ((con_agg, "-", con_ck), (po_agg, "--", po_ck)):
            for run_id, lad in (agg.get("ladders") or {}).items():
                if not run_id.startswith(f"{prefix}-lr"):
                    continue
                rates = lad.get("rates_by_step") or {}
                if not rates:
                    continue
                steps = sorted(int(s) for s in rates)
                tag = run_id.rsplit("-", 1)[-1]
                ax.plot(
                    steps,
                    [rates[str(s)] for s in steps],
                    style,
                    color=colors.get(tag, "0.5"),
                    lw=1.2,
                    label=(
                        f"{LR_LABEL.get(tag, tag)} "
                        f"({'positive-only' if style == '--' else 'contrastive'})"
                    ),
                )
        band = po_agg.get("band") or con_agg.get("band")
        if band:
            ax.axhspan(*band, color="tab:green", alpha=0.10)
        ax.set_title(PLAIN.get(po_ck, po_ck), fontsize=9)
        ax.set_xlabel("optimizer step")
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("Tier-1 judged rate")
    np.atleast_1d(axes)[-1].legend(fontsize=5)
    fig.suptitle("dose ladders — contrastive (solid) vs positive-only (dashed)", fontsize=10)
    return _save(fig, out_dir / "po_ladder_overlays.png")


# ── persona-dose-matched-regime figures (plan v8 §6) ─────────────────────────

BRACKET_PLAIN = {
    "high": "matched-high (po@25 vs con@45)",
    "low": "near-matched-low (po@10 vs con@25)",
}


def fig_dose_hero(contrast: dict, out_dir: Path) -> Path:
    """HERO (plan v8 §6): the persona-context regime contrast D under three
    dose regimes — unmatched (parent verdict arms), matched-high D_hi,
    near-matched-low D_lo — Newcombe 95% CIs, with the dose-matched
    bare-assistant D as the reference point. Per-arm Tier-1 selection rates
    live in the caption, never as overlays."""
    refs = contrast["references"]
    pts = [
        ("Unmatched\n(parent verdict arms)", refs["unmatched_persona"], "tab:orange"),
        ("Matched-high\n(po@25 vs con@45 new)", contrast["brackets"]["high"], "tab:blue"),
        ("Near-matched-low\n(po@10 new vs con@25)", contrast["brackets"]["low"], "tab:cyan"),
        ("Bare-assistant\nreference (matched)", refs["bare_matched"], "0.6"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for i, (_lab, rec, color) in enumerate(pts):
        if rec.get("D") is None:
            continue
        lo, hi = _ci_err(rec["D"], rec["newcombe_95"])
        ax.bar(i, rec["D"], width=0.6, color=color)
        ax.errorbar(i, rec["D"], yerr=[[lo], [hi]], fmt="none", ecolor="k", capsize=4)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(range(len(pts)))
    ax.set_xticklabels([p[0] for p in pts], fontsize=8)
    ax.set_ylabel("D = positive-only - contrastive\n(pooled non-source judged rate)")
    ax.set_title(
        "persona regime contrast D under dose matching — verdict lattice: "
        f"{contrast['brackets']['high'].get('lattice')} (Newcombe 95% CIs)",
        fontsize=9,
    )
    return _save(fig, out_dir / "dose_hero.png")


def fig_dose_cells(contrast: dict, out_dir: Path) -> Path:
    """Exploratory (plan v8 §6 read 4): the 10-cell per-(bracket x read
    context) D dot plot, point-labeled, Newcombe CIs as non-negative offsets."""
    rows = [c for c in contrast.get("cells", []) if c.get("status") == "computed"]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(rows) + 1)), layout="constrained")
    ys = np.arange(len(rows), dtype=float)
    colors = {"high": "tab:blue", "low": "tab:cyan"}
    for y, cell in zip(ys, rows, strict=True):
        lo, hi = _ci_err(cell["D"], cell["newcombe_95"])
        ax.errorbar(
            cell["D"],
            y,
            xerr=[[lo], [hi]],
            fmt="o",
            color=colors.get(cell.get("bracket"), "tab:purple"),
            capsize=3,
        )
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [
            f"{BRACKET_PLAIN.get(c['bracket'], c['bracket'])} @ "
            f"{READ_CTX_PLAIN.get(c['read_ctx'], c['read_ctx'])}"
            for c in rows
        ],
        fontsize=6,
    )
    ax.set_xlabel("D = positive-only - contrastive (judged rate, trained arms)")
    ax.set_title(
        f"dose-matched per-cell regime contrast ({len(rows)} bracket x read cells; Newcombe 95%)",
        fontsize=10,
    )
    return _save(fig, out_dir / "dose_cells.png")


def fig_dose_ladders(po_agg: dict, con_agg: dict, contrast: dict, out_dir: Path) -> Path:
    """Exploratory (plan v8 §6): the two persona-context Tier-1 ladders with
    the FOUR panel rungs marked (existing verdict arms = circles, new
    dose-selected arms = diamonds)."""
    rungs = contrast.get("panel_rungs") or []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
    panels = (
        ("contrastive persona", con_agg, ("ws-pers-lr1e5",)),
        ("positive-only persona", po_agg, ("ws-po-pers-lr1e5", "ws-po-pers-lr3e5")),
    )
    lr_colors = {"lr1e5": "tab:blue", "lr3e5": "tab:green"}
    for ax, (title, agg, run_ids) in zip(np.atleast_1d(axes), panels, strict=True):
        band = agg.get("band")
        if band:
            ax.axhspan(*band, color="tab:green", alpha=0.10, label="band 0.60-0.85")
        for run_id in run_ids:
            rates = (agg.get("ladders") or {}).get(run_id, {}).get("rates_by_step") or {}
            steps = sorted(int(s) for s in rates)
            tag = run_id.rsplit("-", 1)[-1]
            ax.plot(
                steps,
                [rates[str(s)] for s in steps],
                "-",
                color=lr_colors.get(tag, "0.5"),
                lw=1.2,
                label=LR_LABEL.get(tag, tag),
            )
        for r in rungs:
            if r["run_id"] not in run_ids:
                continue
            new = "(new)" in r["role"]
            ax.scatter(
                [r["step"]],
                [r["rate"]],
                marker="D" if new else "o",
                s=55,
                zorder=5,
                facecolor="tab:red" if new else "none",
                edgecolor="k",
                label=r["role"],
            )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("optimizer step")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6)
    np.atleast_1d(axes)[0].set_ylabel("Tier-1 judged rate")
    fig.suptitle("persona dose ladders — the four panel rungs marked", fontsize=10)
    return _save(fig, out_dir / "dose_ladders.png")


def fig_dose_pooled(contrast: dict, out_dir: Path) -> Path:
    """Exploratory (plan v8 §6): pooled non-source po-vs-con judged rates per
    bracket + the shared base bar (Wilson 95% CIs)."""
    from issue1434_cells import wilson

    bars = []
    for bracket in ("high", "low"):
        rec = contrast["brackets"][bracket]
        if rec.get("status") != "computed":
            continue
        for side, color in (("po", "tab:orange"), ("con", "tab:blue")):
            s = rec[side]
            bars.append(
                (
                    f"{BRACKET_PLAIN[bracket].split(' (')[0]}\n{rec[f'{side}_arm']}",
                    s["k"],
                    s["n"],
                    color,
                )
            )
    base = (contrast.get("static_arms") or {}).get("base", {}).get("pooled_nonsource")
    if base:
        bars.append(("shared base panel", base["k"], base["n"], "0.7"))
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for i, (_lab, k, n, color) in enumerate(bars):
        rate = k / n
        lo, hi = _err(rate, list(wilson(k, n)))
        ax.bar(i, rate, width=0.7, color=color)
        ax.errorbar(i, rate, yerr=[[lo], [hi]], fmt="none", ecolor="k", capsize=3)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=7)
    ax.set_ylabel("pooled non-source judged casual-register rate")
    ax.set_ylim(0, 1)
    ax.set_title("dose-matched pooled leakage per bracket (Wilson 95% CIs)", fontsize=10)
    return _save(fig, out_dir / "dose_pooled_bars.png")


def fig_dose_graded(panel: dict, out_dir: Path) -> Path:
    """Exploratory (plan v8 §6): graded 0-100 score distributions (pooled
    non-source per-item means) for the 2 NEW dose arms."""
    arms = panel.get("arms") or {}
    fig, axes = plt.subplots(1, max(1, len(arms)), figsize=(10, 4), layout="constrained")
    for ax, (label, entry) in zip(np.atleast_1d(axes), sorted(arms.items()), strict=True):
        scores = entry.get("pooled_nonsource_scores") or []
        if scores:
            ax.hist(scores, bins=20, range=(0, 100), color="tab:blue")
            ax.axvline(50, color="tab:red", lw=0.9, label="rate threshold 50")
            ax.legend(fontsize=6)
        else:
            ax.text(0.5, 0.5, "no scored items", ha="center", va="center")
        rate = (entry.get("pooled_nonsource") or {}).get("rate")
        ax.set_title(f"{label} — pooled rate {rate:.3f}" if rate is not None else label, fontsize=9)
        ax.set_xlabel("mean graded score (kept draws)")
    np.atleast_1d(axes)[0].set_ylabel("items")
    fig.suptitle("graded score distributions — new dose arms (non-source contexts)", fontsize=10)
    return _save(fig, out_dir / "dose_graded_dist.png")


def dose_figures(
    contrast: dict, panel: dict, po_agg: dict, con_agg: dict, out_dir: Path
) -> list[Path]:
    """All plan-v8 dose figures (hero + exploratory companions)."""
    return [
        fig_dose_hero(contrast, out_dir),
        fig_dose_cells(contrast, out_dir),
        fig_dose_ladders(po_agg, con_agg, contrast, out_dir),
        fig_dose_pooled(contrast, out_dir),
        fig_dose_graded(panel, out_dir),
    ]


# ── icl-read-amplifier-specificity (ctrl) round figures (plan v11 §6) ────────

CTRL_ARM_PLAIN = {
    "imp-installed": "Installed impolite organism (unrelated behavior)",
    "imp-inert": "Inert impolite adapter (install delta 0)",
}
_CTRL_CTX_ORDER = (
    "default",
    "persona_software_engineer",
    "neg_sp_police",
    "neg_sp_ph4",
    "icl_prefix_writing_style",
    "wildchat_prefix_real545",
)


def _ctrl_ctx_order(panel: dict) -> list[str]:
    """Deterministic read-context order (known contexts first, extras appended
    — the smoke's tiny panel is a subset of the production 6)."""
    have = set()
    for entry in (panel.get("arms") or {}).values():
        have.update((entry.get("contexts") or {}).keys())
    ordered = [c for c in _CTRL_CTX_ORDER if c in have]
    return ordered + sorted(have - set(ordered))


def fig_ctrl_hero(spec: dict, panel: dict, con_agg: dict, out_dir: Path) -> Path:
    """HERO (plan v11 §6): the installed impolite arm's 6-context casual-delta
    profile (Newcombe 95%) overlaid with the four casual organisms' committed
    delta profiles; the verdict ICL cell highlighted; base at 0."""
    ctxs = _ctrl_ctx_order(panel)
    x = np.arange(len(ctxs), dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    # Committed casual organisms (reference profiles, Newcombe from counts).
    for cell_key, color in (
        ("ws-pers", "0.55"),
        ("ws-bare", "0.65"),
        ("ws-conv", "0.75"),
        ("ws-icl", "0.85"),
    ):
        entry = (con_agg.get("panel") or {}).get(cell_key)
        if not entry:
            continue
        ds, els, ehs, xs = [], [], [], []
        for i, ctx in enumerate(ctxs):
            rec = (entry.get("contexts") or {}).get(ctx)
            if not rec:
                continue
            t, b = rec["trained"], rec["base"]
            d = t["rate"] - b["rate"]
            ci = cells.newcombe(t["k_positive"], t["n_scored"], b["k_positive"], b["n_scored"])
            xs.append(i)
            ds.append(d)
            lo, hi = _ci_err(d, list(ci))
            els.append(lo)
            ehs.append(hi)
        ax.errorbar(
            xs,
            ds,
            yerr=[els, ehs],
            fmt="o--",
            color=color,
            capsize=2,
            ms=4,
            label=f"casual organism — {PLAIN[cell_key]} (committed)",
        )
    # The NEW installed unrelated-behavior arm (verdict profile).
    installed = (panel.get("arms") or {}).get("imp-installed") or {}
    ds, els, ehs, xs = [], [], [], []
    for i, ctx in enumerate(ctxs):
        rec = (installed.get("contexts") or {}).get(ctx) or {}
        if rec.get("delta") is None:
            continue
        xs.append(i)
        ds.append(rec["delta"])
        lo, hi = _ci_err(rec["delta"], rec["newcombe_95"])
        els.append(lo)
        ehs.append(hi)
    ax.errorbar(
        xs,
        ds,
        yerr=[els, ehs],
        fmt="s-",
        color="tab:red",
        capsize=3,
        ms=6,
        lw=2,
        label=CTRL_ARM_PLAIN["imp-installed"],
    )
    if "icl_prefix_writing_style" in ctxs:
        i = ctxs.index("icl_prefix_writing_style")
        ax.axvspan(i - 0.35, i + 0.35, color="tab:orange", alpha=0.15, label="verdict ICL cell")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([READ_CTX_PLAIN.get(c, c) for c in ctxs], fontsize=7)
    ax.set_ylabel("casual-register delta vs base (judged rate)")
    v = spec.get("verdict_cell") or {}
    ax.set_title(
        "unrelated-organism control: casual delta per read context — "
        f"verdict lattice: {v.get('lattice')} (Newcombe 95% CIs)",
        fontsize=9,
    )
    ax.legend(fontsize=6, loc="upper left")
    return _save(fig, out_dir / "ctrl_hero.png")


def fig_ctrl_arm_bars(panel: dict, out_dir: Path) -> Path:
    """Exploratory (plan v11 §6): both control arms' per-context casual rates
    (Wilson 95%), with the committed base bar at 0."""
    ctxs = _ctrl_ctx_order(panel)
    arms = sorted((panel.get("arms") or {}).items())
    width = 0.8 / max(1, len(arms) + 1)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    colors = {"imp-installed": "tab:red", "imp-inert": "tab:purple"}
    for j, (label, entry) in enumerate(arms):
        for i, ctx in enumerate(ctxs):
            rec = ((entry.get("contexts") or {}).get(ctx) or {}).get("trained") or {}
            if rec.get("rate") is None:
                continue
            xpos = i + (j - len(arms) / 2) * width
            lo, hi = _err(rec["rate"], rec["wilson_95"])
            ax.bar(
                xpos,
                rec["rate"],
                width=width,
                color=colors.get(label, "tab:gray"),
                label=CTRL_ARM_PLAIN.get(label, label) if i == 0 else None,
            )
            ax.errorbar(xpos, rec["rate"], yerr=[[lo], [hi]], fmt="none", ecolor="k", capsize=2)
    base = panel.get("base_panel") or {}
    for i, ctx in enumerate(ctxs):
        b = base.get(ctx)
        if b:
            ax.bar(
                i + (len(arms) - len(arms) / 2) * width,
                b["rate"],
                width=width,
                color="0.7",
                label="committed base panel" if i == 0 else None,
            )
    ax.set_xticks(range(len(ctxs)))
    ax.set_xticklabels([READ_CTX_PLAIN.get(c, c) for c in ctxs], fontsize=7)
    ax.set_ylabel("judged casual-register rate")
    ax.set_ylim(0, 1)
    ax.set_title("control arms: casual rate per read context (Wilson 95% CIs)", fontsize=10)
    ax.legend(fontsize=7)
    return _save(fig, out_dir / "ctrl_arm_bars.png")


def fig_ctrl_graded(panel: dict, out_dir: Path) -> Path:
    """Exploratory (plan v11 §6): graded 0-100 score distributions per
    (arm x read context) cell under the casual pv rubric."""
    ctxs = _ctrl_ctx_order(panel)
    arms = sorted((panel.get("arms") or {}).items())
    fig, axes = plt.subplots(
        max(1, len(arms)),
        max(1, len(ctxs)),
        figsize=(2.1 * max(1, len(ctxs)), 2.4 * max(1, len(arms))),
        layout="constrained",
        squeeze=False,
    )
    for r, (label, entry) in enumerate(arms):
        for c, ctx in enumerate(ctxs):
            ax = axes[r][c]
            rec = ((entry.get("contexts") or {}).get(ctx) or {}).get("trained") or {}
            scores = rec.get("scores") or []
            if scores:
                ax.hist(scores, bins=20, range=(0, 100), color="tab:blue")
                ax.axvline(50, color="tab:red", lw=0.8)
            else:
                ax.text(0.5, 0.5, "no scored items", ha="center", va="center", fontsize=6)
            if r == 0:
                ax.set_title(READ_CTX_PLAIN.get(ctx, ctx), fontsize=6)
            if c == 0:
                ax.set_ylabel(label, fontsize=7)
            ax.tick_params(labelsize=5)
    fig.suptitle("graded casual scores per (arm x read context) — threshold 50", fontsize=10)
    return _save(fig, out_dir / "ctrl_graded_dist.png")


def fig_ctrl_dual_rubric(spec: dict, panel: dict, out_dir: Path) -> Path:
    """Exploratory (plan v11 §6): the dual-rubric grid — casual vs impolite
    judged rate per (arm x {persona, ICL} cell), Wilson 95%."""
    dual_ctxs = [
        c
        for c in ("persona_software_engineer", "icl_prefix_writing_style")
        if any(
            "impolite" in ((e.get("contexts") or {}).get(c) or {})
            for e in (panel.get("arms") or {}).values()
        )
    ]
    arms = sorted((panel.get("arms") or {}).items())
    bars: list[tuple[str, dict, str]] = []
    for label, entry in arms:
        for ctx in dual_ctxs:
            row = (entry.get("contexts") or {}).get(ctx) or {}
            for instrument, color in (("trained", "tab:blue"), ("impolite", "tab:red")):
                rec = row.get(instrument) or {}
                if rec.get("rate") is None:
                    continue
                rubric = "casual" if instrument == "trained" else "impolite"
                bars.append((f"{label}\n{READ_CTX_PLAIN.get(ctx, ctx)}\n{rubric}", rec, color))
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(bars)), 5), layout="constrained")
    for i, (_lab, rec, color) in enumerate(bars):
        lo, hi = _err(rec["rate"], rec["wilson_95"])
        ax.bar(i, rec["rate"], width=0.7, color=color)
        ax.errorbar(i, rec["rate"], yerr=[[lo], [hi]], fmt="none", ecolor="k", capsize=3)
    gate = (spec.get("engagement_gate") or {}).get("threshold")
    if gate is not None:
        ax.axhline(gate, color="tab:green", lw=0.9, ls="--", label=f"engagement gate {gate}")
        ax.legend(fontsize=7)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=6)
    ax.set_ylabel("judged rate")
    ax.set_ylim(0, 1)
    ax.set_title(
        "dual-rubric grid: casual vs impolite rate per (arm x cell), Wilson 95%", fontsize=10
    )
    return _save(fig, out_dir / "ctrl_dual_rubric.png")


def fig_ctrl_drop_table(spec: dict, out_dir: Path) -> Path:
    """Exploratory (plan v11 §6/§8): per-arm dropped-draw report — content
    drops vs transport losses (llm-judging rule 24 split)."""
    rows = []
    for label, rep in sorted((spec.get("per_arm_drop_report") or {}).items()):
        rows.append(
            [
                label,
                rep.get("n_items"),
                rep.get("n_scored"),
                rep.get("n_dropped_draws_content"),
                rep.get("n_transport_lost_draws"),
            ]
        )
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.5 * max(1, len(rows))), layout="constrained")
    ax.axis("off")
    table = ax.table(
        cellText=[[str(v) for v in r] for r in rows] or [["(none)"] * 5],
        colLabels=["arm", "items", "scored", "content drops", "transport losses"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    ax.set_title("per-arm judge drop report (content vs transport)", fontsize=10)
    return _save(fig, out_dir / "ctrl_drop_table.png")


def ctrl_figures(spec: dict, panel: dict, con_agg: dict, out_dir: Path) -> list[Path]:
    """All plan-v11 ctrl figures (hero + exploratory companions)."""
    return [
        fig_ctrl_hero(spec, panel, con_agg, out_dir),
        fig_ctrl_arm_bars(panel, out_dir),
        fig_ctrl_graded(panel, out_dir),
        fig_ctrl_dual_rubric(spec, panel, out_dir),
        fig_ctrl_drop_table(spec, out_dir),
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="#1434 hero figures")
    p.add_argument("--round", default="i1434", choices=("i1434", "i1434po", "dose", "ctrl"))
    p.add_argument("--deliverables", default=None)
    p.add_argument("--projections", default=None)
    p.add_argument("--out-dir", default=str(cells.FIGURES_DIR_1434))
    args = p.parse_args(argv)
    if args.round == "dose":
        deliver = Path(args.deliverables or cells.DOSE_DELIVERABLES_DIR)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams["savefig.dpi"] = 180
        contrast = json.loads((deliver / "regime_contrast_dose_matched.json").read_text())
        panel = json.loads((deliver / "i1434dose_panel.json").read_text())
        po_agg = json.loads((cells.PO_DELIVERABLES_DIR / "i1434po_ladders.json").read_text())
        con_agg = json.loads((cells.DELIVERABLES_DIR_1434 / "i1434_ladders.json").read_text())
        for path in dose_figures(contrast, panel, po_agg, con_agg, out_dir):
            print(path)
        return 0
    if args.round == "ctrl":
        deliver = Path(
            args.deliverables or (cells.DELIVERABLES_DIR_1434 / "icl-read-amplifier-specificity")
        )
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams["savefig.dpi"] = 180
        spec = json.loads((deliver / "icl_specificity.json").read_text())
        panel = json.loads((deliver / "control_panel.json").read_text())
        con_agg = json.loads((cells.DELIVERABLES_DIR_1434 / "i1434_ladders.json").read_text())
        for path in ctrl_figures(spec, panel, con_agg, out_dir):
            print(path)
        return 0
    po = args.round == "i1434po"
    deliver = Path(
        args.deliverables or (cells.PO_DELIVERABLES_DIR if po else cells.DELIVERABLES_DIR_1434)
    )
    proj_path = Path(
        args.projections
        or f"data/issue_1434/cells/pv/{'projections_po.json' if po else 'projections.json'}"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["savefig.dpi"] = 180  # match the prior fig.savefig(..., dpi=180) resolution
    agg = json.loads(
        (deliver / ("i1434po_ladders.json" if po else "i1434_ladders.json")).read_text()
    )
    if po:
        contrast = json.loads((deliver / "regime_contrast.json").read_text())
        print(fig_regime_hero(contrast, out_dir))
        print(fig_regime_cells(contrast, out_dir))
        print(
            fig_install_grid(
                agg,
                out_dir,
                cell_keys=cells.PO_CELL_KEYS,
                runs=cells.I1434PO_RUNS,
                fname="po_install_grid.png",
                title_prefix="writing_style POSITIVE-ONLY",
            )
        )
        con_agg = json.loads((cells.DELIVERABLES_DIR_1434 / "i1434_ladders.json").read_text())
        print(fig_ladder_overlays(agg, con_agg, out_dir))
        return 0
    print(fig_install_grid(agg, out_dir))
    val_path = deliver / "pv_validation.json"
    if val_path.exists() and proj_path.exists():
        val = json.loads(val_path.read_text())
        proj = json.loads(proj_path.read_text())
        print(fig_rb_validity(val, proj, agg, out_dir))
    else:
        print(f"[figures] pv validation/projections missing ({val_path}, {proj_path}) — skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
