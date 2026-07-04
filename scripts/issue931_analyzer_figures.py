"""Analyzer-pass figures for issue #931 (repo-root copies + regenerations).

1. Copies the run-generated figures from the issue-931 worktree to the
   repo-root ``figures/issue_931/`` so main-pinned URLs resolve.
2. Regenerates ``delta_char`` with an un-clipped y-label + embedded points.
3. Adds the low-level per-unit views behind the delta_char aggregate:
   per-novel (Arm A, labeled) and per-story (Arm B) paired correct-vs-swap
   held-out R^2 at layer 19.
4. Revision-round-2 regenerations (interpretation-critic r1): plain-English
   condition labels replacing internal slugs on ``per_novel_r2_scatter``,
   ``hero2_transfer_matrix``, ``matched_vs_fulln_fractions``,
   ``strict_vs_recentered`` (also un-overlapped, now a paired dot plot), and
   ``power_curve_overlay`` (adds the n=5000 endpoint so the non-monotone dip
   is actually visible).

Run from repo root: ``uv run python scripts/issue931_analyzer_figures.py``
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parent.parent
WT_EVAL = REPO / ".claude/worktrees/issue-931/eval_results/issue_931"
WT_FIGS = REPO / ".claude/worktrees/issue-931/figures/issue_931"
OUT = REPO / "figures/issue_931"


def copy_run_figures() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(WT_FIGS.glob("*")):
        shutil.copy2(f, OUT / f.name)
        n += 1
    print(f"[i931-figs] copied {n} files from worktree figures")


def fig_delta_char() -> None:
    set_paper_style()
    pal = paper_palette(4)
    pts = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (arm, label) in enumerate([("armA", "Real novels"), ("armB", "Model-written stories")]):
        d = json.loads((WT_EVAL / f"delta_char_{arm}.json").read_text())
        y, lo, hi = d["delta_r2_char"], d["delta_ci_lo"], d["delta_ci_hi"]
        ax.errorbar(
            [i], [y], yerr=[[y - lo], [hi - y]], fmt="o", color=pal[i], capsize=4, markersize=8
        )
        ax.text(i + 0.06, y, f"{y:+.3f}", va="center", fontsize=11)
        pts.append(
            {
                "arm": label,
                "delta_r2_char": y,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_rows": d["n_rows"],
                "n_groups": d["n_groups"],
            }
        )
    ax.axhline(0.0, color="0.3", lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Real novels", "Model-written stories"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Character-identity gain in held-out $R^2$\n(correct $-$ swap, 95% CI)")
    savefig_paper(fig, "delta_char", dir=OUT)
    (OUT / "delta_char.meta.json").write_text(json.dumps({"points": pts}, indent=2))
    plt.close(fig)


def _paired(arm: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    w = json.loads((WT_EVAL / f"cells_{arm}_within.json").read_text())["per_group_r2_headline"]
    s = json.loads((WT_EVAL / f"cells_{arm}_swap.json").read_text())["per_group_r2_headline"]
    keys = sorted(set(w) & set(s))
    return keys, np.array([w[k] for k in keys]), np.array([s[k] for k in keys])


def fig_delta_char_per_novel() -> None:
    set_paper_style()
    pal = paper_palette(4)
    keys, w, s = _paired("armA")
    order = np.argsort(w - s)
    fig, ax = plt.subplots(figsize=(8, 10))
    for row, idx in enumerate(order):
        ax.plot([s[idx], w[idx]], [row, row], color="0.7", lw=1, zorder=1)
    ax.scatter(
        s[order], np.arange(len(keys)), color=pal[1], s=30, label="swapped pairing", zorder=2
    )
    ax.scatter(
        w[order], np.arange(len(keys)), color=pal[0], s=30, label="correct pairing", zorder=3
    )
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([keys[i] for i in order], fontsize=9)
    ax.axvline(0.0, color="0.3", lw=1)
    ax.set_xlabel("Per-novel held-out $R^2$ @ layer 19")
    ax.legend(loc="lower left")
    savefig_paper(fig, "delta_char_per_novel", dir=OUT)
    (OUT / "delta_char_per_novel.meta.json").write_text(
        json.dumps(
            {
                "note": (
                    "correct-pairing per-novel R2 from cells_armA_within (all 1982 rows); "
                    "swap from cells_armA_swap (1694 derangement-eligible rows) — subsets "
                    "differ slightly; the pooled delta_char_armA.json read matches subsets."
                ),
                "points": [
                    {"novel": k, "r2_correct": float(a), "r2_swap": float(b)}
                    for k, a, b in zip(keys, w, s, strict=True)
                ],
            },
            indent=2,
        )
    )
    plt.close(fig)


def fig_delta_char_per_story() -> None:
    set_paper_style()
    pal = paper_palette(4)
    keys, w, s = _paired("armB")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    lim_lo = float(min(s.min(), w.min())) - 0.1
    lim_hi = float(max(s.max(), w.max())) + 0.1
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], ls="--", color="0.6", lw=1)
    ax.scatter(s, w, s=18, color=pal[0], alpha=0.6)
    ax.set_xlabel("Per-story held-out $R^2$ @ layer 19, swapped pairing")
    ax.set_ylabel("Per-story held-out $R^2$ @ layer 19, correct pairing")
    frac = float((w > s).mean())
    ax.set_title(f"Model-written stories: correct beats swap in {frac:.0%} of {len(keys)} stories")
    savefig_paper(fig, "delta_char_per_story", dir=OUT)
    (OUT / "delta_char_per_story.meta.json").write_text(
        json.dumps(
            {
                "n_stories": len(keys),
                "frac_correct_gt_swap": frac,
                "points": [
                    {"story": k, "r2_correct": float(a), "r2_swap": float(b)}
                    for k, a, b in zip(keys, w, s, strict=True)
                ],
            },
            indent=2,
        )
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Revision-round-2 regenerations (plain-English labels; critic r1 requests)
# ---------------------------------------------------------------------------

_CELL_PLAIN = {
    "chat_ref": "chat",
    "chat": "chat",
    "armA": "real novels",
    "armA_within": "real novels",
    "armA_within_lastpos": "real novels",
    "armB": "model stories",
    "armB_within": "model stories",
    "armB_within_lastpos": "model stories",
    "armC": "separator control",
    "armC_sep": "separator control",
    "armC_prevmean": "preceding-sentence control",
}
_RECIPE_PLAIN = {"lastpos": "single-position X", "spanmean": "span-mean X"}


def _plain_dir(direction: str, recipe: str) -> str:
    a, b = direction.split("->")
    return f"{_CELL_PLAIN[a]} → {_CELL_PLAIN[b]} ({_RECIPE_PLAIN[recipe]})"


def _merge_meta(name: str, extra: dict) -> None:
    """Add fields to savefig_paper's sidecar without dropping its provenance."""
    p = OUT / f"{name}.meta.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d.update(extra)
    p.write_text(json.dumps(d, indent=2))


def _eval(name: str) -> dict:
    return json.loads((WT_EVAL / name).read_text())


def fig_power_curve_overlay() -> None:
    """Chat power curve INCLUDING the n=5000 full-store endpoint (the dip)."""
    set_paper_style()
    pal = paper_palette(4)
    pc = _eval("power_curve_chat.json")
    pts = [(c["n"], c["r2_per_layer"][19]) for c in pc["curve"] if c.get("r2_per_layer")]
    full = _eval("cells_chat_ref.json")
    pts.append((full["n"], full["r2_per_layer_obs"][19]))
    pts.sort()
    ns, r2s = zip(*pts, strict=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ns, r2s, "o-", color=pal[0], label="chat reference (layer 19)")
    for i, (n, v) in enumerate(pts):
        dy = 0.022 if i % 2 == 0 else -0.045
        ax.text(n, v + dy, f"{v:.2f}", ha="center", fontsize=9)
    a = _eval("cells_armA_within.json")
    b = _eval("cells_armB_within.json")
    ax.scatter(
        [a["n"]],
        [a["r2_per_layer_obs"][19]],
        color=pal[1],
        zorder=3,
        label="real-novel character map (full n)",
    )
    ax.scatter(
        [b["n"]],
        [b["r2_per_layer_obs"][19]],
        color=pal[2],
        zorder=3,
        label="model-story character map (full n)",
    )
    ax.set_xlabel("Training rows n")
    ax.set_ylabel("Held-out $R^2$ @ layer 19")
    ax.set_ylim(0, 0.78)
    ax.legend(fontsize=9, loc="upper left")
    savefig_paper(fig, "power_curve_overlay", dir=OUT)
    _merge_meta("power_curve_overlay", {"chat_curve": [{"n": n, "r2": v} for n, v in pts]})
    plt.close(fig)


def fig_per_novel_r2_scatter() -> None:
    """Per-novel held-out R^2 (Arm A within), plain-English axis label."""
    set_paper_style()
    pg = _eval("cells_armA_within.json")["per_group_r2_headline"]
    names = sorted(pg, key=lambda k: pg[k])
    vals = [pg[k] for k in names]
    fig, ax = plt.subplots(figsize=(7.0, 0.28 * len(names) + 1.4))
    ax.scatter(vals, range(len(names)), s=18)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n[:32] for n in names], fontsize=7)
    ax.axvline(0, color="gray", lw=0.7)
    ax.set_xlabel("Per-novel held-out $R^2$ @ layer 19, real-novel character map")
    savefig_paper(fig, "per_novel_r2_scatter", dir=OUT)
    _merge_meta(
        "per_novel_r2_scatter",
        {"points": [{"novel": k, "r2": float(pg[k])} for k in names]},
    )
    plt.close(fig)


def _matched_recentered_rows() -> dict[str, dict]:
    tm = _eval("transfer_matrix.json")
    hl = tm["headline_layer"]
    seen: dict[str, dict] = {}
    for r in tm["rows"]:
        if r["layer"] == hl and r["application"] == "recentered" and r["power_matched"]:
            seen.setdefault(_plain_dir(r["direction"], r["x_recipe"]), r)
    return seen


def fig_hero2_transfer_matrix() -> None:
    """Matched recentered transfer fractions, plain-English rows, diverging scale."""
    set_paper_style()
    seen = _matched_recentered_rows()
    labels = list(seen)
    fracs = np.asarray([seen[k]["fraction_of_ceiling"] for k in labels], dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 0.5 * len(labels) + 1.6), layout="constrained")
    clipped = np.clip(np.nan_to_num(fracs, nan=0.0), -1.0, 1.0)
    im = ax.imshow(clipped[:, None], cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(
        [
            f"{lbl}  [source n={seen[lbl]['n_train']}, "
            f"ceiling n={seen[lbl]['denominator_n_train']}]"
            for lbl in labels
        ],
        fontsize=8,
    )
    ax.set_xticks([0])
    ax.set_xticklabels(["recentered fraction of ceiling @ layer 19"], fontsize=8.5)
    for i, k in enumerate(labels):
        v = seen[k]["fraction_of_ceiling"]
        cell_dark = bool(np.isfinite(v)) and abs(min(max(float(v), -1.0), 1.0)) > 0.6
        ax.text(
            0,
            i,
            "n/a" if not np.isfinite(v) else f"{v:+.2f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="white" if cell_dark else "black",
        )
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("fraction of ceiling (clipped at ±1)", fontsize=8)
    savefig_paper(fig, "hero2_transfer_matrix", dir=OUT)
    _merge_meta(
        "hero2_transfer_matrix",
        {
            "points": [
                {
                    "direction": k,
                    "fraction_of_ceiling": (
                        None
                        if not np.isfinite(seen[k]["fraction_of_ceiling"])
                        else float(seen[k]["fraction_of_ceiling"])
                    ),
                    "transfer_r2": float(seen[k]["transfer_r2"]),
                    "source_n": seen[k]["n_train"],
                    "ceiling_n": seen[k]["denominator_n_train"],
                }
                for k in labels
            ]
        },
    )
    plt.close(fig)


def fig_matched_vs_fulln() -> None:
    """Matched vs full-n recentered fractions per direction (symlog x)."""
    set_paper_style()
    pal = paper_palette(4)
    tm = _eval("transfer_matrix.json")
    hl = tm["headline_layer"]
    by_dir: dict[str, dict] = {}
    for r in tm["rows"]:
        if r["layer"] != hl or r["application"] != "recentered":
            continue
        d = by_dir.setdefault(_plain_dir(r["direction"], r["x_recipe"]), {})
        d["matched" if r["power_matched"] else "full"] = r["fraction_of_ceiling"]
    keys = [k for k, v in by_dir.items() if "matched" in v]
    fig, ax = plt.subplots(figsize=(8.4, 0.4 * len(keys) + 1.4))
    y = np.arange(len(keys))
    ax.scatter(
        [by_dir[k]["matched"] for k in keys],
        y,
        color=pal[0],
        s=26,
        label="power-matched (primary)",
        zorder=3,
    )
    fx = [(by_dir[k].get("full"), i) for i, k in enumerate(keys) if "full" in by_dir[k]]
    ax.scatter(
        [v for v, _ in fx],
        [i for _, i in fx],
        color=pal[1],
        s=26,
        label="full n (secondary)",
        zorder=2,
    )
    ax.set_xscale("symlog", linthresh=1.0)
    ax.axvline(0, color="gray", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(keys, fontsize=8)
    ax.set_xlabel("Recentered fraction of ceiling @ layer 19 (symlog scale)")
    ax.legend(fontsize=9, loc="lower left")
    savefig_paper(fig, "matched_vs_fulln_fractions", dir=OUT)
    _merge_meta(
        "matched_vs_fulln_fractions",
        {
            "points": [
                {
                    "direction": k,
                    "matched": (
                        None
                        if not np.isfinite(by_dir[k].get("matched", np.nan))
                        else float(by_dir[k]["matched"])
                    ),
                    "full_n": (
                        None
                        if not np.isfinite(by_dir[k].get("full", np.nan))
                        else float(by_dir[k]["full"])
                    ),
                }
                for k in keys
            ]
        },
    )
    plt.close(fig)


def fig_strict_vs_recentered() -> None:
    """Strict-frozen vs recentered transfer R^2 as a paired dot plot (no overlap)."""
    set_paper_style()
    pal = paper_palette(4)
    tm = _eval("transfer_matrix.json")
    hl = tm["headline_layer"]
    by_dir: dict[str, dict] = {}
    for r in tm["rows"]:
        if r["layer"] != hl or not r["power_matched"]:
            continue
        by_dir.setdefault(_plain_dir(r["direction"], r["x_recipe"]), {})[r["application"]] = r[
            "transfer_r2"
        ]
    keys = sorted(by_dir, key=lambda k: by_dir[k].get("recentered", np.nan))
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8.4, 0.4 * len(keys) + 1.4))
    rec = [by_dir[k].get("recentered", np.nan) for k in keys]
    st = [by_dir[k].get("strict", np.nan) for k in keys]
    for i in range(len(keys)):
        ax.plot([st[i], rec[i]], [i, i], color="0.75", lw=1, zorder=1)
    ax.scatter(rec, y, color=pal[0], s=26, label="recentered (primary)", zorder=3)
    ax.scatter(st, y, color=pal[1], s=26, label="strict-frozen (secondary)", zorder=2)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.axvline(0, color="gray", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(keys, fontsize=8)
    ax.set_xlabel("Power-matched transfer $R^2$ @ layer 19 (symlog scale)")
    ax.legend(fontsize=9, loc="upper left")
    savefig_paper(fig, "strict_vs_recentered", dir=OUT)
    _merge_meta(
        "strict_vs_recentered",
        {
            "points": [
                {"direction": k, "recentered": float(r), "strict": float(s)}
                for k, r, s in zip(keys, rec, st, strict=True)
            ]
        },
    )
    plt.close(fig)


if __name__ == "__main__":
    copy_run_figures()
    fig_delta_char()
    fig_delta_char_per_novel()
    fig_delta_char_per_story()
    # Revision-round-2 regenerations OVERWRITE the copied run versions.
    fig_power_curve_overlay()
    fig_per_novel_r2_scatter()
    fig_hero2_transfer_matrix()
    fig_matched_vs_fulln()
    fig_strict_vs_recentered()
    print("[i931-figs] done ->", OUT)
