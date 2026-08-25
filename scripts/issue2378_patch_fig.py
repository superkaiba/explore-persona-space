#!/usr/bin/env python3
"""Issue #2378 `causal-patching-arms` fold figures (+ derived round artifacts).

Default run renders the two fold figures from COMMITTED eval JSONs
(`eval_results/issue_2378/causal-patching-arms/`):

- ``patch_family_forest`` — family-level steered-minus-null mean with 95%
  pair-clustered bootstrap CIs: F_act greedy screen (24 families, PASS
  highlighted), F_act independent temp-1.0 confirm (4 families), F_beh
  judge-scored grid (20 families) + confirm (2 families).
- ``patch_pass_pairs`` — the per-question paired differences behind those
  aggregates: F_act per pair for the 4 screen-PASS families (grid vs
  confirm), and F_beh per pair for the two confirmed story families plus the
  nominal grid family.

``--build-derived`` additionally rebuilds, from the staged pod rollouts under
``data/issue_2378/patch_round/`` (HF-backed, not in git):

- ``fact_cells_confirm.json`` — per-cell confirm F_act (mean over K=5 draws;
  the confirm sibling of the pod-harvested ``fact_cells.json``), via the same
  ``issue2378_patch_run._grid_fact`` code path the analysis used.
- ``cjk_audit.json`` — the Step-3.7 language-intrusion audit: per-stage/arm
  CJK counts over the measured answer text, excluded-intrusion recounts of
  every screen/confirm verdict (full-aggregate replay asserted equal to
  ``patch_summary.json`` first), and the think-block leak rates.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Shared-VM thread caps BEFORE any heavy import (#847).
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
ARM_DIR = ROOT / "eval_results" / "issue_2378" / "causal-patching-arms"
PATCH_ROOT = ROOT / "data" / "issue_2378" / "patch_round"
OUT = "issue_2378"

CJK = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")
THINK = re.compile(r"<think>(.*?)</think>", re.S)

CHAR_SHORT = {"Astra": "Astra", "HELIOS": "HELIOS", "Wren": "Wren", "Dana": "Dana", "Vex": "Vex"}
VARIANT_LABEL = {"all": "all layers", "lstar": "layer 51"}


def fam_label(fam: str) -> str:
    """Plain-English family label: '<source> -> <target>, <patched layers>'."""
    pair, char, direction, variant, _arm = fam.split("|")
    if pair == "chat~plain":
        a, b = "chat", "plain"
    else:
        a, b = "chat", f"{CHAR_SHORT[char]} (story)"
    src, tgt = (a, b) if direction == "a2b" else (b, a)
    return f"{src} → {tgt}, {VARIANT_LABEL[variant]}"


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _fam_sort_key(fam: str):
    pair, char, direction, variant, _ = fam.split("|")
    return (pair != "chat~plain", char, direction != "a2b", variant != "all")


# ── derived-artifact build (staged rollouts required) ────────────────────────


def _iter_rollouts(stage: str):
    for fp in sorted((PATCH_ROOT / stage / "rollouts").glob("*.jsonl")):
        with open(fp) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _has_nonempty_think(txt: str) -> bool:
    m = THINK.search(txt)
    if m is not None:
        return bool(m.group(1).strip())
    return "<think>" in txt


def build_derived() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import issue2378_patch_analysis as A
    import issue2378_patch_run as R

    # 1. Per-cell confirm F_act (mean over draws) — the confirm sibling of
    #    the pod-harvested fact_cells.json, same _grid_fact code path.
    r_args = R.build_argparser().parse_args(
        ["--phase", "screen", "--out-root", str(PATCH_ROOT), "--lstar", "51"]
    )
    rows, dropped = R._grid_fact(r_args, "confirm")
    cells: dict[str, dict] = {}
    for r in rows:
        if r["degenerate"]:
            continue
        rec = cells.setdefault(r["cell_id"], {**r, "vals": []})
        rec["vals"].append(r["f_act"])
    cell_rows = []
    for rec in cells.values():
        rec = dict(rec)
        vals = rec.pop("vals")
        rec["f_act"] = float(np.mean(vals))
        rec["n_draws"] = len(vals)
        rec.pop("draw", None)
        cell_rows.append(rec)
    out = {"rows": sorted(cell_rows, key=lambda r: r["cell_id"]), "dropped": dict(dropped)}
    (ARM_DIR / "fact_cells_confirm.json").write_text(json.dumps(out, indent=1))
    print(f"[derived] fact_cells_confirm.json: {len(cell_rows)} cells, dropped={dict(dropped)}")

    # 2. Language-intrusion audit + excluded-intrusion recounts.
    intr: dict[str, set] = {}
    counts: dict[str, dict] = {}
    think: dict[str, str] = {}
    for stage in ("anchors", "grid", "confirm"):
        s = set()
        per: Counter = Counter()
        tot: Counter = Counter()
        th: Counter = Counter()
        th_tot: Counter = Counter()
        for r in _iter_rollouts(stage):
            txt = r.get("answer") or r.get("gen_text") or ""
            if stage == "anchors":
                key = r.get("framing", "?")
            else:
                key = f"{r['pair_type']}|{r['arm']}"
            tot[key] += 1
            if CJK.search(txt):
                per[key] += 1
                s.add((r.get("cell_id", r.get("ctx_id")), r.get("draw")))
            if not r.get("drop_reason"):
                fr = r.get("framing") or r.get("tgt", "?").split(":")[0]
                th_tot[f"{stage}|{fr}"] += 1
                if _has_nonempty_think(txt):
                    th[f"{stage}|{fr}"] += 1
        intr[stage] = s
        counts[stage] = {k: f"{per[k]}/{tot[k]}" for k in sorted(tot)}
        think.update({k: f"{th[k]}/{th_tot[k]}" for k in sorted(th_tot) if th[k]})

    summ = _load(ARM_DIR / "patch_summary.json")

    def fact_family(rows_, exclude):
        cells_: dict[str, dict] = {}
        for r in rows_:
            if r["degenerate"] or (r["cell_id"], r["draw"]) in exclude:
                continue
            rec = cells_.setdefault(r["cell_id"], {**r, "vals": []})
            rec["vals"].append(r["f_act"])
        crows = [{**rec, "f_act": float(np.mean(rec["vals"]))} for rec in cells_.values()]
        return A._family_stats(A._family_table(crows, "f_act"))["steered_vs_null"]

    replay = fact_family(rows, frozenset())
    for fam, ref in summ["f_act_confirm"]["steered_vs_null"].items():
        got = replay[fam]
        assert abs(got["mean_diff"] - ref["mean_diff"]) < 1e-9, (fam, got, ref)
        assert abs(got["ci_lo"] - ref["ci_lo"]) < 1e-9 and abs(got["ci_hi"] - ref["ci_hi"]) < 1e-9
    fact_excl = fact_family(rows, intr["confirm"])

    scores = A._load_scores(ARM_DIR / "judge")
    anchor_deltas = A._anchor_deltas(PATCH_ROOT, scores)
    fbeh_excl = {}
    for stage in ("grid", "confirm"):
        all_rows = A._cell_rows(PATCH_ROOT, stage)
        drops: Counter = Counter()
        base = A._family_stats(
            A._family_table(A._fbeh_cells(stage, all_rows, scores, anchor_deltas, drops), "f_beh")
        )["steered_vs_null"]
        for fam, ref in summ[f"f_beh_{stage}"]["steered_vs_null"].items():
            assert abs(base[fam]["mean_diff"] - ref["mean_diff"]) < 1e-9, (stage, fam)
        filt = [r for r in all_rows if (r["cell_id"], r.get("draw")) not in intr[stage]]
        drops2: Counter = Counter()
        fbeh_excl[stage] = A._family_stats(
            A._family_table(A._fbeh_cells(stage, filt, scores, anchor_deltas, drops2), "f_beh")
        )["steered_vs_null"]

    audit = {
        "regex": CJK.pattern,
        "scanned_field": "answer (fallback gen_text)",
        "intrusion_counts": counts,
        "replay_check": "full-aggregate replay matched patch_summary.json exactly before exclusion",
        "excluded_recount_f_act_confirm": fact_excl,
        "excluded_recount_f_beh": fbeh_excl,
        "think_block_rates_nonzero": think,
    }
    (ARM_DIR / "cjk_audit.json").write_text(json.dumps(audit, indent=1))
    print("[derived] cjk_audit.json written")


# ── figures ──────────────────────────────────────────────────────────────────


def _forest(ax, fams: list[str], stats: dict, color: str, xlim: tuple[float, float]) -> None:
    """Horizontal forest: mean + 95% CI per family; PASS = full color, else grey."""
    ys = np.arange(len(fams))[::-1]
    for y, fam in zip(ys, fams):
        rec = stats[fam]
        c = color if rec["screen_pass"] else "#B8B8B8"
        lo, hi = rec["ci_lo"], rec["ci_hi"]
        clo, chi = max(lo, xlim[0]), min(hi, xlim[1])
        ax.plot([clo, chi], [y, y], color=c, lw=1.6, zorder=2)
        if lo < xlim[0]:
            ax.plot([xlim[0]], [y], marker="<", color=c, ms=5, zorder=3, clip_on=False)
        if hi > xlim[1]:
            ax.plot([xlim[1]], [y], marker=">", color=c, ms=5, zorder=3, clip_on=False)
        ax.plot([rec["mean_diff"]], [y], marker="o", color=c, ms=5.5, zorder=3)
    ax.axvline(0.0, color="#444444", lw=0.9, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([fam_label(f) for f in fams], fontsize=7.6)
    ax.set_ylim(-0.7, len(fams) - 0.3)
    ax.set_xlim(*xlim)


def fig_family_forest(summ: dict) -> None:
    screen = summ["screen_report"]["families"]
    confirm = summ["f_act_confirm"]["steered_vs_null"]
    beh = dict(summ["f_beh_grid"]["steered_vs_null"])
    beh_confirm = summ["f_beh_confirm"]["steered_vs_null"]

    grid_c = paper_palette_role("primary")
    conf_c = paper_palette_role("accent")

    fig, axes = plt.subplots(
        1, 3, figsize=(12.6, 6.0), width_ratios=[1.05, 1.0, 1.05], layout="none"
    )

    fams1 = sorted(screen, key=_fam_sort_key)
    _forest(axes[0], fams1, screen, grid_c, (-0.42, 0.42))
    axes[0].set_title("Activation score, greedy screen\n(24 families)", loc="left", fontsize=9.5)
    axes[0].set_xlabel("steered − null (context-swap fraction)", fontsize=9)

    fams2 = sorted(confirm, key=_fam_sort_key)
    _forest(axes[1], fams2, confirm, conf_c, (-0.42, 0.42))
    axes[1].set_title(
        "Activation score, independent confirm\n(temp 1.0, 4 screened families)",
        loc="left",
        fontsize=9.5,
    )
    axes[1].set_xlabel("steered − null (context-swap fraction)", fontsize=9)

    fams3 = sorted(beh, key=_fam_sort_key)
    _forest(axes[2], fams3, beh, grid_c, (-1.55, 1.55))
    ys = np.arange(len(fams3))[::-1]
    # confirm rows appended below the grid rows
    extra = sorted(beh_confirm, key=_fam_sort_key)
    for i, fam in enumerate(extra):
        rec = beh_confirm[fam]
        y = -1.2 - i
        c = conf_c if rec["screen_pass"] else "#B8B8B8"
        lo, hi = max(rec["ci_lo"], -1.55), min(rec["ci_hi"], 1.55)
        axes[2].plot([lo, hi], [y, y], color=c, lw=1.6)
        axes[2].plot([rec["mean_diff"]], [y], marker="o", color=c, ms=5.5)
    axes[2].set_yticks(list(ys) + [-1.2 - i for i in range(len(extra))])
    axes[2].set_yticklabels(
        [fam_label(f) for f in fams3] + [fam_label(f) + " [confirm]" for f in extra], fontsize=7.6
    )
    axes[2].set_ylim(-1.2 - len(extra) + 0.3, len(fams3) - 0.3)
    axes[2].set_title(
        "Behavior score, judge-scored\n(20 grid + 2 confirm families)", loc="left", fontsize=9.5
    )
    axes[2].set_xlabel("steered − null (anchor-contrast fraction)", fontsize=9)

    fig.subplots_adjust(left=0.155, right=0.985, top=0.80, bottom=0.095, wspace=0.85)
    fig.text(
        0.01,
        0.965,
        "Cross-framing context-vector patching: one activation-level arm replicates;"
        " no judged-behavior carry",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
    )
    fig.text(
        0.01,
        0.905,
        "steered − matched-null family means, 95% pair-clustered bootstrap CIs"
        " (B=10,000); full color = CI excludes zero; whiskers past axis limits clipped",
        ha="left",
        va="top",
        fontsize=9,
        color="#5A5A5A",
    )
    savefig_paper(fig, f"{OUT}/patch_family_forest", dir="figures/")
    plt.close(fig)
    # Forest rows are many unlabeled 2-point Line2D artists the sidecar's
    # artist read-back skips — point the dashboard viewer at the committed
    # source table instead (paper-plots skill § sidecar data / data_path).
    meta_path = ROOT / "figures" / OUT / "patch_family_forest.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["data_path"] = "eval_results/issue_2378/causal-patching-arms/patch_summary.json"
    meta_path.write_text(json.dumps(meta, indent=1))


def _pair_diffs(rows: list[dict], fam_base: str) -> list[float]:
    """Per-question steered-minus-null diffs for one family base (no |arm suffix)."""
    steered, null = {}, {}
    for r in rows:
        v = r.get("f_act", r.get("f_beh"))
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        base = r["family"].rsplit("|", 1)[0]
        if base != fam_base:
            continue
        (steered if r["arm"] == "steered" else null if r["arm"] == "null" else {})[r["qid"]] = v
    return [steered[q] - null[q] for q in sorted(steered) if q in null]


def fig_pass_pairs(summ: dict) -> None:
    grid_rows = _load(ARM_DIR / "fact_cells.json")["rows"]
    confirm_rows = _load(ARM_DIR / "fact_cells_confirm.json")["rows"]
    grid_c = paper_palette_role("primary")
    conf_c = paper_palette_role("accent")

    pass_fams = sorted(summ["screen_report"]["confirm_families"], key=_fam_sort_key)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), width_ratios=[1.45, 1.0], layout="none")
    rng = np.random.default_rng(42)

    ax = axes[0]
    for i, fam in enumerate(pass_fams):
        base = fam.rsplit("|", 1)[0]
        for (
            dx,
            rows,
            c,
        ) in ((-0.16, grid_rows, grid_c), (0.16, confirm_rows, conf_c)):
            d = _pair_diffs(rows, base)
            x = i + dx + rng.uniform(-0.09, 0.09, len(d))
            ax.scatter(x, d, s=16, color=c, alpha=0.65, linewidths=0)
            ax.plot(
                [i + dx - 0.13, i + dx + 0.13],
                [float(np.mean(d))] * 2,
                color=c,
                lw=2.4,
                zorder=3,
            )
    ax.axhline(0.0, color="#444444", lw=0.9)
    ax.set_xticks(range(len(pass_fams)))
    ax.set_xticklabels([fam_label(f).replace(", ", ",\n") for f in pass_fams], fontsize=8.2)
    ax.set_ylabel("activation score, steered − null (per question)")
    ax.set_title(
        "Per-question activation differences, 4 screen-PASS families", loc="left", fontsize=9.5
    )

    ax = axes[1]
    beh_groups = [
        ("chat~story|Astra|b2a|lstar", summ["f_beh_cells_confirm"], conf_c, "confirm"),
        ("chat~story|Vex|b2a|all", summ["f_beh_cells_confirm"], conf_c, "confirm"),
        ("chat~story|Dana|b2a|lstar", summ["f_beh_cells_grid"], grid_c, "grid"),
    ]
    labels = []
    for i, (base, rows, c, stage) in enumerate(beh_groups):
        d = _pair_diffs(rows, base)
        x = i + rng.uniform(-0.11, 0.11, len(d))
        ax.scatter(x, d, s=16, color=c, alpha=0.7, linewidths=0)
        ax.plot([i - 0.16, i + 0.16], [float(np.mean(d))] * 2, color=c, lw=2.4, zorder=3)
        labels.append(fam_label(base + "|x").replace(", ", ",\n") + f"\n[{stage}]")
    ax.axhline(0.0, color="#444444", lw=0.9)
    ax.set_xticks(range(len(beh_groups)))
    ax.set_xticklabels(labels, fontsize=8.2)
    ax.set_ylabel("behavior score, steered − null (per question)")
    ax.set_title("Per-question behavior differences", loc="left", fontsize=9.5)

    handles = [
        plt.Line2D([0], [0], marker="o", lw=0, color=grid_c, label="greedy grid"),
        plt.Line2D([0], [0], marker="o", lw=0, color=conf_c, label="temp-1.0 confirm"),
    ]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8.5)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.17, wspace=0.28)
    fig.text(
        0.01,
        0.97,
        "Per-question paired differences behind the family aggregates",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
    )
    fig.text(
        0.01,
        0.90,
        "each point = one question's steered − matched-null difference;"
        " horizontal bar = family mean",
        ha="left",
        va="top",
        fontsize=9,
        color="#5A5A5A",
    )
    savefig_paper(fig, f"{OUT}/patch_pass_pairs", dir="figures/")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--build-derived", action="store_true")
    args = ap.parse_args()
    if args.build_derived:
        build_derived()
    set_paper_style("blog")
    summ = _load(ARM_DIR / "patch_summary.json")
    fig_family_forest(summ)
    fig_pass_pairs(summ)
    print("[figs] patch_family_forest + patch_pass_pairs written")


if __name__ == "__main__":
    main()
