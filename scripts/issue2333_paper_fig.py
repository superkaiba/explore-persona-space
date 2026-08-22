"""Condensed opening-token (snowball) recovery figure for the context-answer-map
paper (sections/results/c2_context_vector.tex): how much of the full context-end
patch effect a k-token opening transplant recovers, Qwen2.5-7B format cells vs
Qwen3.5-9B language cells.

This paper variant lives in its OWN file because the live #2333 session owns
scripts/issue2333_figures.py on branch issue-2333 (+214/-27 unmerged as of
2026-08-20) — editing the main-tree copy would collide at merge
(.claude/rules/cross-session-writer-arbitration.md). TODO: fold into
issue2333_figures.py as a --style iclr pathway after #2333 merges.

Reads (committed):
- eval_results/issue_2333/f_metrics/q25/stats.json (parent round, main tree)
- eval_results/issue_2333/paper_inputs/q35_language_snowball_stats.json —
  staged VERBATIM from branch issue-2333 @ c4578b2394
  (eval_results/issue_2333/q35_language_snowball/f_metrics/stats.json); staged
  under paper_inputs/ so the branch's own canonical path stays merge-clean.

Writes figures/paper/c2_prefill_recovery.{png,pdf,meta.json}.

Usage:
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
    uv run python scripts/issue2333_paper_fig.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE any heavy import

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    figsize_iclr_full,
    paper_color,
    savefig_paper,
    set_paper_style,
)

Q25_STATS = Path("eval_results/issue_2333/f_metrics/q25/stats.json")
Q35_LANG_STATS = Path("eval_results/issue_2333/paper_inputs/q35_language_snowball_stats.json")

KS = (1, 2, 3)


def _q25_format_points(arms: dict) -> dict[str, list[tuple[int, float, float, float]]]:
    """(k, ratio, lo, hi) per arm kind — RAW same-wave recovery ratio, format
    cells, patch-content donors (the confirmatory scheme)."""
    out: dict[str, list[tuple[int, float, float, float]]] = {"prefill": [], "patch": []}
    for kind in ("prefill", "patch"):
        for k in KS:
            rec = arms[f"{kind}{k}_med"]["recovery_samewave"]
            lo, hi = rec["ratio_ci"]
            out[kind].append((k, float(rec["ratio"]), float(lo), float(hi)))
    return out


def _q35_language_points(
    arms: dict, scheme: str
) -> dict[str, list[tuple[int, float, float, float]]]:
    """(k, net ratio, lo, hi) per arm kind — NULL-ADJUSTED net recovery ratio,
    language cells (a wrong-language opening alone moves the judged score, so
    the shuffled-donor null is subtracted from both numerator and denominator)."""
    out: dict[str, list[tuple[int, float, float, float]]] = {"prefill": [], "patch": []}
    for kind in ("prefill", "patch"):
        for k in KS:
            net = arms[f"{kind}{k}_{scheme}"]["recovery_net_samewave"]
            lo, hi = net["ratio_net_ci"]
            out[kind].append((k, float(net["ratio_net"]), float(lo), float(hi)))
    return out


def main() -> int:
    q25 = json.loads(Q25_STATS.read_text(encoding="utf-8"))["per_set"]["s1"]["arms"]
    q35 = json.loads(Q35_LANG_STATS.read_text(encoding="utf-8"))["per_set"]["s1"]["arms"]

    # Pin the paper fragment's quoted numbers against the committed artifacts.
    p3 = q25["prefill3_med"]
    assert p3["n_pairs"] == 172, p3["n_pairs"]
    assert abs(p3["recovery_samewave"]["ratio"] - 0.67) < 0.005, p3["recovery_samewave"]
    assert p3["p_holm"] < 1e-12, p3["p_holm"]
    for k in KS:  # state patches recover 50-54% on format cells
        r = q25[f"patch{k}_med"]["recovery_samewave"]["ratio"]
        assert 0.49 < r < 0.55, (k, r)
    net3 = q35["prefill3_med"]["recovery_net_samewave"]
    assert abs(net3["ratio_net"] - 0.40) < 0.005, net3
    assert net3["ratio_net_ci"][1] < 1.0, net3  # reliable non-token residual
    assert abs(q35["patch3_bstart"]["recovery_net_samewave"]["ratio_net"] - 1.12) < 0.005
    assert abs(q35["prefill3_bstart"]["recovery_net_samewave"]["ratio_net"] - 0.945) < 0.005

    set_paper_style("iclr")
    blue = paper_color("instruct")
    fig, axes = plt.subplots(
        1, 2, figsize=figsize_iclr_full(0.46), layout="constrained", sharey=True
    )
    panels = (
        ("Qwen2.5-7B, format cells", _q25_format_points(q25), None),
        (
            "Qwen3.5-9B, language cells (null-adjusted)",
            _q35_language_points(q35, "med"),
            _q35_language_points(q35, "bstart"),
        ),
    )
    markers = {"prefill": ("o", "token prefill"), "patch": ("s", "state patch")}
    for ax, (title, med, bstart) in zip(axes, panels, strict=True):
        ax.axhline(1.0, color="#AAAAAA", linewidth=0.6, linestyle="--")
        ax.axhline(0.0, color="#AAAAAA", linewidth=0.6)
        for kind, dx in (("prefill", -0.10), ("patch", +0.10)):
            m, label = markers[kind]
            pts = med[kind]
            ax.errorbar(
                [k + dx for k, *_ in pts],
                [v for _, v, *_ in pts],
                yerr=[
                    [max(0.0, v - lo) for _, v, lo, _ in pts],
                    [max(0.0, hi - v) for _, v, _, hi in pts],
                ],
                fmt=m,
                color=blue,
                capsize=2,
                markersize=4,
                label=f"{label} (patch-content donor)" if title.startswith("Qwen2.5") else None,
            )
            if bstart is not None:
                k, v, lo, hi = bstart[kind][-1]  # 3-position natural-opening donors
                ax.errorbar(
                    [k + dx + 0.22],
                    [v],
                    yerr=[[max(0.0, v - lo)], [max(0.0, hi - v)]],
                    fmt=m,
                    mfc="white",
                    mec=blue,
                    mew=1.0,
                    color=blue,
                    ecolor=blue,
                    capsize=2,
                    markersize=4,
                    label=f"{label} (natural-opening donor)",
                )
        ax.set_title(title)
        ax.set_xticks(list(KS))
        ax.set_xlabel("opening positions transplanted")
    axes[0].set_ylabel("share of the full\ncontext-end patch effect")
    handles0, labels0 = axes[0].get_legend_handles_labels()
    handles1, labels1 = axes[1].get_legend_handles_labels()
    axes[0].legend(handles0, labels0, loc="lower right", handletextpad=0.3)
    axes[1].legend(handles1, labels1, loc="lower right", handletextpad=0.3)
    savefig_paper(fig, "c2_prefill_recovery", dir="figures/paper/")
    plt.close(fig)
    print("[paper-fig] wrote figures/paper/c2_prefill_recovery.{png,pdf,meta.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
