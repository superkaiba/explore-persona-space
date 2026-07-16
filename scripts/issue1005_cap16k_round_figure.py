#!/usr/bin/env python3
"""Issue #1005 `cap16k-compliance-reread` round figure (analyzer fold, 0 GPU).

Left: per-family usable-row rate at the 8,192 production cap vs after the
forced 16,384 re-generation of the 97 cap-hit rows (dashed = the 0.95
compliance bar). Right: the per-context regen outcome for the 42 affected
contexts — recovered vs still-truncated counts, family-colored (the per-unit
data behind the left panel's aggregate).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style  # noqa: E402

EVAL = PROJECT_ROOT / "eval_results" / "issue_1005"
ROUND = EVAL / "cap16k-compliance-reread"
OUT = PROJECT_ROOT / "figures" / "issue_1005" / "cap16k-compliance-reread"

FAM_ORDER = ["icl", "wildchat", "persona", "rephrase", "format", "behavior", "default"]
FAM_LABEL = {
    "icl": "in-context\nlearning",
    "wildchat": "WildChat",
    "persona": "persona",
    "rephrase": "rephrase",
    "format": "format",
    "behavior": "behavior",
    "default": "default",
}
RED_FAMS = {"icl", "wildchat"}


def main() -> int:
    cov_pre = json.loads((EVAL / "coverage_by_family.json").read_text())["families"]
    cov_post_blob = json.loads((ROUND / "coverage_by_family.json").read_text())
    cov_post = cov_post_blob["families"]
    acct = json.loads((ROUND / "regen16k_accounting.json").read_text())

    set_paper_style("blog")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)

    # ── left: per-family usable rate, 8,192 vs 16,384 ────────────────────────
    x = np.arange(len(FAM_ORDER))
    pre = [cov_pre[f]["usable_rate"] for f in FAM_ORDER]
    post = [cov_post[f]["usable_rate"] for f in FAM_ORDER]
    w = 0.38
    ax1.bar(x - w / 2, pre, w, label="8,192-token cap", color="#b8c4d9")
    ax1.bar(x + w / 2, post, w, label="after 16,384 re-generation", color="#2f6db3")
    for xi, (a, b) in zip(x, zip(pre, post, strict=True), strict=True):
        ax1.text(xi - w / 2, a + 0.004, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax1.text(xi + w / 2, b + 0.004, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
    ax1.axhline(0.95, ls="--", lw=1.2, color="#444444")
    ax1.text(
        len(FAM_ORDER) - 0.4, 0.951, "0.95 compliance bar", ha="right", va="bottom", fontsize=9
    )
    ax1.set_xticks(x, [FAM_LABEL[f] for f in FAM_ORDER])
    ax1.set_ylim(0.90, 1.015)
    ax1.set_ylabel("usable-row rate")
    ax1.set_xlabel("context family")
    ax1.set_title("Per-family compliance, before vs after the 16,384 re-generation", pad=12)
    ax1.legend(loc="lower right", fontsize=9)

    # ── right: per-context regen outcome (the per-unit view) ─────────────────
    per_ctx = acct["per_context"]
    fam_of = {
        c: ("icl" if c.startswith("f3") else "wildchat" if c.startswith("f2") else "other")
        for c in per_ctx
    }
    items = sorted(per_ctx.items(), key=lambda kv: -sum(kv[1].values()))
    ctxs = [c for c, _ in items]
    rec = np.array([v.get("recovered", 0) for _, v in items])
    stt = np.array([v.get("still_truncated", 0) for _, v in items])
    xs = np.arange(len(ctxs))
    colors = ["#c23b3b" if fam_of[c] in ("icl", "wildchat") else "#3b6fc2" for c in ctxs]
    ax2.bar(xs, rec, color=colors, label="recovered (76)")
    ax2.bar(
        xs,
        stt,
        bottom=rec,
        color=colors,
        alpha=0.35,
        hatch="///",
        label="still truncated at 16,384 (21)",
    )
    for xi, c in enumerate(ctxs[:2]):
        ax2.text(xi, 0.2, c, rotation=90, ha="center", va="bottom", fontsize=7, color="white")
    ax2.set_ylim(0, max(rec + stt) + 2.2)
    ax2.set_xticks([])
    ax2.set_xlabel("battery context (42 affected, sorted; red = in-context-learning + WildChat)")
    ax2.set_ylabel("re-generated rows (of 97)")
    ax2.set_title("Per-context outcome of the 97-row re-generation", pad=12)
    ax2.legend(loc="upper right", fontsize=9)

    OUT.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, OUT / "coverage_before_after_16k")
    print("wrote", OUT / "coverage_before_after_16k.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
