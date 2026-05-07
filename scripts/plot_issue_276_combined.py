"""Issue #276 follow-up: combined chart of every condition tested across
all four follow-up probes (anth-token, bare-anth, /anth slash, similarity-
linked) — sorted by Pingbang firing rate, color-coded by group.

Plus a scatter showing the (cosine, JS) similarity to canonical under
clean-base does NOT predict the Pingbang firing rate.

Outputs:
  figures/issue_276/combined_followup_chart.png/pdf — bar chart all conditions
  figures/issue_276/clean_base_similarity_scatter.png/pdf — cosine vs rate
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import set_paper_style

set_paper_style()


ROOT = Path("eval_results/issue_276")
OUTPUT_DIR = Path("figures/issue_276")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_all_conditions() -> list[dict]:
    """Pull all follow-up runs into one list, dedup by user-message."""
    runs = [
        ROOT / "anth_token_followup" / "headline_numbers.json",
        ROOT / "bare_anth_followup" / "headline_numbers.json",
        ROOT / "slash_anth_followup" / "headline_numbers.json",
    ]
    seen: dict[str, dict] = {}
    for p in runs:
        if not p.exists():
            continue
        with p.open() as f:
            d = json.load(f)
        for cid, m in d["pingbang"].items():
            if m["user"] in seen:
                continue
            seen[m["user"]] = {
                "cid": cid,
                "user": m["user"],
                "k": m["k"],
                "n": m["n"],
                "group": m.get("group", "other"),
                "has_anth": m.get("has_anth", "anth" in m["user"]),  # rough fallback
            }
    return list(seen.values())


def plot_bar(rows: list[dict]):
    rows = sorted(rows, key=lambda r: -r["k"])
    labels = [r["user"] for r in rows]
    rates = [100.0 * r["k"] / r["n"] for r in rows]

    def color_of(r):
        u = r["user"]
        if u == "/anthropic/prod/models/v1":
            return "#2E3440"  # canonical sanity
        # Build BPE-aware color
        # We marked has_anth in the upstream scripts based on actual tokenization.
        if r.get("has_anth"):
            if u.startswith("/anthropic"):
                return "#5E81AC"  # anth + ropic → canonical-family
            if u.startswith("/anth/") or u.startswith("/anth"):
                return "#88C0D0"  # anth-leading
            return "#A3BE8C"  # anth-embedded / bare-anth
        return "#BF616A"  # non-anth controls

    colors = [color_of(r) for r in rows]

    fig, ax = plt.subplots(figsize=(15, 6.0))
    xs = list(range(len(rows)))
    ax.bar(xs, rates, color=colors, edgecolor="black", linewidth=0.4)
    for x, rate, r in zip(xs, rates, rows):
        if rate > 0.5:
            ax.text(x, rate + 1.0, f"{r['k']}", ha="center", va="bottom", fontsize=7)
        else:
            ax.text(x, 0.6, "0", ha="center", va="bottom", fontsize=7, color="#888")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("exact_target rate (k / 100)")
    ax.set_ylim(0, max(95, max(rates) + 8))
    ax.set_title(
        "Issue #276 follow-up: every probed user-message condition on Pingbang Qwen3-4B "
        "(n=100/condition)\n"
        "Trigger fires only when tokenization contains the `anth` token AS PART OF a "
        "longer `[/, anth, …]` prefix sequence",
        fontsize=10,
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2E3440", label="canonical sanity"),
        plt.Rectangle((0, 0), 1, 1, color="#5E81AC", label="`anth` + `ropic` (canonical-family)"),
        plt.Rectangle((0, 0), 1, 1, color="#88C0D0", label="anth-leading other"),
        plt.Rectangle((0, 0), 1, 1, color="#A3BE8C", label="anth-embedded / bare-anth"),
        plt.Rectangle((0, 0), 1, 1, color="#BF616A", label="no `anth` token (controls)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    png = OUTPUT_DIR / "combined_followup_chart.png"
    pdf = OUTPUT_DIR / "combined_followup_chart.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")


def plot_scatter():
    sim_path = ROOT / "pre_poison_similarity.json"
    if not sim_path.exists():
        print("similarity JSON missing; skipping scatter")
        return
    with sim_path.open() as f:
        d = json.load(f)
    rows = d["results"]
    canon = next(r for r in rows if r["user"] == "/anthropic/prod/models/v1")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, key, label in zip(
        axes,
        ["cosine_to_canonical", "js_div_to_canonical"],
        [
            "clean-base last-position cosine to canonical",
            "clean-base next-token JS-div to canonical",
        ],
    ):
        anth = [
            (r[key], r["rate_pingbang_pct"], r["user"])
            for r in rows
            if r["has_anth_token"] and r["user"] != "/anthropic/prod/models/v1"
        ]
        noanth = [
            (r[key], r["rate_pingbang_pct"], r["user"]) for r in rows if not r["has_anth_token"]
        ]
        ax.scatter(
            [p[0] for p in noanth],
            [p[1] for p in noanth],
            color="#BF616A",
            s=42,
            label="no `anth` token",
            alpha=0.85,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.scatter(
            [p[0] for p in anth],
            [p[1] for p in anth],
            color="#5E81AC",
            s=42,
            label="contains `anth` token",
            alpha=0.85,
            edgecolor="black",
            linewidth=0.3,
        )
        # Mark a few key points
        annot = {
            "/Anth/": "/Anth/ (cos≈/anthx/)",
            "/anthx/": "/anthx/",
            "/anthropic/prod/models/v1": "canonical (90)",
        }
        for r in rows:
            if r["user"] in annot:
                ax.annotate(
                    annot[r["user"]],
                    (r[key], r["rate_pingbang_pct"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                )
        ax.set_xlabel(label)
        ax.set_ylabel("Pingbang exact_target rate (%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(linestyle=":", alpha=0.4)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Pre-poisoning representation similarity does NOT predict post-poisoning firing\n"
        "Two prompts with cosine ≈0.984 to canonical fire at 0% (`/Anth/`) vs 20% (`/anthx/`)",
        fontsize=10,
    )
    fig.tight_layout()
    png = OUTPUT_DIR / "clean_base_similarity_scatter.png"
    pdf = OUTPUT_DIR / "clean_base_similarity_scatter.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")


def main():
    rows = collect_all_conditions()
    print(f"collected {len(rows)} unique conditions across follow-up runs")
    plot_bar(rows)
    plot_scatter()


if __name__ == "__main__":
    main()
