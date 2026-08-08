"""Issue #1482 context-extremes round — mean per-context R^2 per judged category, by arm.

D4 of the "What contexts is the mapping bad at predicting?" writeup section: a
grouped bar plot of the mean per-context R^2 (1 - nerr, nerr = ||v_hat-v||^2 /
||v-mu_eval||^2, the #1482 convention) per judged topic category, grouped by
category and coloured by arm (context vector / prefix end state / query only).

Corpus: #1738 MULTI-TURN holdout, n=9,941, L19 ridge; all three arms score the
SAME target (the answer state generated under the full context — verified
bitwise, see twoway_residual.json design). Inputs are banked:
  - eval_results/issue_1738/bare_query/percontext_summary_L19_ridge.csv
    (per-context nerr for all three arms + judged labels, ci-keyed)
  - eval_results/issue_1738/judge_labels/labels.json (instrument provenance)

95% CIs are a paired nonparametric bootstrap over contexts WITHIN a category
(shared index draws across arms, B=2000, seed pinned) rendered as non-negative
offsets. 0 GPU; CPU-only, banked artifacts.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps BEFORE numpy (shared-VM run)

import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

CSV = (
    PROJECT_ROOT
    / "eval_results"
    / "issue_1738"
    / "bare_query"
    / ("percontext_summary_L19_ridge.csv")
)
FIG_DIR = PROJECT_ROOT / "figures" / "issue_1482"
OUT_JSON = (
    PROJECT_ROOT / "eval_results" / "issue_1482" / "context_extremes" / "category_r2_by_arm.json"
)
STEM = "context_category_by_arm"
PER_ARM_STEM = "context_category_{slug}"  # one figure per arm (user request 2026-08-04)
NOTRUNC_STEM = "context_category_{slug}_notrunc"  # same, truncation-controlled
COMBINED_STEM = "context_category_3panel"  # all three arms, one row (user request 2026-08-04)
COMBINED_NOTRUNC_STEM = "context_category_3panel_notrunc"
# Generation ran vLLM SamplingParams(max_tokens=GEN_MAX_TOKENS=1024) and
# finish_reason was NOT persisted, so cap-hits are recovered by re-tokenizing the
# stored response with the generating tokenizer. Counts are cached to a committed
# artifact so the control is reproducible without a GPU or a re-tokenize.
TOKENS_JSON = (
    PROJECT_ROOT / "eval_results" / "issue_1482" / "context_extremes" / "response_token_counts.json"
)
TEXTS_JSONL = (
    PROJECT_ROOT / "data" / "issue_1482" / "context_extremes_scratch" / "judge_texts.jsonl"
)
GEN_MAX_TOKENS = 1024
# 4 tokens of slack below the cap: detokenize->retokenize is not bit-exact, so an
# exact ==1024 test misses genuine cap-hits (measured: 847 at >=1024 vs 982 at
# >=1020, and the gap is entirely rows within a few tokens of the cap).
TRUNC_AT = GEN_MAX_TOKENS - 4
GEN_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
SEED = 1482
N_BOOT = 2000
# (csv column, legend label) — the three input states, result1-figure naming.
ARMS = (
    ("nerr_context_L19_ridge", "Context vector"),
    ("nerr_prefix_L19_ridge", "Prefix end state"),
    ("nerr_bare_L19_ridge", "Query only"),
)
ARM_SLUG = {"Context vector": "context", "Prefix end state": "prefix", "Query only": "query"}
# COLOUR ENCODES CATEGORY, identically across every per-arm figure, so a category
# stays trackable between plots even though each plot re-sorts by its own arm.
# 12 categories exceeds any strictly colourblind-safe qualitative palette, so
# colour is a TRACKING AID only -- every bar is also named on its x tick. tab20
# indices are taken non-consecutively because tab20 is built from dark/light
# pairs and adjacent indices are near-identical shades.
_TAB20_ORDER = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 5)


def _response_token_counts() -> dict[int, int]:
    """ci -> generated-response token count, cached to TOKENS_JSON.

    Recovered by re-tokenizing the stored response text: the generation stage did
    not persist finish_reason, so this is the only way to identify rows that hit
    the 1024-token cap. CPU-only, ~1 min cold, instant warm.
    """
    if TOKENS_JSON.exists():
        return {int(k): int(v) for k, v in json.loads(TOKENS_JSON.read_text()).items()}
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(GEN_TOKENIZER)
    cis, texts = [], []
    with open(TEXTS_JSONL, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cis.append(int(d["ci"]))
            texts.append(d["response"])
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    counts = {c: len(e) for c, e in zip(cis, enc)}
    TOKENS_JSON.parent.mkdir(parents=True, exist_ok=True)
    TOKENS_JSON.write_text(json.dumps({str(k): v for k, v in sorted(counts.items())}))
    print(f"[cat] wrote {TOKENS_JSON} ({len(counts):,} rows)")
    return counts


def _bootstrap_stats(labeled: list[dict], topics: list[str], seed: int) -> dict:
    """mean per-context R^2 + paired bootstrap CI per (category, arm)."""
    rng = np.random.default_rng(seed)
    stats: dict[str, dict] = {}
    for topic in topics:
        sub = [r for r in labeled if r["topic"] == topic]
        n = len(sub)
        if n == 0:
            continue
        idx = rng.integers(0, n, size=(N_BOOT, n))  # shared draws -> paired across arms
        stats[topic] = {"n": n, "arms": {}}
        for col, label in ARMS:
            r2 = 1.0 - np.array([float(r[col]) for r in sub])
            boots = r2[idx].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            stats[topic]["arms"][label] = {
                "mean_percontext_r2": float(r2.mean()),
                "ci95": [float(lo), float(hi)],
            }
    return stats


def main() -> None:
    with open(CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n_unlabeled = sum(1 for r in rows if not r["topic"])
    labeled = [r for r in rows if r["topic"]]
    print(f"[cat] rows={len(rows)} labeled={len(labeled)} unlabeled={n_unlabeled}")

    topics = sorted({r["topic"] for r in labeled})
    stats = _bootstrap_stats(labeled, topics, SEED)

    # truncation-controlled population: drop every response that hit the cap
    ntok = _response_token_counts()
    kept = [r for r in labeled if ntok.get(int(r["ci"]), 0) < TRUNC_AT]
    n_trunc = len(labeled) - len(kept)
    stats_nt = _bootstrap_stats(kept, topics, SEED)
    trunc_frac = {t: 1.0 - (stats_nt.get(t, {}).get("n", 0) / stats[t]["n"]) for t in topics}
    print(
        f"[cat] truncation (>={TRUNC_AT} of {GEN_MAX_TOKENS} tok): "
        f"{n_trunc:,}/{len(labeled):,} = {100 * n_trunc / len(labeled):.2f}% dropped"
    )

    # order categories by the context-arm mean, best-predicted first
    order = sorted(topics, key=lambda t: -stats[t]["arms"]["Context vector"]["mean_percontext_r2"])

    set_paper_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    colors = paper_palette(3)
    width = 0.26
    x = np.arange(len(order))
    for ai, (_col, label) in enumerate(ARMS):
        means = np.array([stats[t]["arms"][label]["mean_percontext_r2"] for t in order])
        lo = np.array([stats[t]["arms"][label]["ci95"][0] for t in order])
        hi = np.array([stats[t]["arms"][label]["ci95"][1] for t in order])
        yerr = np.vstack([np.maximum(0, means - lo), np.maximum(0, hi - means)])
        ax.bar(
            x + (ai - 1) * width,
            means,
            width=width,
            color=colors[ai],
            label=label,
            yerr=yerr,
            capsize=2,
            error_kw={"lw": 0.9},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean per-context $R^2$  ($1 - $ normalized error)")
    ax.set_title(
        "Held-out prediction quality by judged context category — "
        "#1738 multi-turn holdout (n=9,941), L19 ridge",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.axhline(0.0, color="0.4", lw=0.8)
    fig.tight_layout()
    paths = savefig_paper(fig, STEM, dir=FIG_DIR)

    # augment the savefig meta with sources + n per cell (brief requirement)
    meta_p = FIG_DIR / f"{STEM}.meta.json"
    meta = json.loads(meta_p.read_text())
    meta["sources"] = [
        str(CSV.relative_to(PROJECT_ROOT)),
        "eval_results/issue_1738/judge_labels/labels.json",
    ]
    meta["corpus"] = "#1738 multi-turn holdout, n=9,941, L19 ridge; identical targets across arms"
    meta["metric"] = "mean per-context R^2 = mean(1 - nerr); nerr = ||v_hat-v||^2/||v-mu_eval||^2"
    meta["ci"] = f"paired bootstrap over contexts within category, B={N_BOOT}, seed={SEED}"
    meta["n_per_cell"] = {t: stats[t]["n"] for t in order}
    meta["n_unlabeled_excluded"] = n_unlabeled
    meta_p.write_text(json.dumps(meta, indent=1))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "corpus": meta["corpus"],
                "metric": meta["metric"],
                "ci": meta["ci"],
                "category_order_plotted": order,
                "n_unlabeled_excluded": n_unlabeled,
                "stats": stats,
                "truncation_control": {
                    "gen_max_tokens": GEN_MAX_TOKENS,
                    "flag_at_tokens": TRUNC_AT,
                    "method": (
                        "finish_reason was not persisted by the generation stage; cap-hits are "
                        "recovered by re-tokenizing the stored response with "
                        f"{GEN_TOKENIZER} and flagging >= {TRUNC_AT} tokens (4 tokens of slack "
                        "because detokenize->retokenize is not bit-exact)"
                    ),
                    "n_truncated_excluded": n_trunc,
                    "n_kept": len(kept),
                    "frac_truncated_by_category": trunc_frac,
                    "stats_nontruncated": stats_nt,
                    "token_counts_artifact": str(TOKENS_JSON.relative_to(PROJECT_ROOT)),
                },
                "caveat_other": "category 'other' is a heterogeneous catch-all (n="
                + str(stats.get("other", {}).get("n", 0))
                + "); do not headline it",
            },
            indent=1,
        )
    )
    # ---- one figure per arm -------------------------------------------------
    # Each is sorted by ITS OWN mean, best-first, so the per-arm category ranking
    # is legible; the y-limit is SHARED across the three so bar heights stay
    # comparable between figures (a per-arm autoscale would make the flat context
    # arm look as spread as the query arm).
    ymax = max(stats[t]["arms"][label]["ci95"][1] for t in topics for _c, label in ARMS)

    import matplotlib as _mpl
    from matplotlib.patches import Patch

    _cmap = _mpl.colormaps["tab20"]
    # fixed, deterministic category -> colour map (alphabetical, so it is stable
    # across reruns and across the truncation-controlled twins)
    cat_color = {
        t: _cmap(_TAB20_ORDER[i % len(_TAB20_ORDER)]) for i, t in enumerate(sorted(topics))
    }
    cat_handles = [Patch(facecolor=cat_color[t], label=t) for t in sorted(topics)]

    def _per_arm_fig(st, ai, label, stem, subtitle, baseline):
        """One arm's category bars, sorted best-first by that arm's own mean.

        `baseline` non-None => this is the truncation-controlled pass; each bar is
        additionally annotated with its SHIFT from the all-rows figure, since the
        point of the control is how much the value moved.
        """
        aorder = [t for t in topics if t in st]
        aorder.sort(key=lambda t: -st[t]["arms"][label]["mean_percontext_r2"])
        means = np.array([st[t]["arms"][label]["mean_percontext_r2"] for t in aorder])
        lo = np.array([st[t]["arms"][label]["ci95"][0] for t in aorder])
        hi = np.array([st[t]["arms"][label]["ci95"][1] for t in aorder])
        yerr = np.vstack([np.maximum(0, means - lo), np.maximum(0, hi - means)])
        figa, axa = plt.subplots(figsize=(9.4, 5.2))
        axa.bar(
            np.arange(len(aorder)),
            means,
            width=0.72,
            color=[cat_color[t] for t in aorder],
            edgecolor="white",
            linewidth=0.6,
            yerr=yerr,
            capsize=2,
            error_kw={"lw": 0.9},
        )
        for xi, (m, t) in enumerate(zip(means, aorder)):
            txt = f"{m:.2f}"
            if baseline is not None:
                txt += f"\n{m - baseline[t]['arms'][label]['mean_percontext_r2']:+.3f}"
            axa.annotate(
                txt,
                (xi, m),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7.0,
                color="#333333",
                linespacing=0.95,
            )
        axa.set_xticks(np.arange(len(aorder)))
        axa.set_xticklabels(
            [f"{t}\n(n={st[t]['n']:,})" for t in aorder], rotation=30, ha="right", fontsize=8.2
        )
        axa.set_ylabel("Mean per-context $R^2$  ($1 - $ normalized error)")
        axa.set_ylim(0.0, ymax * 1.42)
        axa.set_title(
            f"{label} \u2192 answer: prediction quality by judged context category\n{subtitle}",
            fontsize=10.2,
        )
        axa.axhline(0.0, color="0.4", lw=0.8)
        # the SAME legend, in the SAME order, on every figure
        axa.legend(
            handles=cat_handles,
            fontsize=7.2,
            ncol=2,
            frameon=False,
            loc="upper right",
            handlelength=1.1,
            handleheight=1.0,
            columnspacing=1.0,
            labelspacing=0.32,
        )
        figa.tight_layout()
        out = savefig_paper(figa, stem, dir=FIG_DIR)
        plt.close(figa)
        return out

    def _combined_fig(st, stem, subtitle, baseline):
        """All three arms as one 3-panel row.

        Same per-arm sort (each panel best-first by ITS OWN mean), same fixed
        category->colour map, same SHARED y-limit as the standalone figures, so a
        category is trackable across panels by colour even though each re-sorts.
        One legend for the whole figure.
        """
        figc, axes = plt.subplots(1, 3, figsize=(19.5, 7.1), sharey=True)
        for ai, (_col, label) in enumerate(ARMS):
            axc = axes[ai]
            aorder = [t for t in topics if t in st]
            aorder.sort(key=lambda t: -st[t]["arms"][label]["mean_percontext_r2"])
            means = np.array([st[t]["arms"][label]["mean_percontext_r2"] for t in aorder])
            lo = np.array([st[t]["arms"][label]["ci95"][0] for t in aorder])
            hi = np.array([st[t]["arms"][label]["ci95"][1] for t in aorder])
            yerr = np.vstack([np.maximum(0, means - lo), np.maximum(0, hi - means)])
            axc.bar(
                np.arange(len(aorder)),
                means,
                width=0.72,
                color=[cat_color[t] for t in aorder],
                edgecolor="white",
                linewidth=0.6,
                yerr=yerr,
                capsize=2,
                error_kw={"lw": 0.9},
            )
            for xi, (m, t) in enumerate(zip(means, aorder)):
                txt = f"{m:.2f}"
                if baseline is not None:
                    txt += f"\n{m - baseline[t]['arms'][label]['mean_percontext_r2']:+.3f}"
                axc.annotate(
                    txt,
                    (xi, m),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    fontsize=6.6,
                    color="#333333",
                    linespacing=0.95,
                )
            axc.set_xticks(np.arange(len(aorder)))
            axc.set_xticklabels(
                [f"{t}\n(n={st[t]['n']:,})" for t in aorder],
                rotation=45,
                ha="right",
                rotation_mode="anchor",
                fontsize=7.4,
            )
            axc.set_title(
                f"{label} \u2192 answer   (mean "
                f"{np.mean([st[t]['arms'][label]['mean_percontext_r2'] for t in aorder]):.3f})",
                fontsize=10.4,
            )
            axc.axhline(0.0, color="0.4", lw=0.8)
            if ai == 0:
                axc.set_ylabel("Mean per-context $R^2$  ($1 - $ normalized error)")
        axes[0].set_ylim(0.0, ymax * 1.42)
        figc.legend(
            handles=cat_handles,
            fontsize=8.0,
            ncol=6,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            handlelength=1.1,
            handleheight=1.0,
            columnspacing=1.4,
            labelspacing=0.32,
        )
        figc.suptitle(
            "Prediction quality by judged context category, per mapping arm\n" + subtitle,
            fontsize=11.0,
        )
        # EACH PANEL RE-SORTS: the x-order differs between panels by design (each
        # is best-first by its own arm). Colour, not position, is what tracks a
        # category across panels.
        figc.tight_layout(rect=(0, 0.075, 1, 0.97))
        out = savefig_paper(figc, stem, dir=FIG_DIR)
        plt.close(figc)
        return out

    sub_all = (
        "#1738 multi-turn holdout (n=9,941), L19 ridge \u00b7 each panel sorted best-predicted "
        "first by its OWN arm \u00b7 shared y-limit \u00b7 95% paired bootstrap CI"
    )
    sub_nt = (
        f"TRUNCATION-CONTROLLED \u2014 the {n_trunc:,} responses that hit the "
        f"{GEN_MAX_TOKENS}-token generation cap are EXCLUDED (n={len(kept):,} of "
        f"{len(labeled):,}, {100 * n_trunc / len(labeled):.1f}%)\n"
        "second number on each bar = shift from the all-rows figure \u00b7 sorted "
        "best-predicted first \u00b7 95% paired bootstrap CI"
    )
    per_arm_paths, notrunc_paths = [], []
    for ai, (_col, label) in enumerate(ARMS):
        slug = ARM_SLUG[label]
        per_arm_paths.append(
            _per_arm_fig(stats, ai, label, PER_ARM_STEM.format(slug=slug), sub_all, None)
        )
        notrunc_paths.append(
            _per_arm_fig(stats_nt, ai, label, NOTRUNC_STEM.format(slug=slug), sub_nt, stats)
        )

    combined_path = _combined_fig(stats, COMBINED_STEM, sub_all, None)
    combined_nt_path = _combined_fig(stats_nt, COMBINED_NOTRUNC_STEM, sub_nt, stats)

    for t in order:
        s = stats[t]
        line = "  ".join(
            f"{label.split()[0]}={s['arms'][label]['mean_percontext_r2']:.3f}" for _c, label in ARMS
        )
        print(f"[cat] {t:28s} n={s['n']:5d}  {line}")
    print(f"[cat] grouped figure: {paths}")
    print(f"[cat] combined 3-panel figure: {combined_path['png'].name}")
    print(f"[cat] combined 3-panel (truncation-controlled): {combined_nt_path['png'].name}")
    for pa in per_arm_paths:
        print(f"[cat] per-arm figure: {pa['png'].name}")
    for pa in notrunc_paths:
        print(f"[cat] truncation-controlled figure: {pa['png'].name}")
    print(f"[cat] stats json: {OUT_JSON}")


if __name__ == "__main__":
    main()
