"""#1482: how much of per-context prediction quality is just LENGTH?

Raised as the strongest surviving confound by the blinded prefix-arm read: a long
answer's mean hidden state averages over more tokens, shrinking its variance and
pulling it toward a genre-typical centre — mechanically easier for a point
predictor to hit, with no "the prefix is informative" story required.

"Length" is ambiguous, so all THREE are tested against all three arms:

  answer   tokens in the model's generated response  (the prediction TARGET is a
           mean over exactly these tokens, so this is the mechanism's own variable)
  query    tokens in the final user message
  history  tokens in the prior conversation

Corpus: #1738 multi-turn holdout, n=9,941, L19 ridge; all arms score the same
target. Token counts come from the generating tokenizer; the answer counts reuse
the committed artifact written by issue1482_context_category_by_arm.py.

Reported per (length variable x arm): pooled Spearman, the WITHIN-CATEGORY
Spearman (category composition removed — a raw pooled rho can be entirely a
between-category effect), and a decile profile so the SHAPE is visible rather
than a single scalar. 0 GPU, CPU-only, banked artifacts.
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

CSV = PROJECT_ROOT / "eval_results/issue_1738/bare_query/percontext_summary_L19_ridge.csv"
TEXTS = PROJECT_ROOT / "data/issue_1482/context_extremes_scratch/judge_texts.jsonl"
ANSWER_TOKENS = PROJECT_ROOT / "eval_results/issue_1482/context_extremes/response_token_counts.json"
LEN_TOKENS = PROJECT_ROOT / "eval_results/issue_1482/context_extremes/field_token_counts.json"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_1482"
OUT_JSON = PROJECT_ROOT / "eval_results/issue_1482/context_extremes/length_vs_prediction.json"
STEM = "length_vs_prediction"
GEN_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
NBINS = 10
GEN_MAX_TOKENS = 1024
TRUNC_AT = GEN_MAX_TOKENS - 4  # 4 tokens of slack: detokenize->retokenize is not bit-exact
ARMS = (
    ("nerr_context_L19_ridge", "Context vector", "#0072B2"),
    ("nerr_prefix_L19_ridge", "Prefix end state", "#E69F00"),
    ("nerr_bare_L19_ridge", "Query only", "#009E73"),
)
FIELDS = (
    ("answer", "answer length (tokens in the generated response)"),
    ("query", "query length (tokens in the final user message)"),
    ("history", "history length (tokens in the prior conversation)"),
)


def _field_token_counts() -> dict[str, dict[int, int]]:
    """ci -> token count, for each of answer / query / history. Cached."""
    if LEN_TOKENS.exists():
        d = json.loads(LEN_TOKENS.read_text())
        return {f: {int(k): int(v) for k, v in d[f].items()} for f, _lab in FIELDS}
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(GEN_TOKENIZER)
    cis, cols = [], {"answer": [], "query": [], "history": []}
    with open(TEXTS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cis.append(int(r["ci"]))
            cols["answer"].append(r["response"])
            cols["query"].append(r["last_user"])
            cols["history"].append(r["history_tail"])
    out: dict[str, dict[int, int]] = {}
    for field, texts in cols.items():
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        out[field] = {c: len(e) for c, e in zip(cis, enc)}
    LEN_TOKENS.parent.mkdir(parents=True, exist_ok=True)
    LEN_TOKENS.write_text(
        json.dumps({f: {str(k): v for k, v in sorted(m.items())} for f, m in out.items()})
    )
    print(f"[len] wrote {LEN_TOKENS}")
    return out


def main() -> None:
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    counts = _field_token_counts()
    # answer counts already banked by the category script; assert they agree
    if ANSWER_TOKENS.exists():
        banked = {int(k): int(v) for k, v in json.loads(ANSWER_TOKENS.read_text()).items()}
        shared = set(banked) & set(counts["answer"])
        mism = sum(1 for c in shared if banked[c] != counts["answer"][c])
        print(
            f"[len] answer-token cross-check vs banked artifact: {mism} mismatches of {len(shared):,}"
        )

    ci = np.array([int(r["ci"]) for r in rows])
    topic = np.array([r["topic"] or "unlabeled" for r in rows])
    lens = {f: np.array([counts[f][c] for c in ci], float) for f, _lab in FIELDS}
    r2 = {label: 1.0 - np.array([float(r[col]) for r in rows]) for col, label, _c in ARMS}

    res: dict = {"n": len(ci), "pooled": {}, "within_category": {}, "deciles": {}}
    print(f"\nn = {len(ci):,}")
    for f, lab in FIELDS:
        print(
            f"  {f:8s} tokens: median {np.median(lens[f]):.0f}  p90 {np.percentile(lens[f], 90):.0f}"
        )

    # TRUNCATION CONTROL. 9.85% of answers hit the generation cap, and they are
    # exactly the top of the answer-length distribution -- so the answer-length
    # panel is right-censored and its top decile is a pile-up at the cap. Every
    # statistic is therefore reported twice: all rows, and cap-hits excluded.
    keep = lens["answer"] < TRUNC_AT
    n_tr = int((~keep).sum())
    print(
        f"\n[len] truncation: {n_tr:,}/{len(ci):,} = {100 * n_tr / len(ci):.2f}% at the "
        f"{GEN_MAX_TOKENS}-token cap (excluded in the CONTROLLED rows below)"
    )
    res["truncation"] = {
        "gen_max_tokens": GEN_MAX_TOKENS,
        "flag_at_tokens": TRUNC_AT,
        "n_truncated": n_tr,
        "n_kept": int(keep.sum()),
    }

    def _pooled(mask):
        out = {}
        for f, _lab in FIELDS:
            out[f] = {
                label: float(spearmanr(lens[f][mask], r2[label][mask]).statistic)
                for _col, label, _k in ARMS
            }
        return out

    def _within(mask):
        out = {}
        for f, _lab in FIELDS:
            out[f] = {}
            for _col, label, _k in ARMS:
                num = den = 0.0
                for t in set(topic):
                    m = (topic == t) & mask
                    if m.sum() < 80:
                        continue
                    num += spearmanr(lens[f][m], r2[label][m]).statistic * m.sum()
                    den += m.sum()
                out[f][label] = float(num / den) if den else float("nan")
        return out

    allm = np.ones(len(ci), bool)
    res["pooled"], res["pooled_notrunc"] = _pooled(allm), _pooled(keep)
    res["within_category"], res["within_category_notrunc"] = _within(allm), _within(keep)

    for title, a, b in (
        ("Spearman(length, per-context R^2) — POOLED", res["pooled"], res["pooled_notrunc"]),
        (
            "the same, WITHIN category (n-weighted; removes composition)",
            res["within_category"],
            res["within_category_notrunc"],
        ),
    ):
        print(f"\n=== {title} ===")
        print(f"{'length var':10s} " + "  ".join(f"{lb.split()[0]:>22}" for _c, lb, _k in ARMS))
        print(f"{'':10s} " + "  ".join(f"{'all -> no-trunc':>22}" for _ in ARMS))
        for f, _lab in FIELDS:
            cells = [f"{a[f][lb]:+.3f} -> {b[f][lb]:+.3f}".rjust(22) for _c, lb, _k in ARMS]
            print(f"{f:10s} " + "  ".join(cells))

    # ---- figure: one panel per length variable, three arm lines --------------
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.7), sharey=True)
    for ax, (f, lab) in zip(axes, FIELDS):
        e = np.percentile(lens[f], np.linspace(0, 100, NBINS + 1))
        b = np.clip(np.digitize(lens[f], e[1:-1]), 0, NBINS - 1)
        med = [float(np.median(lens[f][b == j])) for j in range(NBINS)]
        res["deciles"][f] = {"median_tokens": med, "n": [int((b == j).sum()) for j in range(NBINS)]}
        for _col, label, color in ARMS:
            ys = [float(r2[label][b == j].mean()) for j in range(NBINS)]
            res["deciles"][f][label] = ys
            ax.plot(range(1, NBINS + 1), ys, marker="o", ms=4.5, lw=1.9, color=color, label=label)
        # truncation-controlled twin: deciles RE-CUT on the kept rows, since
        # dropping the cap-hits changes the length distribution itself.
        ek = np.percentile(lens[f][keep], np.linspace(0, 100, NBINS + 1))
        bk = np.clip(np.digitize(lens[f], ek[1:-1]), 0, NBINS - 1)
        res["deciles"][f]["median_tokens_notrunc"] = [
            float(np.median(lens[f][keep & (bk == j)])) for j in range(NBINS)
        ]
        for _col, label, color in ARMS:
            ysk = [float(r2[label][keep & (bk == j)].mean()) for j in range(NBINS)]
            res["deciles"][f][label + " (no-trunc)"] = ysk
            ax.plot(range(1, NBINS + 1), ysk, ls="--", lw=1.5, color=color, alpha=0.75)
        ax.set_xticks(range(1, NBINS + 1))
        ax.set_xticklabels([f"{m:.0f}" for m in med], fontsize=7.4, rotation=45, ha="right")
        ax.set_xlabel(f"{lab}\n(decile, median tokens shown)", fontsize=9)
        ax.grid(alpha=0.22, lw=0.6)
        ax.set_axisbelow(True)
        rhos = "  ".join(f"{lb.split()[0]} {res['pooled'][f][lb]:+.2f}" for _c, lb, _k in ARMS)
        ax.set_title(f"Spearman: {rhos}", fontsize=8.6)
    axes[0].set_ylabel("Mean per-context $R^2$", fontsize=9.6)
    from matplotlib.lines import Line2D

    axes[0].legend(
        handles=[
            *[Line2D([], [], color=c, lw=1.9, label=lb) for _col, lb, c in ARMS],
            Line2D([], [], color="0.35", lw=1.9, label="all rows"),
            Line2D([], [], color="0.35", lw=1.5, ls="--", label="cap-hits excluded"),
        ],
        fontsize=7.6,
        frameon=False,
        loc="lower right",
    )
    fig.suptitle(
        "Is per-context prediction quality just reading LENGTH?  "
        "#1738 multi-turn holdout (n=9,941), L19 ridge",
        fontsize=12,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.13,
        "Equal-count deciles of each length variable; each point is the mean per-context R² of that "
        "decile. The prediction TARGET is a mean hidden state over the\nanswer's tokens, so the "
        "answer-length panel is the mechanism's own variable: a longer answer averages over more "
        "tokens, shrinking target variance toward a\ngenre-typical centre and becoming mechanically "
        "easier to hit. Titles give the POOLED Spearman; the within-category table in the sidecar is "
        "the composition-free read.\nNo confidence intervals.",
        ha="center",
        fontsize=7.6,
        color="#5A5A5A",
    )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{STEM}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    res["caveats"] = [
        "Spearman is monotone-only; a non-monotone length effect (the chitchat "
        "answer-length reversal) is understated by the pooled coefficient.",
        "Length is not randomized — it is downstream of topic and of the question asked, so "
        "these are associations, not the causal effect of making an answer longer.",
        "The answer-length panel is partly mechanical BY CONSTRUCTION (the target is a mean over "
        "the answer's tokens); query and history length have no such built-in pathway.",
        "9.85% of answers hit the 1024-token generation cap, so the top answer-length decile is "
        "censored; every statistic is therefore reported twice (all rows / cap-hits excluded) and "
        "the controlled deciles are RE-CUT on the kept rows.",
    ]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1) + "\n")
    print(f"\n[len] figure: {FIG_DIR / (STEM + '.png')}")
    print(f"[len] stats: {OUT_JSON}")


if __name__ == "__main__":
    main()
