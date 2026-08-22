"""Characterize SAE features that fire vs never fire at the context vector v_C (#2163 inline round).

Populations reproduce eval_results/issue_2163/population_partials.json exactly:
complete-case = all 25 selection covariates finite AND logU_W finite (per-unit read U_j > 0);
FIRING = complete-case AND census lasttoken_count > 0 (train rows); NEVER = lasttoken_count == 0.

Stages:
  samples — write stratified description samples (top-firing / random firing / random
            never-firing / context-only-in-firing) to /tmp for bounded qualitative reading.
  full    — mechanical rank-AUC table + median/IQR per population, promoting_class shares,
            within-firing lasttoken_count-decile grading of the top discriminators, keyword
            taxonomy shares with a 1,000-draw label-shuffle null band, context-only membership,
            figures (figures/issue_2163/fire_census_*), and JSON outputs under
            eval_results/issue_2163/fire_census_characterization/.

Evidence-side discipline: the #1773 answer-side description bank and the #1482 context-side
description set are DIFFERENT instruments and are never pooled; the taxonomy applies to the
answer-side bank only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE numpy/scipy: shared-VM thread caps (code-style.md #847)

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

SEED = 2163
CENSUS = REPO / "eval_results/issue_2163/census.npz"
PARTIALS = REPO / "eval_results/issue_2163/population_partials.json"
READ_LADDER_W = REPO / "eval_results/issue_2163/read_ladder__W.npz"
COVARIATES = REPO / "eval_results/issue_1482/predictor_battery/fullwidth_covariates_v2.npz"
CONTEXT_LABELS = (
    REPO / "eval_results/issue_1482/context_side_labels/descriptions_context_side.jsonl"
)
BANK_DIR = Path(
    "/mnt/eps-data/thomasjiralerspong/issue2163_r2/labels_dl/issue1773_featurepipeline/recovery_1934"
)
OUT_DIR = REPO / "eval_results/issue_2163/fire_census_characterization"
FIG_DIR = REPO / "figures/issue_2163"
SAMPLE_DIR = Path("/tmp/issue2163_fire_samples")
DICT_SIZE = 131072

# Census last-token covariates are definitionally degenerate across the split
# (mean_act_when_active_lasttoken and span_last_ratio are 0 for every never-firing feature;
# lasttoken_count defines the split) — excluded from the AUC contrast.
DEFINITIONAL = ("lasttoken_count", "mean_act_when_active_lasttoken", "span_last_ratio")
PROMOTING_CLASS_NAMES = {0: "other", 1: "promoting", 2: "suppressing", 3: "partition"}

# Keyword/regex taxonomy over #1773 ANSWER-SIDE feature descriptions. Categories distilled
# from a seeded stratified read of 900 descriptions (300 top-firing by lasttoken_count /
# 300 random firing / 300 random never-firing); multi-label; case-insensitive.
TAXONOMY: dict[str, str] = {
    "boundary/transition markers": r"boundar|transition|mark(?:s|ing)? the (?:beginning|start|end)|beginning of (?:a |an |the )?new|new section|section break|structural (?:marker|element|boundar)|turn-taking|discourse boundar",
    "response/turn initiation": r"begin(?:s|ning)? (?:direct |conversational |new |a |an |the )*(?:respons|repl|answer)|start of (?:a |an |the )?(?:respons|repl|answer|assistant)|initiat\w+[^.]{0,40}respons|respons\w*[^.]{0,30}initiat|certainly|acknowledg|affirmative|greeting|salutation|turn in (?:dialogue|conversation)",
    "AI-assistant meta (refusal/disclaimer)": r"AI assistant|AI model|assistant'?s|chatbot|refus|disclaimer|apolog|declin\w+ (?:inappropriate|harmful|request)|ethical (?:boundar|guardrail|disclaim)|safety (?:warning|refusal|response)|meta-commentary|willingness to help|offer\w* (?:to )?(?:help|assist)",
    "list/enumeration structure": r"\blist\b|listing|enumerat|numbered|bullet|itemiz|step-by-step|list item|ordinal",
    "punctuation/whitespace/formatting": r"punctuat|whitespace|newline|line break|comma\b|colon\b|period\b|semicolon|quotation|bracket|parenthes|hyphen|dash\b|ellips|markdown|formatting (?:symbol|element|marker|delimiter)|delimiter",
    "code/programming syntax": r"\bcode\b|coding|programming|syntax|identifier|variable|function (?:name|signature|call)|method|API|HTML|XML|JSON|SQL|CSS|regex|compiler|keyword|import statement|namespace|stack trace|command[- ]line|source file|snippet|markup",
    "word-fragment/subword/morpheme": r"fragment|subword|morpheme|suffix|prefix|syllable|mid-word|word-internal|partial word|word piece|completes? (?:a |the )?word|camel[Cc]ase|snake_case|byte-pair",
    "grammar/function words": r"function word|grammatical|preposition|conjunction|article\b|pronoun|auxiliary|particle|determiner|copula|verb form|tense\b|inflect|morpholog",
    "numbers/math/quantitative": r"digit|numeric|number|math|equation|arithmetic|decimal|percentage|timestamp|date\b|dates\b|year\b|measurement|quantitat|statistic|calculat",
    "named entities/proper nouns": r"proper noun|named entit|brand|company|organization|place name|city|geograph|person(?:al)? name|surname",
    "non-English/multilingual": r"non-English|multilingual|Chinese|Russian|Spanish|French|German|Portuguese|Cyrillic|Arabic|Japanese|Korean|Italian|Vietnamese|foreign language|language boundar|code-switch|translat",
    "domain content (science/med/law/finance)": r"medical|clinical|pharma|chemi|biolog|scientific|legal|law\b|court|financ|econom|business|academic|scholar|engineer|physic",
    "dialogue/conversational/social": r"dialogue|conversation|chat\b|speaker|interlocutor|politeness|polite|second[- ]person|first[- ]person|user (?:message|request|input)|social media",
    "narrative/creative/emotional": r"narrative|story|fiction|creative|poem|poetry|role-?play|emotion|feeling|character(?:s)? (?:in|interact)",
    "instruction/procedural": r"instruction|procedural|tutorial|how[- ]to|step(?:s)? in|guide|recipe|advis|recommend|prescriptive",
}


def load_populations() -> dict:
    """Rebuild the committed complete-case mask + FIRING/NEVER split; hard-reconcile ns."""
    census = np.load(CENSUS)
    cov = np.load(COVARIATES)
    committed = json.loads(PARTIALS.read_text())
    assert np.array_equal(np.asarray(cov["feat_ids"], dtype=np.int64), np.arange(DICT_SIZE))
    assert np.array_equal(np.asarray(census["feat_ids"], dtype=np.int64), np.arange(DICT_SIZE))

    cols: dict[str, np.ndarray] = {}
    for name in sorted(cov.files):
        if name == "feat_ids":
            continue
        v = np.asarray(cov[name], dtype=np.float64)
        fin = np.isfinite(v)
        if (float(np.std(v[fin])) if fin.any() else 0.0) < 1e-6:
            continue  # dec_norm (zero-variance) — matches the committed dropped_columns
        cols[name] = v
    for name in DEFINITIONAL:
        cols[name] = np.asarray(census[name], dtype=np.float64)
    columns = sorted(cols)
    assert columns == committed["selection_columns"], "selection-column set drifted"

    u_w = np.asarray(np.load(READ_LADDER_W)["u"], dtype=np.float64)
    log_u = np.where(u_w > 0, np.log10(np.clip(u_w, 1e-300, None)), np.nan)
    cov_finite = np.isfinite(np.stack([cols[c] for c in columns])).all(axis=0)
    complete = cov_finite & np.isfinite(log_u)
    ltc = np.asarray(census["lasttoken_count"], dtype=np.float64)
    firing = complete & (ltc > 0)
    never = complete & (ltc == 0)

    pops = committed["populations"]
    recon = {
        "n_complete_case": (int(complete.sum()), pops["full"]["n"]),
        "n_firing": (int(firing.sum()), pops["train_active"]["n"]),
        "n_never": (int(never.sum()), pops["never_active"]["n"]),
    }
    for k, (mine, theirs) in recon.items():
        assert mine == theirs, f"{k}: recomputed {mine} != committed {theirs}"
    pre_mask_firing = int((ltc > 0).sum())
    dropped_firing = ltc > 0
    dropped_firing &= ~complete
    drop_reason = {
        "n_premask_firing": pre_mask_firing,
        "n_dropped_by_complete_case": int(dropped_firing.sum()),
        "dropped_logU_nonfinite": int((dropped_firing & ~np.isfinite(log_u)).sum()),
        "dropped_covariate_nonfinite": int((dropped_firing & ~cov_finite).sum()),
    }
    return {
        "cols": cols,
        "columns": columns,
        "complete": complete,
        "firing": firing,
        "never": never,
        "ltc": ltc,
        "reconciliation": {k: {"recomputed": v[0], "committed": v[1]} for k, v in recon.items()},
        "firing_premask_drop": drop_reason,
    }


def load_bank() -> tuple[dict[int, str], dict]:
    """feat_id -> answer-side description; negative sentinel ids dropped and counted."""
    desc: dict[int, str] = {}
    n_rows = 0
    n_neg = 0
    for shard in sorted(BANK_DIR.glob("descriptions_merged.shard*.jsonl")):
        with open(shard) as fh:
            for line in fh:
                d = json.loads(line)
                n_rows += 1
                fid = int(d["feat_id"])
                if fid < 0 or fid >= DICT_SIZE:
                    n_neg += 1
                    continue
                desc[fid] = d["description"]
    meta = {"bank_rows": n_rows, "bank_dropped_out_of_range_ids": n_neg, "bank_valid": len(desc)}
    return desc, meta


def mech_auc(pop: dict) -> tuple[list[dict], list[str]]:
    """Rank-AUC P(random firing > random never-firing) per continuous covariate, via rank-sum."""
    firing, never = pop["firing"], pop["never"]
    both = firing | never
    is_fire = firing[both]
    n1, n0 = int(is_fire.sum()), int((~is_fire).sum())
    rows = []
    contrast_cols = [c for c in pop["columns"] if c not in DEFINITIONAL and c != "promoting_class"]
    mat = np.stack([pop["cols"][c][both] for c in contrast_cols])  # (K, n) — complete-case: finite
    ranks = rankdata(mat, axis=1)  # average ties
    r1 = ranks[:, is_fire].sum(axis=1)
    auc = (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    for k, c in enumerate(contrast_cols):
        vf, vn = pop["cols"][c][firing], pop["cols"][c][never]
        rows.append(
            {
                "covariate": c,
                "auc_firing_gt_never": float(auc[k]),
                "firing_median": float(np.median(vf)),
                "firing_iqr": [float(np.percentile(vf, 25)), float(np.percentile(vf, 75))],
                "never_median": float(np.median(vn)),
                "never_iqr": [float(np.percentile(vn, 25)), float(np.percentile(vn, 75))],
            }
        )
    rows.sort(key=lambda r: -abs(r["auc_firing_gt_never"] - 0.5))
    return rows, contrast_cols


def promoting_shares(pop: dict) -> dict:
    pc = pop["cols"]["promoting_class"]
    out = {}
    for name, mask in (("firing", pop["firing"]), ("never", pop["never"])):
        v = pc[mask].astype(np.int64)
        out[name] = {
            PROMOTING_CLASS_NAMES[k]: round(float((v == k).mean()), 4)
            for k in sorted(PROMOTING_CLASS_NAMES)
        }
    return out


def decile_grading(pop: dict, top_cols: list[str]) -> dict:
    """Within FIRING: median of each top discriminator across lasttoken_count deciles."""
    firing = pop["firing"]
    ltc_f = pop["ltc"][firing]
    edges = np.unique(np.percentile(ltc_f, np.arange(0, 101, 10)))
    dec = np.clip(np.searchsorted(edges, ltc_f, side="right") - 1, 0, len(edges) - 2)
    out = {
        "lasttoken_count_decile_edges": edges.tolist(),
        "n_bins": int(len(edges) - 1),
        "per_bin_n": np.bincount(dec, minlength=len(edges) - 1).tolist(),
        "medians": {},
    }
    for c in top_cols:
        v = pop["cols"][c][firing]
        out["medians"][c] = {
            "never_firing_reference": float(np.median(pop["cols"][c][pop["never"]])),
            "per_decile": [float(np.median(v[dec == b])) for b in range(len(edges) - 1)],
        }
    return out


def _samples(pop: dict, desc: dict[int, str]) -> dict[str, np.ndarray]:
    """Deterministic stratified samples of LABELED features (300/300/300)."""
    rng = np.random.default_rng(SEED)
    labeled = np.zeros(DICT_SIZE, dtype=bool)
    labeled[list(desc.keys())] = True
    fire_lab = np.flatnonzero(pop["firing"] & labeled)
    never_lab = np.flatnonzero(pop["never"] & labeled)
    order = np.lexsort((fire_lab, -pop["ltc"][fire_lab]))
    top = fire_lab[order[:300]]
    rest = np.setdiff1d(fire_lab, top)
    rand_fire = rng.choice(rest, size=300, replace=False)
    rand_never = rng.choice(never_lab, size=300, replace=False)
    return {"top_firing": top, "random_firing": rand_fire, "random_never": rand_never}


def stage_samples(pop: dict, desc: dict[int, str], ctx_rows: list[dict]) -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    samples = _samples(pop, desc)
    for name, ids in samples.items():
        with open(SAMPLE_DIR / f"{name}.txt", "w") as fh:
            for fid in ids:
                fh.write(f"{int(fid)}\t{int(pop['ltc'][fid])}\t{desc[int(fid)][:220]}\n")
    # Context-side (#1482 instrument): ALL of them are outside the complete-case set (their
    # answer-side conditional covariates are NaN by construction), so sample by the PRE-MASK
    # last-token count: every row with lasttoken_count > 0, plus a random contrast sample.
    ctx_fire = [r for r in ctx_rows if pop["ltc"][r["feat_id"]] > 0]
    ctx_rest = [r for r in ctx_rows if pop["ltc"][r["feat_id"]] == 0]
    rng = np.random.default_rng(SEED + 1)
    keep = rng.permutation(len(ctx_rest))[: 100 - len(ctx_fire)]
    with open(SAMPLE_DIR / "context_only_sample.txt", "w") as fh:
        for r in ctx_fire:
            fh.write(
                f"FIRES\t{r['feat_id']}\t{int(pop['ltc'][r['feat_id']])}\t{r['description'][:220]}\n"
            )
        for i in sorted(keep):
            r = ctx_rest[i]
            fh.write(f"never\t{r['feat_id']}\t0\t{r['description'][:220]}\n")
    print(
        f"samples written to {SAMPLE_DIR} (context-side: {len(ctx_fire)} fire pre-mask, "
        f"{100 - len(ctx_fire)} never-firing sampled)"
    )


def taxonomy_shares(pop: dict, desc: dict[int, str], n_draws: int = 1000) -> dict:
    """Per-category share firing vs never among LABELED features + label-shuffle null band."""
    labeled = np.zeros(DICT_SIZE, dtype=bool)
    labeled[list(desc.keys())] = True
    fire_lab = np.flatnonzero(pop["firing"] & labeled)
    never_lab = np.flatnonzero(pop["never"] & labeled)
    pool = np.concatenate([fire_lab, never_lab])
    n_f = len(fire_lab)
    cats = list(TAXONOMY)
    pats = [re.compile(TAXONOMY[c], re.IGNORECASE) for c in cats]
    catmat = np.zeros((len(pool), len(cats)), dtype=np.float64)
    n_other = 0
    for i, fid in enumerate(pool):
        d = desc[int(fid)]
        hit = False
        for k, p in enumerate(pats):
            if p.search(d):
                catmat[i, k] = 1.0
                hit = True
        n_other += not hit
    obs_f = catmat[:n_f].mean(axis=0)
    obs_n = catmat[n_f:].mean(axis=0)
    obs_diff = obs_f - obs_n

    rng = np.random.default_rng(SEED + 7)
    null_diff = np.empty((n_draws, len(cats)))
    for t in range(n_draws):
        perm = rng.permutation(len(pool))
        sel = catmat[perm[:n_f]].mean(axis=0)
        null_diff[t] = sel - (catmat.sum(axis=0) - catmat[perm[:n_f]].sum(axis=0)) / (
            len(pool) - n_f
        )
    band_lo = np.percentile(null_diff, 2.5, axis=0)
    band_hi = np.percentile(null_diff, 97.5, axis=0)
    family_band = float(np.percentile(np.abs(null_diff).max(axis=1), 97.5))

    # Unlabeled share: shuffle over ALL complete-case features.
    unl = ~labeled
    both = pop["firing"] | pop["never"]
    idx = np.flatnonzero(both)
    is_f = pop["firing"][idx]
    u = unl[idx].astype(np.float64)
    obs_unl = float(u[is_f].mean() - u[~is_f].mean())
    nf_all = int(is_f.sum())
    null_unl = np.empty(n_draws)
    for t in range(n_draws):
        perm = rng.permutation(len(idx))
        s = u[perm[:nf_all]].mean()
        null_unl[t] = s - (u.sum() - u[perm[:nf_all]].sum()) / (len(idx) - nf_all)
    rows = []
    for k, c in enumerate(cats):
        rows.append(
            {
                "category": c,
                "firing_share": float(obs_f[k]),
                "never_share": float(obs_n[k]),
                "diff": float(obs_diff[k]),
                "null_band_2p5_97p5": [float(band_lo[k]), float(band_hi[k])],
                "clears_percat_band": bool(obs_diff[k] < band_lo[k] or obs_diff[k] > band_hi[k]),
                "clears_family_band": bool(abs(obs_diff[k]) > family_band),
            }
        )
    rows.sort(key=lambda r: -abs(r["diff"]))
    return {
        "denominators": {"firing_labeled": n_f, "never_labeled": int(len(pool) - n_f)},
        "share_matched_no_category": round(n_other / len(pool), 4),
        "family_band_p97_5_of_max_abs_diff": family_band,
        "categories": rows,
        "unlabeled_row": {
            "firing_unlabeled_share": float(u[is_f].mean()),
            "never_unlabeled_share": float(u[~is_f].mean()),
            "diff": obs_unl,
            "null_band_2p5_97p5": [
                float(np.percentile(null_unl, 2.5)),
                float(np.percentile(null_unl, 97.5)),
            ],
        },
        "n_draws": n_draws,
    }


def context_only(pop: dict, ctx_rows: list[dict]) -> dict:
    ids = np.array([r["feat_id"] for r in ctx_rows], dtype=np.int64)
    ltc = pop["ltc"][ids]
    return {
        "n_context_side_labeled": int(len(ids)),
        "in_firing": int(pop["firing"][ids].sum()),
        "in_never": int(pop["never"][ids].sum()),
        "outside_complete_case": int((~pop["complete"][ids]).sum()),
        "premask_lasttoken_count_gt0": int((ltc > 0).sum()),
        "premask_lasttoken_count_max": float(ltc.max()),
        "nonfinite_covariates_all_1653": ["consistency", "mean_act_cond"],
        "evidence_side_caveat": "context-side descriptions (#1482 instrument) — never pooled with the #1773 answer-side taxonomy",
    }


def make_figures(pop: dict, auc_rows: list[dict], tax: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    c_fire, c_never = "#4477AA", "#EE6677"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # (i) AUC contrast bars.
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    names = [r["covariate"] for r in auc_rows][::-1]
    vals = [r["auc_firing_gt_never"] - 0.5 for r in auc_rows][::-1]
    colors = [c_fire if v > 0 else c_never for v in vals]
    ax.barh(np.arange(len(names)), vals, color=colors)
    ax.set_yticks(np.arange(len(names)), names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("rank AUC − 0.5  (blue: higher for firing, red: higher for never-firing)")
    ax.set_title("How firing vs never-firing features differ, per feature property")
    savefig_paper(fig, "fire_census_auc_contrast", dir=FIG_DIR)
    plt.close(fig)

    # (ii) overlaid distributions for the top 4 discriminators.
    top4 = [r["covariate"] for r in auc_rows[:4]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, c in zip(axes.ravel(), top4):
        vf, vn = pop["cols"][c][pop["firing"]], pop["cols"][c][pop["never"]]
        logx = (
            vf.min() > 0
            and vn.min() > 0
            and (
                np.percentile(np.concatenate([vf, vn]), 99)
                / max(np.percentile(np.concatenate([vf, vn]), 1), 1e-12)
                > 50
            )
        )
        xf, xn, lab = (np.log10(vf), np.log10(vn), f"log10({c})") if logx else (vf, vn, c)
        bins = np.linspace(
            min(xf.min(), xn.min()), max(np.percentile(xf, 99.5), np.percentile(xn, 99.5)), 60
        )
        ax.hist(
            xn,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            color=c_never,
            label="never-firing",
        )
        ax.hist(
            xf,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            color=c_fire,
            label="firing",
        )
        ax.set_xlabel(lab)
        ax.set_ylabel("density")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Top discriminating feature properties: firing vs never-firing at the context vector"
    )
    savefig_paper(fig, "fire_census_top_discriminators", dir=FIG_DIR)
    plt.close(fig)

    # (iii) taxonomy shares + difference with null band.
    rows = tax["categories"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    y = np.arange(len(rows))[::-1]
    axes[0].barh(
        y + 0.2, [r["firing_share"] for r in rows], height=0.38, color=c_fire, label="firing"
    )
    axes[0].barh(
        y - 0.2, [r["never_share"] for r in rows], height=0.38, color=c_never, label="never-firing"
    )
    axes[0].set_yticks(y, [r["category"] for r in rows], fontsize=9)
    axes[0].set_xlabel("share of labeled features in category")
    axes[0].legend(frameon=False)
    diffs = [r["diff"] for r in rows]
    lo = [r["diff"] - r["null_band_2p5_97p5"][0] for r in rows]
    hi = [r["null_band_2p5_97p5"][1] - r["diff"] for r in rows]
    axes[1].barh(y, diffs, height=0.55, color="#777777")
    band_lo = [r["null_band_2p5_97p5"][0] for r in rows]
    band_hi = [r["null_band_2p5_97p5"][1] for r in rows]
    axes[1].barh(
        y,
        np.array(band_hi) - np.array(band_lo),
        left=band_lo,
        height=0.9,
        color="#BBBBBB",
        alpha=0.5,
        label="label-shuffle null band (95%)",
    )
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("share difference (firing − never-firing)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Answer-side description categories: firing vs never-firing features")
    savefig_paper(fig, "fire_census_category_shares", dir=FIG_DIR)
    plt.close(fig)
    _ = (lo, hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["samples", "full"])
    args = ap.parse_args()

    pop = load_populations()
    desc, bank_meta = load_bank()
    ctx_rows = [json.loads(x) for x in open(CONTEXT_LABELS)]

    if args.stage == "samples":
        print(json.dumps(pop["reconciliation"]))
        print(json.dumps(pop["firing_premask_drop"]))
        print(json.dumps(bank_meta))
        stage_samples(pop, desc, ctx_rows)
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    auc_rows, contrast_cols = mech_auc(pop)
    top6 = [r["covariate"] for r in auc_rows[:6]]
    deciles = decile_grading(pop, top6)
    tax = taxonomy_shares(pop, desc)
    ctx = context_only(pop, ctx_rows)

    labeled = np.zeros(DICT_SIZE, dtype=bool)
    labeled[list(desc.keys())] = True
    coverage = {
        name: {
            "n": int(m.sum()),
            "labeled": int((m & labeled).sum()),
            "coverage": round(float((m & labeled).sum() / m.sum()), 4),
        }
        for name, m in (
            ("firing", pop["firing"]),
            ("never", pop["never"]),
            ("complete_case", pop["complete"]),
        )
    }

    make_figures(pop, auc_rows, tax)

    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    summary = {
        "meta": {
            **as_metadata_dict(git_provenance(REPO)),
            "seed": SEED,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": [
                str(p.relative_to(REPO))
                for p in (CENSUS, PARTIALS, READ_LADDER_W, COVARIATES, CONTEXT_LABELS)
            ]
            + [str(BANK_DIR)],
        },
        "populations": pop["reconciliation"],
        "firing_premask_drop": pop["firing_premask_drop"],
        "bank": bank_meta,
        "label_join_coverage": coverage,
        "promoting_class_shares": promoting_shares(pop),
        "decile_grading_within_firing": deciles,
        "context_only_subpopulation": ctx,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT_DIR / "auc_table.json").write_text(
        json.dumps({"contrast_columns": contrast_cols, "rows": auc_rows}, indent=1)
    )
    (OUT_DIR / "category_counts.json").write_text(json.dumps(tax, indent=1))
    print(
        json.dumps(
            {
                "auc_top6": {
                    r["covariate"]: round(r["auc_firing_gt_never"], 3) for r in auc_rows[:6]
                },
                "coverage": coverage,
                "context_only": ctx,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
