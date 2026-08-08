"""Task #1776 dose-round (followup p3p4) analyzer supplement.

Recomputes, on the P4 dose-escalated steering round, the same reads the parent
analyzer supplement (`scripts/issue1776_analyzer_supplement.py`) computed for
the parent alpha<=4 steering leg:

  (a) steering noise — alpha=0 within-baseline pseudo-shift floor + per-stratum
      split-half reliability of the per-cell answer shift (2-vs-2 draw split);
  (b) judged shifts — MATCHED-context steered-minus-baseline deltas per
      (trait rubric, stratum) with a context-clustered bootstrap CI (the
      committed `judge_scores.json` `steered_minus_baseline` subtracts the
      150-context baseline mean from the 50-context control strata; here every
      delta is per-context matched, controls included);
  (c) H2 operator contrast — per (direction, alpha): mean cos(shift, J pred) vs
      cos(shift, M' pred) with the context-clustered paired-difference CI, plus
      magnitude ratios (same estimator as the parent supplement block (c));
  (d) language-intrusion audit — per-arm CJK counts over the dose judged pools
      (raw steered_dose rollouts joined with judge per_cell scores) + zeroed /
      excluded recounts of the matched judged deltas (pure counting; no row
      text enters any output).

Inputs (all committed / HF-mirrored):
  eval_results/issue_1776/followup_p3p4/judge/judge_scores.json   (per_cell)
  data/issue_1776/hf_dl/p3p4_fold/issue1776_jacobian/analysis_tensors/followup_p3p4/
      dose_summaries/*.pt   dose_cells/*.jsonl
  data/issue_1776/hf_dl/p3p4_fold/issue1776_jacobian/raw_completions/steered_dose/*.json

Output: eval_results/issue_1776/followup_p3p4/dose_supplement.json
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps BEFORE numpy/torch import (#847 gate)

import numpy as np  # noqa: E402
import torch  # noqa: E402

BASE = Path("data/issue_1776/hf_dl/p3p4_fold/issue1776_jacobian")
TENS = BASE / "analysis_tensors/followup_p3p4"
RAW = BASE / "raw_completions/steered_dose"
JUDGE = Path("eval_results/issue_1776/followup_p3p4/judge/judge_scores.json")
OUT = Path("eval_results/issue_1776/followup_p3p4/dose_supplement.json")

CJK = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")
ALPHAS = ("11.8401", "23.6803", "47.3606")
DIRS = ("evil", "sycophancy", "hallucination", "w1_mprime", "random")
RNG = np.random.default_rng(42)

out: dict = {}

# ---------------------------------------------------------- (a) steering noise
bl = torch.load(TENS / "dose_summaries/baseline_a0.pt", map_location="cpu", weights_only=False)
blv = bl["v19"]
assert isinstance(blv, dict) and len(blv) == 150, f"baseline v19 shape unexpected: {len(blv)}"
scale = float(np.sqrt(0.4 / (0.5 + 1 / 3)))  # mean5-mean5 vs 2-vs-3 split variance (parent)
pseudo = np.array(
    [float((t[:2].mean(0) - t[2:].mean(0)).norm()) for t in blv.values() if t.shape[0] == 5]
)
rel = {}
for d in DIRS:
    for a in ALPHAS:
        st = f"{d}_a{a}"
        sv = torch.load(TENS / f"dose_summaries/{st}.pt", map_location="cpu", weights_only=False)[
            "v19"
        ]
        cos, dvn = [], []
        for cid, t in sv.items():
            b5 = blv.get(cid)
            if b5 is None or t.shape[0] < 4 or b5.shape[0] < 4:
                continue
            d1 = t[:2].mean(0) - b5[:2].mean(0)
            d2 = t[2:4].mean(0) - b5[2:4].mean(0)
            cos.append(float(torch.nn.functional.cosine_similarity(d1, d2, dim=0)))
            dvn.append(float((t.mean(0) - b5.mean(0)).norm()))
        rel[st] = {
            "n_cells": len(cos),
            "splithalf_dv_cos_median": float(np.median(cos)),
            "splithalf_dv_cos_q10": float(np.quantile(cos, 0.10)),
            "splithalf_dv_cos_q90": float(np.quantile(cos, 0.90)),
            "splithalf_dv_cos": cos,
            "dv_norm_mean": float(np.mean(dvn)),
            "dv_norm": dvn,
        }
out["steering_noise"] = {
    "alpha0_pseudo_shift_norm_median": float(np.median(pseudo)),
    "scaled_noise_floor_median": float(np.median(pseudo) * scale),
    "scale_factor_2v3_to_5v5": scale,
    "common_seed_caveat": (
        "steered and baseline draws share seed_base=42 per (context, sample); the "
        "independent-draw floor OVERestimates steered-minus-baseline noise, so the "
        "split-half reliability read is the binding one (parent convention)"
    ),
    "per_cell_reliability": rel,
}

# ------------------------------------------------- (b) matched judged deltas
jd = json.loads(JUDGE.read_text())
per_cell = jd["per_cell"]
base_mean: dict[tuple[str, str], float] = {}
for r in per_cell:
    if r["stratum"] == "baseline_a0":
        base_mean[(r["trait"], r["context_id"])] = float(r["cell_mean"])
judged = {}
for trait in jd["traits"]:
    judged[trait] = {}
    strata = sorted({r["stratum"] for r in per_cell if r["trait"] == trait})
    for st in strata:
        if st == "baseline_a0":
            continue
        rows = [r for r in per_cell if r["trait"] == trait and r["stratum"] == st]
        deltas, cids = [], []
        for r in rows:
            b = base_mean.get((trait, r["context_id"]))
            assert b is not None, f"no matched baseline for {trait}/{r['context_id']}"
            deltas.append(float(r["cell_mean"]) - b)
            cids.append(r["context_id"])
        deltas = np.array(deltas)
        boots = []
        n = len(deltas)
        for _ in range(1000):
            take = RNG.choice(n, n, replace=True)  # context == cell here (1 cell/context)
            boots.append(float(deltas[take].mean()))
        judged[trait][st] = {
            "matched_delta_mean": float(deltas.mean()),
            "delta_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_contexts": n,
            "per_context_delta": [round(float(x), 4) for x in deltas],
            "context_ids": cids,
        }
out["judged_matched_deltas"] = {
    "note": (
        "per-context matched steered-minus-baseline deltas of judge cell means "
        "(0-100), context-level bootstrap CI (1000 resamples); controls matched to "
        "their own 50-context baseline cells under the same rubric — unlike the "
        "committed steered_minus_baseline block, which subtracts the 150-context "
        "baseline mean"
    ),
    "per_trait": judged,
}

# --------------------------------------------------------------- (c) H2 boot
cells = []
for p in sorted(glob.glob(str(TENS / "dose_cells/*.jsonl"))):
    if "baseline" in p:
        continue
    for line in open(p):
        r = json.loads(line)
        if r.get("cos_pred_jlast") is not None:
            cells.append(r)
h2 = {}
for d in DIRS:
    for a in ALPHAS:
        sub = [r for r in cells if r["direction"] == d and r["alpha"] == float(a)]
        if not sub:
            continue
        ctxs: dict[str, list] = {}
        for r in sub:
            ctxs.setdefault(r["context_id"], []).append(r)
        ids = list(ctxs.keys())
        diffs, mj, mm = [], [], []
        for _ in range(1000):
            take = RNG.choice(len(ids), len(ids), replace=True)
            rows = [r for i in take for r in ctxs[ids[i]]]
            jv = float(np.mean([r["cos_pred_jlast"] for r in rows]))
            mv = float(np.mean([r["cos_pred_mprime"] for r in rows]))
            diffs.append(jv - mv)
            mj.append(jv)
            mm.append(mv)
        h2[f"{d}_a{a}"] = {
            "mean_cos_jlast": float(np.mean([r["cos_pred_jlast"] for r in sub])),
            "mean_cos_mprime": float(np.mean([r["cos_pred_mprime"] for r in sub])),
            "cos_jlast_ci95": [float(np.percentile(mj, 2.5)), float(np.percentile(mj, 97.5))],
            "cos_mprime_ci95": [float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))],
            "paired_diff_ci95": [
                float(np.percentile(diffs, 2.5)),
                float(np.percentile(diffs, 97.5)),
            ],
            "median_mag_ratio_jlast": float(np.median([r["mag_ratio_jlast"] for r in sub]))
            if "mag_ratio_jlast" in sub[0]
            else None,
            "median_mag_ratio_mprime": float(np.median([r["mag_ratio_mprime"] for r in sub]))
            if "mag_ratio_mprime" in sub[0]
            else None,
            "n_cells": len(sub),
            "n_contexts": len(ids),
        }
out["h2_clustered_bootstrap"] = {
    "note": "context-clustered bootstrap (1000 resamples) over dose prefill cells",
    "per_stratum": h2,
}

# -------------------------------------------------- (d) CJK intrusion audit
score_by = {}
for r in per_cell:
    score_by.setdefault((r["trait"], r["stratum"], r["context_id"]), float(r["cell_mean"]))
intr = {}
for f in sorted(RAW.glob("*.json")):
    raw = json.loads(f.read_text())
    st = raw["stratum"]
    n_tot, n_intr, intr_cids = 0, 0, set()
    for c in raw["contexts"]:
        for s in c["samples"]:
            n_tot += 1
            if CJK.search(s):
                n_intr += 1
                intr_cids.add(c["context_id"])
    intr[st] = {
        "intruded_rows": n_intr,
        "total_rows": n_tot,
        "frac": round(n_intr / max(n_tot, 1), 4),
        "n_intruded_contexts": len(intr_cids),
        "intruded_context_ids": sorted(intr_cids),
    }
# recounts of the matched judged deltas excluding intruded contexts (per trait
# rubric x stratum; a context is excluded when EITHER its steered or baseline
# pool at that stratum carries an intruded row)
base_intr = set(intr["baseline_a0"]["intruded_context_ids"])
recounts = {}
for trait, sts in judged.items():
    recounts[trait] = {}
    for st, blk in sts.items():
        bad = set(intr.get(st, {}).get("intruded_context_ids", [])) | base_intr
        keep = [d for d, c in zip(blk["per_context_delta"], blk["context_ids"]) if c not in bad]
        recounts[trait][st] = {
            "matched_delta_mean_excluded": float(np.mean(keep)) if keep else None,
            "n_kept": len(keep),
            "n_excluded": blk["n_contexts"] - len(keep),
        }
out["language_intrusion"] = {
    "regex": CJK.pattern,
    "per_stratum": {
        k: {kk: vv for kk, vv in v.items() if kk != "intruded_context_ids"} for k, v in intr.items()
    },
    "excluded_recounts": recounts,
    "note": (
        "row = one sampled completion; a context is excluded from the recount when "
        "either its steered or its baseline pool carries an intruded row; counts "
        "only — no completion text is persisted here"
    ),
}

OUT.write_text(json.dumps(out, indent=1))
print("wrote", OUT)
print(
    "pseudo floor median",
    out["steering_noise"]["alpha0_pseudo_shift_norm_median"],
    "scaled",
    out["steering_noise"]["scaled_noise_floor_median"],
)
