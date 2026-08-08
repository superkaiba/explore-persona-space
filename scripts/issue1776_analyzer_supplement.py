"""Analyzer supplementary reads for #1776 (VM CPU, re-reductions only).

Computes, from persisted HF artifacts + committed eval JSONs:
  (a) full-rank J_last split-half row cosine + cross-half prediction agreement
      (attenuation ceiling on R2_J) + amplitude-deficit read on the 1,536
      J-fit pairs (in-sample for J);
  (b) steered-shift noise floor: alpha=0 within-baseline pseudo-shift
      (2-vs-3 draw split, scaled to the mean5-vs-mean5 estimator) +
      per-cell split-half delta-v reliability for evil_a4 / random_a4;
  (c) H2 context-clustered bootstrap per direction (mean cos J_last vs
      M-prime over steered prefill cells; 1,000 resamples) + mag ratios;
  (d) WildChat-leg per-context squared errors for the fitted maps + J_last
      (per-unit companion of the transfer aggregate), validated against the
      committed transfer.json aggregates;
  (e) matched-context all-positions judge shifts + CJK-intrusion recounts
      (zeroed / excluded), prompt-language conditioned.

Outputs: eval_results/issue_1776/analyzer/supplement.json (+ per-unit
arrays inside). No GPU, no fits — single GEMMs on <= 1536x3584 matrices.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # bind shared-VM thread caps BEFORE numpy/torch import (#847)

import numpy as np  # noqa: E402
import torch  # noqa: E402

B = Path("data/issue_1776/hf_dl/issue1776_jacobian")
OUT = Path("eval_results/issue_1776/analyzer")
OUT.mkdir(parents=True, exist_ok=True)
CJK = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")

out: dict = {}

# ---------------------------------------------------------------- (a) J noise
J = torch.load(B / "analysis_tensors/jac_full/J_last.pt", map_location="cpu", weights_only=False)
A14 = torch.load(B / "analysis_tensors/jpairs/acts14.pt", map_location="cpu", weights_only=False)
A19 = torch.load(B / "analysis_tensors/jpairs/acts19.pt", map_location="cpu", weights_only=False)
A14 = (A14 if torch.is_tensor(A14) else list(A14.values())[0]).float()
A19 = (A19 if torch.is_tensor(A19) else list(A19.values())[0]).float()
halves = [hs.float() / hc.float().unsqueeze(1) for hs, hc in zip(J["half_sums"], J["half_counts"])]
vb = [v.float() for v in J["v_bar_half"]]
cb = [c.float() for c in J["c_bar_half"]]
Jm = J["J"].float()
row_cos = torch.nn.functional.cosine_similarity(halves[0], halves[1], dim=1)
p1 = (A14 - cb[0]) @ halves[0].T
p2 = (A14 - cb[1]) @ halves[1].T
p1c, p2c = p1 - p1.mean(0), p2 - p2.mean(0)
pooled_agree = float((p1c * p2c).sum() / (p1c.norm() * p2c.norm()))
jterm = (A14 - (cb[0] + cb[1]) / 2) @ Jm.T
resid = A19 - A19.mean(0)
cos_align = torch.nn.functional.cosine_similarity(jterm, resid, dim=1)
out["j_convergence"] = {
    "m_pairs_per_seed": 150,
    "splithalf_row_cos_median": float(row_cos.median()),
    "splithalf_row_cos_q10": float(row_cos.quantile(0.10)),
    "splithalf_row_cos_q90": float(row_cos.quantile(0.90)),
    "crosshalf_prediction_agreement_pooled_cos": pooled_agree,
    "jterm_norm_median_insample": float(jterm.norm(dim=1).median()),
    "resid_norm_median_insample": float(resid.norm(dim=1).median()),
    "cos_jterm_resid_median_insample": float(cos_align.median()),
    "note": (
        "in-sample on the 1,536 J-fit pairs; jterm = J_last @ (c14 - cbar); "
        "amplitude deficit = resid/jterm norm ratio; optimal-rescale R2 bound "
        "= median cos^2"
    ),
    "amplitude_deficit_median": float((resid.norm(dim=1) / jterm.norm(dim=1)).median()),
}

# ------------------------------------------------------- (b) steering noise
bl = torch.load(
    B / "analysis_tensors/phase3/baseline_a0.pt", map_location="cpu", weights_only=False
)["v19"]
scale = float(np.sqrt(0.4 / (0.5 + 1 / 3)))  # mean5-mean5 vs 2-vs-3 split variance
pseudo = np.array(
    [float((t[:2].mean(0) - t[2:].mean(0)).norm()) for t in bl.values() if t.shape[0] == 5]
)
rel = {}
for st in ("evil_a4", "random_a4"):
    sv = torch.load(B / f"analysis_tensors/phase3/{st}.pt", map_location="cpu", weights_only=False)[
        "v19"
    ]
    cos, dv = [], []
    for cid, t in sv.items():
        b5 = bl.get(cid)
        if b5 is None or t.shape[0] < 4 or b5.shape[0] < 4:
            continue
        d1 = t[:2].mean(0) - b5[:2].mean(0)
        d2 = t[2:4].mean(0) - b5[2:4].mean(0)
        cos.append(float(torch.nn.functional.cosine_similarity(d1, d2, dim=0)))
        dv.append(float((t.mean(0) - b5.mean(0)).norm()))
    rel[st] = {
        "splithalf_dv_cos": cos,
        "dv_norm": dv,
        "splithalf_dv_cos_median": float(np.median(cos)),
    }
out["steering_noise"] = {
    "alpha0_pseudo_shift_norm_median": float(np.median(pseudo)),
    "scaled_noise_floor_median": float(np.median(pseudo) * scale),
    "scale_factor_2v3_to_5v5": scale,
    "common_seed_caveat": (
        "steered and baseline draws share seed_base=42 per (context, sample); "
        "common random numbers make the independent-draw floor an OVERestimate "
        "of steered-minus-baseline noise, so the split-half reliability read "
        "is the binding one"
    ),
    "per_cell_reliability": rel,
}

# --------------------------------------------------------------- (c) H2 boot
rng = np.random.default_rng(42)
cells = []
for p in sorted(glob.glob(str(B / "analysis_tensors/phase3/cells/*.jsonl"))):
    if "baseline" in p or "allpos" in p:
        continue
    for line in open(p):
        r = json.loads(line)
        if r.get("cos_pred_jlast") is not None:
            cells.append(r)
h2 = {}
for d in sorted({r["direction"] for r in cells}):
    sub = [r for r in cells if r["direction"] == d]
    ctxs: dict[str, list] = {}
    for r in sub:
        ctxs.setdefault(r["context_id"], []).append(r)
    ids = list(ctxs.keys())
    diffs = []
    for _ in range(1000):
        take = rng.choice(len(ids), len(ids), replace=True)
        rows = [r for i in take for r in ctxs[ids[i]]]
        diffs.append(
            float(
                np.mean([r["cos_pred_jlast"] for r in rows])
                - np.mean([r["cos_pred_mprime"] for r in rows])
            )
        )
    h2[d] = {
        "mean_cos_jlast": float(np.mean([r["cos_pred_jlast"] for r in sub])),
        "mean_cos_mprime": float(np.mean([r["cos_pred_mprime"] for r in sub])),
        "paired_diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "median_mag_ratio_jlast": float(np.median([r["mag_ratio_jlast"] for r in sub])),
        "median_mag_ratio_mprime": float(np.median([r["mag_ratio_mprime"] for r in sub])),
        "n_cells": len(sub),
        "n_contexts": len(ids),
    }
out["h2_clustered_bootstrap"] = {
    "note": "context-clustered bootstrap (1000 resamples), steered prefill cells only",
    "per_direction": h2,
}

# ------------------------------------------------------ (d) wildchat per-ctx
shards = sorted(glob.glob(str(B / "wildchat_fresh/final_token_capture/shard*.pt")))
cx, vx = [], []
for p in shards:
    dd = torch.load(p, map_location="cpu", weights_only=False)
    cx.append(dd["cx_last"])
    vx.append(dd["v_x"])
layers = dd["layers"]
CX = torch.cat(cx).double()
VX = torch.cat(vx).double()
c14 = CX[:, layers.index(14), :]
c19 = CX[:, layers.index(19), :]
y = VX[:, layers.index(19), :]


def apply_ridge(path: Path, x: torch.Tensor) -> torch.Tensor:
    pl = torch.load(path, map_location="cpu", weights_only=False)
    return ((x - pl["xmu"].double()) / pl["xsd"].double()) @ pl["W"].double() + pl["ymu"].double()


preds = {
    "m_ridge_x50k": apply_ridge(B / "analysis_tensors/comparator/m_ridge_x50k.pt", c14),
    "m_ridge_lmsys50k": apply_ridge(B / "analysis_tensors/comparator/m_ridge_lmsys50k.pt", c14),
    "m_shipped": apply_ridge(
        Path("data/issue_1776/hf_dl/issue779_monitoring/n1m_readout/weights/L19/ridge.pt"), c19
    ),
}
# J affine: intercepts approximated by jpairs cbar + the x50k ridge ymu
ymu19 = torch.load(
    B / "analysis_tensors/comparator/m_ridge_x50k.pt", map_location="cpu", weights_only=False
)["ymu"].double()
cbar = ((cb[0] + cb[1]) / 2).double()
preds["J_last"] = (c14 - cbar) @ Jm.double().T + ymu19
ss_tot = float(((y - y.mean(0)) ** 2).sum())
perctx = {}
val = {}
for name, p in preds.items():
    se = ((y - p) ** 2).sum(1)
    perctx[name] = [float(v) for v in se]
    val[name] = 1.0 - float(se.sum()) / ss_tot
out["wildchat_per_context"] = {
    "n": int(y.shape[0]),
    "recomputed_r2": val,
    "committed_r2_reference": {
        "m_ridge_x50k": 0.6611,
        "m_ridge_lmsys50k": 0.6729,
        "m_shipped": 0.7335,
        "J_last": -0.3583,
    },
    "note": (
        "per-context squared error, this analyzer's re-application of persisted "
        "weights; J_last intercepts approximated by jpairs c-bar + x50k ridge ymu "
        "(the pipeline used n1m LMSYS train-pool anchors), so its recomputed R2 is "
        "an approximation of the committed row"
    ),
    "per_context_se": perctx,
}

# ----------------------------------------------------- (e) allpos recounts
j = json.load(open("eval_results/issue_1776/phase3/judge/judge_scores.json"))
cells_j = {(r["trait"], r["stratum"], r["context_id"]): r for r in j["per_cell"]}
RAW = Path("data/issue_1776/hf_dl/raw_completions/steered")


def _flags(st: str):
    d = json.load(open(RAW / f"{st}.json"))
    return {
        c["context_id"]: ([bool(CJK.search(s)) for s in c["samples"]], bool(CJK.search(c["user"])))
        for c in d["contexts"]
    }, [c["context_id"] for c in d["contexts"]]


base_flags, _ = _flags("baseline_a0")
allpos = {}
for tr, st in [
    ("evil", "evil_a4_allpos"),
    ("sycophancy", "sycophancy_a4_allpos"),
    ("hallucination", "hallucination_a4_allpos"),
]:
    f_st, ids = _flags(st)

    def _stats(fmap, stratum):
        tot = n = 0
        tot_ex = n_ex = 0
        tot_z = 0.0
        intr = ntot = 0
        for cid in ids:
            r = cells_j.get((tr, stratum, cid))
            if not r:
                continue
            fl = fmap.get(cid, ([False] * 5, False))[0]
            for k, sc in zip(r["sample_idx"], r["sample_scores"]):
                if sc is None:
                    continue
                hit = fl[k] if k < len(fl) else False
                tot += sc
                n += 1
                intr += hit
                ntot += 1
                tot_z += 0.0 if hit else sc
                if not hit:
                    tot_ex += sc
                    n_ex += 1
        return tot / max(n, 1), tot_ex / max(n_ex, 1), tot_z / max(n, 1), intr, ntot

    s_raw, s_ex, s_z, s_i, s_n = _stats(f_st, st)
    b_raw, b_ex, b_z, b_i, b_n = _stats(base_flags, "baseline_a0")
    allpos[st] = {
        "matched_context_shift_raw": s_raw - b_raw,
        "matched_context_shift_excluded_intrusion": s_ex - b_ex,
        "matched_context_shift_zeroed_intrusion": s_z - b_z,
        "steered_intruded_over_total": [s_i, s_n],
        "baseline_intruded_over_total": [b_i, b_n],
        "committed_smb_unmatched": j["steered_minus_baseline"][tr].get(st),
        "note": "committed steered_minus_baseline used the 200-context baseline mean; these use the same 50 contexts",
    }
out["allpos_matched_recounts"] = allpos

# language-intrusion audit summary (prompt-language conditioned, all strata)
audit = {}
for p in sorted(glob.glob(str(RAW / "*.json"))):
    d = json.load(open(p))
    st = d.get("stratum") or Path(p).stem
    n = i = nn = ii = 0
    for c in d["contexts"]:
        p_cjk = bool(CJK.search(c["user"]))
        for s in c["samples"]:
            hit = bool(CJK.search(s))
            n += 1
            i += hit
            if not p_cjk:
                nn += 1
                ii += hit
    audit[st] = {"intruded": i, "total": n, "noncjk_prompt_intruded": ii, "noncjk_prompt_total": nn}
out["cjk_audit"] = audit

with open(OUT / "supplement.json", "w") as f:
    json.dump(out, f, indent=1)
print("wrote", OUT / "supplement.json")
print(json.dumps({k: (v if not isinstance(v, dict) else "...") for k, v in out.items()}))
print("wildchat recomputed r2:", val)
