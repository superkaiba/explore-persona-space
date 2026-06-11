"""Issue #537 analyzer: registered hypothesis reads from the shipped G tensor.

Computes (plan v6 §3 + §7 success criteria; all zero-GPU, tensor/meta/G_cells only):

- H-structure (success criterion 2 / kill criterion 1; marker row): between-eval-context
  variance of G[i, ->] per train row (off-diagonal), averaged over rows, vs the mean
  per-cell question-level bootstrap noise variance. Pass: (V_between - V_noise) >= 2*V_noise.
  Split-half cross-check uses noise_var_split_half from G_meta.
- H-inoculation (success criterion 4): mean off-diagonal G of the behavior-instruction
  (binst) train row vs the default train row, raw + diagonal-normalized, per behavior.
- H-asymmetry: antisymmetric fraction of off-diagonal variance on the 16x16
  shared-instance block, raw per behavior; question-split corrected for the marker row
  (same K random half-splits of the 32 eval questions applied to every cell;
  cross-half covariance kills independent question noise). Single-seed caveat applies:
  A[i,j] mixes training noise from both adapters, which question splits cannot remove.
- H-behavior-dependence: median pairwise Spearman between per-behavior z-normalized
  off-diagonal G matrices (raw, un-disattenuated; the question-split disattenuation
  needs judge-row per-question splits and ships as a follow-up).

Output: eval_results/issue_537/analysis/registered_reads.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

EVAL = Path("eval_results/issue_537")
OUT = EVAL / "analysis"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(537)

d = np.load(EVAL / "G_tensor/G_tensor.npz", allow_pickle=True)
with open(EVAL / "G_tensor/G_meta.json") as f:
    meta = json.load(f)
pc = meta["per_cell"]

behaviors = [str(b) for b in d["behaviors"]]
train_cids = [[str(c) for c in row] for row in d["train_cids"]]
eval_cids = [str(c) for c in d["eval_cids"][0]]
G = d["G"][..., 0]  # (5, 16, 30)
NV = d["noise_var"][..., 0]
IF = d["implant_failed"][..., 0]

results: dict = {
    "git_commit_tensor": meta["git_commit"],
    "single_seed_caveat": meta["single_seed_caveat"],
}


# ---------------------------------------------------------------- H-structure
def h_structure(bi: int, exclude_failed_rows: bool, exclude_binst_cols: bool) -> dict:
    tc = train_cids[bi]
    rows = []
    for i, ci in enumerate(tc):
        if exclude_failed_rows and IF[bi, i, :].any():
            continue
        cols = []
        for j, cj in enumerate(eval_cids):
            if cj == ci:
                continue  # off-diagonal only
            if exclude_binst_cols and cj.startswith("binst"):
                continue
            cols.append(j)
        g_row = G[bi, i, cols]
        nv_row = NV[bi, i, cols]
        rows.append((float(np.var(g_row, ddof=1)), float(np.mean(nv_row))))
    v_between = float(np.mean([r[0] for r in rows]))
    v_noise = float(np.mean([r[1] for r in rows]))
    return {
        "v_between": v_between,
        "v_noise_floor": v_noise,
        "corrected_over_floor_ratio": (v_between - v_noise) / v_noise if v_noise > 0 else None,
        "pass_2x": bool((v_between - v_noise) >= 2 * v_noise),
        "n_rows": len(rows),
    }


hs = {}
mi = behaviors.index("marker")
hs["marker_primary"] = h_structure(mi, exclude_failed_rows=True, exclude_binst_cols=False)
hs["marker_excl_binst_cols"] = h_structure(mi, exclude_failed_rows=True, exclude_binst_cols=True)
hs["marker_incl_failed_row"] = h_structure(mi, exclude_failed_rows=False, exclude_binst_cols=False)
# split-half cross-check (meta carries noise_var_split_half for marker cells)
sh_floor = []
for i, ci in enumerate(train_cids[mi]):
    if IF[mi, i, :].any():
        continue
    vals = [
        pc[f"marker/{ci}__{cj}"]["noise_var_split_half"]
        for cj in eval_cids
        if cj != ci
        and f"marker/{ci}__{cj}" in pc
        and "noise_var_split_half" in pc[f"marker/{ci}__{cj}"]
    ]
    sh_floor.append(np.mean(vals))
hs["marker_split_half_floor"] = float(np.mean(sh_floor))
# descriptive per-row reads for the other behaviors (registered as texture, not kill)
for b in ["fact", "refusal", "sycophancy", "em"]:
    hs[f"{b}_descriptive"] = h_structure(
        behaviors.index(b), exclude_failed_rows=True, exclude_binst_cols=False
    )
results["h_structure"] = hs

# -------------------------------------------------------------- H-inoculation
inoc = {}
for bi, b in enumerate(behaviors):
    tc = train_cids[bi]
    binst_cid = f"binst_{b}"
    bi_row = tc.index(binst_cid)
    def_row = tc.index("default")
    binst_j = eval_cids.index(binst_cid)
    def_j = eval_cids.index("default")
    binst_off = np.delete(G[bi, bi_row, :], binst_j)
    def_off = np.delete(G[bi, def_row, :], def_j)
    diag_binst = G[bi, bi_row, binst_j]
    diag_def = G[bi, def_row, def_j]
    inoc[b] = {
        "binst_offdiag_mean": float(np.mean(binst_off)),
        "default_offdiag_mean": float(np.mean(def_off)),
        "raw_sign_flip_pass": bool(np.mean(binst_off) < np.mean(def_off)),
        "diag_binst": float(diag_binst),
        "diag_default": float(diag_def),
        "binst_offdiag_diagnorm": float(np.mean(binst_off) / diag_binst)
        if diag_binst != 0
        else None,
        "default_offdiag_diagnorm": float(np.mean(def_off) / diag_def) if diag_def != 0 else None,
    }
results["h_inoculation"] = inoc


# --------------------------------------------------------------- H-asymmetry
def antisym_fraction(M: np.ndarray) -> float:
    """Fraction of off-diagonal (ordered-pair) variance carried by the antisymmetric part."""
    n = M.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    g = np.array([M[i, j] for i, j in pairs])
    a = np.array([0.5 * (M[i, j] - M[j, i]) for i, j in pairs])
    var_g = float(np.mean((g - g.mean()) ** 2))
    return float(np.mean(a**2) / var_g) if var_g > 0 else np.nan


asym = {}
for bi, b in enumerate(behaviors):
    tc = train_cids[bi]
    cols = [eval_cids.index(c) for c in tc]  # 16x16 shared-instance block
    M = G[bi][:, cols]
    asym[b] = {"raw_antisym_fraction": antisym_fraction(M)}
results["h_asymmetry_raw"] = asym

# Question-split corrected (marker row only this round): same K half-splits per cell.
K = 200
cells = {}
for ci in train_cids[mi]:
    for cj in train_cids[mi]:
        p = EVAL / f"G_cells/marker/{ci}__{cj}__seed42.json"
        if p.exists():
            with open(p) as f:
                cell = json.load(f)
            dq = np.array([q["trained"]["logp"] - q["base"]["logp"] for q in cell["per_question"]])
            cells[(ci, cj)] = dq
nq = 32
tc = train_cids[mi]
n = len(tc)
num_a, den_g = [], []
for _ in range(K):
    perm = rng.permutation(nq)
    h1, h2 = perm[: nq // 2], perm[nq // 2 :]
    G1 = np.full((n, n), np.nan)
    G2 = np.full((n, n), np.nan)
    for ii, ci in enumerate(tc):
        for jj, cj in enumerate(tc):
            if (ci, cj) in cells:
                G1[ii, jj] = cells[(ci, cj)][h1].mean()
                G2[ii, jj] = cells[(ci, cj)][h2].mean()
    pairs = [
        (i, j)
        for i in range(n)
        for j in range(n)
        if i != j and np.isfinite(G1[i, j]) and np.isfinite(G1[j, i])
    ]
    a1 = np.array([0.5 * (G1[i, j] - G1[j, i]) for i, j in pairs])
    a2 = np.array([0.5 * (G2[i, j] - G2[j, i]) for i, j in pairs])
    g1 = np.array([G1[i, j] for i, j in pairs])
    g2 = np.array([G2[i, j] for i, j in pairs])
    num_a.append(np.mean(a1 * a2))
    den_g.append(np.mean((g1 - g1.mean()) * (g2 - g2.mean())))
results["h_asymmetry_marker_question_split"] = {
    "corrected_antisym_fraction": float(np.mean(num_a) / np.mean(den_g)),
    "K_splits": K,
    "n_cells_block": len(cells),
    "caveat": "question splits cannot remove training noise; single seed (42)",
}

# ----------------------------------------------------- H-behavior-dependence
offdiag_vecs = {}
mask_cells = []
for bi, b in enumerate(behaviors):
    tc = train_cids[bi]
    vec = []
    for i, ci in enumerate(tc):
        for j, cj in enumerate(eval_cids):
            if cj == ci:
                continue
            vec.append(G[bi, i, j])
    offdiag_vecs[b] = np.array(vec)
z = {b: (v - v.mean()) / v.std() for b, v in offdiag_vecs.items()}
pairwise = {}
rhos = []
for x in range(len(behaviors)):
    for y in range(x + 1, len(behaviors)):
        bx, by = behaviors[x], behaviors[y]
        rho, p = spearmanr(z[bx], z[by])
        pairwise[f"{bx}~{by}"] = {"rho": float(rho), "p": float(p), "n": len(z[bx])}
        rhos.append(rho)
results["h_behavior_dependence"] = {
    "pairwise": pairwise,
    "median_rho_raw": float(np.median(rhos)),
    "note": "raw, un-disattenuated; question-split disattenuation = follow-up",
}

# -------------------------------------------- descriptive: per-row breadth
breadth = {}
for bi, b in enumerate(behaviors):
    tc = train_cids[bi]
    rows = {}
    for i, ci in enumerate(tc):
        j = eval_cids.index(ci)
        off = np.delete(G[bi, i, :], j)
        rows[ci] = {
            "diag": float(G[bi, i, j]),
            "offdiag_mean": float(np.mean(off)),
            "breadth_diagnorm": float(np.mean(off) / G[bi, i, j]) if G[bi, i, j] != 0 else None,
            "implant_failed": bool(IF[bi, i, :].any()),
        }
    breadth[b] = rows
results["per_row_breadth"] = breadth

with open(OUT / "registered_reads.json", "w") as f:
    json.dump(results, f, indent=1)
print(json.dumps({k: v for k, v in results.items() if k != "per_row_breadth"}, indent=1))
