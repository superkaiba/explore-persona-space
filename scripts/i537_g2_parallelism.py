"""Issue #537 G2 registered mechanism test (plan §6.4ii; marker row; zero-GPU).

Tests whether the trained-minus-base residual delta at the post-response slot
points in the SAME direction at every eval context (rank-1 prediction), per
adapter and layer {6, 14, 22, 27}. Registered reads, all from the shipped
activation_deltas dumps (no new forwards):

- parallelism: mean pairwise cosine of dh(a, j) = mean_q(trained - base) across
  the 30 eval contexts, per (adapter, layer);
- noise CEILING: same-cell split-half-over-questions dh cosines (K random
  half-splits; bounds measurement noise only -- single-seed caveat);
- common-mode FLOOR: cross-adapter dh cosines at the SAME context (generic-LoRA
  drift) + a base-side anisotropy null (pairwise cosines of base context-mean
  differences at the same slot/layer);
- registered read = EXCESS of the parallelism cosine over the common-mode floor;
- robustness: remove the across-context mean dh per adapter before the cosine;
- scaling: ||dh(a, j)|| vs the rank-1 projection coefficient
  (v_j . v_a)/||v_a||^2 from the L22 x mean_response clouds (16 shared eval
  contexts only -- partial scope, held-out/instruction contexts lack
  mean_response clouds locally).

Input: /tmp/i537_act/issue537_context_generalization/activation_deltas/marker/
Output: eval_results/issue_537/analysis/g2_parallelism.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ACT = Path("/tmp/i537_act/issue537_context_generalization/activation_deltas/marker")
EVAL = Path("eval_results/issue_537")
rng = np.random.default_rng(537)

LAYERS = ["layer_6", "layer_14", "layer_22", "layer_27"]
adapters = sorted(d.name for d in ACT.iterdir() if d.is_dir() and d.name != "_base")
eval_cids = sorted(p.stem for p in (ACT / "_base").glob("*.npz"))
assert len(adapters) == 16 and len(eval_cids) == 30, (len(adapters), len(eval_cids))


def cid_of(adapter: str) -> str:
    return adapter.removeprefix("i537_marker_").removesuffix("_seed42")


base = {j: np.load(ACT / "_base" / f"{j}.npz") for j in eval_cids}
K_SPLITS = 50


def cos(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


results: dict = {"layers": {}, "n_adapters": len(adapters), "n_eval_contexts": len(eval_cids)}
dh_all: dict[str, dict[tuple[str, str], np.ndarray]] = {layer: {} for layer in LAYERS}
dh_q_all: dict[str, dict[tuple[str, str], np.ndarray]] = {layer: {} for layer in LAYERS}

for a in adapters:
    for j in eval_cids:
        z = np.load(ACT / a / f"{j}.npz")
        for layer in LAYERS:
            dq = z[layer].astype(np.float64) - base[j][layer].astype(np.float64)  # (32, H)
            dh_q_all[layer][(a, j)] = dq
            dh_all[layer][(a, j)] = dq.mean(axis=0)

for layer in LAYERS:
    dh = dh_all[layer]
    dh_q = dh_q_all[layer]

    # (1) registered parallelism: per adapter, mean pairwise cosine across contexts
    par = []
    par_centered = []
    for a in adapters:
        vs = np.stack([dh[(a, j)] for j in eval_cids])
        vc = vs - vs.mean(axis=0)  # robustness: remove across-context mean dh
        cs, cc = [], []
        for x in range(len(eval_cids)):
            for y in range(x + 1, len(eval_cids)):
                cs.append(cos(vs[x], vs[y]))
                cc.append(cos(vc[x], vc[y]))
        par.append(np.mean(cs))
        par_centered.append(np.mean(cc))

    # (2) ceiling: same-cell split-half over questions
    ceil = []
    for a in adapters:
        for j in eval_cids:
            dq = dh_q[(a, j)]
            vals = []
            for _ in range(K_SPLITS):
                perm = rng.permutation(32)
                vals.append(cos(dq[perm[:16]].mean(axis=0), dq[perm[16:]].mean(axis=0)))
            ceil.append(np.mean(vals))

    # (3) floor: cross-adapter cosines at the same context
    floor = []
    for j in eval_cids:
        for x in range(len(adapters)):
            for y in range(x + 1, len(adapters)):
                floor.append(cos(dh[(adapters[x], j)], dh[(adapters[y], j)]))

    # (3b) anisotropy null: base-side context-mean differences
    base_means = {j: base[j][layer].astype(np.float64).mean(axis=0) for j in eval_cids}
    null_vecs = []
    for _ in range(400):
        j1, j2, j3, j4 = rng.choice(eval_cids, 4, replace=False)
        null_vecs.append(cos(base_means[j1] - base_means[j2], base_means[j3] - base_means[j4]))

    results["layers"][layer] = {
        "parallelism_mean_pairwise_cos": float(np.mean(par)),
        "parallelism_per_adapter_min_max": [float(np.min(par)), float(np.max(par))],
        "parallelism_centered_mean": float(np.mean(par_centered)),
        "ceiling_split_half_mean": float(np.mean(ceil)),
        "floor_cross_adapter_mean": float(np.mean(floor)),
        "anisotropy_null_mean": float(np.mean(null_vecs)),
        "excess_over_floor": float(np.mean(par) - np.mean(floor)),
    }

# (4) scaling read at L22: ||dh|| vs rank-1 projection coefficient (16 shared contexts)
SHARED = [
    "sp_swe",
    "sp_doctor",
    "sp_ph1",
    "sp_ph2",
    "wc_short_code",
    "wc_short_advice",
    "wc_long_write",
    "icl_k2",
    "icl_k8",
    "reph_imp",
    "reph_polite",
    "reph_casual",
    "fmt_json",
    "fmt_code",
    "default",
    "binst_marker",
]
vmeans = {}
for cid in SHARED:
    z = np.load(EVAL / "clouds" / f"{cid}__mean_response.npz")
    h = z["hidden"][:, 22, :].astype(np.float64)
    h = h[np.isfinite(h).all(axis=1)]
    vmeans[cid] = h.mean(axis=0)

scal = []
for a in adapters:
    ca = cid_of(a)
    if ca not in vmeans:
        continue
    va = vmeans[ca]
    norms_dh, projs = [], []
    for j in SHARED:
        norms_dh.append(np.linalg.norm(dh_all["layer_22"][(a, j)]))
        projs.append(float(vmeans[j] @ va / (va @ va)))
    rho, p = spearmanr(norms_dh, projs)
    scal.append({"adapter": ca, "spearman_norm_vs_proj": float(rho), "p": float(p)})
results["scaling_l22_shared16"] = {
    "per_adapter": scal,
    "median_rho": float(np.median([s["spearman_norm_vs_proj"] for s in scal])),
    "note": "projection coefficients from L22 x mean_response clouds; 16 shared eval contexts only",
}
results["caveats"] = [
    "single-seed: split-half ceiling bounds MEASUREMENT noise only",
    "late-layer parallelism near-tautological for a working implant (W_U[marker] component); early layers carry the mechanism weight",
]

with open(EVAL / "analysis/g2_parallelism.json", "w") as f:
    json.dump(results, f, indent=1)
print(json.dumps({k: v for k, v in results.items() if k != "scaling_l22_shared16"}, indent=1))
print("median scaling rho:", results["scaling_l22_shared16"]["median_rho"])
