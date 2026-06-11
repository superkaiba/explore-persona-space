"""Issue #537 G1 registered mechanism test (plan §6.4; marker row; zero-GPU).

Joint OLS of dG_anti = 0.5*(G[i->j] - G[j->i]) on dlog||v|| = log||v_j|| - log||v_i|| AND
(s_i - s_j), on the marker 16x16 shared-instance block. Norms come from the
L22 x mean_response P1 clouds (v_c = the context's mean activation; §6.3 primary).
Predicted slope ~= 1 on dlog||v|| under the half-difference convention.

Exclusions per the standing flags: the implant-failed train row (code-comment
wrap) and the saturated behavior-instruction cell (primary); quarantined cells
masked per §4.3. Question-cluster bootstrap (B=2000) CIs from G_cells
per-question deltas; single-seed caveat applies (DV noise widens CIs only).

Output: eval_results/issue_537/analysis/g1_regression.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EVAL = Path("eval_results/issue_537")
rng = np.random.default_rng(537)

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
EXCLUDE_PRIMARY = {"fmt_code", "binst_marker"}

# --- context-mean norms at L22 x mean_response -------------------------------
norms = {}
for cid in SHARED:
    z = np.load(EVAL / "clouds" / f"{cid}__mean_response.npz")
    h = z["hidden"][:, 22, :].astype(np.float64)
    h = h[np.isfinite(h).all(axis=1)]
    v = h.mean(axis=0)
    norms[cid] = float(np.linalg.norm(v))

# --- per-question marker G on the shared block -------------------------------
cells = {}
for ci in SHARED:
    for cj in SHARED:
        p = EVAL / f"G_cells/marker/{ci}__{cj}__seed42.json"
        if p.exists():
            with open(p) as f:
                cell = json.load(f)
            cells[(ci, cj)] = np.array(
                [q["trained"]["logp"] - q["base"]["logp"] for q in cell["per_question"]]
            )

# diagonal implant strength s_i
s = {ci: cells[(ci, ci)].mean() for ci in SHARED}

# quarantine mask
with open(EVAL / "prereg/quarantine_manifest.json") as f:
    qm = json.load(f)
quar = {tuple(c) for c in qm["quarantined_cells"]["marker"]}  # (train_cid, eval_cid) pairs


def pairs_for(exclude: set[str]) -> list[tuple[str, str]]:
    ctx = [c for c in SHARED if c not in exclude]
    out = []
    for a in range(len(ctx)):
        for b in range(a + 1, len(ctx)):
            ci, cj = ctx[a], ctx[b]
            if (ci, cj) in quar or (cj, ci) in quar:
                continue
            if (ci, cj) in cells and (cj, ci) in cells:
                out.append((ci, cj))
    return out


def fit(pairs: list[tuple[str, str]], qidx: np.ndarray | None = None) -> dict:
    y, x1, x2 = [], [], []
    for ci, cj in pairs:
        gij = cells[(ci, cj)][qidx].mean() if qidx is not None else cells[(ci, cj)].mean()
        gji = cells[(cj, ci)][qidx].mean() if qidx is not None else cells[(cj, ci)].mean()
        y.append(0.5 * (gij - gji))
        x1.append(np.log(norms[cj]) - np.log(norms[ci]))
        x2.append(s[ci] - s[cj])
    X = np.column_stack([np.ones(len(y)), x1, x2])
    yv = np.array(y)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    r2 = 1 - resid.var() / yv.var() if yv.var() > 0 else np.nan
    return {
        "intercept": beta[0],
        "slope_dlognorm": beta[1],
        "slope_strengthdiff": beta[2],
        "r2": r2,
        "n_pairs": len(pairs),
    }


results = {}
for name, excl in [
    ("primary_excl_failed_and_saturated", EXCLUDE_PRIMARY),
    ("sensitivity_full_block", set()),
    # Round-2 leverage check (interpretation-critic): fmt_json's L22 norm (78.0)
    # sits far below the 85.9-90.2 cluster of the other 13 primary contexts, so
    # the norm regressor's spread is mostly fmt_json's pairs. Refit without it.
    ("sensitivity_excl_fmt_json_leverage", EXCLUDE_PRIMARY | {"fmt_json"}),
]:
    pr = pairs_for(excl)
    point = fit(pr)
    nq = 32
    boots = {"slope_dlognorm": [], "slope_strengthdiff": []}
    for _ in range(2000):
        qidx = rng.integers(0, nq, nq)
        b = fit(pr, qidx)
        boots["slope_dlognorm"].append(b["slope_dlognorm"])
        boots["slope_strengthdiff"].append(b["slope_strengthdiff"])
    for k in boots:
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        point[f"{k}_ci"] = [float(lo), float(hi)]
    # descriptive scale of the regressors
    point["dlognorm_range"] = [
        float(min(np.log(norms[c]) for c in SHARED if c not in excl)),
        float(max(np.log(norms[c]) for c in SHARED if c not in excl)),
    ]
    results[name] = {
        k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in point.items()
    }

results["norms_l22_mean_response"] = {c: norms[c] for c in SHARED}
results["caveats"] = [
    "single-seed: context-structure vs training-noise not separable via seeds",
    "question-cluster bootstrap CIs bound probe-sampling noise only",
]
out = EVAL / "analysis/g1_regression.json"
with open(out, "w") as f:
    json.dump(results, f, indent=1)
print(json.dumps(results, indent=1))
