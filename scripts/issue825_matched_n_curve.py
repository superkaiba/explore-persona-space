#!/usr/bin/env python
"""#825 matched-n learning curve: does the curated single-turn context->answer map
(within-instruct Xi->Yi @ L19, committed R2=0.673) drop toward the ~0.2-0.3 the
#1092 real-multi-turn per-turn cells show, when subsampled to the turn-cell sizes?

Isolates SAMPLE SIZE from CORPUS: same #825 curated single-turn bundle, same GCV
ridge + grouped 5-fold estimator, just fewer rows. If R2 stays ~0.6 at n=497,
the gap to the turn-depth cells is CORPUS (real multi-turn), not n.

Reuses cm.extract_stem/load_cell + the map_alignment ridge core verbatim.

R4 adoption (onpolicy-turn-depth-map round, plan §4.5/R4): rescue-moved sole
copy adopted from ~/.task-workflow/root-sync-rescue/1784116235-issue1320-
step9c-unblock/ with the FIT MACHINERY verbatim (_r2_at + the load path +
the fixed subsample rngs 1000-1004 / fold seed 0). Additions are output-side
only: a committed results JSON (the anchor value of record, superseding the
marker-quoted "~0.48"), the binding full-n version gate (|full - 0.673| <=
0.002 — W2), the secondary n=497 sanity band (+/-0.02 vs the marker-recorded
~0.48), and --dl-dir/--extra-n/--out-json args. Gate failure DEMOTES the
anchor in the JSON (anchor_status) per plan §12.9/§7 — recorded, not raised —
so the fit script draws the demoted label instead of a hard confirm.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue825_crossmodel_map_transfer as cm  # noqa: E402
import issue825_map_alignment as ma  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

LAYER = 19
SUBSAMPLE_SIZES = [100, 200, 300, 497, 700, 1000, 1500, 2000]
N_SUBSAMPLE_SEEDS = 5
# W2 binding version fingerprint: the committed full-n within-instruct L19 R2
# (eval_results/issue_825/cells_S1.json r2_per_layer_obs[19]).
COMMITTED_FULL_N_R2 = 0.6730940896676356
FULL_N_TOL = 0.002
# Secondary sanity band: the marker-recorded "~0.48" at n=497 (plan §12.9).
MARKER_N497_R2 = 0.48
N497_TOL = 0.02
OUT_JSON_DEFAULT = REPO_ROOT / "eval_results/issue_825/matched_n_curve/results.json"
DL_DIR_DEFAULT = Path("data/issue_825/hf_dl/map_alignment")


def _r2_at(Xf, Yf, conv, idx) -> float:
    """Held-out grouped-5-fold GCV-ridge R2 on the subsample idx (verbatim)."""
    Xi, Yi, cv = Xf[idx], Yf[idx], conv[idx]
    folds = cm._cv_folds(cv, cm.N_FOLDS, 0)

    def preds(tr, te):
        return ma._ridge_predict(ma._ridge_prep(Xi[tr]), Yi[tr], Xi[te])

    return ma._heldout_pooled_r2(Yi, folds, preds)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dl-dir", default=str(DL_DIR_DEFAULT), help="npz cache / download dir")
    ap.add_argument(
        "--extra-n",
        type=int,
        action="append",
        default=[],
        help="additional subsample size(s), e.g. the realized kept-row t1 n after drops",
    )
    ap.add_argument("--out-json", default=str(OUT_JSON_DEFAULT))
    args = ap.parse_args()

    dl = Path(args.dl_dir)
    dl.mkdir(parents=True, exist_ok=True)
    npz_i = cm.extract_stem(ma.STEM_INSTRUCT, dl)
    ci = cm.load_cell(npz_i, ma.ROLE)
    X, Y = ci["X"], ci["Y"]
    layers = [int(v) for v in ci["layers"]]
    conv = None
    for k in ("conv_ids", "conv", "ids", "conversation_ids"):
        if k in ci:
            conv = np.asarray(ci[k])
            break
    if conv is None:
        raise SystemExit(f"no conv-id key in load_cell output; keys={list(ci.keys())}")
    li = layers.index(LAYER)
    dev = ma._fit_device()
    Xf = torch.as_tensor(X[:, li, :], dtype=torch.float64).to(dev)
    Yf = torch.as_tensor(Y[:, li, :], dtype=torch.float64).to(dev)
    N = Xf.shape[0]
    print(f"[data] instruct single-turn: N={N}  D={Xf.shape[1]}  n_convs={len(set(conv.tolist()))}")

    full = _r2_at(Xf, Yf, conv, np.arange(N))
    full_diff = abs(full - COMMITTED_FULL_N_R2)
    full_pass = full_diff <= FULL_N_TOL
    print(
        f"[gate] FULL within-instruct L19 R2 = {full:.4f}  (committed "
        f"{COMMITTED_FULL_N_R2:.6f}; |d|={full_diff:.2e} tol={FULL_N_TOL} -> "
        f"{'PASS' if full_pass else 'FAIL'})"
    )

    sizes = sorted(set(SUBSAMPLE_SIZES + list(args.extra_n)))
    print(
        "[curve] within-instruct L19 R2 vs subsample n (mean +/- sd over "
        f"{N_SUBSAMPLE_SEEDS} draws):"
    )
    curve = []
    for n in [*sizes, N]:
        if n > N:
            continue
        vals = []
        for s in range(N_SUBSAMPLE_SEEDS):
            rs = np.random.default_rng(1000 + s)
            idx = rs.choice(N, size=n, replace=False)
            vals.append(_r2_at(Xf, Yf, conv, idx))
        mean = st.mean(vals)
        sd = st.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"  n={n:>5}: R2 = {mean:.3f} +/- {sd:.3f}   draws={[round(v, 3) for v in vals]}")
        curve.append({"n": int(n), "r2_mean": mean, "r2_sd": sd, "r2_draws": vals})

    n497 = next((row for row in curve if row["n"] == 497), None)
    n497_pass = None
    if n497 is not None:
        n497_diff = abs(n497["r2_mean"] - MARKER_N497_R2)
        n497_pass = n497_diff <= N497_TOL
        print(
            f"[gate] n=497 draw mean = {n497['r2_mean']:.4f} (marker-recorded "
            f"~{MARKER_N497_R2}; |d|={n497_diff:.3f} tol={N497_TOL} -> "
            f"{'PASS' if n497_pass else 'FAIL'})"
        )
    anchor_status = (
        "committed"
        if full_pass and (n497_pass is not False)
        else "demoted — marker-recorded ~0.48 (uncommitted provenance); see plan §12.9/§7"
    )
    print(f"[anchor] status: {anchor_status}")

    payload = {
        "issue": 825,
        "description": (
            "Matched-n learning curve of the #825 curated single-turn within-instruct "
            "context->answer map @ L19 (GCV ridge, grouped 5-fold, fixed subsample "
            "rngs 1000-1004, fold seed 0). The committed anchor of record for the "
            "onpolicy-turn-depth-map round (supersedes the marker-quoted ~0.48)."
        ),
        "layer": LAYER,
        "n_full": int(N),
        "n_folds": int(cm.N_FOLDS),
        "n_subsample_seeds": N_SUBSAMPLE_SEEDS,
        "subsample_rngs": [1000 + s for s in range(N_SUBSAMPLE_SEEDS)],
        "gate_full_n": {
            "value": full,
            "committed": COMMITTED_FULL_N_R2,
            "abs_diff": full_diff,
            "tol": FULL_N_TOL,
            "pass": bool(full_pass),
            "role": "binding version fingerprint (plan W2)",
        },
        "anchor_n497": (
            None
            if n497 is None
            else {
                "r2_mean": n497["r2_mean"],
                "r2_sd": n497["r2_sd"],
                "marker_value": MARKER_N497_R2,
                "tol": N497_TOL,
                "pass": bool(n497_pass),
                "role": "secondary sanity band (plan §12.9)",
            }
        ),
        "anchor_status": anchor_status,
        "curve": curve,
        "input": {
            "stem": ma.STEM_INSTRUCT,
            "role": ma.ROLE,
            "hf_prefix": cm.HF_PREFIX,
            "hf_revision": cm.HF_REV,
        },
        "git_commit": _git_commit(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
