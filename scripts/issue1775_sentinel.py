#!/usr/bin/env python3
"""#1775 results-sentinel composer (pod-side; the /issue Step 7 payload contract).

Writes the ENVELOPED sentinel (sentinel_schema_version=1, kind=epm:results,
version=1 — the poller's #1095 rewrite derives max+1 on collision) whose
``note`` carries all 10 required payload keys. Pod-side code NEVER shells out
to task.py — the VM poller drains this file into the marker.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

from issue1775_common import (  # noqa: E402
    HF_DATA_REPO,
    OUT_HF_PREFIX,
    PROJECT_ROOT,
    eval_dir,
    git_sha,
    out_root,
)


def _maybe(path: Path, *keys):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
        if d is None:
            return None
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description="#1775 results sentinel writer")
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--gpu-hours-used", type=float, required=True)
    ap.add_argument("--gpu-hours-budgeted", type=float, default=14.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    er = out_root() / "eval_results" / "issue_1775"
    eval_paths = sorted(str(p.relative_to(out_root())) for p in er.rglob("*.json") if p.is_file())
    ladder = eval_dir("ladder")
    bil = eval_dir("bilinear")
    det = eval_dir("detection")
    eval_numbers = {
        "gate_c_context_pca48": _find_gate_c(ladder / "linear_fits.json", "pca48"),
        "gate_c_context_ambient": _find_gate_c(ladder / "linear_fits.json", "ambient"),
        "r_star_inner_val": _maybe(
            bil / "bilinear_fits.json", "schemes", "prefix", "r_star_inner_val"
        ),
        "delta_named_prefix": _maybe(
            bil / "bilinear_fits.json", "schemes", "prefix", "delta_named", "delta_r2"
        ),
        "delta_beyond_prefix": _maybe(
            bil / "bilinear_fits.json",
            "schemes",
            "prefix",
            "delta_beyond_mlp_minus_bilinear",
            "delta_r2",
        ),
        "detection_family_size": _maybe(det / "hsic_dcor.json", "registered_family_size"),
        "power_mde": _maybe(det / "hsic_dcor.json", "power_check", "mde"),
    }
    deviations_path = eval_dir("") / "plan_deviations.json"
    plan_deviations = json.loads(deviations_path.read_text()) if deviations_path.exists() else []
    payload = {
        "eval_numbers": eval_numbers,
        "eval_paths": eval_paths,
        "reproducibility_card": {
            "model": "none trained — reads over banked #1092/#779 activation stores",
            "folds": "novel-prefix 6-fold FOLD_SEED=0 + novel-query companion + doubly-novel",
            "layers": {"primary": 14, "bridge": 19},
            "targets": "pooled t1/t2/t3 stacked ambient + FULL-fit-population pca48 "
            "(declared deviation — see plan_deviations)",
            "engines": {
                "ridge": "issue1092_fit_grid PRESS engine (press_fit_predict) verbatim",
                "expansion_ridge": "fit_h.ridge_fit_predict_fast (>=3-slice parity-gated)",
                "krr": "exact RBF, gamma=median-heuristic x {0.25..4}, "
                "lambda in {1e-4..10}, inner group-split selected",
                "rff": "D=16384, KRR-selected gamma, seeds 0-2",
                "mlp": "#779 recipe verbatim: w=8192 lr=3e-4 wd=1e-4 AdamW full-batch "
                "patience=20 max=300, group-respecting early-stop, seeds 0-2",
                "bilinear": "r in {0,1,2,4,8,16,32,64}; warm-start stitch ridge; "
                "wd in {0,1e-4,1e-2}; seeds 0-2; Adam + decoupled per-group decay",
            },
            "seeds": {"fold": 0, "fit": [0, 1, 2], "permutation": 0, "minhash": 0},
            "hsic_dcor": "B=1000 x {prefix-block, query-block, within-prefix derangement}; "
            "Holm over 30 registered tests",
            "store": "issue1092_realistic_crossing (HF data repo) @ L14/L19",
        },
        "wandb_url": "n/a — no training in this task (plan section 0: no model training)",
        "hf_hub_url": f"https://huggingface.co/datasets/{HF_DATA_REPO}/tree/main/{OUT_HF_PREFIX}",
        "worktree_path": str(PROJECT_ROOT),
        "final_commit_sha": git_sha(),
        "gpu_hours_used": args.gpu_hours_used,
        "gpu_hours_budgeted": args.gpu_hours_budgeted,
        "plan_deviations": plan_deviations,
    }
    sentinel = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "smoke": bool(args.smoke),
        "note": payload,
    }
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(sentinel, indent=2), encoding="utf-8")
    os.replace(tmp, args.dest)
    print(f"[sentinel] wrote {args.dest} ({len(eval_paths)} eval paths)", flush=True)
    return 0


def _find_gate_c(path: Path, basis: str):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    for u in d.get("units", []):
        if u.get("arm") == "context_end" and u.get("basis") == basis and u.get("gate_c"):
            return {"r2": u["r2"], **u["gate_c"]}
    return None


if __name__ == "__main__":
    sys.exit(main())
