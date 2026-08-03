#!/usr/bin/env python3
"""Issue #1738 — materialize the floor-adjusted taxonomy contrast family.

Recomputes the P4c floor-adjusted contrasts (adj_i = nerr_i - floor_i/den_i over
the K-resample subsample) with the SAME battery conventions as ``taxonomy.json``
(10k-draw joint context bootstrap, 10k permutation p, BH-FDR q=0.05, seed 1738 —
the #1482 batched implementations, imported), and writes them as a standalone
committed artifact ``eval_results/issue_1738/taxonomy_floor_adjusted.json``
(same per-arm schema as ``taxonomy.json`` arms, plus ``n_joined``).

Inputs (all committed on issue-1738 / local worktree):
- ``eval_results/issue_1738/kresample/floors_L{L}.npz``  (ci, floor, den, share; n=1,988)
- ``eval_results/issue_1738/percontext/{arm}_L{L}_ridge.npz``  (nerr, ci; n=9,941)
- ``eval_results/issue_1738/judge_labels/labels.json``
- the local sampling-manifest mirror (depth/corpus per ci)

Verification: asserts the recomputed rows match the ``floor_adjusted`` block
embedded in the committed ``taxonomy.json`` (same helpers, same seed) so the
standalone artifact is recomputable-by-construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # #847: shared-VM thread caps bind BEFORE numpy import

import numpy as np  # noqa: E402

import issue1482_analysis as A82  # noqa: E402
import issue1738_characterize as CH  # noqa: E402


def _floor_adjusted_arm(out_eval: Path, arm: str, layer: int, labels: dict, fields: dict) -> dict:
    """Recompute one arm's floor-adjusted contrast family (batched batteries).

    Returns the per-arm block: n (post non-finite drop), n_joined (K-resample
    contexts joined to this arm's percontext rows), family, contrasts.
    """
    with np.load(out_eval / "kresample" / f"floors_L{layer}.npz") as fz:
        floors_ci = fz["ci"].copy()
        floors_share = fz["share"].copy()
    z = np.load(out_eval / "percontext" / f"{arm}_L{layer}_ridge.npz")
    nerr, ci_rows = z["nerr"].astype(np.float64), z["ci"]

    pos_of = {int(c): p for p, c in enumerate(ci_rows.tolist())}
    missing = [int(c) for c in floors_ci if int(c) not in pos_of]
    assert not missing, f"floors ci absent from {arm} percontext rows: {missing[:5]}"
    sub_pos = np.asarray([pos_of[int(c)] for c in floors_ci], dtype=np.int64)
    adj_all = nerr[sub_pos] - floors_share
    fin = np.isfinite(adj_all)
    adj = adj_all[fin]
    sub_ci = np.asarray(floors_ci)[fin]

    fmasks = CH._contrast_masks(sub_ci, labels, fields)
    fpv = A82._perm_pvals(adj, [m for _, m in fmasks], CH.N_PERM, CH.STAT_SEED)
    fsig = A82._bh_fdr(fpv, CH.BH_Q)
    rows = []
    for (name, m), p, s in zip(fmasks, fpv, fsig, strict=True):
        deltas = A82._boot_group_delta(adj, m, ~m, CH.N_BOOT, CH.STAT_SEED)
        rows.append(
            {
                "contrast": name,
                "n_group": int(m.sum()),
                "delta_mean_adj_nerr": float(adj[m].mean() - adj[~m].mean()),
                "boot_ci": [
                    float(np.quantile(deltas, 0.025)),
                    float(np.quantile(deltas, 0.975)),
                ],
                "perm_p": float(p),
                "bh_significant": bool(s),
            }
        )
    return {
        "n": int(len(adj)),
        "n_joined": int(len(floors_ci)),
        "n_dropped_nonfinite": int((~fin).sum()),
        "family": [n for n, _ in fmasks],
        "contrasts": rows,
    }


def _verify_against_taxonomy(out_eval: Path, doc: dict) -> None:
    """Assert the standalone recompute matches taxonomy.json's embedded block."""
    tax = json.loads((out_eval / "taxonomy.json").read_text())
    for arm_key, arm_doc in doc["arms"].items():
        emb = tax["arms"][arm_key]["floor_adjusted"]
        emb_by = {c["contrast"]: c for c in emb["contrasts"]}
        assert arm_doc["n"] == emb["n"], (arm_key, arm_doc["n"], emb["n"])
        assert set(c["contrast"] for c in arm_doc["contrasts"]) == set(emb_by), arm_key
        for c in arm_doc["contrasts"]:
            e = emb_by[c["contrast"]]
            for k in ("delta_mean_adj_nerr", "perm_p"):
                assert abs(c[k] - e[k]) < 1e-12, (arm_key, c["contrast"], k, c[k], e[k])
            assert c["bh_significant"] == e["bh_significant"], (arm_key, c["contrast"])
            assert max(abs(a - b) for a, b in zip(c["boot_ci"], e["boot_ci"], strict=True)) < 1e-12
    print("[verify] standalone recompute matches taxonomy.json embedded floor_adjusted block")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-eval", type=Path, default=PROJECT_ROOT / "eval_results" / "issue_1738")
    ap.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue_1738" / "mt100k" / "sampling_manifest",
    )
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    labels = CH._load_labels(args.out_eval)
    fields = CH._manifest_fields(args.manifest_dir)
    doc: dict = {
        "n_boot": CH.N_BOOT,
        "n_perm": CH.N_PERM,
        "bh_q": CH.BH_Q,
        "seed": CH.STAT_SEED,
        "layer": args.layer,
        "definition": (
            "adj_i = nerr_i - floor_i/den_i (map error net of answer-sampling variance); "
            "contexts resampled JOINTLY per draw; contrast families re-enumerated on the "
            "K-resample subsample (topic masks require n>=50 within the subsample)"
        ),
        "inputs": {
            "floors": f"eval_results/issue_1738/kresample/floors_L{args.layer}.npz",
            "percontext": f"eval_results/issue_1738/percontext/{{arm}}_L{args.layer}_ridge.npz",
            "labels": "eval_results/issue_1738/judge_labels/labels.json",
        },
        "arms": {},
    }
    for arm in ("prefix", "context"):
        doc["arms"][f"{arm}_L{args.layer}_ridge"] = _floor_adjusted_arm(
            args.out_eval, arm, args.layer, labels, fields
        )
        print(
            f"[floor-adjusted] {arm}: n={doc['arms'][f'{arm}_L{args.layer}_ridge']['n']} "
            f"contrasts={len(doc['arms'][f'{arm}_L{args.layer}_ridge']['contrasts'])}"
        )
    if not args.no_verify:
        _verify_against_taxonomy(args.out_eval, doc)
    out_path = args.out_eval / "taxonomy_floor_adjusted.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(out_path)
    print(f"[floor-adjusted] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
