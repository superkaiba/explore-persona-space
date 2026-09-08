"""Does the refusal-axis read survive controlling answer length?

Question this answers
---------------------
The comment-3 round found the graded refusal read is collinear with answer
length: the map's signed refusal-axis projection correlates with the observed
refusal-margin change at Spearman 0.79, and with the answer-length change at
-0.80. Refusals are short and compliances are long, so a "refusal" read can be
a length read wearing a costume. The banked #2617 analysis saw the same thing
(``S4_len_partial``: rho 0.72 -> 0.52 once length is partialled, collinearity
0.63, gate tripped, ``authoritative: tercile``) but the tercile read itself was
never computed.

This script computes it. Three reads of increasing strictness:

  1. RAW           Spearman(projection, refusal-margin change) over all pairs.
  2. RANK-PARTIAL  the same, partialling out the answer-length change on ranks.
                   Cheap, but assumes the length relation is monotone and that
                   a linear-in-ranks adjustment removes it.
  3. TERCILE       Spearman within each tercile of answer-length change. Length
                   varies little inside a tercile, so a correlation that
                   survives there is not a length effect. This is the read the
                   banked analysis named authoritative, and it makes no
                   functional-form assumption.

Read 3 is the verdict. Read 2 is reported because it is the conventional
control and its disagreement with read 3, if any, is itself informative.

Both arms are carried throughout. A length artefact would inflate the map and
the copy baseline alike, so the map-minus-copy delta is the quantity that
survives the confound by construction.

No new fit
----------
Every input is a banked per-pair scalar from #2617. Nothing is fitted beyond
rank correlations. Reused rather than reimplemented:
``analysis.paired_ci.paired_bootstrap_rho_delta`` for the paired map-versus-copy
intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/scipy imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS pools freeze at import time.
load_dotenv()

import numpy as np  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

from explore_persona_space.analysis.paired_ci import (  # noqa: E402
    paired_bootstrap_rho_delta,
)
from explore_persona_space.task_workflow import repo_root  # noqa: E402

MAP_ARM = "arm_779ce"
COPY_ARM = "arm_iddelta"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rank_partial(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman partial correlation of x and y controlling z (correlation on ranks)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    r_xy = np.corrcoef(rx, ry)[0, 1]
    r_xz = np.corrcoef(rx, rz)[0, 1]
    r_yz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1.0 - r_xz**2) * (1.0 - r_yz**2))
    if denom <= 0:
        return float("nan")
    return float((r_xy - r_xz * r_yz) / denom)


def _paired(pred: np.ndarray, copy: np.ndarray, dv: np.ndarray, n_boot: int, seed: int) -> dict:
    if pred.size < 6:
        return {"n": int(pred.size), "note": "too few pairs for a stable rank correlation"}
    rho_map, rho_copy, lo, hi = paired_bootstrap_rho_delta(pred, copy, dv, n_boot=n_boot, seed=seed)
    return {
        "n": int(pred.size),
        "rho_map": float(rho_map),
        "rho_copy": float(rho_copy),
        "rho_map_p": float(spearmanr(pred, dv).pvalue),
        "delta_map_minus_copy": float(rho_map - rho_copy),
        "delta_ci95": [float(lo), float(hi)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=26171)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = repo_root()
    src = root / "eval_results/issue_2617/svmp_verbharm"
    rows = _rows(src / "perpair.jsonl")

    margin = np.array([r["margin_delta"] for r in rows], dtype=np.float64)
    length = np.array([r["ans_len_delta"] for r in rows], dtype=np.float64)
    proj = {
        arm_name: np.array(
            [r[f"axis_cos_pred_{arm}"] * r[f"norm_pred_{arm}"] for r in rows], dtype=np.float64
        )
        for arm_name, arm in (("map", MAP_ARM), ("copy", COPY_ARM))
    }

    payload: dict = {
        "question": "does the graded refusal read survive an answer-length control",
        "dv": "margin_delta (continuous judge refusal-margin change)",
        "predictor": "signed projection of the predicted answer shift on the refusal direction",
        "confound": "ans_len_delta (answer-length change, tokens)",
        "arms": {"map": MAP_ARM, "copy": COPY_ARM},
        "n_pairs": len(rows),
        "bootstrap": {"unit": "pair", "draws": args.n_boot, "seed": args.seed},
        "sources": ["eval_results/issue_2617/svmp_verbharm/perpair.jsonl"],
        "collinearity": {
            "spearman_margin_vs_length": float(spearmanr(margin, length).statistic),
            "spearman_proj_map_vs_length": float(spearmanr(proj["map"], length).statistic),
            "spearman_proj_copy_vs_length": float(spearmanr(proj["copy"], length).statistic),
        },
    }

    # --- read 1: raw ----------------------------------------------------------
    payload["raw"] = _paired(proj["map"], proj["copy"], margin, args.n_boot, args.seed)

    # --- read 2: rank-partial, controlling answer length ----------------------
    payload["rank_partial_controlling_length"] = {
        "rho_map": _rank_partial(proj["map"], margin, length),
        "rho_copy": _rank_partial(proj["copy"], margin, length),
        "note": "assumes a monotone length relation removable linearly in ranks",
    }

    # --- read 3: within length terciles (the authoritative read) --------------
    edges = np.quantile(length, [1 / 3, 2 / 3])
    tercile = np.digitize(length, edges)
    terciles: dict = {}
    for t in (0, 1, 2):
        sel = tercile == t
        terciles[f"tercile_{t}"] = {
            "length_range": [float(length[sel].min()), float(length[sel].max())],
            "median_length_delta": float(np.median(length[sel])),
            **_paired(proj["map"][sel], proj["copy"][sel], margin[sel], args.n_boot, args.seed + t),
        }
    payload["length_terciles"] = {
        "edges": [float(e) for e in edges],
        "by_tercile": terciles,
    }

    out = args.out or (root / "eval_results/issue_2617/comment3_length_control/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    c = payload["collinearity"]
    print(
        f"collinearity: margin vs length {c['spearman_margin_vs_length']:+.3f}   "
        f"map projection vs length {c['spearman_proj_map_vs_length']:+.3f}"
    )
    r = payload["raw"]
    print(
        f"\nraw            n={r['n']:3d}  map {r['rho_map']:+.3f}  copy {r['rho_copy']:+.3f}  "
        f"delta {r['delta_map_minus_copy']:+.3f} [{r['delta_ci95'][0]:+.3f},{r['delta_ci95'][1]:+.3f}]"
    )
    rp = payload["rank_partial_controlling_length"]
    print(f"rank-partial         map {rp['rho_map']:+.3f}  copy {rp['rho_copy']:+.3f}")
    print()
    for name, t in terciles.items():
        if "rho_map" not in t:
            print(f"{name:14s} n={t['n']:3d}  {t['note']}")
            continue
        print(
            f"{name:14s} n={t['n']:3d}  map {t['rho_map']:+.3f} (p={t['rho_map_p']:.1e})  "
            f"copy {t['rho_copy']:+.3f}  delta {t['delta_map_minus_copy']:+.3f} "
            f"[{t['delta_ci95'][0]:+.3f},{t['delta_ci95'][1]:+.3f}]  "
            f"len median {t['median_length_delta']:+.0f}"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
