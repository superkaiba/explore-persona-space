"""Issue #2588: ceiling-normalized capability rho for the K3 REFIT maps.

The banked-panel version of this read is
``scripts/issue2588_ceiling_normalized.py`` -> ``ceiling_normalized_capability.json``
(2026-09-08), which normalizes each cell's SINGLE-DRAW-target held-out R2 by that
model's two-draw reliability ceiling and re-runs the capability Spearman. This
script does the same for the K3 round's REFIT maps, which were fit against a
three-rollout averaged training target (epm:followup-scope v1, 2026-09-09).

ZERO new generation. Refit R2 comes from the round's own terminal artifacts on
the HF data repo; ceilings and the capability index come from the banked
ceiling JSON.

THE MATCHING CHOICE, reported both ways rather than decided silently.
The banked panel's test target is a single draw (seed 42), so its ceiling is the
single-draw test-retest reliability r1, which is what the two-draw estimator
measures. The refit's test target is the mean of THREE draws (seeds 42/43/44),
whose reliability is higher than r1. Normalizing refit R2 by r1 therefore
understates how close the refit map sits to its own ceiling. Two readings:

  single-draw ceiling  refit_r2 / r1
      Comparable in denominator to the banked row, so the two normalized
      columns differ only through the numerator. Not the matched quantity.

  three-draw ceiling   refit_r2 / r3,  r3 = 3*r1 / (1 + 2*r1)
      Spearman-Brown for the reliability of a k-draw mean at k=3. This is the
      quantity the refit map is actually bounded by.

Because r3 is a monotone transform of r1 applied per cell, the two readings can
still rank cells differently, so both Spearmans are reported.

Usage:
    uv run python scripts/issue2588_k3_ceiling_normalized_refit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # HF token + thread caps before the scipy import

from scipy.stats import spearmanr  # noqa: E402

from explore_persona_space.orchestrate import hub as HUB  # noqa: E402

REPO_ROOT = _REPO_ROOT
CEILING_JSON = (
    REPO_ROOT
    / "eval_results"
    / "issue_2588"
    / "ceiling_normalized"
    / "ceiling_normalized_capability.json"
)
OUT_DEFAULT = (
    REPO_ROOT
    / "eval_results"
    / "issue_2588"
    / "ceiling_normalized"
    / "k3_refit_ceiling_normalized_capability.json"
)

HF_REPO = "superkaiba1/explore-persona-space-data"
HF_BASE = "issue2588_capability_panel/k3_train_refit/fits"

# The ten paper no-thinking cells, == issue2588_k3_train_refit.TARGET_CELLS.
TEN_CELLS = (
    "q35_0p8b_a",
    "q35_2b_a",
    "q35_4b_a",
    "q35_9b_a",
    "q35_27b_a",
    "q36_27b_a",
    "q38_27b_a",
    "o3_7b_i_a",
    "o31_32b_i_a",
    "q3_32b_a",
)
K_DRAWS = 3


def spearman_brown(r1: float, k: int) -> float:
    """Reliability of a k-draw mean given single-draw reliability r1."""
    return k * r1 / (1.0 + (k - 1) * r1)


def load_refit(cell: str) -> dict | None:
    """Refit + banked reads from the cell's terminal upload-fits artifact."""
    from huggingface_hub import hf_hub_download

    rel = f"{HF_BASE}/{cell}/k3_refit_prompt_last.json"
    try:
        path = HUB.retry_transient(
            lambda: hf_hub_download(HF_REPO, rel, repo_type="dataset"),
            what=f"hf_hub_download {rel}",
        )
    except Exception:
        return None
    payload = json.loads(Path(path).read_text())
    return {
        "refit_r2_avg_target": payload["refit"]["test_r2_avg_target"],
        "banked_r2": payload["banked"]["test_r2"],
        "refit_acc1": payload["refit"]["test_acc1_cos_avg_target_raw"],
        "banked_acc1": payload["banked"]["acc1_raw"],
        "gate_pass": (payload.get("gate") or {}).get("pass"),
        "layer_star": payload["banked"]["layer_star"],
    }


def rho(xs: list[float], ys: list[float]) -> dict:
    result = spearmanr(xs, ys)
    return {"rho": float(result.statistic), "p": float(result.pvalue), "n": len(xs)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args(argv)

    banked = {row["cell"]: row for row in json.loads(CEILING_JSON.read_text())["per_model"]}
    missing_from_ceiling = [c for c in TEN_CELLS if c not in banked]
    if missing_from_ceiling:
        raise SystemExit(f"ceiling JSON is missing cells: {missing_from_ceiling}")

    rows = []
    absent = []
    for cell in TEN_CELLS:
        refit = load_refit(cell)
        if refit is None:
            absent.append(cell)
            continue
        base = banked[cell]
        r1_a = base["ceiling_s42x43"]["ceiling"]
        r1_b = base["ceiling_s43x44_banked"]["two_draw"]["ceiling"]
        row = {
            "cell": cell,
            "model": base.get("model"),
            "aa_index": base["aa_index"],
            "layer_star": refit["layer_star"],
            "gate_pass": refit["gate_pass"],
            "banked_r2": base["test_r2"],
            "refit_r2_avg_target": refit["refit_r2_avg_target"],
            "banked_acc1": refit["banked_acc1"],
            "refit_acc1": refit["refit_acc1"],
            "ceiling_r1_s42x43": r1_a,
            "ceiling_r1_s43x44": r1_b,
            "ceiling_r3_s42x43": spearman_brown(r1_a, K_DRAWS),
            "ceiling_r3_s43x44": spearman_brown(r1_b, K_DRAWS),
            "banked_r2_normalized_s42x43": base["r2_normalized_s42x43"],
            "banked_r2_normalized_s43x44": base["r2_normalized_s43x44"],
        }
        row["refit_r2_norm_singledraw_s42x43"] = refit["refit_r2_avg_target"] / r1_a
        row["refit_r2_norm_singledraw_s43x44"] = refit["refit_r2_avg_target"] / r1_b
        row["refit_r2_norm_threedraw_s42x43"] = (
            refit["refit_r2_avg_target"] / row["ceiling_r3_s42x43"]
        )
        row["refit_r2_norm_threedraw_s43x44"] = (
            refit["refit_r2_avg_target"] / row["ceiling_r3_s43x44"]
        )
        rows.append(row)

    aa = [r["aa_index"] for r in rows]
    trends = {
        "banked_raw_r2_vs_aa": rho(aa, [r["banked_r2"] for r in rows]),
        "refit_raw_r2_vs_aa": rho(aa, [r["refit_r2_avg_target"] for r in rows]),
        "banked_normalized_r2_s42x43_vs_aa": rho(
            aa, [r["banked_r2_normalized_s42x43"] for r in rows]
        ),
        "banked_normalized_r2_s43x44_vs_aa": rho(
            aa, [r["banked_r2_normalized_s43x44"] for r in rows]
        ),
        "refit_normalized_singledraw_s42x43_vs_aa": rho(
            aa, [r["refit_r2_norm_singledraw_s42x43"] for r in rows]
        ),
        "refit_normalized_singledraw_s43x44_vs_aa": rho(
            aa, [r["refit_r2_norm_singledraw_s43x44"] for r in rows]
        ),
        "refit_normalized_threedraw_s42x43_vs_aa": rho(
            aa, [r["refit_r2_norm_threedraw_s42x43"] for r in rows]
        ),
        "refit_normalized_threedraw_s43x44_vs_aa": rho(
            aa, [r["refit_r2_norm_threedraw_s43x44"] for r in rows]
        ),
        "ceiling_r1_s42x43_vs_aa": rho(aa, [r["ceiling_r1_s42x43"] for r in rows]),
        "ceiling_r1_s43x44_vs_aa": rho(aa, [r["ceiling_r1_s43x44"] for r in rows]),
    }

    out = {
        "schema_version": 1,
        "meta": {
            "task": 2588,
            "script": "scripts/issue2588_k3_ceiling_normalized_refit.py",
            "round": "k3_train_refit",
            "k_draws": K_DRAWS,
            "ceiling_source": str(CEILING_JSON.relative_to(REPO_ROOT)),
            "refit_source": f"{HF_REPO}:{HF_BASE}/<cell>/k3_refit_prompt_last.json",
            "estimator": "banked two-draw variance-weighted per-dim Pearson (#1491 "
            "ceiling_var_weighted_r); three-draw ceiling via Spearman-Brown",
            "cells_present": [r["cell"] for r in rows],
            "cells_absent": absent,
            "complete_panel": not absent,
        },
        "per_model": rows,
        "trends": trends,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    status = "COMPLETE 10/10" if not absent else f"PARTIAL {len(rows)}/10 (absent: {absent})"
    print(f"panel: {status}")
    print(
        f"{'cell':14s} {'AA':>3s} {'bank_r2':>8s} {'refit_r2':>8s} "
        f"{'r1':>6s} {'r3':>6s} {'bank_norm':>9s} {'refit/r1':>8s} {'refit/r3':>8s}"
    )
    for r in sorted(rows, key=lambda x: x["aa_index"]):
        print(
            f"{r['cell']:14s} {r['aa_index']:3d} {r['banked_r2']:8.4f} "
            f"{r['refit_r2_avg_target']:8.4f} {r['ceiling_r1_s42x43']:6.3f} "
            f"{r['ceiling_r3_s42x43']:6.3f} {r['banked_r2_normalized_s42x43']:9.4f} "
            f"{r['refit_r2_norm_singledraw_s42x43']:8.4f} "
            f"{r['refit_r2_norm_threedraw_s42x43']:8.4f}"
        )
    print()
    for name, value in trends.items():
        print(f"  {name:44s} rho={value['rho']:+.3f} p={value['p']:.4f} n={value['n']}")
    print(f"\nwrote {args.out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
