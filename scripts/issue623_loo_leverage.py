#!/usr/bin/env python3
"""Issue #623 free-analysis follow-up — leave-one-out leverage check on the headline rho.

ANALYSIS-ONLY. Re-reads ONLY the two artifacts already on disk from the main
phase-6 analysis (``eval_results/issue_623/{cosine_matrix,syc_i}.json``); no
training, no eval generation, no pod, no new model calls.

Motivation (analyzer free-analysis follow-up, Step 9a-ter): the headline Spearman
rho rests on n=35 personas with one high-sycophancy outlier (satirist, syc_i
0.297, far above the n=2 improv_comedian at 0.21). A single high-leverage point
can inflate (or in principle deflate) a rank correlation. This recomputes rho
with each persona dropped in turn (n=34 LOO) and reports the min/max so the
reader can confirm no single persona drives the headline CI.

DV reused verbatim from ``scripts/issue623_analyze.py``: the HEADLINE arm
``lt_persona_lt_syc`` at the steering-selected layer (default 14), cosine metric,
proj_i = cosine(persona_vector_i, sycophancy_vector) vs the reused #612 syc_i.
The ``assistant`` baseline-self is pre-registered-dropped (it is already absent
from the cosine_matrix correlation panel, but we filter on ``is_baseline_self``
defensively as well).

Outputs:
  eval_results/issue_623/rho_loo_leverage.json   (full LOO array + min/max + CI sanity)

Usage (off-pod, single CPU phase, no shards):
  uv run python scripts/issue623_loo_leverage.py
  uv run python scripts/issue623_loo_leverage.py --layer 14 --arm lt_persona_lt_syc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.experiments.persona_decomp_623 import (  # noqa: E402
    reproducibility_metadata,
)

# Headline arm + layer (steering-selected) — defaults match issue623_analyze.py.
DEFAULT_ARM = "lt_persona_lt_syc"
DEFAULT_LAYER = 14
# Headline CI from the main analysis (rho_by_layer.json headline) — used only for
# the containment sanity check, NOT recomputed here (this follow-up re-reads only
# the two named artifacts; the CI is a fixed reference number from the brief).
HEADLINE_CI = (0.494, 0.867)


def load_headline_proj(cosine_matrix_path: Path, arm: str, layer: int) -> dict[str, float]:
    """Load {persona: proj_i} for one (arm, layer) from cosine_matrix.json."""
    doc = json.loads(cosine_matrix_path.read_text())
    cm = doc["cosine_matrix"]
    if arm not in cm:
        raise KeyError(
            f"arm {arm!r} not in cosine_matrix.json (have {sorted(cm)}); "
            "cannot run LOO leverage on a missing headline arm."
        )
    layer_key = str(layer)
    if layer_key not in cm[arm]:
        raise KeyError(
            f"layer {layer_key!r} not under arm {arm!r} in cosine_matrix.json "
            f"(have {sorted(cm[arm])})."
        )
    return dict(cm[arm][layer_key])


def load_correlation_panel(syc_i_path: Path) -> dict[str, float]:
    """Load {persona: syc_i} for the correlation panel (baseline-self dropped)."""
    doc = json.loads(syc_i_path.read_text())
    entries = doc["syc_i"]
    return {p: v["syc_i"] for p, v in entries.items() if not v["is_baseline_self"]}


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho with a hard fail-fast guard (no NaN/degeneracy into the array)."""
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("NaN in proj_i or syc_i array — refusing to run spearmanr.")
    rho = spearmanr(x, y).correlation
    if rho is None or np.isnan(rho):
        raise ValueError(
            "Spearman rho is NaN — degenerate (constant) axis on the full panel; "
            "the headline correlation would be undefined."
        )
    return float(rho)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Issue #623 free-analysis follow-up — LOO leverage on the headline rho."
    )
    parser.add_argument(
        "--arm", default=DEFAULT_ARM, help="Headline arm key in cosine_matrix.json."
    )
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER, help="Headline layer.")
    parser.add_argument(
        "--cosine-matrix",
        default="eval_results/issue_623/cosine_matrix.json",
        help="cosine_matrix.json from phase 6 (relative to repo root).",
    )
    parser.add_argument(
        "--syc-i",
        default="eval_results/issue_623/syc_i.json",
        help="syc_i.json from phase 5 (relative to repo root).",
    )
    parser.add_argument(
        "--out",
        default="eval_results/issue_623/rho_loo_leverage.json",
        help="Output JSON (relative to repo root).",
    )
    args = parser.parse_args()

    def resolve(p: str) -> Path:
        return PROJECT_ROOT / p if not Path(p).is_absolute() else Path(p)

    cosine_matrix_path = resolve(args.cosine_matrix)
    syc_i_path = resolve(args.syc_i)
    out_path = resolve(args.out)

    print(f"[phase=loo_leverage] arm={args.arm} layer={args.layer}", flush=True)

    proj = load_headline_proj(cosine_matrix_path, args.arm, args.layer)
    syc_i = load_correlation_panel(syc_i_path)

    # personas present in BOTH the proj map and the correlation panel
    personas = sorted(p for p in syc_i if p in proj)
    n = len(personas)
    if n < 4:
        raise ValueError(
            f"Only {n} personas in the intersection of proj + syc_i panel; "
            "too few for a meaningful LOO leverage check."
        )

    x_full = np.array([proj[p] for p in personas], dtype=float)
    y_full = np.array([syc_i[p] for p in personas], dtype=float)
    rho_full = spearman_rho(x_full, y_full)
    print(f"[phase=loo_leverage] full N={n} rho={rho_full:.4f}", flush=True)

    # leave-one-out: drop each persona in turn, recompute rho on the n-1 panel
    loo: list[dict] = []
    for i, dropped in enumerate(personas):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rho_i = spearman_rho(x_full[mask], y_full[mask])
        loo.append(
            {
                "dropped_persona": dropped,
                "rho": rho_i,
                "n": n - 1,
                "in_headline_ci": HEADLINE_CI[0] <= rho_i <= HEADLINE_CI[1],
            }
        )

    rhos = np.array([e["rho"] for e in loo], dtype=float)
    min_idx = int(np.argmin(rhos))
    max_idx = int(np.argmax(rhos))
    min_entry = loo[min_idx]
    max_entry = loo[max_idx]
    all_in_ci = all(e["in_headline_ci"] for e in loo)

    print(
        f"[phase=loo_leverage] min rho={min_entry['rho']:.4f} "
        f"(drop {min_entry['dropped_persona']}); "
        f"max rho={max_entry['rho']:.4f} (drop {max_entry['dropped_persona']})",
        flush=True,
    )
    print(
        f"[phase=loo_leverage] all {n} LOO rho in headline CI {HEADLINE_CI}? {all_in_ci}",
        flush=True,
    )

    doc = {
        "schema_version": 1,
        "metadata": reproducibility_metadata(
            {
                "arm": args.arm,
                "layer": args.layer,
                "analysis_only": True,
                "inputs": [str(cosine_matrix_path), str(syc_i_path)],
            }
        ),
        "arm": args.arm,
        "layer": args.layer,
        "metric": "cosine",
        "n_full": n,
        "rho_full": rho_full,
        "headline_ci": list(HEADLINE_CI),
        "min_rho": {
            "rho": min_entry["rho"],
            "dropped_persona": min_entry["dropped_persona"],
        },
        "max_rho": {
            "rho": max_entry["rho"],
            "dropped_persona": max_entry["dropped_persona"],
        },
        "rho_range": float(rhos.max() - rhos.min()),
        "all_loo_rho_in_headline_ci": all_in_ci,
        "loo": loo,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2))
    print(f"[phase=loo_leverage] wrote {out_path}", flush=True)
    print("[phase=loo_leverage] done", flush=True)


if __name__ == "__main__":
    main()
