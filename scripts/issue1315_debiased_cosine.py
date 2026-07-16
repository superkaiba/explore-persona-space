#!/usr/bin/env python
"""#1315 cross-cell direction-cosine half-draw CIs (descriptive; plan §4.6).

The #1112 debiased-cosine machinery verbatim (paired subsample-without-
replacement half-draws, m = 60 of 120, 2000 draws, seed 1112, same-cell
split-half attenuation references), re-pointed at the #1315 ICL panel pairs.

Writes eval_results/issue_1315/geometry/debiased_cosine.json.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from issue1112_debiased_cosine import (  # noqa: E402
    QUANTILES,
    batched_half_cosines,
    half_partition_masks,
)

from explore_persona_space.experiments import issue_1315 as C  # noqa: E402
from explore_persona_space.experiments.issue_1112 import geometry as geo  # noqa: E402

REPO_ROOT = _SCRIPTS_DIR.parent
STAGE = REPO_ROOT / "data" / f"issue_{C.ISSUE}" / "hf_dl" / "analysis_tensors"
OUT = REPO_ROOT / "eval_results" / "issue_1315" / "geometry" / "debiased_cosine.json"

LAYER = 14
N_DRAWS = 2000
SEED = 1112  # the #1112 half-draw convention
PAIRS = C.DIFF_PAIRS  # method pair + both negatives pairs (ICL panel)


def _q(draws: np.ndarray) -> dict[str, float]:
    return {str(q): float(np.nanquantile(draws, q)) for q in QUANTILES}


def main() -> int:
    base = geo.load_store(STAGE / "base_subsets" / "base_own_icl_prefix_impolite.pt")
    cells = sorted({c for p in PAIRS for c in p[1:]})
    clouds = {
        c: geo.delta_cloud(
            geo.load_store(STAGE / "capture" / c / "selected" / "pooled.pt"),
            base,
            "response",
            LAYER,
        )
        for c in cells
    }
    n = next(iter(clouds.values())).shape[0]
    qidx = np.array([int(m["question_idx"]) for m in base["row_meta"]])
    results: dict[str, dict] = {}
    for scheme, kw in (
        ("row_random", {}),
        ("question_aligned", {"question_idx": qidx}),
    ):
        masks = half_partition_masks(n, N_DRAWS, SEED, **kw)
        for name, a, b in PAIRS:
            d = batched_half_cosines(clouds[a], clouds[b], masks)
            cross, ra, rb_ = d["cross"], d["ref_a"], d["ref_b"]
            corrected = cross / np.sqrt(np.clip(ra * rb_, 1e-12, None))
            results.setdefault(name, {})[scheme] = {
                "cell_a": a,
                "cell_b": b,
                "full_cloud_cos_mu": float(
                    np.dot(clouds[a].mean(0), clouds[b].mean(0))
                    / (
                        np.linalg.norm(clouds[a].mean(0)) * np.linalg.norm(clouds[b].mean(0))
                        + 1e-12
                    )
                ),
                "cross_quantiles": _q(cross),
                "ref_a_quantiles": _q(ra),
                "ref_b_quantiles": _q(rb_),
                "corrected_quantiles": _q(corrected),
                "frac_cross_below_min_ref": float(np.mean(cross < np.minimum(ra, rb_))),
                "n_draws": N_DRAWS,
                "m": n // 2,
                "seed": SEED,
            }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"layer": LAYER, "arm": "response", "pairs": results}, indent=1))
    for name, schemes in results.items():
        r = schemes["question_aligned"]
        print(
            name,
            f"full-cloud cos {r['full_cloud_cos_mu']:.3f}; corrected median "
            f"{r['corrected_quantiles']['0.5']:.3f} [{r['corrected_quantiles']['0.025']:.3f}, "
            f"{r['corrected_quantiles']['0.975']:.3f}]; "
            f"frac cross<ref {r['frac_cross_below_min_ref']:.3f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
