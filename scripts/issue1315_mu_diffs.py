#!/usr/bin/env python
"""#1315 registered mean-shift-norm paired differences (D_mag / H5 + descriptive).

The #1112 ``‖μ‖`` companion convention verbatim (scripts/issue1112_geometry.py:
``_mu_norm_draws`` + ``paired_diff_record``, mu_n_boot=2000, seed 653, identical
resample indices per pair): per registered ICL pair, per layer, the paired
cluster-bootstrap difference of mean-shift norms on the response arm.

Writes eval_results/issue_1315/geometry/mu_norm_diffs.json.
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

from issue1112_geometry import MU_N_BOOT, _draw_weight_matrix, _mu_norm_draws  # noqa: E402

from explore_persona_space.experiments import issue_1315 as C  # noqa: E402
from explore_persona_space.experiments.issue_653.spectral import (  # noqa: E402
    bootstrap_index_matrix,
)
from explore_persona_space.experiments.issue_1112 import geometry as geo  # noqa: E402

REPO_ROOT = _SCRIPTS_DIR.parent
STAGE = REPO_ROOT / "data" / f"issue_{C.ISSUE}" / "hf_dl" / "analysis_tensors"
OUT = REPO_ROOT / "eval_results" / "issue_1315" / "geometry" / "mu_norm_diffs.json"

BOOT_SEED = 653  # the #653/#1112 convention
LAYERS = list(range(28))


def main() -> int:
    base_sub = STAGE / "base_subsets" / "base_own_icl_prefix_impolite.pt"
    assert base_sub.exists(), f"run issue1315_geometry.py first (missing {base_sub})"
    base = geo.load_store(base_sub)
    n_rows = len(base["row_meta"])
    cluster_ids = [f"{m['context_id']}__{m['question_idx']}" for m in base["row_meta"]]
    idx_mu = bootstrap_index_matrix(cluster_ids, n_boot=MU_N_BOOT, seed=BOOT_SEED)
    w_mu = _draw_weight_matrix(idx_mu, n_rows)

    cells = ["imp_icl_lora_neg", "imp_icl_lora_pos", "imp_icl_ft_neg", "imp_icl_ft_pos"]
    stores = {c: geo.load_store(STAGE / "capture" / c / "selected" / "pooled.pt") for c in cells}
    mu_draws: dict[str, dict[int, np.ndarray]] = {c: {} for c in cells}
    mu_point: dict[str, dict[int, float]] = {c: {} for c in cells}
    for c in cells:
        for layer in LAYERS:
            cloud = geo.delta_cloud(stores[c], base, "response", layer)
            mu_draws[c][layer] = _mu_norm_draws(cloud, w_mu)
            mu_point[c][layer] = float(np.linalg.norm(cloud.mean(axis=0)))

    def pair_by_layer(a: str, b: str) -> dict[str, dict]:
        return {
            str(layer): geo.paired_diff_record(
                mu_draws[a][layer], mu_draws[b][layer], mu_point[a][layer], mu_point[b][layer]
            )
            for layer in LAYERS
        }

    payload = {
        "convention": "paired cluster bootstrap, mu_n_boot=2000 seed 653, response arm",
        "pairs": {name: pair_by_layer(a, b) for name, a, b in C.DIFF_PAIRS},
        "mu_point_by_cell_layer": {c: {str(li): mu_point[c][li] for li in LAYERS} for c in cells},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    for name, _a, _b in C.DIFF_PAIRS:
        p = payload["pairs"][name]["14"]
        print(name, "L14:", {k: round(v, 3) for k, v in p.items() if isinstance(v, float)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
