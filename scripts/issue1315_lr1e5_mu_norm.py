#!/usr/bin/env python
"""#1315 fu-r1: registered paired mean-shift-norm difference for the lr contrast.

The plan (fu-r1 §3) registers "paired Δrank-k@90 and Δ‖μ‖ at layer 14 with
identical resample indices"; the round geometry runner persisted rank/PR/
top-share diffs only. This script computes the missing Δ‖μ‖ with the #1112
``‖μ‖`` companion convention verbatim (scripts/issue1112_geometry.py
``_mu_norm_draws`` + ``paired_diff_record``, mu_n_boot=2000, seed 653,
identical resample indices for both cells), response arm, all 28 layers,
for the pair (imp_conv_lora_lr1e5, imp_conv_lora) on the WildChat panel.

Writes eval_results/issue_1315/lr1e5_followup/geometry/mu_norm_diffs_lr.json
and injects ``diff_mu_norm`` into the round geometry_per_cell.json's
cross_cell_diffs['LRconv_lr1e5_vs_lr3e5']['reads']['response/L<l>'] entries.
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

from explore_persona_space.experiments.issue_653.spectral import (  # noqa: E402
    bootstrap_index_matrix,
)
from explore_persona_space.experiments.issue_1112 import geometry as geo  # noqa: E402

REPO_ROOT = _SCRIPTS_DIR.parent
STAGE = REPO_ROOT / "data" / "issue_1315" / "hf_dl" / "analysis_tensors"
GEO_DIR = REPO_ROOT / "eval_results" / "issue_1315" / "lr1e5_followup" / "geometry"
OUT = GEO_DIR / "mu_norm_diffs_lr.json"
ROUND_JSON = GEO_DIR / "geometry_per_cell.json"

BOOT_SEED = 653  # the #653/#1112 convention
LAYERS = list(range(28))
PAIR_NAME = "LRconv_lr1e5_vs_lr3e5"
CELL_A, CELL_B = "imp_conv_lora_lr1e5", "imp_conv_lora"  # a - b


def main() -> int:
    base_sub = STAGE / "base_subsets" / "base_own_wildchat_prefix_real545.pt"
    assert base_sub.exists(), f"missing staged base subset {base_sub}"
    base = geo.load_store(base_sub)
    n_rows = len(base["row_meta"])
    cluster_ids = [f"{m['context_id']}__{m['question_idx']}" for m in base["row_meta"]]
    idx_mu = bootstrap_index_matrix(cluster_ids, n_boot=MU_N_BOOT, seed=BOOT_SEED)
    w_mu = _draw_weight_matrix(idx_mu, n_rows)

    stores = {
        c: geo.load_store(STAGE / "capture" / c / "selected" / "pooled.pt")
        for c in (CELL_A, CELL_B)
    }
    mu_draws: dict[str, dict[int, np.ndarray]] = {c: {} for c in stores}
    mu_point: dict[str, dict[int, float]] = {c: {} for c in stores}
    for c in stores:
        for layer in LAYERS:
            cloud = geo.delta_cloud(stores[c], base, "response", layer)
            assert cloud.shape[0] == n_rows, (c, layer, cloud.shape)
            mu_draws[c][layer] = _mu_norm_draws(cloud, w_mu)
            mu_point[c][layer] = float(np.linalg.norm(cloud.mean(axis=0)))

    by_layer = {
        str(layer): geo.paired_diff_record(
            mu_draws[CELL_A][layer],
            mu_draws[CELL_B][layer],
            mu_point[CELL_A][layer],
            mu_point[CELL_B][layer],
        )
        for layer in LAYERS
    }
    payload = {
        "convention": "paired cluster bootstrap, mu_n_boot=2000 seed 653, response arm",
        "pair": {"name": PAIR_NAME, "cell_a": CELL_A, "cell_b": CELL_B},
        "diff_mu_norm_by_layer": by_layer,
        "mu_point_by_cell_layer": {c: {str(li): mu_point[c][li] for li in LAYERS} for c in stores},
    }
    OUT.write_text(json.dumps(payload, indent=1))

    # Inject into the round JSON's LR-pair reads (response arm, all layers).
    round_payload = json.loads(ROUND_JSON.read_text())
    reads = round_payload["cross_cell_diffs"][PAIR_NAME]["reads"]
    for layer in LAYERS:
        key = f"response/L{layer}"
        assert key in reads, key
        reads[key]["diff_mu_norm"] = by_layer[str(layer)]
    ROUND_JSON.write_text(json.dumps(round_payload, indent=1))

    p14 = by_layer["14"]
    print(
        "L14 diff_mu_norm:",
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p14.items()},
        "| points:",
        round(mu_point[CELL_A][14], 3),
        round(mu_point[CELL_B][14], 3),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
