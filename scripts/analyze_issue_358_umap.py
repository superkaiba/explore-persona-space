"""Issue #358 — Step 3 (UMAP).

Reads `eval_results/issue_358/acts_*.pt`, fits UMAP(2) on the layer-18
activations of the binary-pool rows (same input shape and z-scoring as the
PCA step), then projects ALL 109 rows into the embedding for plotting.

Two parameter settings written to disk:
  - `n_neighbors=15` (primary; umap-learn default)
  - `n_neighbors=5`  (small-n sanity per plan §4.5)

`metric="cosine"`, `min_dist=0.1`, `random_state=42` for both.

Output:
  eval_results/issue_358/umap_coords.json — embedding for both panels and
      both models. See module-level for the JSON schema.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from explore_persona_space.metadata import get_run_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("issue_358_umap")

INPUT_DIR = Path("eval_results/issue_358")
HEADLINE_LAYER = 18
PANELS: list[tuple[str, int]] = [
    ("n_neighbors_15", 15),
    ("n_neighbors_5", 5),
]


def _label_for(cond: dict[str, Any]) -> str:
    """Same label scheme as PCA — PERSONA-LONG split off for scatter-only
    alpha handling in the plot script.
    """
    if cond["class"] == "PERSONA-PROMPT" and not cond["binary_pool"]:
        return "PERSONA-LONG"
    if cond["class"] == "PERSONA-PROMPT":
        return "PERSONA-SHORT"
    return cond["class"]


def _fit_one_model(acts_path: Path) -> dict[str, Any]:
    # Lazy import: users without the `viz` extra installed can still run
    # the PCA + probe scripts; only this UMAP step requires umap-learn.
    import umap

    log.info("loading %s", acts_path)
    D = torch.load(acts_path, weights_only=False)
    conditions: list[dict] = D["conditions"]
    acts_all = D["activations"][:, HEADLINE_LAYER, :].numpy()
    pool_mask = np.asarray([c["binary_pool"] for c in conditions], dtype=bool)
    log.info("umap fit-pool: %d / %d rows", int(pool_mask.sum()), len(conditions))

    # Same global z-scoring as PCA — fit on the binary pool, apply to all.
    scaler = StandardScaler(with_mean=True, with_std=True).fit(acts_all[pool_mask])
    Xz_pool = scaler.transform(acts_all[pool_mask])
    Xz_all = scaler.transform(acts_all)

    labels = [_label_for(c) for c in conditions]

    panels: dict[str, dict[str, Any]] = {}
    for tag, n_neighbors in PANELS:
        log.info("fitting umap-learn: %s (n_neighbors=%d)", tag, n_neighbors)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
            n_components=2,
        )
        # Fit on the binary pool, transform all rows. Matches the PCA pattern
        # (fit-on-pool, transform-all) for like-for-like geometry comparison
        # between PCA and UMAP panels.
        reducer.fit(Xz_pool)
        emb = reducer.transform(Xz_all)
        panels[tag] = {
            "n_neighbors": n_neighbors,
            "min_dist": 0.1,
            "metric": "cosine",
            "random_state": 42,
            "coords": emb.tolist(),  # (N, 2)
        }

    return {
        "model_id": D.get("model_id"),
        "revision": D.get("revision"),
        "layer": HEADLINE_LAYER,
        "panels": panels,
        "labels": labels,
        "binary_pool": pool_mask.tolist(),
        "cids": [c["cid"] for c in conditions],
        "n_tokens": [c["n_tokens"] for c in conditions],
        "anth_token_bearing": [c["anth_token_bearing"] for c in conditions],
        "bin": [c.get("bin") for c in conditions],
    }


def main() -> int:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "poisoned": _fit_one_model(INPUT_DIR / "acts_poisoned.pt"),
        "base": _fit_one_model(INPUT_DIR / "acts_base.pt"),
        "metadata": get_run_metadata(),
    }
    out_path = INPUT_DIR / "umap_coords.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
