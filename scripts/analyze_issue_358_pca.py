"""Issue #358 — Step 2 (PCA).

Reads `eval_results/issue_358/acts_poisoned.pt` (and `acts_base.pt` for the
appendix panel), fits PCA(10) on the layer-18 activations of the **binary
pool only** (plan §4.4) — i.e. excludes the 6 long-PERSONA scatter-only
rows from the covariance estimate so the principal-component axes are
defined by the trigger-vs-paraphrase contrast the probe is also testing.
Then projects ALL 109 rows into that basis for plotting.

The PCA-fit `StandardScaler` here is GLOBAL (fit on the binary pool, used
for projecting every row). It MUST NOT be reused as the per-fold scaler
in ``scripts/analyze_issue_358_probe.py`` — the probe's pooled-LOPO uses
a *per-fold* scaler so held-out variance does not leak into normalisation.
This module's scaler is intentionally a different object.

Output:
  eval_results/issue_358/pca_coords.json — Z (N, 10) coords + per-component
      variance explained + labels, for both the poisoned-model (primary)
      and base-model (appendix) panels.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from explore_persona_space.metadata import get_run_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("issue_358_pca")

INPUT_DIR = Path("eval_results/issue_358")
HEADLINE_LAYER = 18
N_COMPONENTS = 10


def _label_for(cond: dict[str, Any]) -> str:
    """Plot-class label. Long PERSONAs are split off as PERSONA-LONG so the
    figure can flag them at 60% alpha (scatter-only — not in the probe pool).
    """
    if cond["class"] == "PERSONA-PROMPT" and not cond["binary_pool"]:
        return "PERSONA-LONG"
    if cond["class"] == "PERSONA-PROMPT":
        return "PERSONA-SHORT"
    return cond["class"]


def _fit_one_model(acts_path: Path) -> dict[str, Any]:
    log.info("loading %s", acts_path)
    D = torch.load(acts_path, weights_only=False)
    conditions: list[dict] = D["conditions"]
    # Layer 18 activations across ALL rows (incl. PERSONA-LONG which will
    # be projected but not used for the fit).
    acts_all = D["activations"][:, HEADLINE_LAYER, :].numpy()  # (N, hidden)
    pool_mask = np.asarray([c["binary_pool"] for c in conditions], dtype=bool)
    log.info(
        "fit pool: %d / %d rows (PCA fit on binary pool only — see module docstring)",
        int(pool_mask.sum()),
        len(conditions),
    )

    # GLOBAL StandardScaler — fit on the binary-pool covariance.
    scaler = StandardScaler(with_mean=True, with_std=True).fit(acts_all[pool_mask])
    Xz_pool = scaler.transform(acts_all[pool_mask])
    pca = PCA(n_components=N_COMPONENTS).fit(Xz_pool)
    Xz_all = scaler.transform(acts_all)
    Z = pca.transform(Xz_all)  # (N, 10)

    labels = [_label_for(c) for c in conditions]

    return {
        "model_id": D.get("model_id"),
        "revision": D.get("revision"),
        "layer": HEADLINE_LAYER,
        "n_components": N_COMPONENTS,
        "variance_explained": pca.explained_variance_ratio_.tolist(),
        "variance_explained_cum": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "coords": Z.tolist(),  # (N, 10)
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
    out_path = INPUT_DIR / "pca_coords.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info("wrote %s", out_path)

    pois = out["poisoned"]
    log.info(
        "poisoned-model PC1+PC2 cumulative variance: %.2f%%",
        100 * pois["variance_explained_cum"][1],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
