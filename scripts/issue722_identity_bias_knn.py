# ruff: noqa: RUF002, RUF003
"""#722-line inline free-analysis (user-chat, 2026-07-22): identity+bias + kNN retrieval
for the PREFIX-LEVEL context→answer map (the #658 50-context × 7-family battery).

The map: #594 last-input-token c_C (query-averaged, per context) → #658 mean
answer summary v_A (query-averaged, per context), leave-one-family-out (LOFO)
ridge — the R²≈0.8@L18 line #722's M0 fit rides on. Adds, per layer:

- ``identity_plus_bias`` — v̂_A = c_C + b, b = train-fold mean(v_A − c_C).
- ``identity_copy`` and ``predict_the_mean`` floors.
- kNN retrieval — P(true v_A within k nearest neighbors of the pooled-OOF
  prediction) among all 50 battery targets, k ∈ {1, 3, 5}, euclidean + cosine.

R² convention: pooled OOF R² over all 50 contexts (SS_tot on the pooled own
mean — the ffc heldout_recon convention). 0-GPU, VM CPU, seconds per layer.
Writes ONLY ``eval_results/issue_722/identity_bias_knn/results.json``.

Usage::

    uv run python scripts/issue722_identity_bias_knn.py
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps BEFORE numpy/torch (shared-VM rule)

import datetime
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue810_common import battery_family_map  # noqa: E402
from issue810_fit_reconstruction import _load_cc, _load_free_summaries  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.experiments.issue_779.fit_h import ridge_fit_predict  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue722_identity_bias_knn")

OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_722" / "identity_bias_knn"
KS = (1, 3, 5)


def _pooled_r2(pred: np.ndarray, true: np.ndarray) -> float:
    mu = true.mean(0)
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - mu) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-12)


def main() -> int:
    t0 = time.time()
    summaries, capture_layers = _load_free_summaries("betley")
    mean_summ = summaries["mean"]  # {ctx_id: (Lc, H)}
    ctx_ids = sorted(mean_summ)
    fam_map = battery_family_map(PROJECT_ROOT / "data" / "issue594" / "battery.json")
    fams = sorted({fam_map[c] for c in ctx_ids})
    assert len(ctx_ids) == 50 and len(fams) == 7, (len(ctx_ids), fams)
    cc = _load_cc(ctx_ids, list(range(len(capture_layers))))
    logger.info("loaded 50-context battery: %d layers, families %s", len(capture_layers), fams)

    fam_of = np.array([fam_map[c] for c in ctx_ids])
    folds = [np.where(fam_of == f)[0] for f in fams]
    arm_names = ("ridge", "identity_copy", "identity_plus_bias", "predict_the_mean")

    per_layer: dict[str, dict] = {}
    for lc, li in enumerate(capture_layers):
        x = np.stack([cc[c][lc] for c in ctx_ids]).astype(np.float64)
        y = np.stack([np.asarray(mean_summ[c][lc].float(), dtype=np.float64) for c in ctx_ids])
        oof = {name: np.zeros_like(y) for name in arm_names}
        for te in folds:
            tr = np.setdiff1d(np.arange(len(ctx_ids)), te)
            oof["ridge"][te] = ridge_fit_predict(x[tr], y[tr], x[te])
            oof["identity_copy"][te] = x[te]
            oof["identity_plus_bias"][te] = identity_bias_predict(x[tr], y[tr], x[te])
            oof["predict_the_mean"][te] = np.tile(y[tr].mean(0), (len(te), 1))
        arms = {}
        for name in arm_names:
            arms[name] = {
                "r2_pooled_oof": _pooled_r2(oof[name], y),
                "knn": {
                    m: knn_retrieval(oof[name], y, ks=KS, metric=m) for m in ("euclidean", "cosine")
                },
            }
        per_layer[str(int(li))] = arms
        logger.info(
            "L%02d  ridge R2 %.3f acc@1/3/5 %.2f/%.2f/%.2f | id+bias R2 %.3f "
            "acc@1/3/5 %.2f/%.2f/%.2f | copy R2 %.3f",
            li,
            arms["ridge"]["r2_pooled_oof"],
            *(arms["ridge"]["knn"]["euclidean"]["acc_at_k"][k] for k in KS),
            arms["identity_plus_bias"]["r2_pooled_oof"],
            *(arms["identity_plus_bias"]["knn"]["euclidean"]["acc_at_k"][k] for k in KS),
            arms["identity_copy"]["r2_pooled_oof"],
        )

    best_li = max(per_layer, key=lambda k: per_layer[k]["ridge"]["r2_pooled_oof"])
    out = {
        "design": {
            "n_contexts": 50,
            "families": fams,
            "folds": "leave-one-family-out",
            "input": "c_C last-input-token, query-averaged (#594 store)",
            "target": "v_A mean answer summary, query-averaged (#658 store/v0_summaries.pt)",
            "ridge": "fit_h.ridge_fit_predict (numpy-SVD, GCV over logspace(-2,4,13))",
            "knn_pool": "all 50 battery targets, mid-rank ties",
            "ks": list(KS),
        },
        "best_ridge_layer": int(best_li),
        "per_layer": per_layer,
        "metadata": {
            "script": "issue722_identity_bias_knn",
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
            ).stdout.strip(),
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "provenance": (
                "user-chat inline free-analysis 2026-07-22; identity+bias baseline + kNN "
                "retrieval on the prefix-level battery map (#594 c_C → #658 mean v_A), LOFO"
            ),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=1))
    logger.info("wrote %s (%.1fs total)", OUT_DIR / "results.json", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
