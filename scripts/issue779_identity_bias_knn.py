# ruff: noqa: RUF002, RUF003
"""#779 inline free-analysis (user-chat, 2026-07-22): identity+bias baseline + kNN retrieval.

Adds two reads to the ffc context→answer map, under the SAME fixed split
(3600/400/1000, seed 42), the SAME per-variant val-selected ridge layer, and
the SAME bootstrap-CI machinery as fair_comparison.json:

- ``identity_plus_bias`` — v̂_A = v_C + b, b = train-mean(v_A − v_C): the
  W=identity learned-bias baseline (the missing member of the existing
  identity family: raw copy / scaled / diagonal).
- kNN retrieval — P(true v_A within k nearest neighbors of the prediction)
  among the 1000 held-out test targets, k ∈ {1, 5, 10}, euclidean + cosine,
  for the ridge map AND every baseline arm.

Read-only on the live n1m round's outputs; writes ONLY
``eval_results/issue_779/identity_bias_knn/results.json``.

Usage::

    uv run python scripts/issue779_identity_bias_knn.py
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
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue779_ffc_baselines as B  # noqa: E402  (_fit_scale / _fit_diag)
import issue779_fitter_fair_comparison as F  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue779_identity_bias_knn")

FC_PATH = PROJECT_ROOT / "eval_results" / "issue_779" / "fitter-fair-comparison"
OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_779" / "identity_bias_knn"
KS = (1, 5, 10)


def main() -> int:
    t0 = time.time()
    fc = json.loads((FC_PATH / "fair_comparison.json").read_text())
    sp = fc["split"]
    assert (sp["n_contexts"], sp["n_train"], sp["n_val"], sp["n_test"], sp["seed"]) == (
        5000,
        3600,
        400,
        1000,
        42,
    ), sp
    train, val, test = F.fixed_split(5000, 3600, 400, 1000, F.SPLIT_SEED)
    bundle = F.load_pass_b()
    dev = torch.device("cpu")

    out: dict = {"split": sp, "ks": list(KS), "inputs": {}}
    for variant in ("last", "mean"):
        li = int(fc["inputs"][variant]["ridge"]["val_selected_layer"])
        logger.info("[%s] val-selected ridge layer %d", variant, li)
        x = F.input_layer(bundle, variant, li)
        y = F.target_vx(bundle, li)
        xtr, xva, xte = x[train], x[val], x[test]
        ytr, yva, yte = y[train], y[val], y[test]

        preds: dict[str, np.ndarray] = {}
        (ridge_te,), lam = F.gram_fit_apply(xtr, ytr, [xte], dev, val=(xva, yva))
        preds["ridge"] = np.asarray(ridge_te)
        preds["identity_copy"] = xte.astype(np.float64)
        preds["identity_plus_bias"] = identity_bias_predict(xtr, ytr, xte)
        preds["scaled_identity"] = B._fit_scale(xtr, ytr) * xte
        preds["diagonal_only"] = xte * B._fit_diag(xtr, ytr)
        preds["predict_the_mean"] = np.tile(ytr.mean(0), (len(test), 1))
        logger.info("[%s] ridge refit done (lambda=%.4g, %.1fs)", variant, lam, time.time() - t0)

        arms = {}
        for i, (name, p) in enumerate(preds.items()):
            r2 = F._bootstrap_recon_ci(p, yte, F.BOOT_N, seed=1000 + i)
            knn = {m: knn_retrieval(p, yte, ks=KS, metric=m) for m in ("euclidean", "cosine")}
            arms[name] = {"r2": r2, "knn": knn}
            logger.info(
                "[%s] %-18s R2 %.4f  acc@1/5/10 (eucl) %.3f/%.3f/%.3f  med-rank %.1f",
                variant,
                name,
                r2["r2"]["point"],
                *(knn["euclidean"]["acc_at_k"][k] for k in KS),
                knn["euclidean"]["median_rank"],
            )
        out["inputs"][variant] = {"layer": li, "ridge_lambda": float(lam), "arms": arms}

    out["metadata"] = {
        "script": "issue779_identity_bias_knn",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
        ).stdout.strip(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "provenance": (
            "user-chat inline free-analysis 2026-07-22; identity+bias baseline + kNN retrieval "
            "on the round-1 ffc pass_b tensors (data/issue_779/pass_b), same fixed split / "
            "val-selected layers / bootstrap machinery as fair_comparison.json; kNN pool = "
            "the 1000 held-out test targets, mid-rank ties (constant predictors score 0, "
            "chance = k/n_pool reported alongside)"
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=1))
    logger.info("wrote %s (%.1fs total)", OUT_DIR / "results.json", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
