#!/usr/bin/env python3
"""Rescore Figure 2's single-turn training-size sweep on five-rollout means.

The fitted linear and nonlinear predictions are banked for nine training-set
sizes. This script holds those predictions fixed and changes only evaluation:

* target = mean(original on-policy answer + four fresh on-policy answers);
* duplicate policy = retain one representative of each exact source-answer
  vector equivalence class (942 of the original 1,000 rows);
* retrieval = whitened cosine + two-sided CSLS (K=10);
* whitening = the banked 963,444-example single-turn training-answer statistics.

No model inference or refitting is performed. Inputs are pinned to the same HF
dataset revision as ``issue1901_singleturn_retrieval_final.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

import issue779_common as C  # noqa: E402
import issue779_fitter_fair_comparison as F79  # noqa: E402
import issue1901_singleturn_retrieval_final as FINAL  # noqa: E402


logger = logging.getLogger("issue1901_figure2_five_rollout_scaling")

REVISION = "83d249cc9d495ca6f5d10f9156a622bcdca29a19"
LAYER = 19
N_TEST = 1_000
TRAIN_SIZES = (5_000, 10_000, 25_000, 50_000, 100_000, 150_000, 250_000, 500_000, 963_444)
PREDICTORS = ("ridge", "mlp")
DEFAULT_STAGE = PROJECT_ROOT / "data/issue_1901/figure2_five_rollout_scaling"
DEFAULT_OUT = PROJECT_ROOT / "eval_results/issue_1901/figure2_five_rollout_scaling.json"

BASE_FILES = {
    "pass_b": "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt",
    "whiten": "issue1901_mlpdense/analysis_tensors/whiten_stats_L19.npz",
    "test_draws": "issue1901_avgpool/analysis_tensors/kresample/V_test_shard00.npz",
}


def _prediction_path(n_train: int, predictor: str) -> str:
    return f"issue1901_mlpdense/analysis_tensors/preds_L19_n{n_train}_{predictor}.npz"


def _sha256(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def _stage(stage_root: Path) -> dict[str, Path]:
    files = dict(BASE_FILES)
    for n_train in TRAIN_SIZES:
        for predictor in PREDICTORS:
            files[f"pred_{predictor}_{n_train}"] = _prediction_path(n_train, predictor)

    out = {}
    for key, filename in files.items():
        logger.info("[stage] %s", filename)
        out[key] = Path(
            hf_hub_download(
                repo_id=C.HF_DATA_REPO,
                repo_type="dataset",
                filename=filename,
                revision=REVISION,
                local_dir=stage_root,
            )
        )
    return out


def _load_five_rollout_target(
    pass_b: Path, draws: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bundle = F79.load_pass_b(pass_b)
    n_contexts = int(bundle["cx_last"].shape[0])
    _train, _val, test = F79.fixed_split(
        n_contexts,
        n_contexts - 400 - N_TEST,
        400,
        N_TEST,
        F79.SPLIT_SEED,
    )
    target = np.asarray(F79.target_vx(bundle, LAYER)[test], dtype=np.float32)
    del bundle

    capture = np.load(draws, allow_pickle=False)
    fresh = np.asarray(capture["V"], dtype=np.float32)
    capture_ids = np.asarray(capture["ci"], dtype=np.int64)
    assert fresh.shape == (N_TEST, FINAL.K_DRAWS, C.EXPECTED_HIDDEN), fresh.shape
    assert len(np.unique(capture_ids)) == N_TEST
    row_for_id = {int(capture_id): i for i, capture_id in enumerate(capture_ids)}
    expected_ids = -(1 + np.arange(N_TEST, dtype=np.int64))
    if set(row_for_id) != set(expected_ids.tolist()):
        raise RuntimeError("five-rollout test capture IDs do not match the pinned test bank")
    fresh = np.stack([fresh[row_for_id[int(capture_id)]] for capture_id in expected_ids])
    target_mean = (
        target.astype(np.float64) + fresh.sum(axis=1, dtype=np.float64)
    ) / (FINAL.K_DRAWS + 1)
    return target, target_mean, np.asarray(test, dtype=np.int64)


def _load_prediction(path: Path, expected_rows: np.ndarray) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    rows = np.asarray(payload["rows"], dtype=np.int64)
    if not np.array_equal(rows, expected_rows):
        raise RuntimeError(f"prediction rows do not match the pinned test split: {path.name}")
    pred = np.asarray(payload["pred_fp16"], dtype=np.float64)
    assert pred.shape == (N_TEST, C.EXPECTED_HIDDEN), pred.shape
    return pred


def _score(paths: dict[str, Path]) -> dict:
    source_target, target_mean, expected_rows = _load_five_rollout_target(
        paths["pass_b"], paths["test_draws"]
    )
    view = FINAL.make_eval_view(source_target, N_TEST, "keep_one")
    if view.diagnostics["realized_n_pool"] != 942 or view.diagnostics["realized_n_query"] != 942:
        raise RuntimeError(f"unexpected deduplicated test geometry: {view.diagnostics}")
    whiten, whitening = FINAL._whitener(paths["whiten"])

    per_n = {}
    for n_train in TRAIN_SIZES:
        per_predictor = {}
        for predictor in PREDICTORS:
            pred_path = paths[f"pred_{predictor}_{n_train}"]
            pred = _load_prediction(pred_path, expected_rows)
            r2, mean_cosine = F79._recon_point(pred, target_mean)
            retrieval = FINAL.score_cell(
                pred,
                target_mean,
                view,
                whiten,
                seed=190_102 + n_train + (0 if predictor == "ridge" else 1),
            )["whiten_csls"]["strict"]
            per_predictor[predictor] = {
                "r2": float(r2),
                "mean_cosine": float(mean_cosine),
                "top1": float(retrieval["acc_at_k"]["1"]),
                "top5": float(retrieval["acc_at_k"]["5"]),
                "top1_ci95": retrieval["acc1_ci95"],
                "prediction_file": pred_path.name,
                "prediction_sha256": _sha256(pred_path),
            }
            logger.info(
                "[score] n=%d predictor=%s R2=%.4f top1=%.4f",
                n_train,
                predictor,
                r2,
                retrieval["acc_at_k"]["1"],
            )
        per_n[str(n_train)] = per_predictor

    return {
        "issue": 1901,
        "analysis": "figure2-five-rollout-scaling",
        "layer": LAYER,
        "target": "mean(original answer vector + four fresh on-policy answer vectors)",
        "n_rollouts": FINAL.K_DRAWS + 1,
        "retrieval": {
            "metric": "whitened cosine + two-sided CSLS",
            "csls_k": FINAL.K_CSLS,
            "duplicate_policy": "keep_one exact source-answer-vector equivalence class",
            "n_query": view.diagnostics["realized_n_query"],
            "n_pool": view.diagnostics["realized_n_pool"],
            "rank": "strict top-k; mid-rank ties and top ties fail top-1",
        },
        "whitening": whitening,
        "data_revision": REVISION,
        "source_sha256": {
            "pass_b": _sha256(paths["pass_b"]),
            "whiten": _sha256(paths["whiten"]),
            "test_draws": _sha256(paths["test_draws"]),
        },
        "test_rows_sha256": hashlib.sha256(expected_rows.tobytes()).hexdigest(),
        "duplicate_audit": view.diagnostics,
        "per_n": per_n,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    paths = _stage(args.stage_root)
    result = _score(paths)
    _write_json(args.out, result)
    print(args.out)


if __name__ == "__main__":
    main()
