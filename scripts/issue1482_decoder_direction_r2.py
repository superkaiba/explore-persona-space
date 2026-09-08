#!/usr/bin/env python3
"""Issue #1482: per-direction held-out R^2 for all 131,072 layer-19 SAE decoder
columns, the decoder-direction twin of the paper's per-feature activation R^2.

A reviewer asked for the SAE property analysis "repeated on decoder directions
rather than feature activations". This produces the swapped dependent variable
and nothing else, so the downstream property analysis can run unchanged.

  paper's DV    held-out R^2 of predicting feature f's MEAN ACTIVATION over the
                answer. Passes the encoder and the BatchTopK gate, so for a rare
                feature the target is mostly zeros and the estimate rests on a
                handful of non-zero answers.
  this DV       held-out R^2 of predicting d_f^T v, the answer state's component
                along f's UNIT decoder column. Dense, defined on all 20,000
                held-out answers, no encoder, no gate, no threshold. Every
                direction is scored on the same rows however rarely its feature
                fires, which closes the firing-rate-dependent noise channel the
                reviewer named.

Inputs are all banked: the recovered answer-state truth (identity-gated on every
one of the 20,000 rows, max relative error 1.8e-3 against the banked per-row
squared error) and the round's own ridge prediction. No new fit is performed
here, so this measures the SAME map the paper's numbers measure, read along a
different direction.

Estimator: 1 - ||E u||^2 / ||Yc u||^2, evaluated as quadratic forms against the
two Gram matrices so the (n, 131072) products never materialize. This recipe
reproduces the banked dense_direction_r2 for the four locally available trait
directions to 3.3e-16 (gate below, run every invocation).

0 GPU, no pod: two 143 MB arrays and a cached decoder.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps BEFORE numpy (shared-VM discipline)

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_sae as S  # noqa: E402

TRUTH = (
    "data/issue_1482/holdout_truth/issue1482_error_analysis/analysis_tensors/"
    "holdout_truth/v_holdout_L19_full.npz"
)
PRED = "data/issue_1482/percontext/refit_holdout__ridge__seed0.npz"
TRAITS = ("apathetic", "humorous", "impolite", "optimistic")
RB_BANKED = "eval_results/issue_1482/rb_as_sae_feature/rb_as_feature.json"
STAGE = "data/issue_1482/twoway_stage"
LAYER, SAE_K, CHUNK = 19, 64, 8192
GATE_TOL = 1e-9


def _log(m: str) -> None:
    print(f"[dec-r2 {datetime.now(UTC).strftime('%H:%M:%S')}] {m}", flush=True)


def per_direction_r2(Yc: np.ndarray, E: np.ndarray, W: np.ndarray) -> np.ndarray:
    """1 - ||E u||^2 / ||Yc u||^2 per unit column of W, chunked over the dictionary."""
    G_y, G_e = Yc.T @ Yc, E.T @ E
    out = np.empty(W.shape[1], dtype=np.float64)
    for s in range(0, W.shape[1], CHUNK):
        U = W[:, s : s + CHUNK]
        out[s : s + CHUNK] = 1.0 - np.einsum("df,df->f", G_e @ U, U) / np.einsum(
            "df,df->f", G_y @ U, U
        )
    return out


def estimator_gate() -> dict:
    """Reproduce banked trait dense_direction_r2 on the #1738 footing. Fails loud."""
    import torch

    with np.load(PROJECT_ROOT / STAGE / f"y_parent_L{LAYER}.npz") as z:
        Y = np.asarray(z["y16"], dtype=np.float64)
    with np.load(PROJECT_ROOT / STAGE / f"pred_context_L{LAYER}_ridge.npz") as z:
        P = np.asarray(z["pred16"], dtype=np.float64)
    Yc, E = Y - Y.mean(axis=0, keepdims=True), Y - P
    banked = json.loads((PROJECT_ROOT / RB_BANKED).read_text())["per_trait"]
    worst, rows = 0.0, {}
    for t in TRAITS:
        d = torch.load(
            PROJECT_ROOT / f"data/issue_779/r_b/{t}.pt", map_location="cpu", weights_only=False
        )
        u = np.asarray(d["r_b"][LAYER], dtype=np.float64)
        u = (u / np.linalg.norm(u))[:, None]
        got = float(per_direction_r2(Yc, E, u)[0])
        ref = banked[t]["dense_direction_r2"]["context"]
        worst = max(worst, abs(got - ref))
        rows[t] = {"recomputed": got, "banked": ref, "abs_diff": abs(got - ref)}
    verdict = "PASS" if worst <= GATE_TOL else "FAIL"
    _log(f"estimator gate {verdict}: max abs diff {worst:.2e} over {len(TRAITS)} traits")
    if verdict != "PASS":
        raise SystemExit(f"ESTIMATOR GATE FAIL: {worst:.3e} > {GATE_TOL:.1e}")
    return {"verdict": verdict, "max_abs_diff": worst, "traits": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sae-cache", default="data/issue_1738/fullwidth/sae_cache")
    ap.add_argument("--out-dir", default="eval_results/issue_1482/decoder_direction")
    args = ap.parse_args()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gate = estimator_gate()

    with np.load(PROJECT_ROOT / TRUTH) as z:
        Y = np.asarray(z["v16"], dtype=np.float64)
        ci = np.asarray(z["ci"], dtype=np.int64)
    with np.load(PROJECT_ROOT / PRED) as z:
        rows = np.asarray(z["holdout_rows"], dtype=np.int64)
        P = np.asarray(z["holdout_pred16"], dtype=np.float64)
    if not np.array_equal(ci, rows):
        raise RuntimeError("recovered truth row order does not match the banked prediction")
    _log(f"aligned {Y.shape[0]:,} rows x {Y.shape[1]} dims")

    Yc, E = Y - Y.mean(axis=0, keepdims=True), Y - P
    pooled = 1.0 - float((E**2).sum()) / float((Yc**2).sum())
    _log(f"pooled dense R^2 = {pooled:.4f}  (banked refit_check reference 0.6531)")

    sae = S.BatchTopKSAE.load(k=SAE_K, layer=LAYER, device="cpu", cache_dir=args.sae_cache)
    W_dec = np.asarray(sae.w_dec.detach().cpu().numpy(), dtype=np.float64)
    if W_dec.shape[0] != Y.shape[1]:
        W_dec = W_dec.T  # want (d, n_feat)
    W = W_dec / (np.linalg.norm(W_dec, axis=0, keepdims=True) + 1e-12)
    _log(f"decoder {W.shape[1]:,} directions, unit-normalized")

    r2 = per_direction_r2(Yc, E, W)
    np.save(out_dir / "decoder_direction_r2_fullwidth.npy", r2)
    meta = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "layer": LAYER,
        "n_rows": int(Y.shape[0]),
        "n_directions": int(W.shape[1]),
        "pooled_dense_r2": pooled,
        "pooled_reference_banked": 0.6531218765443059,
        "estimator_gate": gate,
        "quantiles": {
            q: float(np.quantile(r2, float(q))) for q in ("0.05", "0.25", "0.5", "0.75", "0.95")
        },
        "target": "decoder-direction projection d_f^T v (dense, no encoder/gate)",
    }
    (out_dir / "decoder_direction_r2_meta.json").write_text(json.dumps(meta, indent=1))
    _log(
        f"per-direction R^2 median {np.median(r2):.4f}  "
        f"IQR [{np.quantile(r2, 0.25):.4f}, {np.quantile(r2, 0.75):.4f}]"
    )
    _log(f"wrote {out_dir / 'decoder_direction_r2_fullwidth.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
