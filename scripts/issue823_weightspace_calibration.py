"""#823 free-analysis follow-up: matched-design calibration cells for the weight-space read.

CORRECTIVE APPEND to weightspace_compare.json. The existing within-own
half1-vs-half2 cell (`noise_ceiling_*`) is NOT a valid noise ceiling for the
own-vs-plain full-data comparison: the two half-fits differ in X ROWS while the
own-vs-plain full-data fits share IDENTICAL X rows, so shared-X estimation
structure confounds the comparison (flat cos ~0.04 within-arm-different-rows vs
0.58-0.69 cross-arm-same-rows is the symptom).

Adds, per read-out layer (14/17/26), two matched-design calibration cells using
the SAME deterministic half split (np.random.default_rng(0).permutation of the
valid rows; h1 = first half, h2 = second half), SAME DualRidgeShared solver,
per-arm GCV lambda — imported verbatim from issue823_weightspace_compare:

  1. cross_arm_diff_rows      W_own(half1) vs W_plain(half2)  — different arms
     AND different rows. Apples-to-apples against the existing noise_ceiling cell
     (same arm, different rows): if similar, arm identity adds nothing beyond
     row-resampling noise in Frobenius terms.
  2. cross_arm_same_rows_half W_own(half1) vs W_plain(half1)  — different arms,
     SAME rows, at half n. Compare to the full-data flat cosine (n=4998 analog)
     to see how the shared-X inflation scales with n.

Per cell: flat cosine + k=50 subspace alignment (input=right-SV, output=left-SV
mean cos of principal angles).

Writes `matched_half_calibration` as a NEW top-level key and updates the
`description`; everything already in the JSON is left untouched.

Usage:
  uv run python scripts/issue823_weightspace_calibration.py
"""

from __future__ import annotations

import datetime
import json
import logging
import pathlib

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue823_weightspace_compare import (  # noqa: E402
    K_SWEEP,
    READ_OUT_LAYERS,
    DualRidgeShared,
    _flat_cosine,
    _load_arm,
    _load_bundle_cx_last,
    _subspace_alignment,
    _valid_idx,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue823_weightspace_calibration")

EXPECTED_N = 5000
K = 50  # the team-lead-requested subspace k for the calibration cells


def _cell(A: np.ndarray, B: np.ndarray) -> dict:
    sub = _subspace_alignment(A, B, K_SWEEP)[str(K)]
    return {
        "flat_cosine": _flat_cosine(A, B),
        "k": K,
        "input_mean_cos_principal_angle": sub["input_mean_cos_principal_angle"],
        "output_mean_cos_principal_angle": sub["output_mean_cos_principal_angle"],
    }


def main() -> None:
    torch.set_num_threads(8)
    base = pathlib.Path(__file__).resolve().parent.parent
    json_path = (
        base / "eval_results" / "issue_823" / "crossarm_transfer" / "weightspace_compare.json"
    )
    existing = json.loads(json_path.read_text())

    n = EXPECTED_N
    cx_last = _load_bundle_cx_last(n)
    v_own = _load_arm("a_prime", n)
    v_plain = _load_arm("b2", n)
    valid = _valid_idx(n)

    # IDENTICAL half split to issue823_weightspace_compare (seed 0, first/second half).
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(valid))
    half = len(valid) // 2
    h1, h2 = perm[:half], perm[half:]
    logger.info("valid=%d half=%d (h1=%d h2=%d)", len(valid), half, len(h1), len(h2))

    calib: dict[str, dict] = {}
    for L in sorted(set(READ_OUT_LAYERS.values())):
        X = cx_last[valid, L, :].numpy().astype(np.float64)
        Yo = v_own[valid, L, :].numpy().astype(np.float64)
        Yp = v_plain[valid, L, :].numpy().astype(np.float64)

        # per-half shared eigh (own + plain share X within a half)
        sh1 = DualRidgeShared(X[h1])
        W_own_h1, lam_own_h1 = sh1.W(Yo[h1])
        W_plain_h1, lam_plain_h1 = sh1.W(Yp[h1])
        sh2 = DualRidgeShared(X[h2])
        W_plain_h2, lam_plain_h2 = sh2.W(Yp[h2])

        calib[str(L)] = {
            "n_half": int(half),
            "lambda_own_h1": lam_own_h1,
            "lambda_plain_h1": lam_plain_h1,
            "lambda_plain_h2": lam_plain_h2,
            "cross_arm_diff_rows": _cell(W_own_h1, W_plain_h2),
            "cross_arm_same_rows_half": _cell(W_own_h1, W_plain_h1),
        }
        logger.info(
            "L%d: cross_arm_diff_rows flat=%.4f  cross_arm_same_rows_half flat=%.4f",
            L,
            calib[str(L)]["cross_arm_diff_rows"]["flat_cosine"],
            calib[str(L)]["cross_arm_same_rows_half"]["flat_cosine"],
        )

    existing["matched_half_calibration"] = {
        "description": (
            "Matched-design calibration for the weight-space own-vs-plain comparison. "
            "CONFOUND: the existing own-vs-plain full-data fits share IDENTICAL X rows "
            "(same Gram/eigenbasis), while the existing noise_ceiling cell (own h1 vs "
            "own h2) uses DIFFERENT X rows -- so shared-X estimation structure inflates "
            "the own-vs-plain flat cosine / output-subspace alignment and the "
            "half1-vs-half2 cell is NOT a valid upper bound. These two cells isolate the "
            "confound: cross_arm_diff_rows (W_own h1 vs W_plain h2; different arms AND "
            "rows) is apples-to-apples with noise_ceiling (same arm, different rows) -- "
            "similarity means arm identity adds nothing beyond row-resampling noise; "
            "cross_arm_same_rows_half (W_own h1 vs W_plain h1; different arms, SAME rows, "
            "half n) is the half-n analog of the full-data flat_cosine_own_vs_plain, so "
            "comparing them shows how shared-X inflation scales with n."
        ),
        "half_split": "np.random.default_rng(0).permutation(valid); h1=first, h2=second half",
        "k": K,
        "per_layer": calib,
        "reference_existing": {
            str(L): {
                "flat_cosine_own_vs_plain_full_n4998": existing["per_layer"][str(L)][
                    "flat_cosine_own_vs_plain"
                ],
                "noise_ceiling_own_h1_vs_h2_flat_cosine": existing["per_layer"][str(L)][
                    "noise_ceiling_flat_cosine_own_half1_vs_half2"
                ],
            }
            for L in sorted(set(READ_OUT_LAYERS.values()))
        },
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    existing["description"] = (
        existing["description"]
        + " APPENDED matched_half_calibration (see that key): the disjoint-half "
        "noise_ceiling cell uses different X rows while own-vs-plain shares X rows, so "
        "it is NOT a valid ceiling; the calibration cells (cross_arm_diff_rows, "
        "cross_arm_same_rows_half) isolate the shared-X confound at matched design."
    )
    json_path.write_text(json.dumps(existing, indent=1))
    logger.info("Appended matched_half_calibration to %s", json_path)


if __name__ == "__main__":
    main()
