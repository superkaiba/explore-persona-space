"""Plain-identity vs identity+bias spot check on ONE #1336 turnstore cell.

The pair files persist the identity+BIAS within-stage baseline
(``per_layer.<L>.baselines.within.identity_bias_r2``, the canonical
``analysis/mapping_baselines.identity_bias_predict`` read, out-of-fold) but
PLAIN identity (v̂ = x, no learned bias) is computed nowhere in the repo and
is NOT recoverable from the stored R² values: the gap between the two is the
squared mean-offset term ``n·||mean(y − x)||² / TSS``, which no persisted
field carries.

Recomputing plain identity therefore needs the raw activations, and the full
within-class sweep (4 target stages × 8 corpus-format cells) is ~300-470 GB
— a pod job. This script is the bounded SPOT CHECK on the single cheapest
complete cell instead: it reproduces the target's identity+bias value from
the staged store and, if that matches the number the pair file already
persists, the store + slot convention are confirmed and the plain-identity
number computed beside it is trustworthy. If it does NOT match, the check
reports itself inconclusive rather than shipping a measurement.

Every step reuses the production code path — the same loader
(``issue1336_metric_ladder._load_surface_xy``), the same row alignment
(``issue1336_ladder_alignment._align_rows``), the same seeded
conversation-grouped fold split (``issue825_fit_cells._cv_folds``), the same
canonical baseline helper, and the same pooled-R² reduction
(``issue825_fit_cells._pooled_r2``) — so a mismatch cannot be a
re-implementation artifact.

Example:
    uv run python scripts/issue1336_identity_spotcheck.py \\
        --ts-dir /mnt/eps-data/$USER/issue1336_identity/ts_flat \\
        --source base --target dpo --format chat --corpus gsm8k_test1319
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue825_fit_cells as fc  # noqa: E402
import issue1336_ladder_alignment as la  # noqa: E402
import issue1336_metric_ladder as ml  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
)
from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

# The committed per-cell identity+bias cache the plotter renders (round 4).
IDB_PATH = (
    _REPO_ROOT
    / "eval_results"
    / "issue_1336"
    / "metric_ladder_source_target"
    / "identity_bias_within_l30.json"
)

# Agreement bar for "the recomputed value reproduces the stored one". The two
# come from the same code path on the same rows, so only float ordering
# differs; 1e-4 is far tighter than the 0.0879 spread across pairs reaching
# the same (target, corpus) cell and far looser than fp noise.
MATCH_TOL = 1e-4


def _oof_baselines(
    x_t: np.ndarray, y_t: np.ndarray, conv_ids: np.ndarray, n_folds: int, seed: int
) -> tuple[float, float]:
    """(plain identity R², identity+bias R²), both out-of-fold and pooled.

    Plain identity needs no fitting (v̂ = x everywhere) but is evaluated on
    the SAME rows through the SAME pooled reduction so the two numbers are
    directly comparable. Raises on any row left unfitted.
    """
    folds = fc._cv_folds(conv_ids, n_folds, seed)
    pred_bias = np.empty_like(y_t)
    fitted = np.zeros(len(y_t), dtype=bool)
    for f in range(n_folds):
        te = np.where(folds == f)[0]
        tr = np.where(folds != f)[0]
        if len(te) == 0:
            continue
        pred_bias[te] = identity_bias_predict(x_t[tr], y_t[tr], x_t[te]).astype(y_t.dtype)
        fitted[te] = True
    assert fitted.all(), f"unfitted rows: {int((~fitted).sum())}"
    # Plain identity is fold-independent (no train-fold parameter at all), so
    # the OOF prediction matrix IS x_t.
    return fc._pooled_r2(x_t, y_t), fc._pooled_r2(pred_bias, y_t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ts-dir", type=Path, required=True)
    ap.add_argument("--source", required=True, help="pair SOURCE stage (for the row intersection)")
    ap.add_argument("--target", required=True, help="pair TARGET stage (the within-stage map)")
    ap.add_argument("--format", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    load_kw = dict(
        smoke=False,
        wave1_dir=args.ts_dir,
        gen_root=None,
        expected_layers=cm.EXPECTED_LAYERS,
    )
    xy_s = ml._load_surface_xy(args.ts_dir, args.source, args.format, args.corpus, **load_kw)
    xy_t = ml._load_surface_xy(args.ts_dir, args.target, args.format, args.corpus, **load_kw)

    ids_s = np.asarray([str(c) for c in xy_s["conv_ids"]])
    ids_t = np.asarray([str(c) for c in xy_t["conv_ids"]])
    common, i_s, i_t = la._align_rows(ids_s, ids_t)
    li = args.layer
    x_t = np.asarray(xy_t["X"][i_t][:, li, :], dtype=np.float32)
    y_t = np.asarray(xy_t["Y"][i_t][:, li, :], dtype=np.float32)
    assert x_t.shape == y_t.shape, (x_t.shape, y_t.shape)

    r2_identity, r2_identity_bias = _oof_baselines(
        x_t, y_t, np.asarray(common), cm.N_FOLDS, cm.FIT_SEED
    )

    key = f"{args.source}__{args.target}|{args.format}|{args.corpus}"
    stored = json.loads(IDB_PATH.read_text()).get(key) if IDB_PATH.is_file() else None
    delta = None if stored is None else abs(r2_identity_bias - float(stored))
    verdict = (
        "inconclusive — no stored value for this cell"
        if stored is None
        else ("confirmed" if delta <= MATCH_TOL else "inconclusive — recomputation disagrees")
    )

    rec = {
        "cell": key,
        "layer": li,
        "n_rows": int(len(common)),
        "d": int(x_t.shape[1]),
        "n_folds": cm.N_FOLDS,
        "fit_seed": cm.FIT_SEED,
        "identity_r2": r2_identity,
        "identity_bias_r2_recomputed": r2_identity_bias,
        "identity_bias_r2_stored": None if stored is None else float(stored),
        "abs_delta": delta,
        "match_tol": MATCH_TOL,
        "verdict": verdict,
    }
    print(json.dumps(rec, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
