"""Cap-excluded pooled ("trained on everything") read for the #2054 plain-text cell.

The assistant plain-text on-policy instruct cell
(``conversation_paired_stories_assistant__on_policy__bare_text__qwen2.5-7b-instruct``)
carries a generation-cap defect: the plain-text render has no end-of-turn token,
so 3,401 of its 8,000 rows ran to the 4,096-token cap and their answer vectors
are means over truncated completions. #2054 banked the WITHIN-CELL refit with
those rows removed (0.390 against the raw 0.209), but not the corresponding read
for the POOLED map — the "trained on everything" arm of the Results-2 Plot 2
figure. Without it that figure compares a cap-excluded bar against a raw one.

This script produces the missing number from EXISTING data (0 GPU-h of new
generation): it re-fits the pooled map from the banked activation store and
scores it on the target cell's held-out rows twice, once over all rows and once
over the cap-excluded rows.

Two deliberate design choices, both recorded in the output:

* The capped rows stay IN the pooled TRAINING set, so the fitted map is the
  banked one and every other cell's pooled read is unchanged. They are 3,401 of
  ~448,000 pooled rows (0.76%); dropping them would perturb all 100 cells'
  numbers to correct one cell's eval.
* Only the ``context`` arm is computed — the arm both Results-2 figures read.

Validation gate: the all-rows pooled read must reproduce the banked ladder value
for this cell (``eval_results/issue_2054/specialization_ladder/ladder.json``,
``r2.pooled``) to within ``--gate-tol``. That gate exercises the exact code path
the cap-excluded number comes from, so a PASS is what makes the restricted read
trustworthy. A miss FAILS the run rather than writing an unvalidated number.

Usage::

    uv run python scripts/issue2054_caphit_pooled_refit.py \
        --activations-dir <flat dir of the 100 lattice .npz cells> \
        --on-policy-jsonl <target cell's on-policy generations .jsonl> \
        --out-root eval_results/issue_2054/caphit_pooled_refit
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    # script mode puts scripts/ (not the repo root) on sys.path[0] (gotchas.md).
    sys.path.insert(0, str(_REPO))

import numpy as np

from explore_persona_space.atomic_io import atomic_replace
from explore_persona_space.experiments.issue_779.fit_h import reconstruction_metrics
from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance
from scripts.issue2054_ctx2ctx_fit import ARM_VEC_KEY, discover_cells, load_fold_map
from scripts.issue2054_pool_specialize import (
    accumulate_pooled_moments,
    fit_pooled_per_fold,
    join_cell,
    load_cell_with_answer,
)

SCRIPT_VERSION = "issue2054_caphit_pooled_refit_v1"
ARM = "context"
TARGET_CELL = "conversation_paired_stories_assistant__on_policy__bare_text__qwen2.5-7b-instruct"
# The banked cap-hit split, asserted so a mask built from a different generation
# run can never be mistaken for the one the banked within-cell refit used.
EXPECTED_N_CAPPED = 3401
EXPECTED_N_KEPT = 4599
LADDER = _REPO / "eval_results/issue_2054/specialization_ladder/ladder.json"


def _log(msg: str) -> None:
    print(msg, flush=True)


def capped_conv_ids(jsonl: Path) -> tuple[set[str], set[str]]:
    """Split the target cell's conv_ids by vLLM ``finish_reason``.

    ``length`` means the generation ran to the cap; every other value means the
    model stopped on its own. Returns (capped, kept) and fails loud when the
    split does not match the banked one.
    """
    capped: set[str] = set()
    kept: set[str] = set()
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            (capped if row["finish_reason"] == "length" else kept).add(row["conv_id"])
    if len(capped) != EXPECTED_N_CAPPED or len(kept) != EXPECTED_N_KEPT:
        raise RuntimeError(
            f"cap-hit split {len(capped)} capped / {len(kept)} kept does not match the banked "
            f"{EXPECTED_N_CAPPED} / {EXPECTED_N_KEPT} — wrong generations file for {TARGET_CELL}?"
        )
    _log(f"[caphit] mask: {len(capped)} capped, {len(kept)} kept (matches banked split)")
    return capped, kept


def banked_pooled_r2() -> float:
    """The ladder's pooled read for the target cell — the validation target."""
    payload = json.loads(LADDER.read_text())
    for unit in payload["units"]:
        if unit["arm"] == ARM and unit["cell"] == TARGET_CELL:
            return float(unit["r2"]["pooled"])
    raise KeyError(f"no {ARM}-arm unit for {TARGET_CELL} in {LADDER}")


def evaluate_target(cell, fold_map: dict, pooled_models: dict, kept: set[str]) -> dict:
    """Per-fold pooled-map R^2 on the target cell, over all rows and kept rows."""
    k = int(fold_map["k"])
    act = load_cell_with_answer(cell)
    j = join_cell(act, fold_map["fold_of"], k, ARM)
    x_all = np.asarray(act[ARM_VEC_KEY[ARM]][j["rows"]], dtype=np.float64)
    y_all = np.asarray(act["v_A"][j["rows"]], dtype=np.float64)
    # j["order"] is the joined conv_id order, row-aligned with x_all / y_all.
    keep_mask = np.fromiter((cid in kept for cid in j["order"]), dtype=bool, count=len(j["order"]))
    _log(f"[caphit] target join n={j['n_join']} kept_in_join={int(keep_mask.sum())}")

    folds = []
    for f in range(k):
        te = j["fold_rows"][f]
        m0 = pooled_models[f]
        pred = m0.predict_np(x_all[te])
        y_te = y_all[te]
        sub = keep_mask[te]
        n_sub = int(sub.sum())
        if n_sub < 2:
            raise RuntimeError(f"fold {f}: only {n_sub} kept rows — restricted R^2 undefined")
        folds.append(
            {
                "fold": f,
                "n_test": int(len(te)),
                "n_test_kept": n_sub,
                "r2_all": reconstruction_metrics(pred, y_te)["r2"],
                "r2_kept": reconstruction_metrics(pred[sub], y_te[sub])["r2"],
            }
        )
        _log(
            f"[caphit] fold {f}: n={len(te)} kept={n_sub} "
            f"r2_all={folds[-1]['r2_all']:+.4f} r2_kept={folds[-1]['r2_kept']:+.4f}"
        )
    return {
        "per_fold": folds,
        "r2_all_mean": float(np.mean([r["r2_all"] for r in folds])),
        "r2_kept_mean": float(np.mean([r["r2_kept"] for r in folds])),
        "n_join": j["n_join"],
        "n_kept_in_join": int(keep_mask.sum()),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Not argparse-required so --import-check runs standalone; checked in main().
    p.add_argument("--activations-dir", type=Path, default=None)
    p.add_argument("--on-policy-jsonl", type=Path, default=None)
    p.add_argument(
        "--out-root", type=Path, default=_REPO / "eval_results/issue_2054/caphit_pooled_refit"
    )
    p.add_argument("--fold-map-file", type=Path, default=None)
    p.add_argument("--fold-map-ref", default="origin/main")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--gate-tol",
        type=float,
        default=1e-3,
        help="max |all-rows read - banked ladder read| before the run FAILS",
    )
    p.add_argument("--import-check", action="store_true")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _log("[caphit] import-check OK")
        return 0

    for flag, value in (
        ("--activations-dir", args.activations_dir),
        ("--on-policy-jsonl", args.on_policy_jsonl),
    ):
        if value is None:
            raise SystemExit(f"{flag} is required (omit only with --import-check)")

    t_start = time.time()
    fold_map = load_fold_map(args.fold_map_file, args.fold_map_ref)
    k = int(fold_map["k"])
    _log(f"[caphit] fold map {fold_map['_source']} k={k} sha={fold_map['_sha256'][:12]}")

    _, kept = capped_conv_ids(args.on_policy_jsonl)

    cells = discover_cells(args.activations_dir)
    target = [c for c in cells if c.key == TARGET_CELL]
    if not target:
        raise FileNotFoundError(
            f"{TARGET_CELL} not among {len(cells)} cells in {args.activations_dir}"
        )
    _log(f"[caphit] {len(cells)} cells; target present")

    acc = accumulate_pooled_moments(cells, fold_map["fold_of"], k, [ARM], args.device)
    pooled = fit_pooled_per_fold(acc["mom"][ARM], list(range(k)), k)
    _log(f"[caphit] pooled maps fit for {len(pooled)} folds")

    result = evaluate_target(target[0], fold_map, pooled, kept)

    banked = banked_pooled_r2()
    delta = abs(result["r2_all_mean"] - banked)
    result["validation"] = {
        "banked_pooled_r2": banked,
        "recomputed_all_rows_r2": result["r2_all_mean"],
        "abs_delta": delta,
        "tol": args.gate_tol,
        "passed": bool(delta <= args.gate_tol),
    }
    _log(
        f"[caphit] VALIDATION recomputed_all={result['r2_all_mean']:+.5f} "
        f"banked={banked:+.5f} delta={delta:.2e} tol={args.gate_tol:.0e}"
    )

    args.out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "script_version": SCRIPT_VERSION,
            "arm": ARM,
            "target_cell": TARGET_CELL,
            "fold_map_sha256": fold_map["_sha256"],
            "n_cells_pooled": len(cells),
            "capped_rows_left_in_pool_train": True,
            "wall_s": round(time.time() - t_start),
            **as_metadata_dict(git_provenance(), phase="caphit-pooled-refit"),
        },
        "result": result,
    }
    out = args.out_root / "caphit_pooled_refit.json"
    with atomic_replace(out) as tmp:
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _log(f"[caphit] wrote {out}")

    if not result["validation"]["passed"]:
        _log(
            "[caphit] FAIL: the all-rows recompute does not reproduce the banked pooled read; "
            "the cap-excluded number is NOT trustworthy and is not to be plotted."
        )
        return 3
    _log(
        f"[caphit] PASS  pooled all rows {result['r2_all_mean']:+.4f} -> "
        f"cap-excluded {result['r2_kept_mean']:+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
