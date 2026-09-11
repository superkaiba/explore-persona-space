"""Pooled ("shared") context->answer map REFIT on a restricted cell subset (#2054).

The manuscript's Section 4.4 says one linear map is fit on all settings and
reports its recovery across the SIX instruction-tuned settings of Figure 8
(``figures/paper/c4_shared_speakers``). The banked pooled map behind that
figure was instead fit on the union of all 56 lattice cells (both models, all
three provenances). This script refits the pooled map on a NAMED cell subset
and rescores the six settings, so the measurement matches the description.

Everything except the pooled TRAINING cell list is held identical to the banked
run: the production shared fold map (K=5 conversation-grouped, seed 137), the
``issue2054_pool_specialize`` streamed-moment GCV ridge (standardize-X on
population sd, center-Y, eigh, GCV under the #1887 dof cap 0.9), the
``issue2054_pool_rungs`` per-cell bias rung, layer 19, the ``context`` arm, and
the same held-out rows per fold.

Two rungs per target cell, mirroring the ladder's ``pooled`` / ``bias``:

    pooled  z = M(x)                      (the shared map as is)
    bias    z + mean_tr(y - z)            (exact LS per-setting intercept)

Cap-hit handling for the assistant plain-text cell mirrors
``issue2054_caphit_pooled_refit``: the capped rows stay IN the pooled training
pool, and that one cell additionally reports kept-row reads (fit of the bias
rung on kept TRAIN rows, scored on kept TEST rows) — the regime the published
sidecar plots for that setting.

``--pool-preset all`` reproduces the banked 56-cell pooled fit through this exact
code path and gates every recomputed all-rows read against ``ladder.json``. That
gate is what makes the restricted-subset numbers trustworthy.

Usage::

    uv run python scripts/issue2054_sixsetting_pooled_refit.py \
        --activations-dir <flat dir of the 56 lattice .npz cells> \
        --pool-preset six --out-root eval_results/issue_2054/pooled_refit_six_settings
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
from scripts.issue2054_caphit_pooled_refit import TARGET_CELL as CAPHIT_CELL
from scripts.issue2054_caphit_pooled_refit import capped_conv_ids
from scripts.issue2054_ctx2ctx_fit import ARM_VEC_KEY, D_AMBIENT, discover_cells, load_fold_map
from scripts.issue2054_pool_specialize import (
    accumulate_pooled_moments,
    fit_pooled_per_fold,
    join_cell,
    load_cell_with_answer,
)

SCRIPT_VERSION = "issue2054_sixsetting_pooled_refit_v1"
ARM = "context"
LADDER = _REPO / "eval_results/issue_2054/specialization_ladder/ladder.json"

# The six instruction-tuned settings of manuscript Figure 8, resolved from
# scripts/issue2054_paper_r2_figs.POSITIONS (character, framing) x provenance
# on_policy x model qwen2.5-7b-instruct. Display labels live in that module;
# these are the cell ids the labels resolve to.
SIX_SETTINGS: list[tuple[str, str]] = [
    ("conversation_paired_stories_assistant__on_policy__chat__qwen2.5-7b-instruct",
     "Assistant chat template"),
    ("conversation_paired_stories_assistant__on_policy__bare_text__qwen2.5-7b-instruct",
     "Assistant plain text"),
    ("char_helios__on_policy__attrib_quoted__qwen2.5-7b-instruct", "HELIOS ship AI"),
    ("char_wren__on_policy__attrib_quoted__qwen2.5-7b-instruct", "Wren helpful"),
    ("char_dana__on_policy__attrib_quoted__qwen2.5-7b-instruct", "Dana ordinary"),
    ("char_vex__on_policy__attrib_quoted__qwen2.5-7b-instruct", "Vex villain"),
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def banked_ladder() -> dict[str, dict[str, float]]:
    """Per-cell banked ``pooled`` / ``bias`` reads for the context arm."""
    payload = json.loads(LADDER.read_text())
    out: dict[str, dict[str, float]] = {}
    for unit in payload["units"]:
        if unit["arm"] != ARM:
            continue
        out[unit["cell"]] = {r: float(unit["r2"][r]) for r in ("pooled", "bias")}
    return out


def evaluate_cell(cell, fold_map: dict, pooled_models: dict, kept: set[str] | None) -> dict:
    """Per-fold held-out reads of the (re)fit pooled map on one cell.

    ``kept`` restricts the SECOND set of reads to the cap-kept rows (the bias
    rung refit on kept train rows), mirroring issue2054_caphit_pooled_refit.
    """
    k = int(fold_map["k"])
    act = load_cell_with_answer(cell)
    j = join_cell(act, fold_map["fold_of"], k, ARM)
    x_all = np.asarray(act[ARM_VEC_KEY[ARM]][j["rows"]], dtype=np.float64)
    y_all = np.asarray(act["v_A"][j["rows"]], dtype=np.float64)
    n_rows = len(j["order"])
    if kept is not None:
        keep_mask = np.fromiter(
            (cid in kept for cid in j["order"]), dtype=bool, count=n_rows
        )
    else:
        keep_mask = None

    folds = []
    for f in range(k):
        te = j["fold_rows"][f]
        te_mask = np.zeros(n_rows, dtype=bool)
        te_mask[te] = True
        tr = np.flatnonzero(~te_mask)
        m0 = pooled_models[f]
        z_te = m0.predict_np(x_all[te])
        z_tr = m0.predict_np(x_all[tr])
        y_te, y_tr = y_all[te], y_all[tr]
        b_cf = (y_tr - z_tr).mean(axis=0)
        row = {
            "fold": f,
            "n_pooled_train": int(m0.n_train),
            "n_cell_train": int(len(tr)),
            "n_test": int(len(te)),
            "d_ambient": D_AMBIENT,
            "pooled_info": m0.info(),
            "r2_all_pooled": reconstruction_metrics(z_te, y_te)["r2"],
            "r2_all_bias": reconstruction_metrics(z_te + b_cf, y_te)["r2"],
            "bias_norm_all": float(np.linalg.norm(b_cf)),
        }
        if keep_mask is not None:
            sub_te, sub_tr = keep_mask[te], keep_mask[tr]
            if int(sub_te.sum()) < 2 or int(sub_tr.sum()) < 2:
                raise RuntimeError(f"fold {f}: <2 kept rows — restricted read undefined")
            b_kept = (y_tr[sub_tr] - z_tr[sub_tr]).mean(axis=0)
            row.update(
                {
                    "n_test_kept": int(sub_te.sum()),
                    "n_train_kept": int(sub_tr.sum()),
                    "r2_kept_pooled": reconstruction_metrics(z_te[sub_te], y_te[sub_te])["r2"],
                    "r2_kept_bias": reconstruction_metrics(
                        z_te[sub_te] + b_kept, y_te[sub_te]
                    )["r2"],
                    "bias_norm_kept": float(np.linalg.norm(b_kept)),
                }
            )
        folds.append(row)
        extra = (
            f" kept_pooled={row['r2_kept_pooled']:+.4f} kept_bias={row['r2_kept_bias']:+.4f}"
            if keep_mask is not None
            else ""
        )
        _log(
            f"[six] {cell.key} fold {f}: n_test={len(te)} "
            f"pooled={row['r2_all_pooled']:+.4f} bias={row['r2_all_bias']:+.4f}{extra}"
        )

    out = {"cell": cell.key, "n_join": j["n_join"], "per_fold": folds}
    keys = ["r2_all_pooled", "r2_all_bias"]
    if keep_mask is not None:
        keys += ["r2_kept_pooled", "r2_kept_bias"]
        out["n_kept_in_join"] = int(keep_mask.sum())
    for key in keys:
        out[f"{key}_mean"] = float(np.mean([r[key] for r in folds]))
    return out


def subtract_capped_moments(
    acc: dict, cell, fold_of: dict, k: int, device: str, capped: set[str]
) -> int:
    """Remove one cell's CAPPED rows from the accumulated pooled moments.

    Second moments are sums over rows, so a row subset can be removed by
    subtraction — the ADDITIVITY the ``issue2054_loco_pooled`` hold-out
    variants already rely on. The per-fold row selection mirrors
    ``accumulate_pooled_moments`` exactly (same join, same ``rows``/``fold_rows``
    indexing), restricted to the capped conv_ids. Returns the rows removed.
    """
    import torch

    dev = torch.device(device)
    act = load_cell_with_answer(cell)
    j = join_cell(act, fold_of, k, ARM)
    vec = ARM_VEC_KEY[ARM]
    removed = 0
    for f in range(k):
        sel = j["fold_rows"][f]
        drop = np.asarray([i for i in sel if j["order"][i] in capped], dtype=np.int64)
        if drop.size == 0:
            continue
        idx = j["rows"][drop]
        x = torch.as_tensor(act[vec][idx].astype(np.float64), device=dev)
        y = torch.as_tensor(act["v_A"][idx].astype(np.float64), device=dev)
        m = acc["mom"][ARM][f]
        m["n"] -= int(x.shape[0])
        m["sum_x"] -= x.sum(0)
        m["sum_y"] -= y.sum(0)
        m["yss"] -= float((y * y).sum())
        m["c_xx"] -= x.T @ x
        m["c_xy"] -= x.T @ y
        removed += int(x.shape[0])
        _log(f"[six] dropped {x.shape[0]} capped rows from pooled fold {f}")
    return removed


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--activations-dir", type=Path, default=None)
    p.add_argument("--pool-preset", choices=("six", "all"), default="six")
    p.add_argument("--on-policy-jsonl", type=Path, default=None,
                   help="plain-text cell generations (finish_reason) for the cap mask")
    p.add_argument("--out-root", type=Path,
                   default=_REPO / "eval_results/issue_2054/pooled_refit_six_settings")
    p.add_argument("--out-name", default=None)
    p.add_argument("--fold-map-file", type=Path, default=None)
    p.add_argument("--fold-map-ref", default="origin/main")
    p.add_argument("--device", default="cpu")
    p.add_argument("--gate-tol", type=float, default=1e-3)
    p.add_argument("--drop-capped-from-pool", action="store_true",
                   help="sensitivity arm: remove the plain-text cell's capped rows from the "
                        "pooled TRAINING moments (they are 8.9%% of a six-cell pool vs 0.76%% "
                        "of the banked 56-cell pool)")
    p.add_argument("--import-check", action="store_true")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _log("[six] import-check OK")
        return 0
    if args.activations_dir is None:
        raise SystemExit("--activations-dir is required (omit only with --import-check)")
    if args.on_policy_jsonl is None:
        raise SystemExit("--on-policy-jsonl is required (the cap mask for the plain-text cell)")

    t_start = time.time()
    fold_map = load_fold_map(args.fold_map_file, args.fold_map_ref)
    k = int(fold_map["k"])
    _log(f"[six] fold map {fold_map['_source']} k={k} sha={fold_map['_sha256'][:12]}")

    _, kept = capped_conv_ids(args.on_policy_jsonl)

    cells = discover_cells(args.activations_dir)
    by_key = {c.key: c for c in cells}
    six_keys = [key for key, _ in SIX_SETTINGS]
    missing = [key for key in six_keys if key not in by_key]
    if missing:
        raise FileNotFoundError(f"missing target cells in {args.activations_dir}: {missing}")

    pool_cells = [by_key[key] for key in six_keys] if args.pool_preset == "six" else cells
    _log(
        f"[six] pool_preset={args.pool_preset}: {len(pool_cells)} cells in the pooled "
        f"training pool ({len(cells)} discovered)"
    )

    t_mom = time.time()
    acc = accumulate_pooled_moments(pool_cells, fold_map["fold_of"], k, [ARM], args.device)
    _log(f"[six] moments accumulated in {time.time() - t_mom:.1f}s")
    n_dropped = 0
    if args.drop_capped_from_pool:
        capped, _ = capped_conv_ids(args.on_policy_jsonl)
        n_dropped = subtract_capped_moments(
            acc, by_key[CAPHIT_CELL], fold_map["fold_of"], k, args.device, capped
        )
        _log(f"[six] removed {n_dropped} capped rows from the pooled training moments")
    pooled = fit_pooled_per_fold(acc["mom"][ARM], list(range(k)), k)

    results = []
    for key, label in SIX_SETTINGS:
        mask = kept if key == CAPHIT_CELL else None
        res = evaluate_cell(by_key[key], fold_map, pooled, mask)
        res["label"] = label
        results.append(res)

    banked = banked_ladder()
    checks = []
    for res in results:
        for rung, field in (("pooled", "r2_all_pooled_mean"), ("bias", "r2_all_bias_mean")):
            recomputed = float(res[field])
            ref = banked[res["cell"]][rung]
            delta = abs(recomputed - ref)
            checks.append(
                {
                    "cell": res["cell"],
                    "rung": rung,
                    "banked_56cell": ref,
                    "recomputed": recomputed,
                    "abs_delta": delta,
                    # Only meaningful as a GATE when the pool is the banked 56-cell union.
                    "passed": bool(delta <= args.gate_tol),
                }
            )
    gate_applies = args.pool_preset == "all"
    gate_passed = all(c["passed"] for c in checks) if gate_applies else None

    payload = {
        "metadata": {
            "script_version": SCRIPT_VERSION,
            "arm": ARM,
            "layer": 19,
            "pool_preset": args.pool_preset,
            "pool_cells": [c.key for c in pool_cells],
            "n_cells_pooled": len(pool_cells),
            "fold_map_source": fold_map["_source"],
            "fold_map_sha256": fold_map["_sha256"],
            "k": k,
            "seed": fold_map.get("seed"),
            "capped_rows_left_in_pool_train": not args.drop_capped_from_pool,
            "n_capped_rows_dropped_from_pool": n_dropped,
            "activations_dir": str(args.activations_dir),
            "wall_s": round(time.time() - t_start),
            **as_metadata_dict(git_provenance(), phase="sixsetting-pooled-refit"),
        },
        "results": results,
        "validation": {
            "reference": "eval_results/issue_2054/specialization_ladder/ladder.json (r2.pooled/r2.bias)",
            "tol": args.gate_tol,
            "gate_applies": gate_applies,
            "passed": gate_passed,
            "per_check": checks,
        },
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    name = args.out_name or f"pooled_refit_{args.pool_preset}.json"
    out = args.out_root / name
    with atomic_replace(out) as tmp:
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _log(f"[six] wrote {out}")

    if gate_applies and not gate_passed:
        for c in checks:
            if not c["passed"]:
                _log(
                    f"[six] GATE FAIL {c['cell']} {c['rung']}: "
                    f"recomputed={c['recomputed']:+.6f} banked={c['banked_56cell']:+.6f} "
                    f"delta={c['abs_delta']:.3e}"
                )
        return 3
    if gate_applies:
        _log("[six] GATE PASS: the 56-cell recompute reproduces every banked ladder read")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
