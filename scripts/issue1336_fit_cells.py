#!/usr/bin/env python
"""Issue #1336 — Phase F: within-cell held-out ridge fits (thin #825 driver).

Thin driver over ``issue825_fit_cells`` cores (`heldout_r2_sweep`, controls,
bootstrap) with the Llama frozen set {16, 21, 22, 30} threaded through the
default-preserving ``frozen_layers`` parametrization (the fact-checker
must-do: the module-global Qwen set would otherwise silently persist preds at
the wrong layers).

RESUME recipe (plan v9 §4 route 1, `resume_on_recalibrated_dv`): each cell's
PRIMARY within-stage read is the E1-validated held-out cross-fitted per-dim
affine-recalibrated pooled R^2 (`experiments/issue_1336/recal.py` — the E1
functions, imported), UNIFORM across all stages, with raw pooled R^2 always
reported as companion; the per-stage usable-strength bar rides the persisted
Qwen exchange rate (`cm.load_qwen_recal_cal` — E1.d, never recomputed).
``collect_lambdas=True`` + the selected-lambda audit are threaded through
every production sweep (committed grid retained — plan v9 drops the v7-R1
widened grid: A_v = 0.000). Preds persist at ``cm.preds_layers(frozen)``
(frozen + the E1 verdict layer L29 — default-preserving extension).

Modes (one per invocation):
  --g0 [--g0-probe-only|--g0-local-dir D]   G0 fit-core reuse gate (plan §7):
        refit the committed Qwen S1 cell (pinned #825 turnstore stems @
        deb7a452) through THIS generalized fit path; PASS <=> layer-19
        held-out R^2 within ±0.01 of the committed 0.6731. Exit 3 on FAIL.
  --cells <id,...|all|smoke> [--matched-n]  per-cell sweeps -> cells/*.json,
        nulls/*.json, preds npz (+ manifest), prefix-slot degeneracy check.
  --g1-check                                G1 rig-transfer kill gate (plan §7,
        re-adjudicated on the RECALIBRATED primary per plan v9 route 1: kill
        bar = persisted bar_r, marginal = 0.3 x the same exchange rate) from
        the After-RLVR lmsys5k-chat cell JSON (+ naturalistic when the chat
        read is marginal/below). Exit 0 pass / 3 KILL / 4 need-nat.

v2 mode (plan v13, `full-corpora-stage-evals-metric-ladder` round): pass
``--v2`` with ``--cells``. Default-preserving: v1 invocations are unchanged.
Under --v2: cells resolve against the CELLS_V2 registry (incl. the
``*_xprefix`` naturalistic prefix-arm cells, x_slot from the registry or
``--x-slot``); the sweep runs on the 23-point grid cm.LAMBDAS_23 under the
adaptive edge rule (<=2 one-decade extensions/side, whole-cell
selection-symmetric re-runs, `estimator-limited: lambda-edge` label);
``fc.N_INNER_LAMBDA_FOLDS`` is patched to 2 (module-global patch style);
outputs land under ``cells_v2/`` with preds staged to
``data/issue_1336/preds_v2`` (manifest ``preds_manifest_v2.json``);
``--matched-n`` companions become persist-layer-subset refits at n=7,350
(seed-1336 subsample) on the four above-size corpora (plan §4 Phase FIT).

Under --v3-pooled (plan v15 Phase FIT_pool): cells resolve to POOLED units
``pooled_<ckpt>_arm_<on|off>`` derived from the CELLS_V3 registry — one pooled
ridge per (activation checkpoint x on/off-policy arm) over the union of the
round-3 corpora. Rows are selected + fold-grouped by the Phase C_pool split
manifest (cluster-grouped 80/20 + 5 folds); the manifest fold ids ride into
the #825 core as the conv_ids GROUP KEY (``_cv_folds`` over exactly n_folds
unique values is a bijective relabeling, so the realized CV partition IS the
persisted one). Same 23-pt grid + adaptive edge rule + inner-group-CV
(n_inner=2) as --v2; a train-side final fit predicts the pooled 20% test side
for the per-corpus slice read; identity+bias + kNN retrieval per fitted map
(Guideline 11). Outputs land under ``cells_pooled_v3/`` with preds staged to
``data/issue_1336/preds_pooled_v3/{on,off}-policy`` (manifest
``preds_pooled_v3_manifest.json``). Gates: ``--g0v3`` (pooled-split
reproducibility vs the round-3 per-corpus read, tol 0.05 x ex_v2 at the
headline layer) and ``--g1v3-check`` (pooled rig-health kill vs bar_v2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import issue825_fit_cells as fc  # noqa: E402
import issue1336_extract_turnstore as et  # noqa: E402
import issue1336_pooled_split as ps  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402
from explore_persona_space.experiments.issue_1336 import recal as rc  # noqa: E402

# Merged v1 + v2 registry lookup: ids are disjoint except the 5 shared
# gsm8k_test1319 cells (fully-reused wave-1 cells — the v2 dict is the same
# cell plus the x_slot field, so v2-wins merge order is behavior-identical).
CELL_BY_ID = {c["cell_id"]: c for c in [*cm.CELLS, *cm.CELLS_V2]}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--g0", action="store_true", help="run the G0 fit-core reuse gate")
    ap.add_argument("--g0-probe-only", action="store_true", help="Hub existence probe only")
    ap.add_argument("--g0-local-dir", type=Path, default=None, help="pre-staged G0 bundle dir")
    ap.add_argument("--g0-dl-dir", type=Path, default=Path("data/issue_1336/g0_qwen"))
    ap.add_argument("--g1-check", action="store_true", help="evaluate the G1 kill gate")
    ap.add_argument(
        "--g0v2",
        action="store_true",
        help="run the G0' three-leg fit-core parity gate (plan v13 §7): (a) legacy-pinned "
        "Qwen S1 refit, (b) Gram-vs-primal equality, (c) v2-recipe anchor -> v2 bars JSON",
    )
    ap.add_argument(
        "--g1v2-check",
        action="store_true",
        help="evaluate the G1' v2 kill gate (After-RLVR lmsys23k chat cell vs bar_v2)",
    )
    ap.add_argument(
        "--v3-pooled",
        action="store_true",
        help="v3 pooled multi-dataset fit mode (plan v15 Phase FIT_pool): --cells selects "
        "pooled units pooled_<ckpt>_arm_<on|off> | all | smoke",
    )
    ap.add_argument(
        "--g0v3",
        action="store_true",
        help="G0v3 pooled-split reproducibility gate (round-3 RLVR x lmsys23k-chat refit "
        "under the pooled split vs the per-corpus round-3 read; tol 0.05 x ex_v2)",
    )
    ap.add_argument(
        "--g1v3-check",
        action="store_true",
        help="evaluate the G1' v3 pooled rig-health kill gate (first pooled cell vs bar_v2)",
    )
    ap.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Phase C_pool split manifest (default: data/issue_1336/pooled_split_v3"
        "[_smoke]/split_manifest.json)",
    )
    ap.add_argument(
        "--offpolicy-root",
        type=Path,
        default=None,
        help="root holding the Phase EXT_off turnstore_offpolicy_<i>_chat_<j>[_smoke] "
        "trees (default: data/issue_1336)",
    )
    ap.add_argument("--cells", default=None, help="comma cell ids | all | smoke")
    ap.add_argument("--turnstore-dir", type=Path, default=None)
    ap.add_argument(
        "--wave1-turnstore-dir",
        type=Path,
        default=None,
        help="v2 concat loader: wave-1 stem dir (default: --turnstore-dir)",
    )
    ap.add_argument(
        "--gen-root",
        type=Path,
        default=None,
        help="v2 concat loader: gen answers root for the wave-1 text-sha join "
        "(default: data/issue_1336/gen in production; None under --smoke)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1336"))
    ap.add_argument("--preds-dir", type=Path, default=None)
    ap.add_argument("--folds", type=int, default=cm.N_FOLDS)
    ap.add_argument("--seed", type=int, default=cm.FIT_SEED)
    ap.add_argument("--null-draws", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--frozen-layers", default=None, help="comma ints (default: registry set)")
    ap.add_argument(
        "--matched-n", action="store_true", help="also refit at the matched-n subsample"
    )
    ap.add_argument(
        "--matched-n-size",
        type=int,
        default=None,
        help="subsample size (default: cm.MATCHED_N; under --v2: cm.MATCHED_N_V2)",
    )
    ap.add_argument(
        "--matched-n-seed",
        type=int,
        default=None,
        help="subsample seed (default: --seed; under --v2: cm.MATCHED_N_V2_SEED)",
    )
    ap.add_argument(
        "--v2",
        action="store_true",
        help="v2 full-corpora recipe: CELLS_V2 registry, 23-pt grid + adaptive "
        "edge rule, n_inner=2, cells_v2/ outputs, preds_v2 staging",
    )
    ap.add_argument(
        "--x-slot",
        choices=("context", "prefix"),
        default=None,
        help="X-slot override (default: the cell registry's x_slot, else context)",
    )
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def _metadata(seed: int, n: int) -> dict:
    return {
        "git_commit": fc._git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(seed),
        "n": int(n),
        "script": "scripts/issue1336_fit_cells.py",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[fit1336] wrote {path}")


# ---------------------------------------------------------------------------
# G0 — fit-core reuse gate (artifact-reuse check h-iv: the CONSUMER'S OWN
# loader runs against the real pinned stems before any GPU spend)
# ---------------------------------------------------------------------------
def _g0_probe() -> bool:
    """Cheap Hub existence probe of the pinned stems (single-path file_exists)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    ok = True
    for name in ("instruct_chat_s_shard000.pt", "instruct_chat_s_shard000.json"):
        path = f"{cm.G0['hf_prefix']}/{name}"
        found = hub.retry_transient(
            # HUB_VERIFY_RETRY_EXEMPT: single-path probe wrapped in hub.retry_transient here
            lambda p=path: api.file_exists(
                cm.HF_DATA_REPO, p, repo_type="dataset", revision=cm.G0["revision"]
            ),
            what=f"g0 probe {path}",
        )
        print(f"[g0-probe] {path} @ {cm.G0['revision'][:8]}: {'OK' if found else 'MISSING'}")
        ok = ok and found
    return ok


def _g0_stage(dl_dir: Path) -> Path:
    """Stage the pinned Qwen S1 stems: scoped list_repo_tree + per-file download.

    NEVER snapshot_download on the ~1M-file data repo (full-tree enumeration
    wedge — gotchas.md #833); the prefix-scoped tree walk is seconds. Every
    Hub call rides hub.retry_transient (the listing MATERIALIZES inside the
    thunk — hub list APIs are lazy generators; gotchas.md #779 n50k).
    """
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    entries = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: scoped (path_in_repo) walk, retried via hub.retry_transient
            api.list_repo_tree(
                cm.HF_DATA_REPO,
                path_in_repo=cm.G0["hf_prefix"],
                repo_type="dataset",
                revision=cm.G0["revision"],
                recursive=False,
            )
        ),
        what="g0 stage: scoped tree walk",
    )
    stem = cm.G0["stem"]
    wanted = [e.path for e in entries if Path(e.path).name.startswith(f"{stem}_shard")]
    assert wanted, f"no {stem} shards under {cm.G0['hf_prefix']} @ {cm.G0['revision'][:8]}"
    for rel in sorted(wanted):
        hub.retry_transient(
            lambda r=rel: hf_hub_download(
                repo_id=cm.HF_DATA_REPO,
                repo_type="dataset",
                filename=r,
                revision=cm.G0["revision"],
                local_dir=dl_dir,
            ),
            what=f"g0 stage: download {rel}",
        )
    bundle_dir = dl_dir / cm.G0["hf_prefix"]
    print(f"[g0] staged {len(wanted)} files -> {bundle_dir}")
    return bundle_dir


def run_g0(args) -> int:
    """Refit the committed Qwen S1 cell through the generalized fit path."""
    if args.g0_probe_only:
        return 0 if _g0_probe() else 1
    bundle_dir = args.g0_local_dir or _g0_stage(args.g0_dl_dir)
    bundle = fc._load_bundle_any(bundle_dir, "instruct", "chat", "s")
    # Parent Track-S normalization (issue825_fit_cells._normalize_cell):
    # assistant slot (index 0) -> a1 profile (index 1). The #825 core asserts
    # against ITS Qwen EXPECTED_LAYERS global; scope it to the G0 store's
    # expected value (pinned Qwen 28; a local fixture asserts its own count).
    exp_layers = (
        int(cm.G0["expected_layers"]) if args.g0_local_dir is None else _bundle_n_layers(bundle)
    )
    with cm.fc_expected_layers(fc, exp_layers):
        xy = fc._cell_xy(bundle, {"slot_index": 0, "target_turn_index": 1})
    X, Y, conv_ids = xy["X"], xy["Y"], xy["conv_ids"]
    n_layers = X.shape[1]
    layer = int(cm.G0["layer"])
    if args.g0_local_dir is None:
        assert n_layers == cm.G0["expected_layers"], (n_layers, cm.G0["expected_layers"])
        assert X.shape[2] == cm.G0["expected_hidden"], (X.shape[2], cm.G0["expected_hidden"])
    else:
        # Smoke fixture: keep the gate arithmetic on ITS own last layer.
        layer = min(layer, n_layers - 1)
    sweep = fc.heldout_r2_sweep(
        X[:, [layer], :],
        Y[:, [layer], :],
        conv_ids,
        n_folds=cm.N_FOLDS,
        seed=cm.FIT_SEED,
        null_draws=0,
        collect_cosines=False,
        frozen_layers=(),
    )
    r2 = float(sweep["r2_obs"][0])
    committed, tol = float(cm.G0["committed_r2"]), float(cm.G0["tol"])
    ok = abs(r2 - committed) <= tol
    payload = {
        "metadata": _metadata(cm.FIT_SEED, len(conv_ids)),
        "gate": "G0",
        "stem": cm.G0["stem"],
        "revision": cm.G0["revision"],
        "layer": layer,
        "r2_layer": r2,
        "committed_r2": committed,
        "tol": tol,
        "abs_dev": abs(r2 - committed),
        "pass": bool(ok),
        "local_dir_fixture": args.g0_local_dir is not None,
    }
    _write_json(args.out_dir / "gates" / "g0_gate.json", payload)
    print(
        f"[g0] layer-{layer} R2={r2:.4f} vs committed {committed} (tol {tol}) -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 3


def _g0_xy_at_gate_layer(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """(X_layer, Y_layer, conv_ids, layer, fixture) for the pinned Qwen S1 cell.

    Shared staging/normalization for run_g0v2 (mirrors run_g0's load path
    byte-for-byte; run_g0 itself is left untouched — v1 gate unchanged).
    """
    bundle_dir = args.g0_local_dir or _g0_stage(args.g0_dl_dir)
    bundle = fc._load_bundle_any(bundle_dir, "instruct", "chat", "s")
    fixture = args.g0_local_dir is not None
    exp_layers = int(cm.G0["expected_layers"]) if not fixture else _bundle_n_layers(bundle)
    with cm.fc_expected_layers(fc, exp_layers):
        xy = fc._cell_xy(bundle, {"slot_index": 0, "target_turn_index": 1})
    X, Y, conv_ids = xy["X"], xy["Y"], xy["conv_ids"]
    layer = int(cm.G0["layer"])
    if not fixture:
        assert X.shape[1] == cm.G0["expected_layers"], (X.shape[1], cm.G0["expected_layers"])
        assert X.shape[2] == cm.G0["expected_hidden"], (X.shape[2], cm.G0["expected_hidden"])
    else:
        layer = min(layer, X.shape[1] - 1)
    return X[:, [layer], :], Y[:, [layer], :], conv_ids, layer, fixture


def run_g0v2(args) -> int:
    """G0' three-leg fit-core parity gate (plan v13 §4/§7).

    (a) LEGACY parity: Qwen S1 refit under the explicit pre-#1887 pins
        (lambda_selection="gcv", GCV_DOF_CAP=None, LEGACY_UNGUARDED_GCV=True,
        13-pt logspace(-2,4) grid, FORCED Gram route) — layer-19 R^2 within
        +-tol of the committed 0.6731.
    (b) Gram-vs-primal equality: the SAME cell under the FULL v2 recipe
        (23-pt grid, inner-group-CV n_inner=2) run through BOTH routes
        (fc.FORCE_GRAM True/False, matched grid + folds): |dR^2| <= 1e-6.
    (c) v2-recipe anchor: S_qwen_v2 = the primal leg-(b) read, recorded
        BEFORE any Llama read (driver ordering); writes gates_v2/v2_bars.json
        via cm.v2_bars (ex_v2 / bar_v2 / bands).

    Fixture mode (--g0-local-dir): legs run identically but the (a) anchor
    tolerance is INFORMATIONAL (production-n-calibrated verdicts are demoted
    under smoke — the #1345 gate-calibration rule); the (b) numerics-equality
    leg stays ENFORCED at any n. Returns 0 on pass, 3 on gate failure.
    """
    Xl, Yl, conv_ids, layer, fixture = _g0_xy_at_gate_layer(args)
    n = len(conv_ids)
    d = int(Xl.shape[2])
    n_train_min = n - int(np.ceil(n / cm.N_FOLDS))  # smallest train fold (approx, group folds)
    committed, tol = float(cm.G0["committed_r2"]), float(cm.G0["tol"])
    common = dict(
        n_folds=cm.N_FOLDS,
        seed=cm.FIT_SEED,
        null_draws=0,
        collect_cosines=False,
        frozen_layers=(),
    )

    # Leg (a): legacy pins, module-patched + restored (the documented
    # issue825_fit_cells patch convention; FORCE_GRAM pins the legacy route).
    saved = (fc.GCV_DOF_CAP, fc.LEGACY_UNGUARDED_GCV, fc.FORCE_GRAM, fc.N_INNER_LAMBDA_FOLDS)
    try:
        fc.GCV_DOF_CAP = None
        fc.LEGACY_UNGUARDED_GCV = True
        fc.FORCE_GRAM = True
        sweep_a = fc.heldout_r2_sweep(
            Xl, Yl, conv_ids, lambda_selection="gcv", lambdas=np.logspace(-2, 4, 13), **common
        )
    finally:
        fc.GCV_DOF_CAP, fc.LEGACY_UNGUARDED_GCV, fc.FORCE_GRAM, fc.N_INNER_LAMBDA_FOLDS = saved
    r2_legacy = float(sweep_a["r2_obs"][0])
    dev_a = abs(r2_legacy - committed)
    pass_a = dev_a <= tol

    # Legs (b) + (c): full v2 recipe, both routes on the matched grid + folds.
    v2_grid = np.asarray(cm.LAMBDAS_23, dtype=np.float64)
    try:
        fc.N_INNER_LAMBDA_FOLDS = cm.N_INNER_LAMBDA_FOLDS_V2
        fc.FORCE_GRAM = True
        sweep_gram = fc.heldout_r2_sweep(
            Xl, Yl, conv_ids, lambda_selection="inner-group-cv", lambdas=v2_grid, **common
        )
        fc.FORCE_GRAM = False
        sweep_primal = fc.heldout_r2_sweep(
            Xl, Yl, conv_ids, lambda_selection="inner-group-cv", lambdas=v2_grid, **common
        )
    finally:
        fc.GCV_DOF_CAP, fc.LEGACY_UNGUARDED_GCV, fc.FORCE_GRAM, fc.N_INNER_LAMBDA_FOLDS = saved
    r2_gram = float(sweep_gram["r2_obs"][0])
    r2_primal = float(sweep_primal["r2_obs"][0])
    delta_b = abs(r2_gram - r2_primal)
    pass_b = delta_b <= 1e-6
    lam_g, lam_p = sweep_gram.get("gcv_lambda"), sweep_primal.get("gcv_lambda")
    lambda_match = (
        bool(np.allclose(np.nan_to_num(np.asarray(lam_g)), np.nan_to_num(np.asarray(lam_p))))
        if lam_g is not None and lam_p is not None
        else None
    )
    assert n_train_min > d, (
        f"G0'(b) expects the primal regime (n_train {n_train_min} > d {d}) — "
        "the equality leg must exercise the production regime switch"
    )

    # Leg (c): the primal v2-recipe read IS the anchor; bars ride cm.v2_bars.
    s_qwen_v2 = r2_primal
    bars = cm.v2_bars(s_qwen_v2)

    enforced_a = not fixture
    ok = (pass_a or not enforced_a) and pass_b
    payload = {
        "metadata": _metadata(cm.FIT_SEED, n),
        "gate": "G0v2",
        "stem": cm.G0["stem"],
        "revision": cm.G0["revision"],
        "layer": layer,
        "recorded_before_llama_reads": True,
        "leg_a_legacy": {
            "r2": r2_legacy,
            "committed_r2": committed,
            "tol": tol,
            "abs_dev": dev_a,
            "pass": bool(pass_a),
            "enforced": bool(enforced_a),
            "pins": {
                "lambda_selection": "gcv",
                "GCV_DOF_CAP": None,
                "LEGACY_UNGUARDED_GCV": True,
                "FORCE_GRAM": True,
                "grid": "logspace(-2,4,13)",
            },
        },
        "leg_b_gram_vs_primal": {
            "r2_gram": r2_gram,
            "r2_primal": r2_primal,
            "abs_delta": delta_b,
            "tol": 1e-6,
            "pass": bool(pass_b),
            "enforced": True,
            "lambda_match": lambda_match,
            "n_train_min": int(n_train_min),
            "d": d,
        },
        "leg_c_v2_anchor": {"s_qwen_v2": s_qwen_v2, "bars": bars},
        "local_dir_fixture": fixture,
        "pass": bool(ok),
    }
    _write_json(args.out_dir / "gates_v2" / "g0v2.json", payload)
    bars_payload = {"metadata": _metadata(cm.FIT_SEED, n), "fixture": fixture, **bars}
    _write_json(args.out_dir / "gates_v2" / "v2_bars.json", bars_payload)
    print(
        f"[g0v2] (a) legacy R2={r2_legacy:.4f} dev={dev_a:.4f} "
        f"{'PASS' if pass_a else 'FAIL'}{'' if enforced_a else ' (informational: fixture)'} | "
        f"(b) |gram-primal|={delta_b:.2e} {'PASS' if pass_b else 'FAIL'} | "
        f"(c) s_qwen_v2={s_qwen_v2:.4f} ex_v2={bars['ex_v2']:.4f} bar_v2={bars['bar_v2']:.4f}"
    )
    return 0 if ok else 3


def run_g1v2_check(out_dir: Path) -> int:
    """G1' v2 rig-health kill gate (plan v13 §7).

    KILL <=> BOTH the raw AND the recalibrated best-swept-layer R^2 of the
    After-RLVR lmsys23k chat cell sit below bar_v2 = 0.20 * ex_v2 (the G0'(c)
    exchange-rate-scaled bar). Writes gates_v2/g1v2_gate.json; returns 3 on
    KILL, 0 on pass. A NaN read never counts as below-bar (fail-safe: a
    missing/degenerate recal read must not manufacture a kill).
    """
    bars_path = out_dir / "gates_v2" / "v2_bars.json"
    assert bars_path.exists(), (
        f"{bars_path} missing — run the G0' v2 gate first (driver-enforced ordering)"
    )
    bar = float(json.loads(bars_path.read_text())["bar_v2"])
    cell_id = cm.v2_cell_id("rlvr", "chat", "lmsys23k")
    cell_path = out_dir / "cells_v2" / f"cells_{cell_id}.json"
    assert cell_path.exists(), f"{cell_path} missing — fit the G1' cell before the check"
    cell = json.loads(cell_path.read_text())
    raw = np.asarray(cell["r2_per_layer_obs"], dtype=float)
    raw_best = float(np.nanmax(raw)) if np.isfinite(raw).any() else float("nan")
    recal_best = float((cell.get("recal") or {}).get("s_recal", float("nan")))
    below = lambda v: bool(np.isfinite(v) and v < bar)  # noqa: E731
    kill = below(raw_best) and below(recal_best)
    payload = {
        "metadata": _metadata(cm.FIT_SEED, int(cell.get("n", 0) or 0)),
        "gate": "G1v2",
        "cell": cell_id,
        "raw_best_r2": raw_best,
        "recal_best_r2": recal_best,
        "bar_v2": bar,
        "verdict": "kill" if kill else "pass",
    }
    _write_json(out_dir / "gates_v2" / "g1v2_gate.json", payload)
    print(
        f"[g1v2] raw_best={raw_best:.4f} recal_best={recal_best:.4f} vs bar_v2={bar:.4f} -> "
        f"{'KILL' if kill else 'PASS'}"
    )
    return 3 if kill else 0


# ---------------------------------------------------------------------------
# Per-cell fits
# ---------------------------------------------------------------------------
def _bundle_n_layers(bundle: dict) -> int:
    """Layer count realized by the bundle (tiny smoke stores / G0 fixtures)."""
    return int(np.asarray(bundle["arrays"]["slots"]).shape[2])


def _cell_xy_1336(bundle: dict, expected_layers: int, x_slot: str = "context") -> dict:
    """(X, Y, conv_ids, nll) for one fit arm of a bundle.

    ``x_slot="context"`` (default, byte-preserving): X = a1-header slot
    (index 1) — the end-of-context activation. ``x_slot="prefix"``: X =
    prefix-header slot (index 0) — the naturalistic prefix-arm cells (plan
    v13 §4 divergence 7). Y is the a1 answer profile either way.

    The #1336 extractor writes slots ordered by position (prefix=0, a1=1) and
    turns by span start (u1=0, a1=1) — asserted here against the bundle shape.
    ``fc._cell_xy`` asserts the layer axis against the #825 module's Qwen
    global (28); scope-rebind it to THIS ladder's expected value (production
    32; smoke stores their own realized count) so the check stays fail-loud
    on the right invariant.
    """
    arrays = bundle["arrays"]
    assert arrays["slots"].shape[1] == 2, f"n_slots {arrays['slots'].shape[1]} != 2"
    assert arrays["profiles"].shape[1] == 2, f"n_turns {arrays['profiles'].shape[1]} != 2"
    si = {"context": 1, "prefix": 0}[x_slot]
    with cm.fc_expected_layers(fc, expected_layers):
        return fc._cell_xy(bundle, {"slot_index": si, "target_turn_index": 1})


def _prefix_degeneracy(bundle: dict, frozen_layers: tuple[int, ...]) -> dict:
    """Registered prefix-arm degeneracy check (plan §4 stated deviation).

    Exact max pairwise cosine DISTANCE across rows of the prefix-header slot
    activation per frozen layer (expected ~0: the prefix token sequence is
    row-constant, so under causal attention the slot activation is too).
    """
    slots = np.asarray(bundle["arrays"]["slots"], dtype=np.float32)  # (N, 2, L, D)
    dev = fc._fit_device()
    out = {}
    for li in frozen_layers:
        if li >= slots.shape[2]:
            continue
        v = torch.as_tensor(slots[:, 0, li, :], dtype=torch.float32).to(dev)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
        gram = v @ v.T
        n = gram.shape[0]
        gram.fill_diagonal_(1.0)
        min_cos = float(gram.min())
        out[str(li)] = {"n": n, "min_pairwise_cos": min_cos, "max_pairwise_cos_dist": 1.0 - min_cos}
        del v, gram
    return out


def _recal_block(
    sweep: dict, Y: np.ndarray, persist_layers: tuple[int, ...], qwen_cal: dict
) -> dict:
    """PRIMARY within-stage read (plan v9 route 1): held-out cross-fitted
    per-dim affine-recalibrated pooled R^2 per persisted layer, raw pooled
    R^2 (fold-local test mean) as companion. Separate reads — never blended.
    """
    fitted = sweep["fitted_mask"]
    folds = np.asarray(sweep["folds"])[fitted]
    per_layer: dict[str, dict] = {}
    for li in persist_layers:
        if li not in sweep["preds_frozen"]:
            continue  # out of range on this store (tiny smoke) — guard-skipped
        pred = sweep["preds_frozen"][li][fitted]
        truth = Y[fitted, li, :]
        rec = rc.crossfit_recal_direct(pred, truth, folds)
        per_layer[str(li)] = {
            "heldout_recal_r2": float(rec["r2"]),
            "raw_r2": float(rc.raw_pooled_r2(pred, truth, folds)),
            "insample_recal_r2": float(rc.insample_recal_r2(pred, truth)),
        }
    finite = {
        int(k): v["heldout_recal_r2"]
        for k, v in per_layer.items()
        if np.isfinite(v["heldout_recal_r2"])
    }
    best = max(finite, key=finite.get) if finite else None
    s_recal = finite[best] if best is not None else float("nan")
    return {
        "primary": "heldout_crossfit_perdim_affine_recal_r2",
        "companion": "raw_pooled_r2 (fold-local test mean, committed convention)",
        "per_layer": per_layer,
        "s_recal": s_recal,
        "s_recal_argmax_layer": best,
        "bar_r": float(qwen_cal["bar_r"]),
        "above_bar": bool(best is not None and s_recal >= qwen_cal["bar_r"]),
        "qwen_exchange": {
            k: qwen_cal[k] for k in ("s_qwen_recal", "committed_anchor", "rate", "path")
        },
    }


def _lambda_audit(
    sweep: dict,
    frozen_layers: tuple[int, ...],
    grid: np.ndarray | list[float] | None = None,
) -> dict:
    """Selected-lambda audit: histogram of the observed-fit selections over
    the REALIZED grid, edge counts + fractions, within-one-grid-step counts
    + fractions (plan v13 §4 per-cell edge-fraction reporting), per-frozen-
    layer rows, and the full (layer x fold) matrix (E1 lambda-join shape).

    ``grid=None`` (default, byte-preserving) audits against the module
    COMMITTED grid ``fc.LAMBDAS``; the v2 path passes the realized (possibly
    edge-extended) grid — the plan §10 named must-fix for the L332 hardcode.
    """
    lam = sweep.get("gcv_lambda")
    assert lam is not None, "lambda audit requires heldout_r2_sweep(collect_lambdas=True)"
    grid = [float(v) for v in (fc.LAMBDAS if grid is None else np.asarray(grid, dtype=np.float64))]
    lamf = np.asarray(lam, dtype=np.float64)
    finite = lamf[np.isfinite(lamf)]
    matrix = [[None if not np.isfinite(v) else float(v) for v in row] for row in lamf]
    n_sel = int(finite.size)
    n_low = int(np.sum(finite == grid[0]))
    n_high = int(np.sum(finite == grid[-1]))
    # "Within one step" INCLUDES the exact-edge selections (min(selected) <=
    # grid[1] — the #1887 tripwire arm-(a) convention).
    n_low1 = int(np.sum(finite <= grid[1])) if len(grid) > 1 else n_low
    n_high1 = int(np.sum(finite >= grid[-2])) if len(grid) > 1 else n_high
    return {
        "grid": grid,
        "selected_hist": {f"{g:g}": int(np.sum(finite == g)) for g in grid},
        "n_selected": n_sel,
        "n_at_low_edge": n_low,
        "n_at_high_edge": n_high,
        "frac_at_low_edge": (n_low / n_sel) if n_sel else None,
        "frac_at_high_edge": (n_high / n_sel) if n_sel else None,
        "n_within_one_step_low": n_low1,
        "n_within_one_step_high": n_high1,
        "frac_within_one_step_low": (n_low1 / n_sel) if n_sel else None,
        "frac_within_one_step_high": (n_high1 / n_sel) if n_sel else None,
        "frozen_layer_rows": {str(li): matrix[li] for li in frozen_layers if li < lamf.shape[0]},
        "gcv_lambda_layer_x_fold": matrix,
    }


def _edge_extend_grid(grid: np.ndarray, low: bool, high: bool) -> np.ndarray:
    """One-DECADE extension (2 points, half-decade spacing preserved) on each
    flagged side of a strictly-ascending log-spaced grid (plan §4 edge rule)."""
    grid = np.asarray(grid, dtype=np.float64)
    lg0, lg1 = float(np.log10(grid[0])), float(np.log10(grid[-1]))
    pre = np.power(10.0, [lg0 - 1.0, lg0 - 0.5]) if low else np.empty(0)
    post = np.power(10.0, [lg1 + 0.5, lg1 + 1.0]) if high else np.empty(0)
    out = np.concatenate([pre, grid, post])
    fc._validate_lambda_grid(out)
    return out


def _run_sweep_edge(
    X: np.ndarray,
    Y: np.ndarray,
    conv_ids: np.ndarray,
    *,
    base_grid: np.ndarray | None,
    sweep_kwargs: dict,
    sweep_fn=None,
) -> tuple[dict, dict | None, np.ndarray | None]:
    """Adaptive edge rule around one full-cell sweep (plan v13 §4 Phase FIT).

    ``base_grid=None`` runs exactly ONE sweep on the module default grid and
    returns ``(sweep, None, None)`` — the byte-identical v1 path. With a
    grid: after each sweep, if ANY observed (layer, fold) selection sits AT
    the grid min (max), extend that side one decade (2 points, half-decade
    spacing) and RE-RUN the FULL cell — observed + every null draw
    (selection stays symmetric: the ``lambdas=`` grid threads into the null
    scans inside ``heldout_r2_sweep``). At most ``cm.MAX_EDGE_EXTENSIONS``
    extensions per side; a cell still at an edge afterwards is labeled
    ``estimator-limited: lambda-edge`` in the returned edge block.
    """
    sweep_fn = sweep_fn or fc.heldout_r2_sweep
    if base_grid is None:
        return sweep_fn(X, Y, conv_ids, **sweep_kwargs), None, None
    base_grid = np.asarray(base_grid, dtype=np.float64)
    grid = base_grid
    ext_low = ext_high = 0
    history: list[dict] = []
    while True:
        sweep = sweep_fn(X, Y, conv_ids, lambdas=grid, **sweep_kwargs)
        lam = np.asarray(sweep["gcv_lambda"], dtype=np.float64)
        finite = lam[np.isfinite(lam)]
        n_low = int(np.sum(finite == grid[0]))
        n_high = int(np.sum(finite == grid[-1]))
        history.append(
            {
                "grid_min": float(grid[0]),
                "grid_max": float(grid[-1]),
                "grid_len": int(len(grid)),
                "n_at_low_edge": n_low,
                "n_at_high_edge": n_high,
            }
        )
        want_low = n_low > 0 and ext_low < cm.MAX_EDGE_EXTENSIONS
        want_high = n_high > 0 and ext_high < cm.MAX_EDGE_EXTENSIONS
        if not (want_low or want_high):
            limited = bool(n_low or n_high)
            edge_block = {
                "rule": (
                    "one-decade (2-pt half-decade) extension per flagged side; "
                    "full-cell selection-symmetric re-run; <=2 extensions/side"
                ),
                "base_grid_min": float(base_grid[0]),
                "base_grid_max": float(base_grid[-1]),
                "realized_grid": [float(v) for v in grid],
                "extensions_low": ext_low,
                "extensions_high": ext_high,
                "estimator_limited": "lambda-edge" if limited else None,
                "history": history,
            }
            return sweep, edge_block, grid
        ext_low += int(want_low)
        ext_high += int(want_high)
        grid = _edge_extend_grid(grid, want_low, want_high)
        print(
            f"[fit1336] edge rule: extend(low={want_low}, high={want_high}) -> "
            f"[{grid[0]:g}, {grid[-1]:g}] (ext {ext_low}/{ext_high})",
            flush=True,
        )


def _persist_preds(
    preds_dir: Path,
    cell_id: str,
    sweep: dict,
    conv_ids,
    tag: str = "",
    manifest_name: str = "preds_manifest.json",
) -> None:
    """fp16 held-out prediction matrices + manifest (round-5 preds pattern).

    ``manifest_name`` default preserves the v1 manifest; the v2 path writes
    ``preds_manifest_v2.json`` (plan §3 row-coverage contract).
    """
    preds_dir.mkdir(parents=True, exist_ok=True)
    fname = f"preds_{cell_id}{tag}.npz"
    arrays = {f"preds_l{li}": p.astype(np.float16) for li, p in sweep["preds_frozen"].items()}
    arrays["fitted_mask"] = sweep["fitted_mask"]
    arrays["conv_ids"] = np.asarray([str(c) for c in conv_ids])
    arrays["folds"] = sweep["folds"]
    path = preds_dir / fname
    np.savez(path, **arrays)  # plain savez: client compression OFF for Xet (#813)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = preds_dir / manifest_name
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[fname] = {
        "sha256": sha,
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "dtype_preds": "float16",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[fit1336] persisted {path} (sha256 {sha[:12]}…)")


def run_one_cell(
    cell: dict,
    ts_dir: Path,
    out_dir: Path,
    preds_dir: Path,
    *,
    frozen_layers: tuple[int, ...],
    n_folds: int,
    seed: int,
    null_draws: int,
    n_boot: int,
    matched_n: int | None,
    expected_layers: int | None,
    qwen_cal: dict,
    x_slot: str = "context",
    lambda_grid: np.ndarray | None = None,
    v2: bool = False,
    matched_n_seed: int | None = None,
    use_concat: bool = False,
    wave1_dir: Path | None = None,
    gen_root: Path | None = None,
) -> dict:
    """One cell's full fit battery. All new kwargs are default-preserving:
    the v1 call shape (no ``v2``/``lambda_grid``/``x_slot``) is byte-identical
    to the committed behavior. Under ``v2``: the adaptive edge rule wraps the
    sweep on ``lambda_grid``, outputs land under ``cells_v2/``, the manifest
    is ``preds_manifest_v2.json``, and matched-n companions refit the
    persist-layer subset only at the seed-1336 subsample (plan v13 §4).
    ``use_concat`` (v2 production) routes the two EXTENDED corpora through the
    wave-1 + extension concat loader (boundary/disjointness + sha-join
    asserts, plan v13 §4 Phase EXT)."""
    cell_id = cell["cell_id"]
    if use_concat and cell["corpus"] in et.CONCAT_SOURCES:
        bundle = et.load_bundle_concat(
            ts_dir,
            cell["model"],
            cell["format"],
            cell["corpus"],
            wave1_dir=wave1_dir,
            gen_root=gen_root,
        )
    else:
        bundle = fc._load_bundle_any(ts_dir, cell["model"], cell["format"], cell["corpus"])
    exp = expected_layers if expected_layers is not None else _bundle_n_layers(bundle)
    xy = _cell_xy_1336(bundle, exp, x_slot=x_slot)
    X, Y, conv_ids = xy["X"], xy["Y"], xy["conv_ids"]
    cells_subdir = "cells_v2" if v2 else "cells"
    manifest_name = "preds_manifest_v2.json" if v2 else "preds_manifest.json"
    print(f"[fit1336] cell={cell_id} n={len(conv_ids)} x_slot={x_slot}", flush=True)

    # Preds/cosine capture + the recal primary run at frozen + the E1 verdict
    # layer (cm.preds_layers); every REGISTERED statistic below (selection-
    # symmetric frozen table, headline-rule domain, cosine/CI reads) stays on
    # the frozen set — the L29 extension is capture-only (plan v9 route 1).
    persist_layers = cm.preds_layers(frozen_layers)
    sweep, edge_block, realized_grid = _run_sweep_edge(
        X,
        Y,
        conv_ids,
        base_grid=lambda_grid,
        sweep_kwargs=dict(
            n_folds=n_folds,
            seed=seed,
            null_draws=null_draws,
            frozen_layers=persist_layers,
            collect_lambdas=True,
        ),
    )
    r2_obs, r2_null = sweep["r2_obs"], sweep["r2_null"]
    summary = fc.selection_symmetric_summary(r2_obs, r2_null, frozen_layers=frozen_layers)

    fl = [li for li in frozen_layers if li < X.shape[1]]
    rp = fc.random_projection_control(X, Y, conv_ids, layers=fl, n_folds=n_folds, seed=seed)
    mb = fc.mean_baseline_r2(Y, conv_ids, layers=fl, n_folds=n_folds, seed=seed)

    cosine_stats, r2_cis = {}, {}
    fitted = sweep["fitted_mask"]
    for li in fl:
        cos = sweep["cosines"][li][fitted]
        cosine_stats[str(li)] = fc.bootstrap_ci(cos, n_boot=n_boot, seed=seed + li)
        pred = sweep["preds_frozen"][li][fitted]
        r2_cis[str(li)] = fc.bootstrap_r2_ci(
            pred, Y[fitted, li, :], n_boot=n_boot, seed=seed + 100 + li
        )

    skill_over_mean = {
        str(li): float(r2_obs[li]) - float(mb.get(str(li), float("nan"))) for li in fl
    }
    nll = xy.get("nll")
    nll_stats = None
    if nll is not None and len(nll):
        nll_stats = {
            "mean": float(np.mean(nll)),
            "median": float(np.median(nll)),
            "p90": float(np.quantile(nll, 0.9)),
        }
    degeneracy = _prefix_degeneracy(bundle, frozen_layers)

    payload = {
        "metadata": _metadata(seed, len(conv_ids)),
        "cell": dict(cell),
        "frozen_layers": list(frozen_layers),
        "preds_layers": list(persist_layers),
        "r2_per_layer_obs": [float(v) for v in r2_obs],
        "recal": _recal_block(sweep, Y, persist_layers, qwen_cal),
        "lambda_audit": _lambda_audit(sweep, frozen_layers, grid=realized_grid),
        "selection_symmetric": summary,
        "random_projection_control_r2": rp,
        "mean_baseline_r2": mb,
        "skill_over_mean": skill_over_mean,
        "cosine_frozen_layers": cosine_stats,
        "r2_bootstrap_ci_frozen_layers": r2_cis,
        "nll_a1": nll_stats,
        "prefix_slot_degeneracy": degeneracy,
        "n_folds": n_folds,
        "null_draws": null_draws,
    }
    if v2:
        payload["x_slot"] = x_slot
        payload["lambda_edge_rule"] = edge_block
        if edge_block is not None and edge_block["estimator_limited"]:
            payload["estimator_limited"] = edge_block["estimator_limited"]
    _write_json(out_dir / cells_subdir / f"cells_{cell_id}.json", payload)
    _write_json(
        out_dir / cells_subdir / f"nulls_{cell_id}.json",
        {
            "metadata": _metadata(seed, len(conv_ids)),
            "cell_id": cell_id,
            "layers": list(range(len(r2_obs))),
            "observed_row": [float(v) for v in r2_obs],
            "null_matrix": [[float(v) for v in row] for row in r2_null],
            "null_layer_max_per_draw": summary["null_layer_max_r2_per_draw"],
        },
    )
    _persist_preds(preds_dir, cell_id, sweep, conv_ids, manifest_name=manifest_name)

    if v2 and matched_n is not None and len(conv_ids) > matched_n:
        # v2 matched-n companion (plan §4 Phase FIT): persist-LAYER-SUBSET
        # refit only (headline layer is among the frozen set by the
        # pre-registered rule — refitting the subset keeps the companion
        # available whichever frozen layer the rule selects), n=7,350,
        # seed-1336 subsample, same edge rule + grid as the main sweep.
        sub_seed = matched_n_seed if matched_n_seed is not None else seed
        rng = np.random.default_rng(sub_seed)
        keep = np.sort(rng.choice(len(conv_ids), size=matched_n, replace=False))
        sub = [li for li in persist_layers if li < X.shape[1]]
        Xm, Ym = X[keep][:, sub, :], Y[keep][:, sub, :]
        sweep_m, edge_m, grid_m = _run_sweep_edge(
            Xm,
            Ym,
            conv_ids[keep],
            base_grid=lambda_grid,
            sweep_kwargs=dict(
                n_folds=n_folds,
                seed=seed,
                null_draws=null_draws,
                frozen_layers=tuple(range(len(sub))),
                collect_lambdas=True,
            ),
        )
        pos2layer = {i: li for i, li in enumerate(sub)}
        recal_m = _recal_block(sweep_m, Ym, tuple(range(len(sub))), qwen_cal)
        recal_m["per_layer"] = {str(pos2layer[int(k)]): v for k, v in recal_m["per_layer"].items()}
        if recal_m["s_recal_argmax_layer"] is not None:
            recal_m["s_recal_argmax_layer"] = pos2layer[int(recal_m["s_recal_argmax_layer"])]
        audit_m = _lambda_audit(sweep_m, tuple(range(len(sub))), grid=grid_m)
        _write_json(
            out_dir / cells_subdir / f"cells_matchedn_{cell_id}.json",
            {
                "metadata": _metadata(seed, matched_n),
                "cell": dict(cell),
                "x_slot": x_slot,
                "matched_n": matched_n,
                "subsample_seed": int(sub_seed),
                "layers_fit": [int(li) for li in sub],
                "r2_obs_by_layer": {
                    str(pos2layer[i]): float(v) for i, v in enumerate(sweep_m["r2_obs"])
                },
                "recal": recal_m,
                "lambda_audit": audit_m,
                "lambda_edge_rule": edge_m,
                "n_folds": n_folds,
                "null_draws": null_draws,
            },
        )
        # Remap position-keyed preds back to true layer ids so the persisted
        # npz keys stay layer-true (preds_l16, ... not preds_l0).
        sweep_m_named = dict(sweep_m)
        sweep_m_named["preds_frozen"] = {
            pos2layer[i]: p for i, p in sweep_m["preds_frozen"].items()
        }
        _persist_preds(
            preds_dir,
            cell_id,
            sweep_m_named,
            conv_ids[keep],
            tag="_matchedn",
            manifest_name=manifest_name,
        )
    elif matched_n is not None and len(conv_ids) > matched_n:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(conv_ids), size=matched_n, replace=False))
        sweep_m = fc.heldout_r2_sweep(
            X[keep],
            Y[keep],
            conv_ids[keep],
            n_folds=n_folds,
            seed=seed,
            null_draws=null_draws,
            frozen_layers=persist_layers,
            collect_lambdas=True,
        )
        summary_m = fc.selection_symmetric_summary(
            sweep_m["r2_obs"], sweep_m["r2_null"], frozen_layers=frozen_layers
        )
        _write_json(
            out_dir / "cells" / f"cells_matchedn_{cell_id}.json",
            {
                "metadata": _metadata(seed, matched_n),
                "cell": dict(cell),
                "matched_n": matched_n,
                "subsample_seed": seed,
                "r2_per_layer_obs": [float(v) for v in sweep_m["r2_obs"]],
                "recal": _recal_block(sweep_m, Y[keep], persist_layers, qwen_cal),
                "lambda_audit": _lambda_audit(sweep_m, frozen_layers),
                "selection_symmetric": summary_m,
                "n_folds": n_folds,
                "null_draws": null_draws,
            },
        )
        _persist_preds(preds_dir, cell_id, sweep_m, conv_ids[keep], tag="_matchedn")
    return payload


# ---------------------------------------------------------------------------
# G1 — rig-transfer kill gate (plan §7)
# ---------------------------------------------------------------------------
def _g1_cell_reads(path: Path) -> tuple[float, float]:
    """(recal primary best, raw companion best) for one G1 cell JSON.

    Fail-loud on a pre-resume JSON without the recal block — a stale cells
    file must never silently feed the re-adjudicated gate (plan v9 route 1).
    """
    cell = json.loads(path.read_text())
    assert "recal" in cell, (
        f"{path} lacks the recal block — stale pre-resume fit output; re-run the fit for this "
        "cell under the plan-v9 resume recipe before evaluating G1"
    )
    return float(cell["recal"]["s_recal"]), float(np.nanmax(cell["r2_per_layer_obs"]))


def run_g1_check(out_dir: Path) -> int:
    """KILL <=> best RECALIBRATED within-stage R^2 < bar_r on the After-RLVR model.

    Plan v9 route 1 re-adjudication: the gate reads the held-out cross-fitted
    per-dim affine-recalibrated primary (best over the persisted layer set —
    the E1 S_r convention) against the exchange-rate-carried bars: kill =
    persisted bar_r (0.20 x rate), marginal = 0.3 x rate. Raw full-sweep best
    stays a reported companion, never the verdict input. Verdict shape is
    unchanged: chat >= marginal -> PASS; chat in [kill, marginal) -> marginal
    (exit 4 asks the dispatcher for the naturalistic read first); chat < kill
    -> KILL only when BOTH formats read < kill (a false KILL is the costly
    error).
    """
    qwen_cal = cm.load_qwen_recal_cal(out_dir)
    kill_bar, marginal_bar = float(qwen_cal["bar_r"]), float(qwen_cal["marginal_r2"])
    chat_path = out_dir / "cells" / "cells_rlvr_chat_lmsys5k.json"
    assert chat_path.exists(), f"G1 requires {chat_path} — fit the RLVR chat cell first"
    chat_best, chat_best_raw = _g1_cell_reads(chat_path)
    nat_path = out_dir / "cells" / "cells_rlvr_naturalistic_lmsys5k.json"
    nat_best = nat_best_raw = None
    if nat_path.exists():
        nat_best, nat_best_raw = _g1_cell_reads(nat_path)

    need_nat = chat_best < marginal_bar and nat_best is None
    if need_nat:
        print(f"[g1] chat best R2={chat_best:.4f} < {marginal_bar:.4f} — need naturalistic read")
        return 4
    if chat_best >= marginal_bar:
        verdict, kill = "pass", False
    elif chat_best >= kill_bar:
        verdict, kill = "pass_marginal", False
    else:
        kill = nat_best < kill_bar
        verdict = "kill" if kill else "pass_marginal_naturalistic_carries"
    payload = {
        "metadata": _metadata(cm.FIT_SEED, 0),
        "gate": "G1",
        "primary_scale": "recal",
        "chat_best_r2": chat_best,
        "naturalistic_best_r2": nat_best,
        "kill_threshold": kill_bar,
        "marginal_threshold": marginal_bar,
        "verdict": verdict,
        "kill": bool(kill),
        "raw_companion": {
            "chat_best_r2_raw": chat_best_raw,
            "naturalistic_best_r2_raw": nat_best_raw,
            "kill_threshold_raw": cm.G1_KILL_R2,
            "marginal_threshold_raw": cm.G1_MARGINAL_R2,
        },
        "qwen_exchange": {
            k: qwen_cal[k] for k in ("s_qwen_recal", "committed_anchor", "rate", "path")
        },
    }
    _write_json(out_dir / "gates" / "g1_gate.json", payload)
    print(
        f"[g1] recal chat={chat_best:.4f} nat={nat_best} (bars kill={kill_bar:.4f} "
        f"marginal={marginal_bar:.4f}; raw companion chat={chat_best_raw:.4f}) -> {verdict}"
    )
    return 3 if kill else 0


# ---------------------------------------------------------------------------
# v3 pooled multi-dataset fits (plan v15 Phase FIT_pool) + gates G0v3 / G1'v3
# ---------------------------------------------------------------------------
# NAMING COLLISION GUARD: the split manifest's per-row "arm" field means the
# train/test SIDE of the pooled 80/20 split; a v3 cell's "arm" field means the
# on/off-POLICY text-source arm (cm.CELLS_V3). This module reads the manifest
# field into a "side" key immediately and never mixes the two.


def _load_split_manifest(path: Path) -> dict:
    """Fail-loud load + row-schema check of the Phase C_pool split manifest."""
    assert path.exists(), (
        f"{path} missing — run Phase C_pool (scripts/issue1336_pooled_split.py) before any "
        "pooled fit (driver-enforced ordering)"
    )
    man = json.loads(path.read_text())
    assert man.get("row_index"), f"{path} carries no row_index — malformed split manifest"
    for e in man["row_index"]:
        assert e.get("prompt_idx") is not None, f"manifest row without prompt_idx: {e}"
        side = e["arm"]  # manifest "arm" == train/test SIDE (collision guard above)
        assert side in ("train", "test"), f"unknown split side {side!r}"
        assert (e.get("fold") is not None) == (side == "train"), (
            f"fold/side mismatch in manifest row: {e}"
        )
    return man


def _manifest_rows_by_corpus(man: dict) -> dict[str, list[dict]]:
    """row_index entries grouped by corpus, manifest order preserved (determinism)."""
    by: dict[str, list[dict]] = {}
    for e in man["row_index"]:
        by.setdefault(e["corpus"], []).append(e)
    return by


def _pooled_units_for(models=None, text_sources=None) -> list[dict]:
    """The pooled fit units — (activation checkpoint x on/off arm) — from CELLS_V3.

    Groups ``cm.cells_v3_for(...)`` pairs (the CONSUMED registry — never
    redefined here) into fit units: the full grid is 10 units (5 checkpoints
    x {on, off}); an on unit carries the single diagonal text source, an off
    unit the off-diagonal sources (plan v15 §4 Phase FIT_pool).
    """
    units: dict[tuple[str, str], dict] = {}
    for p in cm.cells_v3_for(models, text_sources):
        key = (p["model"], p["arm"])
        u = units.setdefault(
            key,
            {
                "cell_id": f"pooled_{p['model']}_arm_{p['arm']}",
                "model": p["model"],
                "hf_id": p["hf_id"],
                "format": p["format"],
                "arm": p["arm"],
                "text_sources": [],
            },
        )
        u["text_sources"].append(p["text_source"])
    return list(units.values())


def _pooled_smoke_units() -> list[dict]:
    """Smoke units derived from cm.SMOKE_OFFDIAG_PAIRS_V3 (PASS_UNIFIED seam):
    one diagonal on-policy unit + one off-policy unit per registered smoke pair."""
    models = tuple(dict.fromkeys(i for i, _ in cm.SMOKE_OFFDIAG_PAIRS_V3))
    sources = tuple(dict.fromkeys([*models, *(j for _, j in cm.SMOKE_OFFDIAG_PAIRS_V3)]))
    return _pooled_units_for(models, sources)


def _pooled_bundle(
    unit_model: str,
    text_source: str,
    corpus: str,
    *,
    ts_dir: Path,
    off_root: Path | None,
    smoke: bool,
    wave1_dir: Path | None,
    gen_root: Path | None,
) -> dict:
    """Load one (activation checkpoint, text source, corpus) capture bundle.

    Diagonal (on-policy) pairs reuse the round-3 v2 turnstores verbatim —
    incl. the wave-1 concat loader for the extended corpora in production
    (run_one_cell's use_concat contract). Off-diagonal (off-policy) pairs
    read the Phase EXT_off trees ``<off_root>/turnstore_offpolicy_<i>_chat_<j>``;
    stems inside keep standard naming so the #825 loaders read them unchanged
    (cm.offpolicy_ts_dirname docstring).
    """
    fmt = cm.V3_TEXT_FORMAT
    if text_source == unit_model:
        if (not smoke) and corpus in et.CONCAT_SOURCES:
            return et.load_bundle_concat(
                ts_dir, unit_model, fmt, corpus, wave1_dir=wave1_dir, gen_root=gen_root
            )
        return fc._load_bundle_any(
            ts_dir, unit_model, fmt, corpus, wanted_keys=("slots", "profiles", "nll")
        )
    assert off_root is not None, "off-diagonal pair needs --offpolicy-root"
    off_dir = off_root / (
        cm.offpolicy_ts_dirname(unit_model, text_source) + ("_smoke" if smoke else "")
    )
    return fc._load_bundle_any(
        off_dir, unit_model, fmt, corpus, wanted_keys=("slots", "profiles", "nll")
    )


def _pooled_xy_from_bundle(
    bundle: dict, entries: list[dict], expected_layers: int | None, x_slot: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(X, Y, nll) for one bundle, row-selected to the manifest entries IN
    MANIFEST ORDER — the pooled row-coverage contract (plan v15 §3): every
    manifest row must resolve in the bundle, NaN-free; fail-loud otherwise.

    Dtype-preserving deviation from ``_cell_xy_1336``: stored fp16 arrays are
    NOT cast to fp32 here (the pooled off arm quadruples the row count, and
    the fit core converts per (layer, fold) to fp64 anyway — fp16 -> fp64 is
    exact, so the numbers are unchanged while peak RSS halves).
    """
    arrays, sidecar = bundle["arrays"], bundle["sidecar"]
    slots, profiles = arrays["slots"], arrays["profiles"]
    assert slots.shape[1] == 2, f"expected 2 slots (prefix, context), got {slots.shape}"
    assert profiles.shape[1] == 2, f"expected 2 turn profiles, got {profiles.shape}"
    if expected_layers is not None:
        assert slots.shape[2] == expected_layers, (
            f"layer axis {slots.shape[2]} != expected {expected_layers}"
        )
    si = {"context": 1, "prefix": 0}[x_slot]
    pos = {str(c): i for i, c in enumerate(sidecar["conv_ids"])}
    missing = [e for e in entries if "s{}".format(e["prompt_idx"]) not in pos]
    if missing:
        ex = ["s{}".format(e["prompt_idx"]) for e in missing[:5]]
        raise AssertionError(
            f"{len(missing)} manifest rows missing from the bundle (e.g. {ex}) — "
            "pooled row-coverage break (plan v15 §3)"
        )
    sel = np.asarray([pos["s{}".format(e["prompt_idx"])] for e in entries], dtype=np.int64)
    X = np.asarray(slots)[sel, si, :, :]
    Y = np.asarray(profiles)[sel, 1, :, :]
    bad = np.isnan(X).any(axis=(1, 2)) | np.isnan(Y).any(axis=(1, 2))
    if bad.any():
        ex = [entries[i]["prompt_sha"][:12] for i in np.flatnonzero(bad)[:5]]
        raise AssertionError(
            f"{int(bad.sum())} NaN rows among manifest-selected rows (prompt_sha e.g. {ex}) "
            "— refusing to silently drop pooled rows"
        )
    nll = None
    if "nll" in arrays:
        nll_arr = np.asarray(arrays["nll"], dtype=np.float32)
        assert nll_arr.shape[1] >= 2, f"nll turns {nll_arr.shape} lack target turn 1"
        nll = nll_arr[sel, 1]
    return X, Y, nll


def _assemble_pooled_rows(
    unit: dict,
    by_corpus: dict[str, list[dict]],
    *,
    corpora: tuple[str, ...],
    ts_dir: Path,
    off_root: Path | None,
    smoke: bool,
    wave1_dir: Path | None,
    gen_root: Path | None,
    x_slot: str,
    expected_layers: int | None,
    frozen_layers: tuple[int, ...],
) -> dict:
    """Concatenate the unit's (text source x corpus) captures in registry order.

    Preallocates the pooled (N, L, D) arrays from manifest counts so peak RSS
    is bounded by pooled + ONE bundle (the off arm multiplies row count by
    its source count). Returns X, Y, nll, per-row metadata, and the prefix
    degeneracy read (computed on the FIRST text source only: the prefix slot
    at a fixed activation checkpoint is text-source-invariant by construction).
    """
    for c in corpora:
        assert by_corpus.get(c), f"split manifest has no rows for corpus {c!r}"
    n_per_corpus = {c: len(by_corpus[c]) for c in corpora}
    n_total = sum(n_per_corpus.values()) * len(unit["text_sources"])
    X_all = Y_all = None
    nll_all = np.full(n_total, np.nan, dtype=np.float32)
    any_nll = False
    rows: list[dict] = []
    degeneracy: dict[str, dict] = {}
    ofs = 0
    for j in unit["text_sources"]:
        for c in corpora:
            entries = by_corpus[c]
            bundle = _pooled_bundle(
                unit["model"],
                j,
                c,
                ts_dir=ts_dir,
                off_root=off_root,
                smoke=smoke,
                wave1_dir=wave1_dir,
                gen_root=gen_root,
            )
            X, Y, nll = _pooled_xy_from_bundle(bundle, entries, expected_layers, x_slot)
            if X_all is None:
                shape = (n_total, X.shape[1], X.shape[2])
                X_all = np.empty(shape, dtype=X.dtype)
                Y_all = np.empty(shape, dtype=Y.dtype)
            assert X.shape[1:] == X_all.shape[1:], (X.shape, X_all.shape)
            n_c = X.shape[0]
            X_all[ofs : ofs + n_c] = X
            Y_all[ofs : ofs + n_c] = Y
            if nll is not None:
                nll_all[ofs : ofs + n_c] = nll
                any_nll = True
            if j == unit["text_sources"][0]:
                degeneracy[f"{j}/{c}"] = _prefix_degeneracy(bundle, frozen_layers)
            for e in entries:
                rows.append(
                    {
                        "row_id": "{}:{}:s{}".format(j, c, e["prompt_idx"]),
                        "text_source": j,
                        "corpus": c,
                        "conv_id": "s{}".format(e["prompt_idx"]),
                        "prompt_sha": e["prompt_sha"],
                        "cluster": int(e["cluster"]),
                        "side": e["arm"],  # manifest "arm" == train/test SIDE
                        "fold": e["fold"],
                    }
                )
            del bundle, X, Y
            ofs += n_c
    assert ofs == n_total, (ofs, n_total)
    return {
        "X": X_all,
        "Y": Y_all,
        "nll": nll_all if any_nll else None,
        "rows": rows,
        "degeneracy": degeneracy,
    }


def _final_fit_test_preds(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    groups_tr: np.ndarray,
    Xte: np.ndarray,
    layers: list[int],
    grid: np.ndarray,
    seed: int,
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    """Full-train-side final fit per persisted layer, predicting the pooled
    TEST side (plan v15 §4 per-corpus slice read). Reuses the fit core's own
    within-fit path VERBATIM — _prep_fold + _prep_inner_lambda (inner-group-CV
    lambda on the realized grid) + _ridge_predict_cached — so the estimator
    matches the CV sweep exactly (no re-implemented solve loop).
    """
    preds: dict[int, np.ndarray] = {}
    lams: dict[str, float] = {}
    for li in layers:
        cache = fc._prep_fold(Xtr[:, li, :], Xte[:, li, :])
        inner = fc._prep_inner_lambda(Xtr[:, li, :], groups_tr, fc.N_INNER_LAMBDA_FOLDS, seed)
        if inner is not None:
            cache["inner"] = inner
        pred, lam = fc._ridge_predict_cached(cache, Ytr[:, li, :], return_lam=True, lambdas=grid)
        preds[li] = np.asarray(pred)
        lams[str(li)] = float(lam)
    return preds, lams


def _pooled_test_block(
    preds_te: dict[int, np.ndarray], Yte: np.ndarray, rows_te: list[dict], layers: list[int]
) -> dict:
    """Held-out 20% test-side R2 — pooled + per-corpus + per-text-source slices."""
    corpus_arr = np.asarray([r["corpus"] for r in rows_te])
    source_arr = np.asarray([r["text_source"] for r in rows_te])
    out: dict = {"r2_pooled": {}, "r2_per_corpus": {}, "r2_per_text_source": {}}
    for li in layers:
        pred = preds_te[li]
        true = Yte[:, li, :]
        out["r2_pooled"][str(li)] = fc._pooled_r2(pred, true)
        out["r2_per_corpus"][str(li)] = {
            c: fc._pooled_r2(pred[corpus_arr == c], true[corpus_arr == c])
            for c in dict.fromkeys(corpus_arr.tolist())
        }
        out["r2_per_text_source"][str(li)] = {
            s: fc._pooled_r2(pred[source_arr == s], true[source_arr == s])
            for s in dict.fromkeys(source_arr.tolist())
        }
    return out


def _mapping_baselines_block(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xte: np.ndarray,
    Yte: np.ndarray,
    preds_te: dict[int, np.ndarray],
    layers: list[int],
    *,
    n_pool_max: int = 2000,
    seed: int = 1336,
) -> dict:
    """Guideline-11 pair per fitted map: identity+learned-bias baseline
    (input/output dims match — applicable) + kNN retrieval of the prediction
    among the held-out TEST pool (euclidean + cosine; chance stated by the
    helper). Pool subsampled to <=2000 rows at seed 1336 — the round-5 metric
    ladder's kNN convention (issue1336_metric_ladder.py).
    """
    from explore_persona_space.analysis import mapping_baselines as mbase

    rng = np.random.default_rng(seed)
    n_te = int(Xte.shape[0])
    keep = np.sort(rng.choice(n_te, size=min(n_pool_max, n_te), replace=False))
    out: dict = {}
    for li in layers:
        true = Yte[:, li, :]
        pred_id = mbase.identity_bias_predict(Xtr[:, li, :], Ytr[:, li, :], Xte[:, li, :])
        blk: dict = {
            "identity_bias_r2_test": fc._pooled_r2(pred_id, true),
            "ridge_r2_test": fc._pooled_r2(preds_te[li], true),
            "knn_pool_rows": int(len(keep)),
            "knn": {},
        }
        for name, pred in (("ridge", np.asarray(preds_te[li])), ("identity_bias", pred_id)):
            for metric in ("euclidean", "cosine"):
                blk["knn"][f"{name}_{metric}"] = mbase.knn_retrieval(
                    np.asarray(pred)[keep], true[keep], ks=(1, 5), metric=metric
                )
        out[str(li)] = blk
    return out


def run_pooled_cell(
    unit: dict,
    man: dict,
    man_path: Path,
    ts_dir: Path,
    off_root: Path | None,
    out_dir: Path,
    preds_root: Path,
    *,
    corpora: tuple[str, ...],
    frozen_layers: tuple[int, ...],
    n_folds: int,
    seed: int,
    null_draws: int,
    n_boot: int,
    matched_n: int | None,
    matched_n_seed: int,
    expected_layers: int | None,
    qwen_cal: dict,
    x_slot: str,
    lambda_grid: np.ndarray,
    smoke: bool,
    wave1_dir: Path | None,
    gen_root: Path | None,
) -> dict:
    """One pooled (checkpoint x arm) fit unit — plan v15 §4 Phase FIT_pool.

    ONE pooled ridge X -> Y through the shared #825 batched core: the split
    manifest's cluster-grouped folds ride into ``heldout_r2_sweep`` as the
    conv_ids GROUP KEY (``_cv_folds`` over exactly ``n_folds`` unique values
    is a bijective relabeling, so the REALIZED partition is the manifest's
    persisted partition, never a re-derived one); the 23-pt grid + adaptive
    edge rule wrap the sweep; a train-side final fit predicts the pooled 20%
    test side for the per-corpus slice read (+ Guideline-11 baselines).
    """
    cell_id = unit["cell_id"]
    man_sha = hashlib.sha256(man_path.read_bytes()).hexdigest()
    by_corpus = _manifest_rows_by_corpus(man)
    asm = _assemble_pooled_rows(
        unit,
        by_corpus,
        corpora=corpora,
        ts_dir=ts_dir,
        off_root=off_root,
        smoke=smoke,
        wave1_dir=wave1_dir,
        gen_root=gen_root,
        x_slot=x_slot,
        expected_layers=expected_layers,
        frozen_layers=frozen_layers,
    )
    rows = asm["rows"]
    train_mask = np.asarray([r["side"] == "train" for r in rows])
    ids = np.asarray([r["row_id"] for r in rows])
    folds_man = np.asarray([-1 if r["fold"] is None else int(r["fold"]) for r in rows])
    Xtr, Ytr = asm["X"][train_mask], asm["Y"][train_mask]
    Xte, Yte = asm["X"][~train_mask], asm["Y"][~train_mask]
    asm["X"] = asm["Y"] = None  # release the pooled originals (peak-RSS bound)
    n_tr, n_te = int(Xtr.shape[0]), int(Xte.shape[0])
    L, d = int(Xtr.shape[1]), int(Xtr.shape[2])
    assert n_te > 0, "pooled split has no test rows"
    groups_tr = folds_man[train_mask]
    assert (groups_tr >= 0).all(), "train row without a manifest fold"
    uniq_folds = np.unique(groups_tr)
    assert len(uniq_folds) == n_folds, (
        f"manifest carries {len(uniq_folds)} folds but --folds={n_folds}; pass --folds equal "
        "to the split manifest's n_folds"
    )
    if not smoke:
        # estimator-validity regime statement (plan v15 §7 G1'): primal route
        assert n_tr > d, f"pooled production fit needs n_train > d, got n={n_tr} d={d}"
    persist_layers = cm.preds_layers(frozen_layers)
    print(
        f"[fit1336] pooled cell={cell_id} n_train={n_tr} n_test={n_te} d={d} "
        f"sources={unit['text_sources']} x_slot={x_slot}",
        flush=True,
    )
    sweep, edge_block, realized_grid = _run_sweep_edge(
        Xtr,
        Ytr,
        groups_tr,
        base_grid=lambda_grid,
        sweep_kwargs=dict(
            n_folds=n_folds,
            seed=seed,
            null_draws=null_draws,
            frozen_layers=persist_layers,
            collect_lambdas=True,
        ),
    )
    r2_obs, r2_null = sweep["r2_obs"], sweep["r2_null"]
    summary = fc.selection_symmetric_summary(r2_obs, r2_null, frozen_layers=frozen_layers)
    fl = [li for li in frozen_layers if li < L]
    rp = fc.random_projection_control(Xtr, Ytr, groups_tr, layers=fl, n_folds=n_folds, seed=seed)
    mb = fc.mean_baseline_r2(Ytr, groups_tr, layers=fl, n_folds=n_folds, seed=seed)
    cosine_stats, r2_cis = {}, {}
    fitted = sweep["fitted_mask"]
    for li in fl:
        cos = sweep["cosines"][li][fitted]
        cosine_stats[str(li)] = fc.bootstrap_ci(cos, n_boot=n_boot, seed=seed + li)
        pred = sweep["preds_frozen"][li][fitted]
        r2_cis[str(li)] = fc.bootstrap_r2_ci(
            pred, Ytr[fitted, li, :], n_boot=n_boot, seed=seed + 100 + li
        )
    skill_over_mean = {
        str(li): float(r2_obs[li]) - float(mb.get(str(li), float("nan"))) for li in fl
    }
    nll_stats = None
    if asm["nll"] is not None:
        finite = np.isfinite(asm["nll"])
        if finite.any():
            v = asm["nll"][finite]
            nll_stats = {
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "p90": float(np.quantile(v, 0.9)),
                "n_finite": int(finite.sum()),
                "n_total": int(asm["nll"].shape[0]),
            }
    persist_in_range = [li for li in persist_layers if li < L]
    preds_te, lams_te = _final_fit_test_preds(
        Xtr, Ytr, groups_tr, Xte, persist_in_range, realized_grid, seed
    )
    rows_te = [r for r in rows if r["side"] == "test"]
    test_block = _pooled_test_block(preds_te, Yte, rows_te, persist_in_range)
    test_block["final_fit_lambda_per_layer"] = lams_te
    mapping_block = _mapping_baselines_block(Xtr, Ytr, Xte, Yte, preds_te, fl)
    payload = {
        "metadata": _metadata(seed, n_tr + n_te),
        "cell": {k: unit[k] for k in ("cell_id", "model", "hf_id", "format", "arm")},
        "text_sources": list(unit["text_sources"]),
        "corpora": list(corpora),
        "x_slot": x_slot,
        "split_manifest": {
            "path": str(man_path),
            "sha256": man_sha,
            "n_train_rows": n_tr,
            "n_test_rows": n_te,
        },
        "n_train": n_tr,
        "n_test": n_te,
        "d": d,
        "frozen_layers": list(frozen_layers),
        "preds_layers": list(persist_layers),
        "r2_per_layer_obs": [float(v) for v in r2_obs],
        "recal": _recal_block(sweep, Ytr, persist_layers, qwen_cal),
        "lambda_audit": _lambda_audit(sweep, frozen_layers, grid=realized_grid),
        "lambda_edge_rule": edge_block,
        "selection_symmetric": summary,
        "random_projection_control_r2": rp,
        "mean_baseline_r2": mb,
        "skill_over_mean": skill_over_mean,
        "cosine_frozen_layers": cosine_stats,
        "r2_bootstrap_ci_frozen_layers": r2_cis,
        "nll_a1": nll_stats,
        "prefix_slot_degeneracy": asm["degeneracy"],
        "test": test_block,
        "mapping_baselines": mapping_block,
        "n_folds": n_folds,
        "null_draws": null_draws,
    }
    if edge_block is not None and edge_block.get("estimator_limited"):
        payload["estimator_limited"] = edge_block["estimator_limited"]
    _write_json(out_dir / "cells_pooled_v3" / f"cells_{cell_id}.json", payload)
    _write_json(
        out_dir / "cells_pooled_v3" / f"nulls_{cell_id}.json",
        {
            "metadata": _metadata(seed, n_tr),
            "cell_id": cell_id,
            "layers": list(range(len(r2_obs))),
            "observed_row": [float(v) for v in r2_obs],
            "null_matrix": [[float(v) for v in row] for row in r2_null],
            "null_layer_max_per_draw": summary["null_layer_max_r2_per_draw"],
        },
    )
    arm_dir = preds_root / cm.POOLED_ARM_DIRS[unit["arm"]]
    ids_tr = ids[train_mask]
    _persist_preds(arm_dir, cell_id, sweep, ids_tr, manifest_name="preds_pooled_v3_manifest.json")
    test_sweep = {
        "preds_frozen": preds_te,
        "fitted_mask": np.ones(n_te, dtype=bool),
        "folds": np.full(n_te, -1, dtype=np.int64),
    }
    _persist_preds(
        arm_dir,
        cell_id,
        test_sweep,
        ids[~train_mask],
        tag="_test",
        manifest_name="preds_pooled_v3_manifest.json",
    )
    cols = ("row_id", "text_source", "corpus", "conv_id", "prompt_sha", "cluster", "fold", "side")
    _write_json(
        arm_dir / f"rows_{cell_id}.json",
        {
            "metadata": _metadata(seed, n_tr + n_te),
            "cell_id": cell_id,
            "split_manifest_sha256": man_sha,
            "note": (
                "npz 'folds' are the fit core's bijective relabeling of the manifest folds; "
                "this manifest carries the manifest-side fold ids"
            ),
            "columns": {k: [r[k] for r in rows] for k in cols},
        },
    )
    if matched_n is not None and n_tr > matched_n:
        rng = np.random.default_rng(matched_n_seed)
        keep = np.sort(rng.choice(n_tr, size=matched_n, replace=False))
        sub = persist_in_range
        Xm, Ym = Xtr[keep][:, sub, :], Ytr[keep][:, sub, :]
        gm = groups_tr[keep]
        assert len(np.unique(gm)) == n_folds, "matched-n subsample lost a manifest fold"
        sweep_m, edge_m, grid_m = _run_sweep_edge(
            Xm,
            Ym,
            gm,
            base_grid=lambda_grid,
            sweep_kwargs=dict(
                n_folds=n_folds,
                seed=seed,
                null_draws=null_draws,
                frozen_layers=tuple(range(len(sub))),
                collect_lambdas=True,
            ),
        )
        pos2layer = {i: li for i, li in enumerate(sub)}
        recal_m = _recal_block(sweep_m, Ym, tuple(range(len(sub))), qwen_cal)
        recal_m["per_layer"] = {str(pos2layer[int(k)]): v for k, v in recal_m["per_layer"].items()}
        if recal_m.get("s_recal_argmax_layer") is not None:
            recal_m["s_recal_argmax_layer"] = pos2layer[int(recal_m["s_recal_argmax_layer"])]
        _write_json(
            out_dir / "cells_pooled_v3" / f"cells_matchedn_{cell_id}.json",
            {
                "metadata": _metadata(seed, matched_n),
                "cell": {k: unit[k] for k in ("cell_id", "model", "arm")},
                "x_slot": x_slot,
                "matched_n": matched_n,
                "subsample_seed": int(matched_n_seed),
                "layers_fit": [int(li) for li in sub],
                "r2_obs_by_layer": {
                    str(pos2layer[i]): float(v) for i, v in enumerate(sweep_m["r2_obs"])
                },
                "recal": recal_m,
                "lambda_audit": _lambda_audit(sweep_m, tuple(range(len(sub))), grid=grid_m),
                "lambda_edge_rule": edge_m,
                "n_folds": n_folds,
                "null_draws": null_draws,
            },
        )
        sweep_m_named = dict(sweep_m)
        sweep_m_named["preds_frozen"] = {
            pos2layer[i]: p for i, p in sweep_m["preds_frozen"].items()
        }
        _persist_preds(
            arm_dir,
            cell_id,
            sweep_m_named,
            ids_tr[keep],
            tag="_matchedn",
            manifest_name="preds_pooled_v3_manifest.json",
        )
    return payload


def run_pooled_v3(args) -> int:
    """Dispatch the pooled (checkpoint x arm) fit units (plan v15 Phase FIT_pool)."""
    assert not args.v2, "--v3-pooled and --v2 are mutually exclusive"
    assert args.cells, "--v3-pooled requires --cells (pooled unit ids | all | smoke)"
    smoke = args.smoke
    # v3 estimator defaults mirror the v2 branch (module-global patch style,
    # issue825_fit_cells.py L78 convention; process-scoped, like main()'s v2 set).
    fc.N_INNER_LAMBDA_FOLDS = cm.N_INNER_LAMBDA_FOLDS_V2
    lambda_grid = np.asarray(cm.LAMBDAS_23, dtype=np.float64)
    sfx = "_smoke" if smoke else ""
    ts_dir = args.turnstore_dir or Path(f"data/issue_1336/turnstore_v2{sfx}")
    off_root = args.offpolicy_root or Path("data/issue_1336")
    preds_root = args.preds_dir or Path(f"data/issue_1336/preds_pooled_v3{sfx}")
    man_path = args.split_manifest or (
        ps.DATA_ROOT
        / (ps.POOLED_OUT_SUBDIR_SMOKE if smoke else ps.POOLED_OUT_SUBDIR)
        / "split_manifest.json"
    )
    man = _load_split_manifest(man_path)
    corpora = tuple(cm.SMOKE_CORPORA_V2) if smoke else tuple(cm.V2_CORPORA)
    if args.frozen_layers:
        frozen = tuple(int(x) for x in args.frozen_layers.split(",") if x.strip())
    else:
        frozen = cm.SMOKE_FROZEN_LAYERS if smoke else cm.FROZEN_LAYERS
    null_draws = (
        args.null_draws
        if args.null_draws is not None
        else (cm.SMOKE_NULL_DRAWS if smoke else cm.N_NULL_DRAWS)
    )
    n_boot = (
        args.n_boot if args.n_boot is not None else (cm.SMOKE_N_BOOT if smoke else cm.N_BOOTSTRAP)
    )
    if args.cells == "all":
        units = _pooled_units_for()
    elif args.cells == "smoke":
        units = _pooled_smoke_units()
    else:
        by_id = {u["cell_id"]: u for u in _pooled_units_for()}
        wanted = [c.strip() for c in args.cells.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in by_id]
        assert not unknown, f"unknown pooled cell ids {unknown}; known: {sorted(by_id)}"
        units = [by_id[c] for c in wanted]
    matched = None
    if args.matched_n:
        # No v3 registry constant exists for the pooled matched-n companion —
        # plan v15 §6 names n=15,000; the caller passes it explicitly.
        assert args.matched_n_size is not None, (
            "--v3-pooled --matched-n requires an explicit --matched-n-size "
            "(plan v15 §6 pooled companion: 15000)"
        )
        matched = int(args.matched_n_size)
    matched_seed = args.matched_n_seed if args.matched_n_seed is not None else cm.MATCHED_N_V2_SEED
    qwen_cal = cm.load_qwen_recal_cal(args.out_dir)
    for unit in units:
        run_pooled_cell(
            unit,
            man,
            man_path,
            ts_dir,
            off_root,
            args.out_dir,
            preds_root,
            corpora=corpora,
            frozen_layers=frozen,
            n_folds=args.folds,
            seed=args.seed,
            null_draws=null_draws,
            n_boot=n_boot,
            matched_n=matched,
            matched_n_seed=matched_seed,
            expected_layers=None if smoke else cm.EXPECTED_LAYERS,
            qwen_cal=qwen_cal,
            x_slot=(args.x_slot or "context"),
            lambda_grid=lambda_grid,
            smoke=smoke,
            wave1_dir=(args.wave1_turnstore_dir or ts_dir),
            gen_root=(args.gen_root or (None if smoke else Path("data/issue_1336/gen"))),
        )
    return 0


def run_g0v3(args) -> int:
    """G0v3 — pooled-split reproducibility gate (plan v15 §6/§7).

    Refits the round-3 RLVR x lmsys23k-chat cell UNDER THE POOLED SPLIT (the
    manifest's lmsys23k train-side rows + their cluster-grouped folds, 23-pt
    grid + edge rule) and asserts |R2_pooled_slice - R2_round3| <= 0.05 x
    ex_v2 at the headline layer (argmax of the ROUND-3 cell's
    r2_per_layer_obs over its frozen set). Under --smoke the production-
    calibrated verdict is demoted to informational (the #1345
    gate-calibration rule); the computation runs identically.
    """
    smoke = args.smoke
    bars_path = args.out_dir / "gates_v2" / "v2_bars.json"
    assert bars_path.exists(), (
        f"{bars_path} missing — run the G0' v2 gate first (driver-enforced ordering)"
    )
    ex_v2 = float(json.loads(bars_path.read_text())["ex_v2"])
    ref_id = cm.v2_cell_id("rlvr", "chat", "lmsys23k")
    ref_path = args.out_dir / "cells_v2" / f"cells_{ref_id}.json"
    assert ref_path.exists(), (
        f"{ref_path} missing — G0v3 compares against the round-3 per-corpus cell; fit it "
        "(or stage the committed copy) first"
    )
    ref = json.loads(ref_path.read_text())
    r2_ref = np.asarray(ref["r2_per_layer_obs"], dtype=np.float64)
    ref_frozen = [int(li) for li in ref.get("frozen_layers", cm.FROZEN_LAYERS) if li < len(r2_ref)]
    assert ref_frozen and np.isfinite(r2_ref[ref_frozen]).any(), (
        "round-3 cell has no finite frozen-layer R2"
    )
    head = int(ref_frozen[int(np.nanargmax(r2_ref[ref_frozen]))])
    r2_r3 = float(r2_ref[head])
    sfx = "_smoke" if smoke else ""
    ts_dir = args.turnstore_dir or Path(f"data/issue_1336/turnstore_v2{sfx}")
    man_path = args.split_manifest or (
        ps.DATA_ROOT
        / (ps.POOLED_OUT_SUBDIR_SMOKE if smoke else ps.POOLED_OUT_SUBDIR)
        / "split_manifest.json"
    )
    man = _load_split_manifest(man_path)
    entries = [e for e in man["row_index"] if e["corpus"] == "lmsys23k" and e["arm"] == "train"]
    assert entries, "split manifest carries no lmsys23k train rows"
    bundle = _pooled_bundle(
        "rlvr",
        "rlvr",
        "lmsys23k",
        ts_dir=ts_dir,
        off_root=None,
        smoke=smoke,
        wave1_dir=(args.wave1_turnstore_dir or ts_dir),
        gen_root=(args.gen_root or (None if smoke else Path("data/issue_1336/gen"))),
    )
    X, Y, _nll = _pooled_xy_from_bundle(
        bundle, entries, None if smoke else cm.EXPECTED_LAYERS, "context"
    )
    groups = np.asarray([int(e["fold"]) for e in entries])
    assert len(np.unique(groups)) == args.folds, "manifest folds != --folds"
    saved = fc.N_INNER_LAMBDA_FOLDS
    try:
        fc.N_INNER_LAMBDA_FOLDS = cm.N_INNER_LAMBDA_FOLDS_V2
        sweep, _edge, _grid = _run_sweep_edge(
            X,
            Y,
            groups,
            base_grid=np.asarray(cm.LAMBDAS_23, dtype=np.float64),
            sweep_kwargs=dict(
                n_folds=args.folds,
                seed=args.seed,
                null_draws=0,
                collect_cosines=False,
                frozen_layers=(),
                collect_lambdas=True,
            ),
        )
    finally:
        fc.N_INNER_LAMBDA_FOLDS = saved
    if head < len(sweep["r2_obs"]):
        r2_pool = float(sweep["r2_obs"][head])
    else:
        # smoke fixture with fewer layers than the production ref — informational
        r2_pool = float(np.nanmax(np.asarray(sweep["r2_obs"], dtype=np.float64)))
    delta = abs(r2_pool - r2_r3)
    tol = 0.05 * ex_v2
    ok = bool(np.isfinite(delta) and delta <= tol)
    enforced = not smoke
    payload = {
        "metadata": _metadata(args.seed, int(X.shape[0])),
        "gate": "G0v3",
        "cell": ref_id,
        "headline_layer": head,
        "r2_pooled_slice": r2_pool,
        "r2_per_corpus_round3": r2_r3,
        "abs_delta": float(delta),
        "tolerance": float(tol),
        "ex_v2": ex_v2,
        "split_manifest": str(man_path),
        "enforced": enforced,
        "pass": ok,
        "verdict": ("pass" if ok else "fail") + ("" if enforced else " (informational — smoke)"),
    }
    _write_json(args.out_dir / "gates_v3" / "g0v3.json", payload)
    print(
        f"[g0v3] pooled slice R2={r2_pool:.4f} vs round-3 {r2_r3:.4f} at L{head} "
        f"(|delta|={delta:.4f} tol={tol:.4f}) -> {payload['verdict']}",
        flush=True,
    )
    return 0 if (ok or not enforced) else 3


def run_g1v3_check(out_dir: Path) -> int:
    """G1' v3 — pooled rig-health kill gate (plan v15 §7).

    KILL <=> BOTH the raw AND recalibrated best held-out R2 of the FIRST
    pooled cell (RLVR on-policy) sit below bar_v2 (0.20 x ex_v2 — the same
    exchange-rate bar as G1'v2; grounds: pooled n_train >> d, so the v8 GCV
    pathology cannot arise). A NaN read never kills (fail-safe, mirroring
    G1'v2).
    """
    bars_path = out_dir / "gates_v2" / "v2_bars.json"
    assert bars_path.exists(), (
        f"{bars_path} missing — run the G0' v2 gate first (driver-enforced ordering)"
    )
    bar = float(json.loads(bars_path.read_text())["bar_v2"])
    cell_path = out_dir / "cells_pooled_v3" / "cells_pooled_rlvr_arm_on.json"
    assert cell_path.exists(), (
        f"{cell_path} missing — fit the first pooled cell (RLVR on-policy) before the check"
    )
    recal_best, raw_best = _g1_cell_reads(cell_path)

    def below(v: float) -> bool:
        return bool(np.isfinite(v) and v < bar)

    kill = below(raw_best) and below(recal_best)
    payload = {
        "metadata": _metadata(cm.FIT_SEED, 0),
        "gate": "G1v3",
        "cell": "pooled_rlvr_arm_on",
        "raw_best_r2": raw_best,
        "recal_best_r2": recal_best,
        "bar_v2": bar,
        "verdict": "kill" if kill else "pass",
        "kill": bool(kill),
    }
    _write_json(out_dir / "gates_v3" / "g1v3_gate.json", payload)
    print(
        f"[g1v3] raw={raw_best:.4f} recal={recal_best:.4f} bar_v2={bar:.4f} "
        f"-> {payload['verdict']}",
        flush=True,
    )
    return 3 if kill else 0


def main() -> int:
    args = parse_args()
    if args.g0 or args.g0_probe_only:
        return run_g0(args)
    if args.g0v2:
        return run_g0v2(args)
    if args.g1_check:
        return run_g1_check(args.out_dir)
    if args.g1v2_check:
        return run_g1v2_check(args.out_dir)
    if args.g0v3:
        return run_g0v3(args)
    if args.g1v3_check:
        return run_g1v3_check(args.out_dir)
    if args.v3_pooled:
        return run_pooled_v3(args)
    assert args.cells, (
        "--cells is required (or --g0 / --g0v2 / --g0v3 / --g1-check / --g1v2-check / "
        "--g1v3-check / --v3-pooled)"
    )
    smoke = args.smoke
    v2 = args.v2
    if v2:
        # v2 estimator defaults (plan v13 §4 Phase FIT): inner-group-CV with
        # n_inner=2 — module-global patch style, the documented convention
        # (issue825_fit_cells.py L78 comment). Selection stays the module
        # default "inner-group-cv"; the 23-pt grid threads per sweep below.
        fc.N_INNER_LAMBDA_FOLDS = cm.N_INNER_LAMBDA_FOLDS_V2
    lambda_grid = np.asarray(cm.LAMBDAS_23, dtype=np.float64) if v2 else None
    ts_base = ("turnstore_v2" if v2 else "turnstore") + ("_smoke" if smoke else "")
    preds_base = ("preds_v2" if v2 else "preds") + ("_smoke" if smoke else "")
    ts_dir = args.turnstore_dir or Path(f"data/issue_1336/{ts_base}")
    preds_dir = args.preds_dir or Path(f"data/issue_1336/{preds_base}")
    if args.frozen_layers:
        frozen = tuple(int(x) for x in args.frozen_layers.split(",") if x.strip())
    else:
        frozen = cm.SMOKE_FROZEN_LAYERS if smoke else cm.FROZEN_LAYERS
    null_draws = (
        args.null_draws
        if args.null_draws is not None
        else (cm.SMOKE_NULL_DRAWS if smoke else cm.N_NULL_DRAWS)
    )
    n_boot = (
        args.n_boot if args.n_boot is not None else (cm.SMOKE_N_BOOT if smoke else cm.N_BOOTSTRAP)
    )
    if args.cells == "all":
        cells = cm.CELLS_V2 if v2 else cm.CELLS
    elif args.cells == "smoke":
        cells = (
            cm.cells_v2_for(cm.SMOKE_MODELS, cm.SMOKE_CORPORA_V2)
            if v2
            else cm.cells_for(cm.SMOKE_MODELS, cm.SMOKE_CORPORA)
        )
    else:
        cells = [CELL_BY_ID[c.strip()] for c in args.cells.split(",") if c.strip()]
    # Persisted E1.d exchange-rate calibration (plan v9 route 1) — fail-loud
    # when absent; the smoke dispatch stages a fixture cal at the SAME
    # relative path so both modes exercise this exact load.
    qwen_cal = cm.load_qwen_recal_cal(args.out_dir)
    matched = None
    if args.matched_n:
        matched = int(
            args.matched_n_size
            if args.matched_n_size is not None
            else (cm.MATCHED_N_V2 if v2 else cm.MATCHED_N)
        )
    matched_seed = (
        args.matched_n_seed
        if args.matched_n_seed is not None
        else (cm.MATCHED_N_V2_SEED if v2 else args.seed)
    )
    for cell in cells:
        cell_matched = matched
        if v2:
            # Matched-n v2 companions run only on the four above-size corpora
            # (plan §4 Phase FIT); realized-n gating stays inside run_one_cell.
            if matched is not None and cell["corpus"] not in cm.MATCHED_N_V2_CORPORA:
                cell_matched = None
        elif matched is not None and cm.CORPORA[cell["corpus"]]["n"] <= matched and not smoke:
            cell_matched = None  # gsm8k_test1319 is already at matched size
        run_one_cell(
            cell,
            ts_dir,
            args.out_dir,
            preds_dir,
            frozen_layers=frozen,
            n_folds=args.folds,
            seed=args.seed,
            null_draws=null_draws,
            n_boot=n_boot,
            matched_n=cell_matched,
            # Production stores assert the ladder's 32 layers; a smoke store
            # asserts its own realized count (tiny-model rebinding pattern).
            expected_layers=None if smoke else cm.EXPECTED_LAYERS,
            qwen_cal=qwen_cal,
            x_slot=(args.x_slot or cell.get("x_slot", "context")),
            lambda_grid=lambda_grid,
            v2=v2,
            matched_n_seed=matched_seed,
            # v2 PRODUCTION routes the two extended corpora through the
            # concat loader; smoke fixtures are single complete stems (the
            # concat seam is pinned by its own tests). v1 is byte-unchanged.
            use_concat=(v2 and not smoke),
            wave1_dir=(args.wave1_turnstore_dir or ts_dir),
            gen_root=(args.gen_root or (None if smoke else Path("data/issue_1336/gen"))),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
