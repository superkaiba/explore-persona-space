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
    ap.add_argument("--cells", default=None, help="comma cell ids | all | smoke")
    ap.add_argument("--turnstore-dir", type=Path, default=None)
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
) -> dict:
    """One cell's full fit battery. All new kwargs are default-preserving:
    the v1 call shape (no ``v2``/``lambda_grid``/``x_slot``) is byte-identical
    to the committed behavior. Under ``v2``: the adaptive edge rule wraps the
    sweep on ``lambda_grid``, outputs land under ``cells_v2/``, the manifest
    is ``preds_manifest_v2.json``, and matched-n companions refit the
    persist-layer subset only at the seed-1336 subsample (plan v13 §4)."""
    cell_id = cell["cell_id"]
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


def main() -> int:
    args = parse_args()
    if args.g0 or args.g0_probe_only:
        return run_g0(args)
    if args.g1_check:
        return run_g1_check(args.out_dir)
    assert args.cells, "--cells is required (or --g0 / --g1-check)"
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
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
