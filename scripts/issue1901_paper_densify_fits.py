#!/usr/bin/env python3
"""Task #1901 paper-densify fits (user-chat inline free analysis, 2026-08-20).

Densifies the two C1 paper figures (figures/paper/c1_scaling_train_pool +
c1_layer_profile) from BANKED stores — 0 GPU-h, no new generation:

- Phase ``layer``: dense 28-layer curve at the round-1 ffc fixed split
  (n_train=3,600 / val 400 / test 1,000, seed 42) over the pass_b tensors
  (HF issue779_monitoring/analysis_tensors/pass_b). Per layer: ridge via the
  EXACT banked core ``issue779_fitter_fair_comparison.gram_fit_apply``
  (13-lambda val-selected grid; parity-gated per layer against
  eval_results/issue_779/fitter-fair-comparison/fair_comparison.json),
  identity+bias (``analysis.mapping_baselines.identity_bias_predict``), and
  kNN retrieval for both (``analysis.mapping_baselines.knn_retrieval``,
  pool = the 1,000 held-out test targets).

- Phase ``ladder``: dense train-size ladder at layer 19 (optionally 14/26)
  over the #1491 scale7_refit store (HF issue1491_scale_ladder/scale7_refit;
  train_25k / val_400 / test_1000): n_train in {50,100,250,500,1000,2500}
  x 3 seeded draws + {5k,10k,15k,20k,25k} x 1 file-order-prefix draw
  (ladder_local_id < n — the #2330 train_5k/train_10k convention, asserted
  against eval_results/issue_2330/split_ids.json), ridge via the EXACT
  anchor core ``issue779_ffc_n1m_fits.fit_ridge`` (streaming primal,
  23-lambda logspace(-3,8,23) val_400 sweep), anchored fail-loud against
  fits_scale7_refit.json (n=25k) and the #2330 matched fits (5k/10k).

Estimator-validity note: rungs with n_train < d=3,584 are DELIBERATE
under-determined scaling points; lambda is selected on val_400 at every
cell (never pure GCV at n < d, #1887).

Per-cell JSONs land incrementally (atomic tmp+replace, generating-parameter
resume keys) under eval_results/issue_1901/paper_densify/, then the
assembled layer_curve_n3600.json / scaling_ladder_L<li>.json are written.

Launch (VM, CPU):
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
  uv run python scripts/issue1901_paper_densify_fits.py --phase layer
  ... --phase ladder
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
ROOT = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps BEFORE numpy/torch (shared-VM rule, #847)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as F  # noqa: E402
import issue779_ffc_n50k_fits as N50  # noqa: E402
import issue779_fitter_fair_comparison as FFC  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402

from explore_persona_space.analysis import mapping_baselines as MB  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue1901_paper_densify")

OUT_DIR = ROOT / "eval_results" / "issue_1901" / "paper_densify"
STAGE_DEFAULT = Path("/mnt/eps-data/thomasjiralerspong/issue1901_paper_densify")
LADDER_HF_PREFIX = "issue1491_scale_ladder/scale7_refit"
LADDER_LAYERS_STORED = (14, 19, 26)
LADDER_LAMBDAS = np.logspace(-3, 8, 23)  # the #1491/#2330 reuse-chain grid
RIDGE_BLOCK = 50_000
KS = (1, 5, 10, 50)

# Generating-parameter resume keys (never float hashes — the #1336 rule).
REGIME_LAYER = {
    "split": [5000, 3600, 400, 1000, 42],
    "lambdas": ["logspace", -2, 4, 13],  # FFC.LAMBDAS — banked-parity grid
    "input_variant": "last",
    "target": "v_x",
    "selection": "val-lambda (gram/dual, FFC.gram_fit_apply)",
}
REGIME_LADDER = {
    "store": LADDER_HF_PREFIX,
    "lambdas": ["logspace", -3, 8, 23],
    "selection": "val-lambda (primal streaming, F.fit_ridge)",
    "draw_seed_formula": "np.random.default_rng(19010000 + n*10 + draw)",
    "prefix_rule": "ladder_local_id < n (file-order prefix, #2330 convention)",
}

GRID_SMALL_NS = (50, 100, 250, 500, 1000, 2500)
GRID_SMALL_DRAWS = (0, 1, 2)
GRID_BIG_NS = (5000, 10000, 15000, 20000, 25000)


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def _resume_ok(path: Path, regime: dict) -> dict | None:
    """Return the cell dict when a completed cell with a MATCHING regime exists."""
    if not path.exists():
        return None
    try:
        cell = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return cell if cell.get("regime") == regime else None


def _knn_both(pred: np.ndarray, true: np.ndarray) -> dict:
    return {m: MB.knn_retrieval(pred, true, ks=KS, metric=m) for m in ("euclidean", "cosine")}


def _meta(phase: str) -> dict:
    return {
        "script": "issue1901_paper_densify_fits",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **as_metadata_dict(git_provenance(), phase=phase),
    }


# ---------------------------------------------------------------------------
# Phase LAYER: dense 28-layer curve at n=3,600 (pass_b, banked-parity core)
# ---------------------------------------------------------------------------


def phase_layer(args) -> int:
    cells_dir = OUT_DIR / "layer_curve_cells"
    bundle = FFC.load_pass_b(args.pass_b)
    n_ctx = int(bundle["cx_last"].shape[0])
    assert n_ctx == 5000, n_ctx
    tr, val, te = FFC.fixed_split(n_ctx, 3600, 400, 1000, 42)
    banked = json.loads(
        (ROOT / "eval_results/issue_779/fitter-fair-comparison/fair_comparison.json").read_text()
    )
    banked_pl = banked["inputs"]["last"]["ridge"]["per_layer"]
    dev = torch.device("cpu")
    layers = args.layers if args.layers else list(range(28))

    for k, li in enumerate(layers):
        cell_path = cells_dir / f"L{li:02d}.json"
        if _resume_ok(cell_path, REGIME_LAYER):
            print(f"[layer] unit {k + 1}/{len(layers)} L{li} SKIP (resume)", flush=True)
            continue
        t0 = time.time()
        X = FFC.input_layer(bundle, "last", li)
        Y = FFC.target_vx(bundle, li)
        (pred_val, pred_te), lam = FFC.gram_fit_apply(
            X[tr], Y[tr], [X[val], X[te]], dev, val=(X[val], Y[val])
        )
        ridge_test = PR._pooled_r2(pred_te, Y[te])
        ridge_val = PR._pooled_r2(pred_val, Y[val])
        b = banked_pl.get(str(li))
        parity_abs = abs(ridge_test - b["test_r2"]) if b else None
        pred_idb = MB.identity_bias_predict(X[tr], Y[tr], X[te]).astype(np.float32)
        idb_r2 = PR._pooled_r2(pred_idb, Y[te])
        cell = {
            "layer": int(li),
            "regime": REGIME_LAYER,
            "ridge": {
                "test_r2": ridge_test,
                "val_r2": ridge_val,
                "selected_lambda": float(lam),
                "banked_test_r2": (b or {}).get("test_r2"),
                "banked_selected_lambda": (b or {}).get("selected_lambda"),
                "parity_abs_diff": parity_abs,
            },
            "identity_bias": {"test_r2": idb_r2},
            "knn": {
                "ridge": _knn_both(pred_te, Y[te]),
                "identity_bias": _knn_both(pred_idb, Y[te]),
            },
        }
        _write_json_atomic(cell_path, cell)
        print(
            f"[layer] unit {k + 1}/{len(layers)} L{li} ridge_test_r2={ridge_test:.4f} "
            f"parity_abs={parity_abs if parity_abs is None else f'{parity_abs:.2e}'} "
            f"idb_r2={idb_r2:.4f} acc1={cell['knn']['ridge']['euclidean']['acc_at_k'][1]:.3f} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    if args.layers:  # pilot / partial run — no assembly
        return 0

    # Assemble + parity gate over all 28 layers.
    per_layer: dict[str, dict] = {}
    worst = (None, -1.0)
    for li in range(28):
        cell = _resume_ok(cells_dir / f"L{li:02d}.json", REGIME_LAYER)
        if cell is None:
            raise RuntimeError(f"layer cell L{li} missing/regime-mismatched at assembly")
        per_layer[str(li)] = {k: v for k, v in cell.items() if k != "regime"}
        pa = cell["ridge"]["parity_abs_diff"]
        if pa is not None and pa > worst[1]:
            worst = (li, pa)
    gate_pass = worst[1] <= args.parity_tol
    out = {
        "split": {"n_contexts": 5000, "n_train": 3600, "n_val": 400, "n_test": 1000, "seed": 42},
        "input_variant": "last (cx_last — the paper's v_C)",
        "target": "v_x (mean-response answer summary, same layer)",
        "lambdas": [float(x) for x in FFC.LAMBDAS],
        "selection": REGIME_LAYER["selection"],
        "knn_pool": "held-out test targets (pool == true, n=1000)",
        "per_layer": per_layer,
        "parity_gate": {
            "reference": "eval_results/issue_779/fitter-fair-comparison/fair_comparison.json "
            "inputs.last.ridge.per_layer",
            "max_abs_diff": worst[1],
            "worst_layer": worst[0],
            "tol": args.parity_tol,
            "pass": bool(gate_pass),
        },
        "metadata": _meta("layer-curve"),
    }
    _write_json_atomic(OUT_DIR / "layer_curve_n3600.json", out)
    print(
        f"[layer] assembled 28 layers -> {OUT_DIR / 'layer_curve_n3600.json'} "
        f"(parity max_abs={worst[1]:.2e} @ L{worst[0]}, pass={gate_pass})",
        flush=True,
    )
    if not gate_pass:
        raise RuntimeError(
            f"banked-parity gate FAIL: max |refit - banked| = {worst[1]:.3e} @ layer "
            f"{worst[0]} > tol {args.parity_tol} — investigate before using the curve"
        )
    return 0


# ---------------------------------------------------------------------------
# Phase LADDER: dense train-size ladder over the scale7_refit store
# ---------------------------------------------------------------------------


def _stage_ladder_split(split: str, layers: tuple[int, ...], staged_dir: Path, cache_dir: Path):
    """Stream one ladder split's capture chunks from HF, slicing ALL requested
    layers in ONE download pass (mirrors issue779_ffc_n1m_fits._stream_hf_chunks'
    download->load->unlink shape; multi-layer slice avoids re-downloading the
    2.9 GB store once per layer). Persists a local npz so resumes never re-pull.

    Returns {layer: (cx (n,H) fp32, vx (n,H) fp32)}, ci (list[int])."""
    staged = staged_dir / f"{split}.npz"
    if staged.exists():
        z = np.load(staged)
        ci = [int(x) for x in z["ci"]]
        return {li: (z[f"cx_L{li}"], z[f"vx_L{li}"]) for li in layers}, ci

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    prefix = f"{LADDER_HF_PREFIX}/{split}/final_token_capture"
    names = hub.retry_transient(
        lambda: sorted(
            f.path.rsplit("/", 1)[-1]
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient right here
            for f in HfApi().list_repo_tree(
                C.HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True
            )
            if getattr(f, "size", None) is not None and f.path.endswith(".pt")
        ),
        what=f"ladder chunk listing ({prefix})",
    )
    if not names:
        raise FileNotFoundError(f"no capture chunks under HF {prefix}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cx_parts: dict[int, list[np.ndarray]] = {li: [] for li in layers}
    vx_parts: dict[int, list[np.ndarray]] = {li: [] for li in layers}
    ci: list[int] = []
    for i, name in enumerate(names):
        got = Path(F._download_chunk_with_retry(C.HF_DATA_REPO, f"{prefix}/{name}", cache_dir))
        b = FFC._mmap_load(got)
        for li in layers:
            cx_parts[li].append(N50._slice_layer(b, "cx_last", li))
            vx_parts[li].append(N50._slice_layer(b, "v_x", li))
        ci.extend(int(x) for x in b["ci"])
        del b
        got.unlink()
        if (i + 1) % 25 == 0:
            print(f"[ladder-stage] {split}: {i + 1}/{len(names)} chunks", flush=True)
    out = {li: (np.concatenate(cx_parts[li]), np.concatenate(vx_parts[li])) for li in layers}
    staged_dir.mkdir(parents=True, exist_ok=True)
    tmp = staged.with_name(f"{split}.tmp.npz")  # suffix stays .npz (np.savez appends it otherwise)
    np.savez(
        tmp,
        ci=np.array(ci, dtype=np.int64),
        **{f"cx_L{li}": out[li][0] for li in layers},
        **{f"vx_L{li}": out[li][1] for li in layers},
    )
    tmp.replace(staged)
    print(f"[ladder-stage] {split}: staged {len(ci)} rows -> {staged}", flush=True)
    return out, ci


def _ladder_cells(n_train_rows: int) -> list[tuple[int, object]]:
    cells: list[tuple[int, object]] = []
    for n in GRID_SMALL_NS:
        cells.extend((n, d) for d in GRID_SMALL_DRAWS)
    cells.extend((n, "prefix") for n in GRID_BIG_NS if n <= n_train_rows)
    return cells


def phase_ladder(args) -> int:
    cells_dir = OUT_DIR / "ladder_cells"
    staged_dir = args.staged_dir
    cache_dir = args.cache_dir
    layers = tuple(args.ladder_layers)
    assert all(li in LADDER_LAYERS_STORED for li in layers), layers

    expected = {"train_25k": 25000, "val_400": 400, "test_1000": 1000}
    data: dict[str, dict] = {}
    cis: dict[str, list[int]] = {}
    for split in ("train_25k", "val_400", "test_1000"):
        arrs, ci = _stage_ladder_split(split, LADDER_LAYERS_STORED, staged_dir, cache_dir)
        if len(ci) != expected[split]:
            raise RuntimeError(
                f"split count shortfall: {split} realized={len(ci)} expected={expected[split]}"
            )
        data[split], cis[split] = arrs, ci

    # File-order-prefix draws: ladder_local_id < n (#2330 convention). Assert the
    # 5k/10k prefixes equal the pinned #2330 split id lists.
    tr_ids = cis["train_25k"]
    id2row = {cid: i for i, cid in enumerate(tr_ids)}
    assert len(id2row) == 25000, "duplicate context ids in train_25k stream"
    split_ids = json.loads((ROOT / "eval_results/issue_2330/split_ids.json").read_text())
    for key, n in (("train_5k", 5000), ("train_10k", 10000)):
        pinned = set(int(x) for x in split_ids["splits"][key])
        mine = set(range(n))
        if pinned != mine:
            raise RuntimeError(
                f"prefix convention mismatch vs #2330 {key}: {len(pinned ^ mine)} ids differ"
            )

    val_n, te_n = expected["val_400"], expected["test_1000"]
    for li in layers:
        X = np.concatenate(
            [data["train_25k"][li][0], data["val_400"][li][0], data["test_1000"][li][0]]
        )
        Y = np.concatenate(
            [data["train_25k"][li][1], data["val_400"][li][1], data["test_1000"][li][1]]
        )
        n_tr_rows = len(tr_ids)
        val_idx = np.arange(n_tr_rows, n_tr_rows + val_n, dtype=np.int64)
        te_idx = np.arange(n_tr_rows + val_n, n_tr_rows + val_n + te_n, dtype=np.int64)
        Yte = Y[te_idx]
        cells = _ladder_cells(n_tr_rows)
        if args.max_cells:
            cells = cells[: args.max_cells]
        dev = torch.device("cpu")
        for k, (n, draw) in enumerate(cells):
            tag = f"L{li}_n{n}_{'prefix' if draw == 'prefix' else f'd{draw}'}"
            cell_path = cells_dir / f"{tag}.json"
            if _resume_ok(cell_path, REGIME_LADDER):
                print(f"[ladder] unit {k + 1}/{len(cells)} {tag} SKIP (resume)", flush=True)
                continue
            t0 = time.time()
            if draw == "prefix":
                rows = np.array([id2row[i] for i in range(n)], dtype=np.int64)
            else:
                rng = np.random.default_rng(19010000 + n * 10 + int(draw))
                rows = rng.choice(n_tr_rows, size=n, replace=False).astype(np.int64)
            pred_te, meta = F.fit_ridge(
                X, Y, rows, val_idx, te_idx, LADDER_LAMBDAS, dev, RIDGE_BLOCK
            )
            r2 = PR._pooled_r2(pred_te, Yte)
            pred_idb = MB.identity_bias_predict(X[rows], Y[rows], X[te_idx]).astype(np.float32)
            idb_r2 = PR._pooled_r2(pred_idb, Yte)
            cell = {
                "layer": int(li),
                "n_train": int(n),
                "draw": draw if draw == "prefix" else int(draw),
                "n_vs_d": {
                    "n_train": int(n),
                    "d": int(X.shape[1]),
                    "underdetermined": n < X.shape[1],
                },
                "regime": REGIME_LADDER,
                "ridge": {"test_r2": r2, "meta": meta},
                "identity_bias": {"test_r2": idb_r2},
                "knn": {
                    "ridge": _knn_both(pred_te, Yte),
                    "identity_bias": _knn_both(pred_idb, Yte),
                },
            }
            _write_json_atomic(cell_path, cell)
            print(
                f"[ladder] unit {k + 1}/{len(cells)} {tag} ridge_r2={r2:.4f} "
                f"lam={meta['selected_lambda']:.3g} idb_r2={idb_r2:.4f} "
                f"acc1={cell['knn']['ridge']['euclidean']['acc_at_k'][1]:.3f} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
        del X, Y

    if args.max_cells:
        return 0

    # Assemble per layer + anchor gates (L19 only — the anchors live there).
    for li in layers:
        rows_out = []
        for n, draw in _ladder_cells(25000):
            tag = f"L{li}_n{n}_{'prefix' if draw == 'prefix' else f'd{draw}'}"
            cell = _resume_ok(cells_dir / f"{tag}.json", REGIME_LADDER)
            if cell is None:
                raise RuntimeError(f"ladder cell {tag} missing/regime-mismatched at assembly")
            rows_out.append({k: v for k, v in cell.items() if k != "regime"})
        out = {
            "store": LADDER_HF_PREFIX,
            "layer": int(li),
            "splits": expected,
            "lambdas": [float(x) for x in LADDER_LAMBDAS],
            "selection": REGIME_LADDER["selection"],
            "draw_convention": {
                "small_ns": list(GRID_SMALL_NS),
                "draws_small": list(GRID_SMALL_DRAWS),
                "seed_formula": REGIME_LADDER["draw_seed_formula"],
                "big_ns": list(GRID_BIG_NS),
                "big_draw": REGIME_LADDER["prefix_rule"],
            },
            "knn_pool": "held-out test targets (pool == true, n=1000)",
            "cells": rows_out,
            "metadata": _meta("scaling-ladder"),
        }
        anchors = {}
        if li == 19:
            f7 = json.loads(
                (ROOT / "eval_results/issue_1491/scale_ladder/fits_scale7_refit.json").read_text()
            )
            m5 = json.loads(
                (ROOT / "eval_results/issue_2330/matched_fits_q25_n5k.json").read_text()
            )
            m10 = json.loads(
                (ROOT / "eval_results/issue_2330/matched_fits_q25_n10k.json").read_text()
            )
            by_key = {(c["n_train"], c["draw"]): c for c in rows_out}
            checks = [
                (
                    "n25k_vs_fits_scale7_refit",
                    by_key[(25000, "prefix")],
                    f7["predictors"]["ridge"]["test_r2"],
                    f7["knn_retrieval"]["ridge"]["euclidean"]["acc_at_k"]["1"],
                ),
                (
                    "n5k_vs_2330",
                    by_key[(5000, "prefix")],
                    m5["per_layer"]["19"]["ridge"]["test_r2"],
                    m5["per_layer"]["19"]["knn_retrieval"]["ridge"]["euclidean"]["acc_at_k"]["1"],
                ),
                (
                    "n10k_vs_2330",
                    by_key[(10000, "prefix")],
                    m10["per_layer"]["19"]["ridge"]["test_r2"],
                    m10["per_layer"]["19"]["knn_retrieval"]["ridge"]["euclidean"]["acc_at_k"]["1"],
                ),
            ]
            failures = []
            for name, cell, ref_r2, ref_acc1 in checks:
                dr2 = abs(cell["ridge"]["test_r2"] - ref_r2)
                # JSON round-trip stringifies the acc_at_k int keys — always read "1".
                da1 = abs(cell["knn"]["ridge"]["euclidean"]["acc_at_k"]["1"] - ref_acc1)
                ok = dr2 <= args.anchor_tol_r2 and da1 <= args.anchor_tol_acc1
                anchors[name] = {
                    "ref_r2": ref_r2,
                    "ours_r2": cell["ridge"]["test_r2"],
                    "abs_diff_r2": dr2,
                    "ref_acc1": ref_acc1,
                    "ours_acc1": cell["knn"]["ridge"]["euclidean"]["acc_at_k"]["1"],
                    "abs_diff_acc1": da1,
                    "pass": bool(ok),
                }
                if not ok:
                    failures.append(name)
            out["anchor_gates"] = {
                "tol_r2": args.anchor_tol_r2,
                "tol_acc1": args.anchor_tol_acc1,
                "checks": anchors,
                "pass": not failures,
            }
        _write_json_atomic(OUT_DIR / f"scaling_ladder_L{li}.json", out)
        print(f"[ladder] assembled -> {OUT_DIR / f'scaling_ladder_L{li}.json'}", flush=True)
        if li == 19 and not out["anchor_gates"]["pass"]:
            raise RuntimeError(f"anchor gates FAILED: {failures} — investigate before use")
    return 0


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True, choices=["layer", "ladder"])
    ap.add_argument(
        "--pass-b",
        type=Path,
        default=STAGE_DEFAULT
        / "pass_b/issue779_monitoring/analysis_tensors/pass_b"
        / "train_context_vectors.pt",
        help="staged pass_b bundle path (layer phase)",
    )
    ap.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="layer phase: restrict to these layers (pilot); default all 28 + assembly",
    )
    ap.add_argument(
        "--ladder-layers",
        type=int,
        nargs="*",
        default=[19],
        help="ladder phase: which stored layers (of 14/19/26) to fit",
    )
    ap.add_argument("--staged-dir", type=Path, default=STAGE_DEFAULT / "staged")
    ap.add_argument("--cache-dir", type=Path, default=STAGE_DEFAULT / "ladder_cache")
    ap.add_argument("--max-cells", type=int, default=0, help="ladder pilot: first K cells only")
    ap.add_argument("--parity-tol", type=float, default=1e-3)
    ap.add_argument("--anchor-tol-r2", type=float, default=1e-3)
    ap.add_argument("--anchor-tol-acc1", type=float, default=0.005)
    ap.add_argument("--import-check", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("import-check OK")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return phase_layer(args) if args.phase == "layer" else phase_ladder(args)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
