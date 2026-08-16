#!/usr/bin/env python3
"""Task #2330 P3: matched context->answer ridge fits, 7B vs 9B (plan §4 P3).

Thin orchestration importing the #1491/#779 cores — no re-implemented fit math:

- ``issue1491_ladder_fits._assemble_scale_layer``  (7B banked-store assembly,
  #2130 count pins at the banked grain 25000/400/1000/999)
- ``issue1491_ladder_fits._stream_ladder_split``   (per-split HF streaming,
  delegating to ``issue779_ffc_n1m_fits._stream_hf_chunks``)
- ``issue1491_ladder_fits._fit_floors``            (shuffled-pairing null refit,
  train-mean, identity-copy, scaled-identity, identity+learned-bias)
- ``issue1491_ladder_fits._knn_reads``             (euclidean+cosine retrieval,
  chance k/n_pool)
- ``issue1491_ladder_fits._reliability_ceiling``   (two-draw seeds 43/44)
- ``issue779_ffc_n1m_fits.fit_ridge``              (fp64 primal, val-selected λ
  over ``np.logspace(-3, 8, 23)``)

Per cell (q25_n5k / q25_n10k / q35_n5k / q35_n10k): stream the model's store
from HF, pin per-split realized counts against the POST-DROP
``eval_results/issue_2330/split_ids.json`` (the single source of truth — P1's
length scan wrote its over-budget drops there; realized wc_test_1k = 998),
subset train rows to the split_ids prefix lists with a matched-ID assert
(row-aligned identity across the 7B and 9B cells), then fit ridge + the five
floors + kNN + the two-draw ceiling at each of the model's 3 layers.

PORT-PARITY ANCHOR GATE (runs FIRST): reproduce the 7B n=25,000 layer-19 fit
through this path and assert |R² − 0.7250873| ≤ 0.01 — a miss is a hard halt
(``failure_class: code`` — the port is broken, not the data). The realized
deviation, selected λ, and val/test R² are persisted in every cell JSON's
``port_parity_anchor`` block; a passing deviation ≥ 1e-3 is flagged
``investigate_before_narrate`` for the analyzer.

λ-grid-edge disposition (plan §11): a fit selecting an edge λ extends the grid
one decade on that side and refits (recorded in fit meta; bounded, fail-loud
past MAX_GRID_EXTENSIONS).

Outputs:
- ``eval_results/issue_2330/matched_fits_<cell>.json`` (4 files; incremental
  atomic rewrite after every (cell, layer) unit)
- ``data/issue_2330/preds/<cell>_test_preds_ridge.npz`` — per-context test AND
  WildChat-fold preds+targets at all 3 layers (keys: ``ci_te``,
  ``pred_te_L{l}``, ``target_te_L{l}``, ``ci_wc``, ``pred_wc_L{l}``,
  ``target_wc_L{l}``, ``layers``, ``primary_layer``) — mirrored to
  ``issue2330_matched/analysis_tensors/preds/`` on the HF data repo.

Modes:
- default: the production battery (pod, repo venv, ``--device cuda``).
- ``--smoke-chunk-dir <dir>``: plan §4 P1 step 4 fits-shape smoke on the LOCAL
  500-row smoke chunk(s) written by the P1 smoke shard (count pins opted out —
  the documented ``expected_split_n=None`` downgrade; labeled smoke).
- ``--selftest``: CPU failure-path exercises (count-pin mismatch, matched-ID
  miss/shuffle, anchor-gate halt, λ-grid extension) on synthetic tensors —
  plan §4 P3 implementer duty (statistics-reconciler rec 3).
- ``--synthetic-e2e``: shape-faithful synthetic CPU end-to-end (d=3584/4096,
  tiny n — a DELIBERATELY under-determined n<d smoke shape, schema check only)
  through the production cell-fit + writer bodies, faking only the HF boundary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

# Heavy imports AFTER load_dotenv() — shared-VM thread caps (#847) bind at
# torch import (pinned by tests/test_shared_vm_thread_caps.py).
import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_ffc_n1m_fits as F  # noqa: E402
import issue779_ffc_n50k_fits as N50  # noqa: E402
import issue779_fitter_fair_comparison as FFC  # noqa: E402
import issue1491_ladder_fits as LF  # noqa: E402

logger = logging.getLogger("issue2330_matched_fits")

REPO_ROOT = _SCRIPTS.parent

# ---------------------------------------------------------------------------
# Constants (plan §4 P3 / §10 / §11)
# ---------------------------------------------------------------------------

HF_PREFIX_7B = "issue1491_scale_ladder/scale7_refit"
HF_PREFIX_9B = "issue2330_matched/qwen35_9b"
# Plan §10 revision pin for the banked 7B store. The reuse streaming core
# (F._stream_hf_chunks) consumes the repo at its default revision; the pin is
# enforced CONTENT-wise by the anchor gate (exact-R² reproduction) + the count
# pins, and recorded here for the repro card.
STORE_REVISION_PIN_7B = "815ff6d976c686af8672b27cfdfb1ce6b419c02c"

# Committed anchor (eval_results/issue_1491/scale_ladder/fits_scale7_refit.json
# predictors.ridge.test_r2) — plan §4 P3 port-parity anchor gate.
ANCHOR_EXPECTED_R2 = 0.7250873220237553
ANCHOR_TOL = 0.01
ANCHOR_INVESTIGATE_DEVIATION = 1e-3  # passing-but-large ⇒ investigate-before-narrate

PREDS_HF_PREFIX = "issue2330_matched/analysis_tensors/preds"

# λ-grid-edge disposition: extend ONE DECADE per pass (grid spacing is 0.5
# decades ⇒ 2 extra points), refit, record; fail loud past the cap.
GRID_DECADE_POINTS = 2
MAX_GRID_EXTENSIONS = 4

MODELS: dict[str, dict] = {
    "qwen25_7b": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "hf_prefix": HF_PREFIX_7B,
        "store_revision_pin": STORE_REVISION_PIN_7B,
        "layers": [19, 14, 26],  # primary FIRST (anchor gate runs at 19 before cell fits)
        "primary_layer": 19,
        "h_dim": 3584,
        "store_train_split": "train_25k",  # banked 25k store; cells subset by id prefix
        "store_expected_n": dict(LF.EXPECTED_SPLIT_N),  # banked grain incl. wc 999
        "cells": {"q25_n5k": "train_5k", "q25_n10k": "train_10k"},
        "anchor": True,
    },
    "qwen35_9b": {
        "model": "Qwen/Qwen3.5-9B",
        "hf_prefix": HF_PREFIX_9B,
        "store_revision_pin": None,  # produced by this issue's P2; consumed at head
        "layers": [22, 16, 30],  # primary FIRST (depth-fraction matched to 7B {14,19,26})
        "primary_layer": 22,
        "h_dim": 4096,
        "store_train_split": "train_10k",  # P2 generates exactly the split_ids rows
        "store_expected_n": None,  # derived from split_ids at run time
        "cells": {"q35_n5k": "train_5k", "q35_n10k": "train_10k"},
        "anchor": False,
    },
}
MODEL_ORDER = ["qwen25_7b", "qwen35_9b"]  # 7B first: the anchor gate halts cheap


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _write_json_atomic(path: Path, obj: dict) -> None:
    """Atomic JSON write (tmp + os.replace) — checkpoint-per-unit carrier."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def _repro_meta() -> dict:
    """Reproducibility metadata block (git commit + dirty flag + env versions)."""
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    meta = dict(as_metadata_dict(git_provenance()))
    meta.update(
        {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "script": "scripts/issue2330_matched_fits.py",
        }
    )
    return meta


def _phase(name: str) -> None:
    print(f"[phase={name}]", flush=True)


def load_split_ids(path: Path) -> dict:
    """Load + schema-check the P0 split_ids.json (single source of truth)."""
    if not path.is_file():
        raise RuntimeError(
            f"split_ids.json missing at {path} — run scripts/issue2330_split_ids.py (P0) "
            "first and check out the issue branch carrying it"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("splits", "counts", "sha256", "dropped_overlength"):
        if key not in payload:
            raise RuntimeError(
                f"split_ids.json schema drift: missing {key!r} (has {sorted(payload)})"
            )
    for split, ids in payload["splits"].items():
        if int(payload["counts"][split]) != len(ids):
            raise RuntimeError(
                f"split_ids.json internal count mismatch: counts[{split}]="
                f"{payload['counts'][split]} vs len(splits[{split}])={len(ids)}"
            )
    return payload


# ---------------------------------------------------------------------------
# Count pins + matched-ID subsetting (fail-loud; exercised by --selftest)
# ---------------------------------------------------------------------------


def _pin_counts(
    realized: dict[str, int], expected: dict[str, int] | None, store_label: str
) -> None:
    """#2130-style fail-loud count pin against the POST-DROP split_ids grain.

    ``expected=None`` is the explicit smoke opt-out (plan §4 P1 step 4 /
    smoke-enumeration item (d)); the pin re-arms at P3 production assemble.
    """
    if expected is None:
        logger.warning("[matched-fits] count pins OPTED OUT (expected=None) for %s", store_label)
        return
    for split, exp in expected.items():
        got = realized.get(split)
        if got != int(exp):
            raise RuntimeError(
                f"split count pin miss (#2130 fail-loud): store={store_label} split={split!r} "
                f"realized={got} expected={exp} (post-drop split_ids grain) — a killed/partial "
                "capture shard or a stale store, never a smaller design; refusing to fit."
            )


def _subset_by_ids(
    cx: np.ndarray, vx: np.ndarray, ci: list[int], ids: list[int], what: str
) -> tuple[np.ndarray, np.ndarray]:
    """Gather rows for ``ids`` IN ids ORDER (row-aligned). Raises on any miss.

    A missing id means manifest/split_ids/store drift — the matched-ID
    property is broken, so this fails loud rather than fitting on a
    silently-mismatched row set.
    """
    by_ci = {int(c): i for i, c in enumerate(ci)}
    missing = [i for i in ids if int(i) not in by_ci]
    if missing:
        raise RuntimeError(
            f"matched-ID assert failed for {what}: {len(missing)}/{len(ids)} split_ids ids "
            f"absent from the streamed store (first: {missing[:10]}) — store/split_ids drift."
        )
    rows = np.array([by_ci[int(i)] for i in ids], dtype=np.int64)
    return cx[rows], vx[rows]


def _assert_matched_ids(realized_ci: list[int], expected_ids: list[int], what: str) -> None:
    """Row-aligned identity: realized context-id LIST == split_ids LIST exactly."""
    if [int(c) for c in realized_ci] != [int(i) for i in expected_ids]:
        set_eq = {int(c) for c in realized_ci} == {int(i) for i in expected_ids}
        raise RuntimeError(
            f"matched-ID assert failed for {what}: realized context-id list != split_ids list "
            f"(n_realized={len(realized_ci)}, n_expected={len(expected_ids)}, "
            f"set_equal={set_eq}) — {'row ORDER differs' if set_eq else 'id SETS differ'}."
        )


# ---------------------------------------------------------------------------
# Ridge with λ-grid-edge one-decade extension (plan §11 disposition)
# ---------------------------------------------------------------------------


def _extended_lambdas(grid: np.ndarray, side: str) -> np.ndarray:
    """Extend the log-spaced λ grid ONE DECADE on ``side`` ('low'|'high')."""
    step = np.log10(grid[1]) - np.log10(grid[0])
    n_new = GRID_DECADE_POINTS
    if side == "low":
        lo = np.log10(grid[0])
        new = 10.0 ** (lo - step * np.arange(n_new, 0, -1))
        return np.concatenate([new, grid])
    if side == "high":
        hi = np.log10(grid[-1])
        new = 10.0 ** (hi + step * np.arange(1, n_new + 1))
        return np.concatenate([grid, new])
    raise ValueError(f"unknown grid edge side {side!r}")


def fit_ridge_edge_extended(
    X: np.ndarray,
    Y: np.ndarray,
    tr: np.ndarray,
    val: np.ndarray,
    ev: np.ndarray,
    dev,
    block: int,
) -> tuple[np.ndarray, dict]:
    """``F.fit_ridge`` + the λ-grid-edge disposition: extend one decade on the
    selected edge side and refit, recording every extension in the meta.
    Fail-loud past MAX_GRID_EXTENSIONS (a λ still at the edge after 4 decades
    of extension is a data/shape defect, not a grid-width problem)."""
    grid = LF.LAMBDAS
    extensions: list[dict] = []
    for _ in range(MAX_GRID_EXTENSIONS + 1):
        pred, meta = F.fit_ridge(X, Y, tr, val, ev, grid, dev, block)
        edge = meta.get("lambda_grid_edge")
        meta = {**meta, "lambda_grid": [float(x) for x in grid], "grid_extensions": extensions}
        if edge is None:
            return pred, meta
        extensions.append(
            {
                "side": edge,
                "selected_lambda_at_edge": float(meta["selected_lambda"]),
                "grid_len_before": int(len(grid)),
            }
        )
        logger.warning(
            "[matched-fits] λ at grid edge (%s, λ=%.3g) — extending one decade + refitting",
            edge,
            meta["selected_lambda"],
        )
        grid = _extended_lambdas(grid, edge)
    raise RuntimeError(
        f"λ still at the grid edge after {MAX_GRID_EXTENSIONS} one-decade extensions "
        f"(grid now spans [{grid[0]:.3g}, {grid[-1]:.3g}]) — refusing to report an "
        "edge-selected ridge fit (plan §11 disposition exhausted)."
    )


# ---------------------------------------------------------------------------
# Port-parity anchor gate (plan §4 P3; runs FIRST)
# ---------------------------------------------------------------------------


def run_anchor_gate(
    X: np.ndarray,
    Y: np.ndarray,
    tr: np.ndarray,
    val: np.ndarray,
    te: np.ndarray,
    dev,
    expected_r2: float = ANCHOR_EXPECTED_R2,
    tol: float = ANCHOR_TOL,
) -> dict:
    """Reproduce the 7B n=25,000 layer-19 ridge fit through THIS path and hard-halt
    on |R² − anchor| > tol (the port is broken, not the data). Returns the anchor
    record persisted in every cell JSON's fit meta (statistics-reconciler recs 1–2)."""
    pred, meta = F.fit_ridge(X, Y, tr, val, te, LF.LAMBDAS, dev, LF.RIDGE_BLOCK)
    r2 = LF._pooled_r2(pred, Y[te])
    deviation = abs(r2 - expected_r2)
    record = {
        "expected_r2": float(expected_r2),
        "realized_r2": float(r2),
        "abs_deviation": float(deviation),
        "tol": float(tol),
        "selected_lambda": float(meta["selected_lambda"]),
        "val_r2_at_selected": float(meta["val_r2_at_selected"]),
        "lambda_grid_edge": meta.get("lambda_grid_edge"),
        "n_train": int(meta["n_train"]),
        # Expected deterministic-reproduction band is ~1e-6–1e-4; a PASSING
        # deviation ≥ 1e-3 is flagged for the analyzer (investigate before
        # narrating any cross-model contrast).
        "investigate_before_narrate": bool(deviation >= ANCHOR_INVESTIGATE_DEVIATION),
    }
    if deviation > tol:
        raise RuntimeError(
            f"PORT-PARITY ANCHOR GATE MISS (hard halt, failure_class: code): reproduced 7B "
            f"n=25k layer-19 ridge R²={r2:.7f} vs committed {expected_r2:.7f} "
            f"(|Δ|={deviation:.3g} > tol {tol}) — the fits port is broken, not the data. "
            f"Anchor record: {json.dumps(record)}"
        )
    logger.info(
        "[matched-fits] anchor gate PASS: R²=%.7f (|Δ|=%.3g, λ=%.3g)",
        r2,
        deviation,
        record["selected_lambda"],
    )
    return record


# ---------------------------------------------------------------------------
# Store assembly (per model × layer)
# ---------------------------------------------------------------------------

STORE_SPLITS = ["val_400", "test_1000", "wc_test_1k"]  # + the model's train split


def assemble_store(
    hf_prefix: str,
    train_split: str,
    layer: int,
    cache_dir: Path,
    expected_n: dict[str, int] | None,
) -> dict[str, dict]:
    """Stream one model's four splits at one layer from HF via the reuse cores.

    Returns {split: {"cx": (n,H) fp32, "vx": (n,H) fp32, "ci": list[int]}} with
    the #2130-style count pins applied at the STORE grain (``expected_n`` maps
    store split names → expected realized counts; None = smoke opt-out).
    """
    store: dict[str, dict] = {}
    realized: dict[str, int] = {}
    for split in [train_split, *STORE_SPLITS]:
        cx, vx, ci = LF._stream_ladder_split(hf_prefix, split, layer, cache_dir)
        store[split] = {"cx": cx, "vx": vx, "ci": list(ci)}
        realized[split] = int(cx.shape[0])
        logger.info("[matched-fits]   %s/%s L%d: n=%d", hf_prefix, split, layer, realized[split])
    _pin_counts(realized, expected_n, f"{hf_prefix} (layer {layer})")
    return store


def assemble_cell_arrays(
    store: dict[str, dict],
    train_split: str,
    split_ids: dict,
    train_key: str,
) -> dict:
    """Build the (X, Y, tr, val, te, wc_te) bundle for ONE cell from a streamed
    store, subsetting every split to the split_ids lists (matched-ID asserts).
    """
    ids = split_ids["splits"]
    tr_ids = ids[train_key]
    cx_tr, vx_tr = _subset_by_ids(
        store[train_split]["cx"],
        store[train_split]["vx"],
        store[train_split]["ci"],
        tr_ids,
        train_key,
    )
    parts_x, parts_y = [cx_tr], [vx_tr]
    idx: dict[str, np.ndarray] = {}
    n = len(tr_ids)
    idx["tr"] = np.arange(0, n, dtype=np.int64)
    realized_cis: dict[str, list[int]] = {train_key: [int(i) for i in tr_ids]}
    for split, key in (("val_400", "val"), ("test_1000", "te"), ("wc_test_1k", "wc_te")):
        want = ids[split]
        cx_s, vx_s = _subset_by_ids(
            store[split]["cx"], store[split]["vx"], store[split]["ci"], want, split
        )
        parts_x.append(cx_s)
        parts_y.append(vx_s)
        idx[key] = np.arange(n, n + len(want), dtype=np.int64)
        realized_cis[split] = [int(i) for i in want]
        n += len(want)
    X = np.concatenate(parts_x, axis=0)
    Y = np.concatenate(parts_y, axis=0)
    return {
        "X": X,
        "Y": Y,
        "tr": idx["tr"],
        "val": idx["val"],
        "te": idx["te"],
        "wc_te": idx["wc_te"],
        "cis": realized_cis,
        "n_realized": {k: int(len(v)) for k, v in realized_cis.items()},
    }


# ---------------------------------------------------------------------------
# One (cell, layer) fit unit
# ---------------------------------------------------------------------------


def fit_cell_layer(bundle: dict, dev, ceiling: dict, h_dim: int) -> dict:
    """Ridge (edge-extended) + floors + kNN + WildChat transfer fold for one
    (cell, layer). Returns the per-layer result dict + the preds arrays."""
    X, Y = bundle["X"], bundle["Y"]
    tr, val, te, wc_te = bundle["tr"], bundle["val"], bundle["te"], bundle["wc_te"]
    assert Y.shape[1] == h_dim, (Y.shape, h_dim)

    pred_ridge, meta_ridge = fit_ridge_edge_extended(X, Y, tr, val, te, dev, LF.RIDGE_BLOCK)
    ridge_r2 = LF._pooled_r2(pred_ridge, Y[te])

    floors = LF._fit_floors(X, Y, tr, val, te, dev, LF.RIDGE_BLOCK)
    # λ-edge disposition SCOPE: the primary + WildChat ridge fits ONLY — the
    # fits whose selected λ carries meaning. The shuffled-pairing NULL lives at
    # the HIGH edge BY DESIGN (its val-optimum is shrink-to-mean, approached
    # monotonically as λ→∞, so extension can never terminate) — the committed
    # parent run has floors.shuffled_pairing.meta.lambda_grid_edge == "high"
    # (fits_scale7_refit.json, λ=1e8) and keeps the fixed grid; parity kept.

    knn_arms = {"ridge": pred_ridge}
    for name in ("identity_bias", "identity_copy", "scaled_identity", "train_mean"):
        knn_arms[name] = floors[name]["pred_te"]
    knn = LF._knn_reads(knn_arms, Y[te])

    # WildChat corpus-transfer fold — ridge ONLY (LINEAR-by-default standing
    # rule: the parent's MLP leg is deliberately NOT carried). Plan §9: the
    # fold's realized count + fold label ride the output schema.
    pred_wc, meta_wc = fit_ridge_edge_extended(X, Y, tr, val, wc_te, dev, LF.RIDGE_BLOCK)
    wc_transfer = {
        "available": True,
        "fold_label": "wildchat_corpus_transfer",
        "n_wc_test": int(len(wc_te)),
        "ridge_test_r2": LF._pooled_r2(pred_wc, Y[wc_te]),
        "ridge_meta": meta_wc,
    }

    n_train, d = int(len(tr)), int(Y.shape[1])
    result = {
        "ridge": {"test_r2": float(ridge_r2), "meta": meta_ridge},
        "floors": {
            name: {"test_r2": float(f["test_r2"]), "meta": f["meta"]} for name, f in floors.items()
        },
        "knn_retrieval": knn,
        "ceiling_two_draw": ceiling,
        "wc_transfer": wc_transfer,
        "n_vs_d": {
            "n_train": n_train,
            "d": d,
            "n_train_over_d": float(n_train / d),
            "underdetermined": bool(n_train < d),
        },
    }
    preds = {
        "pred_te": pred_ridge.astype(np.float32),
        "target_te": Y[te].astype(np.float32),
        "pred_wc": pred_wc.astype(np.float32),
        "target_wc": Y[wc_te].astype(np.float32),
    }
    return {"result": result, "preds": preds}


def save_cell_preds(
    preds_dir: Path,
    cell: str,
    layers: list[int],
    primary_layer: int,
    ci_te: list[int],
    ci_wc: list[int],
    per_layer_preds: dict[int, dict],
) -> Path:
    """Write one npz per cell: test + WildChat preds/targets at all 3 layers."""
    preds_dir.mkdir(parents=True, exist_ok=True)
    path = preds_dir / f"{cell}_test_preds_ridge.npz"
    arrays: dict[str, np.ndarray] = {
        "ci_te": np.array(ci_te, dtype=np.int64),
        "ci_wc": np.array(ci_wc, dtype=np.int64),
        "layers": np.array(sorted(layers), dtype=np.int64),
        "primary_layer": np.array(primary_layer, dtype=np.int64),
    }
    for layer, p in per_layer_preds.items():
        arrays[f"pred_te_L{layer}"] = p["pred_te"]
        arrays[f"target_te_L{layer}"] = p["target_te"]
        arrays[f"pred_wc_L{layer}"] = p["pred_wc"]
        arrays[f"target_wc_L{layer}"] = p["target_wc"]
    np.savez(path, **arrays)
    return path


# ---------------------------------------------------------------------------
# Battery driver (production + synthetic e2e share this body)
# ---------------------------------------------------------------------------


def run_battery(
    split_ids: dict,
    store_fn,
    ceiling_fn,
    dev,
    out_dir: Path,
    preds_dir: Path,
    models: dict[str, dict] | None = None,
    anchor_fn=run_anchor_gate,
) -> dict[str, Path]:
    """Run the 4-cell × 3-layer battery.

    ``store_fn(model_cfg, layer)`` → the per-split store dict (production: HF
    streaming via the reuse cores; synthetic e2e: rng arrays — the ONLY faked
    boundary). ``ceiling_fn(model_cfg, layer)`` → the two-draw ceiling record.
    Cell JSONs are atomically rewritten after EVERY (cell, layer) unit.
    """
    models = models or MODELS
    t0 = time.time()
    cell_paths: dict[str, Path] = {}
    results: dict[str, dict] = {}
    anchor_record: dict | None = None
    # Cross-model matched-ID ledger: realized ci lists per (split-key) — every
    # model must realize the IDENTICAL row-aligned lists (they all come from
    # split_ids verbatim, and this asserts it end-to-end).
    id_ledger: dict[str, list[int]] = {}

    unit_i, unit_total = 0, sum(len(m["layers"]) * len(m["cells"]) for m in models.values())
    for model_key in [k for k in MODEL_ORDER if k in models]:
        mcfg = models[model_key]
        per_cell_preds: dict[str, dict[int, dict]] = {c: {} for c in mcfg["cells"]}
        for layer in mcfg["layers"]:
            _phase(f"assemble_{model_key}_L{layer}")
            store = store_fn(mcfg, layer)
            ceiling = ceiling_fn(mcfg, layer)
            if mcfg.get("anchor") and layer == mcfg["primary_layer"] and anchor_fn is not None:
                _phase("anchor_gate")
                anchor_record = _run_anchor_on_store(store, mcfg, dev, anchor_fn)
            for cell, train_key in mcfg["cells"].items():
                unit_i += 1
                t_unit = time.time()
                bundle = assemble_cell_arrays(
                    store, mcfg["store_train_split"], split_ids, train_key
                )
                for split_name, cis in bundle["cis"].items():
                    _assert_matched_ids(
                        cis, split_ids["splits"][split_name], f"{cell}/{split_name}"
                    )
                    prior = id_ledger.setdefault(split_name, cis)
                    if prior != cis:
                        raise RuntimeError(
                            f"cross-model matched-ID assert failed at {cell}/{split_name}: "
                            "realized id list differs from the ledger set by the first model."
                        )
                fit = fit_cell_layer(bundle, dev, ceiling, mcfg["h_dim"])
                per_cell_preds[cell][layer] = fit["preds"]

                rec = results.setdefault(
                    cell,
                    {
                        "cell": cell,
                        "model_key": model_key,
                        "model": mcfg["model"],
                        "hf_prefix": mcfg["hf_prefix"],
                        "store_revision_pin": mcfg.get("store_revision_pin"),
                        "train_key": train_key,
                        "layers": sorted(mcfg["layers"]),
                        "primary_layer": int(mcfg["primary_layer"]),
                        "h_dim": int(mcfg["h_dim"]),
                        "n_realized": bundle["n_realized"],
                        "counts_expected": {
                            s: int(split_ids["counts"][s]) for s in split_ids["counts"]
                        },
                        "split_ids_sha256": split_ids.get("sha256"),
                        "dropped_overlength": split_ids.get("dropped_overlength"),
                        "per_layer": {},
                        "fit_config": {
                            "lambdas": [float(x) for x in LF.LAMBDAS],
                            "ridge_block": int(LF.RIDGE_BLOCK),
                            "device": str(dev),
                            "grid_edge_disposition": "extend one decade + refit (plan §11)",
                        },
                        "_meta": _repro_meta(),
                    },
                )
                rec["per_layer"][str(layer)] = fit["result"]
                rec["port_parity_anchor"] = anchor_record or {
                    "skipped": "anchor gate not run in this mode"
                }
                path = out_dir / f"matched_fits_{cell}.json"
                _write_json_atomic(path, rec)
                cell_paths[cell] = path
                print(
                    f"[fits] unit {unit_i}/{unit_total} {cell}_L{layer} "
                    f"ridge_r2={fit['result']['ridge']['test_r2']:.4f} "
                    f"elapsed={time.time() - t_unit:.1f}s",
                    flush=True,
                )
        # All layers done for this model → write the per-cell preds npz.
        for cell in mcfg["cells"]:
            npz_path = save_cell_preds(
                preds_dir,
                cell,
                sorted(mcfg["layers"]),
                mcfg["primary_layer"],
                split_ids["splits"]["test_1000"],
                split_ids["splits"]["wc_test_1k"],
                per_cell_preds[cell],
            )
            rec = results[cell]
            rec["preds_path"] = str(npz_path)
            rec["preds_hf_mirror"] = f"{PREDS_HF_PREFIX}/{npz_path.name}"
            _write_json_atomic(cell_paths[cell], rec)
            logger.info("[matched-fits] preds: %s", npz_path)
    logger.info("[matched-fits] battery done in %.1fs", time.time() - t0)
    return cell_paths


def _run_anchor_on_store(store: dict[str, dict], mcfg: dict, dev, anchor_fn) -> dict:
    """Assemble the FULL banked train split (n=25k) + val/test for the anchor fit."""
    tr_store = store[mcfg["store_train_split"]]
    val_store, te_store = store["val_400"], store["test_1000"]
    X = np.concatenate([tr_store["cx"], val_store["cx"], te_store["cx"]], axis=0)
    Y = np.concatenate([tr_store["vx"], val_store["vx"], te_store["vx"]], axis=0)
    n_tr, n_val = tr_store["cx"].shape[0], val_store["cx"].shape[0]
    n_te = te_store["cx"].shape[0]
    tr = np.arange(0, n_tr, dtype=np.int64)
    val = np.arange(n_tr, n_tr + n_val, dtype=np.int64)
    te = np.arange(n_tr + n_val, n_tr + n_val + n_te, dtype=np.int64)
    return anchor_fn(X, Y, tr, val, te, dev)


# ---------------------------------------------------------------------------
# Production wiring (HF streaming + ceiling via the reuse cores)
# ---------------------------------------------------------------------------


def _production_store_fn(args, split_ids: dict):
    def store_fn(mcfg: dict, layer: int) -> dict[str, dict]:
        cache_dir = args.cache_dir / mcfg["hf_prefix"].replace("/", "_")
        cache_dir.mkdir(parents=True, exist_ok=True)
        expected = mcfg["store_expected_n"]
        if expected is None:
            # 9B store: P2 generates EXACTLY the split_ids rows.
            expected = {
                mcfg["store_train_split"]: int(split_ids["counts"][mcfg["store_train_split"]]),
                **{s: int(split_ids["counts"][s]) for s in STORE_SPLITS},
            }
        return assemble_store(
            mcfg["hf_prefix"], mcfg["store_train_split"], layer, cache_dir, expected
        )

    return store_fn


def _production_ceiling_fn(args, split_ids: dict):
    def ceiling_fn(mcfg: dict, layer: int) -> dict:
        cache_dir = args.cache_dir / mcfg["hf_prefix"].replace("/", "_")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return LF._reliability_ceiling(
            mcfg["hf_prefix"],
            layer,
            cache_dir,
            expected_n=int(split_ids["counts"]["test_1000"]),
        )

    return ceiling_fn


def _resolve_device(name: str):
    """Fail-loud device resolution (no silent cuda→cpu fallback; parent parity)."""
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested but torch.cuda.is_available() is False — refusing "
            "to silently fall back to CPU while a GPU pod bills idle. Pass --device cpu "
            "explicitly if a CPU run is genuinely intended."
        )
    return torch.device(name)


def _upload_preds_mirror(preds_dir: Path) -> None:
    """Mirror the preds npz dir to the HF data repo (ONE folder upload)."""
    import os

    from explore_persona_space.orchestrate import hub

    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing — cannot mirror preds to HF"
    _phase("upload_preds_mirror")
    hub._upload(
        local_path=preds_dir,
        repo_id="superkaiba1/explore-persona-space-data",
        repo_type="dataset",
        path_in_repo=PREDS_HF_PREFIX,
        raise_on_error=True,
    )
    logger.info("[matched-fits] mirrored %s -> %s", preds_dir, PREDS_HF_PREFIX)


# ---------------------------------------------------------------------------
# P1 step-4 fits-shape smoke (local 500-row chunk; count pins opted out)
# ---------------------------------------------------------------------------


def _load_local_chunks(chunk_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load the P1 smoke shard's LOCAL .pt capture chunks at one layer (same
    bundle schema as the HF chunks: cx_last/v_x (n, L, H), ci, layers)."""
    names = sorted(p for p in chunk_dir.glob("*.pt"))
    if not names:
        raise FileNotFoundError(f"no local capture chunks under {chunk_dir}")
    cxs, vxs, cis = [], [], []
    for p in names:
        b = FFC._mmap_load(p)
        cxs.append(N50._slice_layer(b, "cx_last", layer))
        vxs.append(N50._slice_layer(b, "v_x", layer))
        cis.extend(int(x) for x in b["ci"])
        del b
    return np.concatenate(cxs, axis=0), np.concatenate(vxs, axis=0), cis


def run_fits_smoke(args) -> int:
    """Plan §4 P1 step 4: run the fits port on the LOCAL 500-row smoke chunk.

    expected_split_n=None semantics (count pins opted out — the documented
    smoke downgrade), labeled smoke; exercises the 9B d=4096 shape + the
    subset/matched-ID code on the production device class."""
    chunk_dir = Path(args.smoke_chunk_dir)
    dev = _resolve_device(args.device)
    chunk_names = sorted(chunk_dir.glob("*.pt"))
    if not chunk_names:
        raise FileNotFoundError(f"no local capture chunks under {chunk_dir}")
    # Layers come from the bundle itself (self-describing).
    first = FFC._mmap_load(chunk_names[0])
    layers = [int(x) for x in first["layers"]]
    h_dim = int(first["cx_last"].shape[-1])
    del first
    if args.h_dim is not None and h_dim != args.h_dim:
        raise RuntimeError(f"smoke chunk h_dim={h_dim} != --h-dim {args.h_dim}")
    out: dict = {
        "label": "SMOKE — plan §4 P1 step 4 fits-shape smoke (expected_split_n=None opt-out)",
        "chunk_dir": str(chunk_dir),
        "layers": layers,
        "h_dim": h_dim,
        "device": str(dev),
        "per_layer": {},
        "_meta": _repro_meta(),
    }
    for layer in layers:
        cx, vx, ci = _load_local_chunks(chunk_dir, layer)
        n = cx.shape[0]
        if n < 20:
            raise RuntimeError(f"smoke chunk too small (n={n}) for a fits-shape smoke")
        _pin_counts({"smoke": n}, None, f"local smoke chunk (layer {layer})")  # explicit opt-out
        # Deterministic tiny split over the smoke rows: ~80/10/10.
        n_val = max(4, n // 10)
        n_te = max(4, n // 10)
        n_tr = n - n_val - n_te
        # Exercise the subset/matched-ID path on the smoke rows too.
        tr_ids = ci[:n_tr]
        cx_tr, vx_tr = _subset_by_ids(cx, vx, ci, tr_ids, "smoke_train")
        _assert_matched_ids(tr_ids, tr_ids, "smoke_train")
        assert cx_tr.shape == (n_tr, h_dim), cx_tr.shape
        X, Y = cx, vx
        tr = np.arange(0, n_tr, dtype=np.int64)
        val = np.arange(n_tr, n_tr + n_val, dtype=np.int64)
        te = np.arange(n_tr + n_val, n, dtype=np.int64)
        # FIXED grid under smoke — the λ-edge one-decade-extension disposition is
        # DEMOTED to an informational meta field here (smoke blind-spot,
        # deliberately enumerated): at the smoke slice n_train (≈400) < d (4096),
        # the junk-fit optimum is shrink-to-mean, so an edge λ is EXPECTED and
        # the production extension loop would spuriously exhaust + halt the P1
        # gate (gotchas.md smoke/production GATE-CALIBRATION parity, #1345).
        # The disposition re-arms at P3 production fits (n_train > d).
        pred, meta = F.fit_ridge(X, Y, tr, val, te, LF.LAMBDAS, dev, LF.RIDGE_BLOCK)
        meta = {**meta, "lambda_grid_edge_disposition": "informational under smoke (n<d slice)"}
        floors = LF._fit_floors(X, Y, tr, val, te, dev, LF.RIDGE_BLOCK)
        knn = LF._knn_reads({"ridge": pred, "train_mean": floors["train_mean"]["pred_te"]}, Y[te])
        out["per_layer"][str(layer)] = {
            "n": int(n),
            "ridge": {"test_r2": float(LF._pooled_r2(pred, Y[te])), "meta": meta},
            "floors": {k: float(v["test_r2"]) for k, v in floors.items()},
            "knn_ridge_acc_at_1_euclidean": knn["ridge"]["euclidean"]["acc_at_k"][1],
            "n_vs_d": {
                "n_train": int(n_tr),
                "d": int(h_dim),
                "underdetermined": bool(n_tr < h_dim),
            },
        }
        print(
            f"[fits-smoke] layer {layer}: n={n} ridge_r2="
            f"{out['per_layer'][str(layer)]['ridge']['test_r2']:.4f}",
            flush=True,
        )
    out_json = Path(args.out_json) if args.out_json else chunk_dir / "fits_smoke.json"
    _write_json_atomic(out_json, out)
    print(f"[fits-smoke] OK — wrote {out_json}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# --selftest: failure-path exercises (statistics-reconciler rec 3)
# ---------------------------------------------------------------------------


def _expect_raise(fn, needle: str, what: str) -> None:
    try:
        fn()
    except RuntimeError as e:
        assert needle in str(e), (what, needle, str(e)[:200])
        print(f"[selftest] PASS {what}: raised as designed ({needle!r})", flush=True)
        return
    raise AssertionError(f"[selftest] FAIL {what}: no RuntimeError raised")


def run_selftest() -> int:
    """CPU failure-path exercises on synthetic tensors (fail-loud paths PROVEN
    to fire): count-pin mismatch, matched-ID miss, matched-ID shuffle,
    anchor-gate halt, λ-grid extension arithmetic, happy-path subsetting."""
    rng = np.random.default_rng(0)

    # 1. Count-pin failure: mismatched split length ⇒ loud RuntimeError.
    _expect_raise(
        lambda: _pin_counts({"train_10k": 9_999}, {"train_10k": 10_000}, "selftest-store"),
        "count pin miss",
        "count-pin mismatch",
    )
    # 1b. Opt-out path is a no-op (the documented smoke downgrade).
    _pin_counts({"train_10k": 7}, None, "selftest-optout")
    print("[selftest] PASS count-pin opt-out (expected=None) is a no-op", flush=True)

    # 2. Matched-ID failure: id absent from the store ⇒ loud.
    cx = rng.standard_normal((4, 8)).astype(np.float32)
    vx = rng.standard_normal((4, 8)).astype(np.float32)
    ci = [10, 11, 12, 13]
    _expect_raise(
        lambda: _subset_by_ids(cx, vx, ci, [10, 99], "selftest"),
        "matched-ID assert failed",
        "matched-ID missing id",
    )
    # 2b. Shuffled id LIST (set equal, order differs) ⇒ loud (row alignment).
    _expect_raise(
        lambda: _assert_matched_ids([12, 11, 10], [10, 11, 12], "selftest"),
        "row ORDER differs",
        "matched-ID shuffled order",
    )
    # 2c. Different id SETS ⇒ loud.
    _expect_raise(
        lambda: _assert_matched_ids([10, 11, 14], [10, 11, 12], "selftest"),
        "id SETS differ",
        "matched-ID set mismatch",
    )
    # 2d. Happy path: rows gathered in ids order.
    sub_cx, sub_vx = _subset_by_ids(cx, vx, ci, [12, 10], "selftest-happy")
    assert np.array_equal(sub_cx[0], cx[2]) and np.array_equal(sub_cx[1], cx[0])
    assert np.array_equal(sub_vx[0], vx[2]) and np.array_equal(sub_vx[1], vx[0])
    print("[selftest] PASS subset happy path (rows in ids order)", flush=True)

    # 3. λ-grid one-decade extension arithmetic (both sides).
    lo = _extended_lambdas(LF.LAMBDAS, "low")
    hi = _extended_lambdas(LF.LAMBDAS, "high")
    assert len(lo) == len(LF.LAMBDAS) + GRID_DECADE_POINTS
    assert len(hi) == len(LF.LAMBDAS) + GRID_DECADE_POINTS
    assert np.isclose(lo[0], LF.LAMBDAS[0] / 10.0), lo[0]
    assert np.isclose(hi[-1], LF.LAMBDAS[-1] * 10.0), hi[-1]
    print("[selftest] PASS λ-grid one-decade extension (low & high)", flush=True)

    # 4. Anchor-gate halt: a wrong expected R² on a real tiny fit ⇒ hard halt;
    #    the SAME fit with the measured expectation ⇒ pass + deviation record.
    n_tr, n_val, n_te, d = 60, 12, 12, 16
    Xs = rng.standard_normal((n_tr + n_val + n_te, d)).astype(np.float32)
    Ys = (0.7 * Xs + 0.05 * rng.standard_normal(Xs.shape)).astype(np.float32)
    tr = np.arange(0, n_tr, dtype=np.int64)
    val = np.arange(n_tr, n_tr + n_val, dtype=np.int64)
    te = np.arange(n_tr + n_val, n_tr + n_val + n_te, dtype=np.int64)
    dev = torch.device("cpu")
    _expect_raise(
        lambda: run_anchor_gate(Xs, Ys, tr, val, te, dev, expected_r2=-0.5, tol=0.01),
        "PORT-PARITY ANCHOR GATE MISS",
        "anchor-gate hard halt",
    )
    pred, _ = F.fit_ridge(Xs, Ys, tr, val, te, LF.LAMBDAS, dev, LF.RIDGE_BLOCK)
    measured = LF._pooled_r2(pred, Ys[te])
    rec = run_anchor_gate(Xs, Ys, tr, val, te, dev, expected_r2=measured, tol=0.01)
    assert rec["abs_deviation"] < 1e-9 and rec["investigate_before_narrate"] is False
    print("[selftest] PASS anchor-gate pass path (deviation record present)", flush=True)

    print("[selftest] ALL PASS", flush=True)
    return 0


# ---------------------------------------------------------------------------
# --synthetic-e2e: shape-faithful CPU end-to-end (schema check; n<d smoke shape)
# ---------------------------------------------------------------------------


def run_synthetic_e2e(args) -> int:
    """Shape-faithful synthetic CPU e2e through the PRODUCTION battery body
    (run_battery → assemble_cell_arrays → fit_cell_layer → JSON/npz writers),
    faking ONLY the HF boundary (store_fn/ceiling_fn) — verifies the JSON +
    npz schemas at d=3584/4096. n_train = d+128 (well-posed n>d, the
    production λ-selection regime — an n<d slice selects the HIGH grid edge
    by construction, junk-fit shrink-to-mean, and would spuriously exhaust
    the plan-§11 one-decade extension loop); eval splits tiny. Schema/wiring
    check only, never a signal read."""
    n_val, n_te, n_wc = 8, 10, 6
    d_max = max(m["h_dim"] for m in MODELS.values())
    n_tr_full, n_tr_half = d_max + 128, d_max + 64
    ids = {
        "train_10k": list(range(0, n_tr_full)),
        "train_5k": list(range(0, n_tr_half)),
        "val_400": list(range(100_000, 100_000 + n_val)),
        "test_1000": list(range(200_000, 200_000 + n_te)),
        "wc_test_1k": list(range(300_000, 300_000 + n_wc)),
    }
    split_ids = {
        "splits": ids,
        "counts": {k: len(v) for k, v in ids.items()},
        "sha256": {k: "synthetic" for k in ids},
        "dropped_overlength": {},
    }

    def _mk_store(h_dim: int, seed: int):
        r = np.random.default_rng(seed)

        def one(id_list):
            cx = r.standard_normal((len(id_list), h_dim)).astype(np.float32)
            vx = (0.6 * cx + 0.1 * r.standard_normal(cx.shape)).astype(np.float32)
            return {"cx": cx, "vx": vx, "ci": list(id_list)}

        return {
            "train_10k": one(ids["train_10k"]),
            "val_400": one(ids["val_400"]),
            "test_1000": one(ids["test_1000"]),
            "wc_test_1k": one(ids["wc_test_1k"]),
        }

    models = {
        k: {
            **MODELS[k],
            "store_train_split": "train_10k",  # synthetic stores carry the split_ids grain
            "store_expected_n": None,
            "anchor": False,  # anchor body covered by --selftest (halt + pass paths)
        }
        for k in MODEL_ORDER
    }

    def store_fn(mcfg: dict, layer: int) -> dict:
        # Fresh store per (model, layer) — each layer is visited exactly once,
        # so no cache (keeps peak RSS ~one store). Deterministic seed
        # (hash() is PYTHONHASHSEED-randomized; never use it for rng seeds).
        return _mk_store(mcfg["h_dim"], seed=mcfg["h_dim"] * 1000 + layer)

    def ceiling_fn(mcfg: dict, layer: int) -> dict:
        return {
            "available": True,
            "n_pairs": n_te,
            "ceiling_var_weighted_r": 0.9,
            "mean_per_dim_r": 0.85,
            "synthetic": True,
        }

    out_dir = Path(args.out_dir) / "synthetic_e2e"
    preds_dir = Path(args.preds_dir) if args.preds_dir else out_dir / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    print(
        "[synthetic-e2e] shape-faithful schema check (never a signal read): "
        f"n_train={n_tr_full} (> d, production λ-selection regime) at d=3584/4096",
        flush=True,
    )
    paths = run_battery(
        split_ids, store_fn, ceiling_fn, dev, out_dir, preds_dir, models=models, anchor_fn=None
    )

    # Schema verification: JSON + npz keys.
    for cell, path in paths.items():
        rec = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "cell",
            "model",
            "per_layer",
            "n_realized",
            "counts_expected",
            "port_parity_anchor",
            "preds_path",
            "preds_hf_mirror",
            "_meta",
        ):
            assert key in rec, (cell, key)
        for layer_rec in rec["per_layer"].values():
            for key in (
                "ridge",
                "floors",
                "knn_retrieval",
                "ceiling_two_draw",
                "wc_transfer",
                "n_vs_d",
            ):
                assert key in layer_rec, (cell, key)
            assert layer_rec["wc_transfer"]["fold_label"] == "wildchat_corpus_transfer"
            assert layer_rec["wc_transfer"]["n_wc_test"] == n_wc
            assert set(layer_rec["floors"]) == {
                "shuffled_pairing",
                "train_mean",
                "identity_copy",
                "scaled_identity",
                "identity_bias",
            }
        z = np.load(rec["preds_path"])
        layers = [int(x) for x in z["layers"]]
        assert len(layers) == 3 and int(z["primary_layer"]) in layers
        for layer in layers:
            for stem in ("pred_te", "target_te", "pred_wc", "target_wc"):
                key = f"{stem}_L{layer}"
                assert key in z, (cell, key)
            assert z[f"pred_te_L{layer}"].shape == (n_te, rec["h_dim"])
            assert z[f"pred_wc_L{layer}"].shape == (n_wc, rec["h_dim"])
        assert z["ci_te"].tolist() == ids["test_1000"]
        assert z["ci_wc"].tolist() == ids["wc_test_1k"]
        print(f"[synthetic-e2e] {cell}: JSON + npz schema OK ({path})", flush=True)
    print("[synthetic-e2e] ALL PASS", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Task #2330 P3: matched fits battery (7B vs 9B)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument(
        "--split-ids",
        type=Path,
        default=REPO_ROOT / "eval_results" / "issue_2330" / "split_ids.json",
        help="P0 split_ids.json (single source of truth for counts + subsets)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "eval_results" / "issue_2330",
        help="cell JSON output dir (canonical: eval_results/issue_2330/)",
    )
    ap.add_argument(
        "--preds-dir",
        type=Path,
        default=None,
        help="preds npz dir (default: <repo>/data/issue_2330/preds)",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HF streaming scratch (default: <out-dir>/.cache; pod: keep under /workspace)",
    )
    ap.add_argument(
        "--no-upload",
        action="store_true",
        help="skip the HF preds mirror (smokes / local verification)",
    )
    ap.add_argument(
        "--smoke-chunk-dir",
        default=None,
        help="P1 step-4 fits-shape smoke on a LOCAL capture-chunk dir (count pins opted out)",
    )
    ap.add_argument("--h-dim", type=int, default=None, help="smoke mode: assert the chunk h_dim")
    ap.add_argument(
        "--out-json",
        default=None,
        help="smoke mode: output JSON path (default <chunk>/fits_smoke.json)",
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="CPU failure-path exercises (count pins, matched-ID, anchor halt) — plan §4 P3",
    )
    ap.add_argument(
        "--synthetic-e2e",
        action="store_true",
        help="shape-faithful synthetic CPU e2e (JSON/npz schema check; n<d smoke shape)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="argparse-attribute completeness + deferred-import resolution (pre-flight)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def _run_import_check() -> int:
    """Execute every deferred import + the args-attribute completeness assert."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    from explore_persona_space.orchestrate import hub  # noqa: F401
    from explore_persona_space.orchestrate.provenance import (  # noqa: F401
        as_metadata_dict,
        git_provenance,
    )

    print("[import-check] OK: argparse attrs complete; deferred imports resolved")
    return 0


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    if args.import_check:
        return _run_import_check()
    if args.selftest:
        return run_selftest()
    if args.synthetic_e2e:
        return run_synthetic_e2e(args)
    if args.smoke_chunk_dir:
        return run_fits_smoke(args)

    # Production battery.
    if args.preds_dir is None:
        args.preds_dir = REPO_ROOT / "data" / "issue_2330" / "preds"
    if args.cache_dir is None:
        args.cache_dir = args.out_dir / ".cache"
    dev = _resolve_device(args.device)
    _phase("load_split_ids")
    split_ids = load_split_ids(args.split_ids)
    _phase("battery")
    store_fn = _production_store_fn(args, split_ids)
    ceiling_fn = _production_ceiling_fn(args, split_ids)
    paths = run_battery(split_ids, store_fn, ceiling_fn, dev, args.out_dir, args.preds_dir)
    if not args.no_upload:
        _upload_preds_mirror(args.preds_dir)
    _phase("done")
    for cell, path in sorted(paths.items()):
        print(f"OK {cell}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(_rc)
