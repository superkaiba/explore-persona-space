"""Issue #2587 unit 4 — P4 all-layer ridge fits on the Qwen3.5-9B store + the plan-§4.5 7B arms.

Phases (``--phase``):

* ``fits`` — per layer ℓ in ``--layers`` (default 0-31): fp64 primal ridge
  (``issue779_ffc_n1m_fits.fit_ridge_with_weights`` — ONE Gram eigh per layer)
  with X = cx_last(ℓ), Y = v_x(ℓ) on the 9B store's q35-surviving splits
  (row sets pinned to ``split_ids.json`` by ordered-ID equality — fail-loud),
  λ val-selected over ``LF.LAMBDAS`` (np.logspace(-3, 8, 23)) with the #2330
  one-decade grid-edge extension (max ``MF.MAX_GRID_EXTENSIONS``, fail-loud
  past the cap); five floors (``LF._fit_floors``) + kNN retrieval
  (``LF._knn_reads``) + wc_test_1k transfer R² via ``F.apply_map`` on the
  SAME persisted payload. Per-layer checkpoints (``<out-root>/percell/L{l}.json``)
  with a regime-keyed resume (generating-params key — never recomputed float
  bytes, #1336); ridge payloads at ``<out-root>/ridge_payloads/L{l}.pt`` and
  test/wc predictions at ``<out-root>/preds/L{l}_preds.pt`` (atomic ``.tmp``
  writes whose names never match the ``L*.pt`` upload glob, #2336). A pilot
  gate times the FIRST computed layer end-to-end and aborts (SystemExit 7 +
  ``pilot_gate_report.json``) when the projected shard wall exceeds 2x
  ``--pilot-budget-s`` (#1415 halt-routing shape).
* ``finalize`` — reads ALL 32 per-layer checkpoints fail-loud, asserts ONE
  regime key, freezes L* = argmax val-R² (persisted ``frozen: true`` — never
  runtime-recomputed downstream), computes the two-draw reliability ceiling
  (seeds 43/44) at L* + the ``--ceiling-twins`` layers, merges everything into
  ``--out-json`` (default ``eval_results/issue_2587/map_layer_sweep.json``),
  uploads payloads + preds to HF under ``--payloads-prefix``/``--preds-prefix``
  when ``--upload hf``, writes the ``fits_done.json`` sentinel.
* ``matched7b`` — the plan-§4.5 TWO deliberately distinct 7B fits (MF3 fix):
  (a) the port-parity ANCHOR GATE (``MF.run_anchor_gate``) on the FULL banked
  ``issue1491_scale_ladder/scale7_refit`` rows (train_25k/val_400/test_1000 at
  the ``LF.EXPECTED_SPLIT_N`` grain, revision-pinned) reproducing
  R² = 0.7250873220237553 +/- 0.01 — a parity gate, NEVER a headline arm,
  never in a cross-model contrast, never in the H1 read; a miss HALTS
  (failure_class: code). Runs FIRST; its record is written to ``--anchor-out``
  immediately after the pass. (b) the headline ``arm_7b_matched25k``: a second
  7B ridge fit at L19 whose train/val/test/wc row sets are EXACTLY the
  q35-surviving IDs from ``split_ids.json`` (ordered-ID-set equality asserted
  per split, recomputed sha256 checked against ``split_ids["sha256"]``,
  per-split manifests persisted; any mismatch HALTS — never a near-match or
  intersection-on-the-fly). The matched payload is applied to the banked
  #2564 vc store (``vc2564``, L19) and the mapped predictions are persisted;
  the ported minimal-pair battery READS belong to unit 5, not this script.

Smoke mode (``--smoke-chunk-dir``): thin dispatch to
``issue2330_matched_fits.run_fits_smoke`` — byte-compatible with unit 2's
``_run_fits_smoke`` subprocess hook (``--smoke-chunk-dir <dir> --device cuda
--h-dim 4096 --out-json <path>``). The smoke exercises the fit/floors/kNN core
on local capture chunks only; it does NOT execute the HF streaming, anchor,
upload, or sentinel phases (enumerated in plan §4's smoke blind-spot block),
and it demotes the λ-grid-edge verdict to informational meta (n < d smoke
slice; #1345 gate-calibration).

Streaming: ONE ``F._stream_n1m_multilayer`` pass per SPLIT (all requested
layers per pass; zero-row pb_head) — never the per-layer
``LF._stream_ladder_split`` on the dense 9B store (which would re-download the
store once per layer, ~31x). The banked 7B store IS read per-split at the
single layer L19 via ``LF._stream_ladder_split`` (revision-pinned).

CLI examples:
  uv run python scripts/issue2587_fits.py --phase fits --layers 0-15 --device cuda
  uv run python scripts/issue2587_fits.py --phase finalize --upload hf \
      --payloads-prefix issue2587_q35_map/analysis_tensors/ridge_payloads \
      --preds-prefix issue2587_q35_map/analysis_tensors/preds
  uv run python scripts/issue2587_fits.py --phase matched7b --upload hf \
      --preds7b-prefix issue2587_minpair/analysis_tensors/preds_7b_matched
  uv run python scripts/issue2587_fits.py --smoke-chunk-dir /tmp/smoke_chunks \
      --device cpu --h-dim 16 --out-json /tmp/smoke_fits.json
  uv run python scripts/issue2587_fits.py --import-check
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from explore_persona_space.orchestrate.env import load_dotenv

# Thread caps + creds BEFORE numpy/torch import (tests/test_shared_vm_thread_caps.py).
load_dotenv()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

import issue779_ffc_n1m_fits as F  # noqa: E402
import issue1491_ladder_fits as LF  # noqa: E402
import issue2330_matched_fits as MF  # noqa: E402
from explore_persona_space.atomic_io import save_pt_atomic  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded  # noqa: E402

logger = logging.getLogger("issue2587_fits")

ISSUE = 2587
HF_DATA_REPO = F.C.HF_DATA_REPO  # consumer's own constant, never a re-typed literal
STORE_PREFIX_9B = "issue2587_q35_map/qwen35_9b"
HF_PREFIX_7B = "issue1491_scale_ladder/scale7_refit"
STORE_REVISION_PIN_7B = "815ff6d976c686af8672b27cfdfb1ce6b419c02c"
H_DIM_9B = 4096
H_DIM_7B = 3584
L19 = 19
N_LAYERS_9B = 32
SPLITS = ("train_25k", "val_400", "test_1000", "wc_test_1k")
CEILING_SEEDS = (43, 44)
ARM_7B_MATCHED = "arm_7b_matched25k"
VC2564_HF_PATH = "issue2564_minpair/analysis_tensors/vc2564/vc2564_bank.pt"
BANK2564_MANIFEST_HF_PATH = "issue2564_minpair/manifests/bank2564_manifest.json"
VC2564_EXPECTED_N = 984
# Machine-stable λ-grid identity for regime keys: GENERATING PARAMS of
# LF.LAMBDAS = np.logspace(-3, 8, 23) — never recomputed float bytes (#1336).
LAMBDA_GRID_KEY = ["logspace", -3.0, 8.0, 23]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _sha_ids(ids) -> str:
    """sha256 of the compact-JSON id list (unit 2 ``_sha_ids`` convention —
    the #2330 split_ids domain; numpy ints coerced so the digest matches the
    plain-int JSON the producer hashed)."""
    plain = [x.item() if hasattr(x, "item") else x for x in ids]
    return hashlib.sha256(json.dumps(plain, separators=(",", ":")).encode()).hexdigest()


def _jsonable(o):
    """Recursively convert numpy scalars/arrays so percell rows stay JSON-native."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def regime_key(
    *, store_prefix: str, split_sha: dict, h_dim: int, selector: str, ridge_block: int, device: str
) -> str:
    """Machine-stable resume/regime key hashed from GENERATING PARAMETERS
    (λ grid as ``LAMBDA_GRID_KEY``, never recomputed float bytes — #1336)."""
    params = {
        "issue": ISSUE,
        "lambda_grid": LAMBDA_GRID_KEY,
        "store_prefix": str(store_prefix),
        "split_sha256": dict(split_sha),
        "h_dim": int(h_dim),
        "selector": str(selector),
        "ridge_block": int(ridge_block),
        "device_requested": str(device),
    }
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _rows_for(ci, ids, what: str) -> np.ndarray:
    """Gather row indices for ``ids`` IN ids ORDER; fail-loud on any miss,
    then assert exact ordered-ID equality (MF._assert_matched_ids). This is
    the §4.5 row-identity assertion: mismatch HALTS — never a near-match,
    warning, or intersection-on-the-fly."""
    by_ci = {int(c): i for i, c in enumerate(ci)}
    missing = [i for i in ids if int(i) not in by_ci]
    if missing:
        raise RuntimeError(
            f"matched-ID assert failed for {what}: {len(missing)}/{len(ids)} split_ids ids "
            f"absent from the streamed store (first: {missing[:10]}) — store/split_ids drift."
        )
    rows = np.array([by_ci[int(i)] for i in ids], dtype=np.int64)
    realized = [int(ci[int(r)]) for r in rows]
    MF._assert_matched_ids(realized, [int(i) for i in ids], what)
    return rows


def _verify_split_sha(ids, recorded_sha: str, split: str) -> None:
    """Recompute the per-split id-list sha and compare against the recorded
    ``split_ids["sha256"][split]`` — a mismatch is manifest self-drift."""
    got = _sha_ids(ids)
    if got != recorded_sha:
        raise RuntimeError(
            f"split_ids sha256 mismatch for {split!r}: recomputed {got} vs recorded "
            f"{recorded_sha} — split_ids.json self-inconsistent (drift); refusing to fit."
        )


def _torch_save_atomic(obj: dict, path: Path) -> None:
    """torch.save via ``atomic_io.save_pt_atomic`` — PROCESS-UNIQUE same-dir
    temp (pid + uuid fragment) + os.replace, so concurrent writers of one
    destination never collide (#2336; the old fixed ``<name>.tmp`` was the
    process-shared temp-name class). The temp still never matches the
    ``L*.pt`` / ``*.pt`` upload globs (suffix ``.tmp``)."""
    save_pt_atomic(path, obj)


def _parse_layers(spec: str) -> list[int]:
    """Parse '0-31' / '0,5,19' / mixed into a sorted unique layer list."""
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError(f"empty --layers spec {spec!r}")
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Ridge fit: edge-extended, weights-returning, cuSOLVER CPU fallback
# ---------------------------------------------------------------------------


def _fit_once(fit_fn, X, Y, tr, val, ev, grid, dev, block):
    """One fit call with the gotchas.md cuSOLVER eigh CPU fallback (exact
    numerical-backend swap; never a Gram jitter). Returns (result, device_realized)."""
    try:
        return fit_fn(X, Y, tr, val, ev, grid, dev, block), str(dev)
    except torch.linalg.LinAlgError:
        print(f"[p4-fits] cuSOLVER LinAlgError on {dev} — retrying the fit on CPU", flush=True)
        return fit_fn(X, Y, tr, val, ev, grid, torch.device("cpu"), block), "cpu"


def fit_ridge_edge_extended_weights(
    X, Y, tr, val, ev, dev, *, block: int | None = None, extend: bool = True, fit_fn=None
):
    """``MF.fit_ridge_edge_extended`` ported to the weights-returning core
    ``F.fit_ridge_with_weights`` (8-arg shape: X, Y, tr, val, ev, lambdas,
    dev, block — the unit-4 required pin; block defaults to LF.RIDGE_BLOCK).

    Returns whatever ``fit_fn`` returns (pred, meta[, payload]) with meta
    extended by ``lambda_grid`` / ``grid_extensions`` / ``device_realized``.
    Fail-loud past MF.MAX_GRID_EXTENSIONS one-decade extensions; with
    ``extend=False`` the first fit returns with the edge recorded
    informationally (--no-edge-extension).
    """
    if block is None:
        block = int(LF.RIDGE_BLOCK)
    if fit_fn is None:
        fit_fn = F.fit_ridge_with_weights
    grid = np.asarray(LF.LAMBDAS, dtype=np.float64)
    extensions: list[dict] = []
    for _ in range(int(MF.MAX_GRID_EXTENSIONS) + 1):
        result, device_realized = _fit_once(fit_fn, X, Y, tr, val, ev, grid, dev, block)
        meta = dict(result[1])
        meta["lambda_grid"] = [float(x) for x in grid]
        meta["grid_extensions"] = list(extensions)
        meta["device_realized"] = device_realized
        edge = meta.get("lambda_grid_edge")
        if edge is None or not extend:
            return (result[0], meta, *result[2:])
        extensions.append(
            {
                "side": edge,
                "selected_lambda_at_edge": float(meta["selected_lambda"]),
                "grid_len_before": int(len(grid)),
            }
        )
        print(
            f"[p4-fits] lambda at grid edge ({edge}, lam={meta['selected_lambda']:.3g}) — "
            "extending one decade + refitting",
            flush=True,
        )
        grid = MF._extended_lambdas(grid, edge)
    raise RuntimeError(
        f"lambda still at the grid edge after {MF.MAX_GRID_EXTENSIONS} one-decade extensions "
        f"(grid now spans [{grid[0]:.3g}, {grid[-1]:.3g}]) — refusing to report an "
        "edge-selected ridge fit (#2330 disposition exhausted)."
    )


def _fit_floors_robust(X, Y, tr, val, te, dev, block):
    """``LF._fit_floors`` with the same cuSOLVER CPU fallback as the main fit."""
    try:
        return LF._fit_floors(X, Y, tr, val, te, dev, block)
    except torch.linalg.LinAlgError:
        print(f"[p4-fits] cuSOLVER LinAlgError in floors on {dev} — retrying on CPU", flush=True)
        return LF._fit_floors(X, Y, tr, val, te, torch.device("cpu"), block)


def _edge_selected_layers(per_layer: dict[int, dict]) -> dict[int, str]:
    """Layers whose PERSISTED ridge meta records a non-null lambda_grid_edge.

    Only producible under ``--no-edge-extension`` (with extension enabled the
    fit either resolves the edge or raises past MAX_GRID_EXTENSIONS), so any
    hit here is a diagnostic-only fit that must never be FROZEN into L* /
    the merged sweep (r1 g5: --no-edge-extension had no finalize backstop)."""
    out: dict[int, str] = {}
    for li, row in per_layer.items():
        edge = ((row.get("ridge") or {}).get("meta") or {}).get("lambda_grid_edge")
        if edge is not None:
            out[int(li)] = str(edge)
    return out


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _chunk_prefix(store_prefix: str, split: str) -> str:
    """The unit-2 upload layout: <store_prefix>/<split>/final_token_capture."""
    return f"{store_prefix}/{split}/final_token_capture"


def stream_store_multilayer(store_prefix, split, layers, cache_dir, mm_dir, h_dim, local_dir=None):
    """ONE multi-layer memmap stream of ONE dense-store split via
    ``F._stream_n1m_multilayer`` (zero-row pb_head carries the hidden dim;
    NEVER the per-layer ``LF._stream_ladder_split`` on a dense store — that
    would re-download the store once per layer, ~31x). Returns (arrays, n_rows)
    with arrays keyed ("cx"|"vx", layer) -> np.memmap plus "ci" -> int64."""
    zero = np.zeros((0, int(h_dim)), dtype=np.float32)
    pb_head = {int(li): (zero, zero) for li in layers}
    ld = None
    if local_dir is not None:
        ld = Path(local_dir) / split / "final_token_capture"
    arrays, n_rows = F._stream_n1m_multilayer(
        _chunk_prefix(store_prefix, split),
        [int(x) for x in layers],
        Path(cache_dir) / "dl" / split,
        Path(mm_dir) / split,
        pb_head,
        local_dir=ld,
    )
    return arrays, int(n_rows)


def _ensure_hf_file(rel_path: str, cache_dir: Path) -> Path:
    """Download one HF data-repo file into the issue cache (retried)."""
    dest = Path(cache_dir) / "hf_files"
    dest.mkdir(parents=True, exist_ok=True)
    got = hub.retry_transient(
        lambda: hf_hub_download(HF_DATA_REPO, rel_path, repo_type="dataset", local_dir=str(dest)),
        what=f"issue2587 input {rel_path}",
    )
    return Path(got)


# ---------------------------------------------------------------------------
# Reliability ceiling (mirrors LF._reliability_ceiling arithmetic exactly,
# parametrized on the banked draw grain + a matched id subset)
# ---------------------------------------------------------------------------


def ceiling_from_draws(ci_a, vx_a, ci_b, vx_b, ids, expected_banked_n: int, what: str) -> dict:
    """Two-draw reliability ceiling on the rows of ``ids`` (gathered by ci from
    both draws). Arithmetic mirrors ``LF._reliability_ceiling`` exactly:
    per-dim Pearson r_d between draws, pooled by Var of the two-draw MEAN
    (ddof=0), ceiling = sum(Vd*r_d)/sum(Vd). #2130 fail-loud pins: each BANKED
    draw must hold exactly ``expected_banked_n`` rows, and every id must pair."""
    n_a, n_b = len(ci_a), len(ci_b)
    if n_a != int(expected_banked_n) or n_b != int(expected_banked_n):
        raise RuntimeError(
            "reliability-ceiling pairing shortfall (#2130 fail-loud pin): "
            f"len(ci_a)={n_a}, len(ci_b)={n_b}, expected_n={expected_banked_n} for {what} — "
            "a short draw means a partial upload or killed capture shard, never absence."
        )
    by_a = {int(c): i for i, c in enumerate(ci_a)}
    by_b = {int(c): i for i, c in enumerate(ci_b)}
    missing = [i for i in ids if int(i) not in by_a or int(i) not in by_b]
    if missing:
        raise RuntimeError(
            "reliability-ceiling pairing shortfall (#2130 fail-loud pin): "
            f"{len(missing)}/{len(ids)} ids absent from a draw for {what} "
            f"(first: {missing[:10]})."
        )
    rows_a = np.array([by_a[int(i)] for i in ids], dtype=np.int64)
    rows_b = np.array([by_b[int(i)] for i in ids], dtype=np.int64)
    A = np.asarray(vx_a, dtype=np.float64)[rows_a]
    B = np.asarray(vx_b, dtype=np.float64)[rows_b]
    a_c = A - A.mean(axis=0, keepdims=True)
    b_c = B - B.mean(axis=0, keepdims=True)
    num = (a_c * b_c).sum(axis=0)
    den = np.sqrt((a_c**2).sum(axis=0) * (b_c**2).sum(axis=0)) + 1e-30
    r_d = (num / den).astype(np.float32)
    Vd = ((A + B) / 2.0).var(axis=0, ddof=0).astype(np.float32)
    ceiling = float((Vd * r_d).sum() / (Vd.sum() + 1e-30))
    return {
        "available": True,
        "n_pairs": int(len(ids)),
        "banked_n_a": int(n_a),
        "banked_n_b": int(n_b),
        "ceiling_var_weighted_r": ceiling,
        "mean_per_dim_r": float(r_d.mean()),
    }


# ---------------------------------------------------------------------------
# vc2564 loader (observed schema — probed from the real banked artifacts)
# ---------------------------------------------------------------------------


def load_vc2564(store_path, manifest_path, layer: int, expected_n: int):
    """Load the banked #2564 vc store + manifest (LOCAL paths; the phase
    downloads them first). Observed store schema (probed 2026-08-25 from
    ``issue2564_minpair/analysis_tensors/vc2564/vc2564_bank.pt``): keys
    ['context_ids','dtype','issue','layers','position','repro','vc'],
    layers=[14,19,26], vc float32 (984, 3, 3584), string context_ids.
    Manifest: n_contexts=984 + contexts rows carrying 'id'. Returns
    (X float32 (n, h), context_ids). Fail-loud on schema/count/membership."""
    # weights_only=False: sha-pinned self-produced #2564 bundle carrying python
    # lists/dicts (torch>=2.6 weights_only default refuses those).
    bank = torch.load(str(store_path), map_location="cpu", weights_only=False)
    for key in ("context_ids", "layers", "vc"):
        if key not in bank:
            raise RuntimeError(
                f"vc2564 store schema drift: missing {key!r} (has {sorted(bank)}) — "
                "re-probe the banked artifact before consuming."
            )
    layers = [int(x) for x in bank["layers"]]
    if int(layer) not in layers:
        raise RuntimeError(f"vc2564 store lacks layer {layer} (has {layers}).")
    col = layers.index(int(layer))
    vc = bank["vc"]
    X = np.asarray(vc[:, col, :].to(torch.float32).numpy())
    ctx = [str(c) for c in bank["context_ids"]]
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    n_manifest = int(manifest["n_contexts"])
    if len(ctx) != int(expected_n) or n_manifest != len(ctx) or X.shape[0] != len(ctx):
        raise RuntimeError(
            f"vc2564 count violation: store n={X.shape[0]} context_ids={len(ctx)} "
            f"manifest n_contexts={n_manifest} expected={expected_n}."
        )
    contexts = manifest["contexts"]
    if isinstance(contexts, dict):
        known = {str(k) for k in contexts}
    else:
        known = {str(r["id"]) for r in contexts}
    unknown = [c for c in ctx if c not in known]
    if unknown:
        raise RuntimeError(
            f"vc2564 membership violation: {len(unknown)}/{len(ctx)} store context_ids "
            f"absent from the bank manifest (first: {unknown[:5]})."
        )
    return X, ctx


# ---------------------------------------------------------------------------
# L* freeze
# ---------------------------------------------------------------------------


def compute_lstar(per_layer: dict) -> dict:
    """Freeze L* = argmax over layers of ridge val_r2_at_selected (ties break
    to the LOWEST layer index; deterministic). The returned block is persisted
    with ``frozen: true`` — downstream consumers READ it, never recompute."""
    if not per_layer:
        raise RuntimeError("compute_lstar: empty per-layer map")
    val = {
        int(li): float(row["ridge"]["meta"]["val_r2_at_selected"]) for li, row in per_layer.items()
    }
    lstar = max(sorted(val), key=lambda li: val[li])
    return {
        "lstar": int(lstar),
        "criterion": "argmax over layers of ridge val_r2_at_selected",
        "tie_break": "lowest layer index",
        "frozen": True,
        "val_r2_by_layer": {str(li): val[li] for li in sorted(val)},
    }


# ---------------------------------------------------------------------------
# Phase: fits (P4)
# ---------------------------------------------------------------------------


def _load_and_verify_split_ids(args) -> dict:
    split_ids = MF.load_split_ids(Path(args.split_ids))
    for s in SPLITS:
        if s not in split_ids["splits"]:
            raise RuntimeError(
                f"split_ids.json missing split {s!r} (has {sorted(split_ids['splits'])})."
            )
        _verify_split_sha(split_ids["splits"][s], split_ids["sha256"][s], s)
    return split_ids


def run_fits(args) -> int:
    dev = MF._resolve_device(args.device)
    split_ids = _load_and_verify_split_ids(args)
    layers = _parse_layers(args.layers)
    rk = regime_key(
        store_prefix=args.store_prefix,
        split_sha=split_ids["sha256"],
        h_dim=int(args.h_dim),
        selector="val_r2",
        ridge_block=int(LF.RIDGE_BLOCK),
        device=str(args.device),
    )
    out_root = Path(args.out_root)
    percell_dir = out_root / "percell"
    payloads_dir = out_root / "ridge_payloads"
    preds_dir = out_root / "preds"
    percell_dir.mkdir(parents=True, exist_ok=True)

    MF._phase("stream")
    arrays: dict = {}
    gather: dict = {}
    for split in SPLITS:
        arr, n_rows = stream_store_multilayer(
            args.store_prefix,
            split,
            layers,
            Path(args.cache_dir),
            Path(args.cache_dir) / "mm",
            int(args.h_dim),
            local_dir=args.local_dir,
        )
        MF._pin_counts(
            {split: n_rows},
            {split: int(split_ids["counts"][split])},
            f"9B store {args.store_prefix}",
        )
        gather[split] = _rows_for(arr["ci"], split_ids["splits"][split], f"9B {split}")
        arrays[split] = arr

    n_tr = len(gather["train_25k"])
    n_val = len(gather["val_400"])
    n_te = len(gather["test_1000"])
    tr = np.arange(n_tr)
    val = np.arange(n_tr, n_tr + n_val)
    te = np.arange(n_tr + n_val, n_tr + n_val + n_te)
    d = int(args.h_dim)
    # Estimator-validity statement: n_train vs feature dim (must be n > d here).
    print(
        f"[p4-fits] n_train={n_tr} vs d={d} ({'n>d ok' if n_tr > d else 'UNDER-DETERMINED'}); "
        f"selector=val_r2 grid=logspace(-3,8,23) regime_key={rk}",
        flush=True,
    )
    if n_tr <= d:
        raise RuntimeError(
            f"n_train={n_tr} <= d={d}: estimator-degenerate regime — refusing the production fit."
        )

    todo: list[int] = []
    for li in layers:
        cpath = percell_dir / f"L{li}.json"
        if cpath.is_file():
            row = json.loads(cpath.read_text(encoding="utf-8"))
            if row.get("regime_key") != rk:
                raise RuntimeError(
                    f"resume regime mismatch at {cpath}: checkpoint regime_key="
                    f"{row.get('regime_key')} vs current {rk} — a resume must never mix "
                    "regimes (#722); quarantine the out-root or pass a fresh --out-root."
                )
            if (payloads_dir / f"L{li}.pt").is_file() and (preds_dir / f"L{li}_preds.pt").is_file():
                print(f"[p4-fits] L{li} already complete (resume skip)", flush=True)
                continue
        todo.append(li)

    MF._phase("fits")
    pilot_checked = False
    for k, li in enumerate(todo):
        t0 = time.time()
        Xl = np.concatenate(
            [
                np.asarray(arrays[s][("cx", li)])[gather[s]]
                for s in ("train_25k", "val_400", "test_1000")
            ],
            axis=0,
        )
        Yl = np.concatenate(
            [
                np.asarray(arrays[s][("vx", li)])[gather[s]]
                for s in ("train_25k", "val_400", "test_1000")
            ],
            axis=0,
        )
        assert Xl.shape == (n_tr + n_val + n_te, d), Xl.shape
        assert Yl.shape == Xl.shape, Yl.shape
        pred_te, meta, payload = fit_ridge_edge_extended_weights(
            Xl, Yl, tr, val, te, dev, extend=not args.no_edge_extension
        )
        test_r2 = float(LF._pooled_r2(pred_te, Yl[te]))
        floors = _fit_floors_robust(Xl, Yl, tr, val, te, dev, int(LF.RIDGE_BLOCK))
        knn = LF._knn_reads(
            {
                "ridge": pred_te,
                "identity_bias": floors["identity_bias"]["pred_te"],
                "train_mean": floors["train_mean"]["pred_te"],
            },
            Yl[te],
        )
        X_wc = np.asarray(arrays["wc_test_1k"][("cx", li)])[gather["wc_test_1k"]]
        Y_wc = np.asarray(arrays["wc_test_1k"][("vx", li)])[gather["wc_test_1k"]]
        pred_wc = F.apply_map(payload, X_wc, dev)
        wc_r2 = float(LF._pooled_r2(pred_wc, Y_wc))

        _torch_save_atomic(
            {
                **payload,
                "issue": ISSUE,
                "layer": int(li),
                "regime_key": rk,
                "repro": MF._repro_meta(),
            },
            payloads_dir / f"L{li}.pt",
        )
        _torch_save_atomic(
            {
                "issue": ISSUE,
                "layer": int(li),
                "regime_key": rk,
                "selected_lambda": float(meta["selected_lambda"]),
                "ci_te": [int(x) for x in split_ids["splits"]["test_1000"]],
                "pred_te": torch.from_numpy(np.asarray(pred_te, dtype=np.float32)),
                "target_te": torch.from_numpy(np.asarray(Yl[te], dtype=np.float32)),
                "ci_wc": [int(x) for x in split_ids["splits"]["wc_test_1k"]],
                "pred_wc": torch.from_numpy(np.asarray(pred_wc, dtype=np.float32)),
                "target_wc": torch.from_numpy(np.asarray(Y_wc, dtype=np.float32)),
                "repro": MF._repro_meta(),
            },
            preds_dir / f"L{li}_preds.pt",
        )
        row = {
            "issue": ISSUE,
            "layer": int(li),
            "regime_key": rk,
            "n_train": int(n_tr),
            "d": d,
            "ridge": {"meta": _jsonable(meta), "test_r2": test_r2, "wc_test_1k_r2": wc_r2},
            "floors": {
                name: {"test_r2": _jsonable(rec.get("test_r2")), "meta": _jsonable(rec.get("meta"))}
                for name, rec in floors.items()
            },
            "knn": _jsonable(knn),
            "timing_s": round(time.time() - t0, 2),
            "repro": MF._repro_meta(),
        }
        MF._write_json_atomic(percell_dir / f"L{li}.json", row)
        elapsed = time.time() - t0
        print(
            f"[p4-fits] unit {k + 1}/{len(todo)} L{li} val_r2={meta['val_r2_at_selected']:.4f} "
            f"test_r2={test_r2:.4f} wc_r2={wc_r2:.4f} lam={meta['selected_lambda']:.3g} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        if not pilot_checked:
            pilot_checked = True
            projected = elapsed * len(todo)
            if not args.skip_pilot_gate and projected > 2.0 * float(args.pilot_budget_s):
                report = {
                    "verdict": "ABORT",
                    "t_layer_s": elapsed,
                    "layers_todo": len(todo),
                    "projected_wall_s": projected,
                    "budget_s": float(args.pilot_budget_s),
                    "rule": "projected > 2x --pilot-budget-s",
                    "repro": MF._repro_meta(),
                }
                MF._write_json_atomic(out_root / "pilot_gate_report.json", report)
                print(
                    f"[p4-fits] PILOT GATE ABORT: projected {projected:.0f}s > 2x budget "
                    f"{args.pilot_budget_s}s (t_layer={elapsed:.1f}s x {len(todo)} layers) — "
                    f"report at {out_root / 'pilot_gate_report.json'}",
                    flush=True,
                )
                raise SystemExit(7)
    MF._phase("fits_shard_done")
    return 0


# ---------------------------------------------------------------------------
# Phase: finalize (L* freeze + ceilings + merge + upload + sentinel)
# ---------------------------------------------------------------------------


def run_finalize(args) -> int:
    split_ids = _load_and_verify_split_ids(args)
    out_root = Path(args.out_root)
    percell_dir = out_root / "percell"
    per_layer: dict[int, dict] = {}
    rks: set[str] = set()
    for li in range(N_LAYERS_9B):
        p = percell_dir / f"L{li}.json"
        if not p.is_file():
            raise RuntimeError(
                f"finalize: missing per-layer checkpoint {p} — run --phase fits for all "
                f"{N_LAYERS_9B} layers first (fail-loud, never a partial sweep)."
            )
        row = json.loads(p.read_text(encoding="utf-8"))
        rks.add(str(row.get("regime_key")))
        per_layer[li] = row
    if len(rks) != 1:
        raise RuntimeError(
            f"finalize: mixed regime keys across per-layer checkpoints: {sorted(rks)} — "
            "a resume must never mix regimes (#722)."
        )
    rk = next(iter(rks))
    edges = _edge_selected_layers(per_layer)
    if edges:
        raise RuntimeError(
            f"finalize: EDGE-SELECTED ridge fits at layers {sorted(edges)} "
            f"(lambda_grid_edge={edges}) — --no-edge-extension fits are diagnostic-only and "
            "must never be frozen into L*; re-run --phase fits WITHOUT --no-edge-extension "
            "for these layers (#2330 disposition)."
        )
    lstar_block = compute_lstar(per_layer)
    lstar = int(lstar_block["lstar"])
    ceil_layers = sorted({lstar, *_parse_layers(args.ceiling_twins)})
    print(f"[p4-fits] L*={lstar} (frozen); ceiling layers={ceil_layers}", flush=True)

    MF._phase("ceiling")
    draws: dict[int, dict] = {}
    for seed in CEILING_SEEDS:
        arr, _n = stream_store_multilayer(
            args.store_prefix,
            f"ceiling_draws/seed{seed}",
            ceil_layers,
            Path(args.cache_dir),
            Path(args.cache_dir) / "mm",
            int(args.h_dim),
            local_dir=args.local_dir,
        )
        draws[seed] = arr
    test_ids = split_ids["splits"]["test_1000"]
    expected_banked = int(split_ids["counts"]["test_1000"])
    ceilings = {
        str(li): ceiling_from_draws(
            draws[43]["ci"],
            draws[43][("vx", li)],
            draws[44]["ci"],
            draws[44][("vx", li)],
            test_ids,
            expected_banked,
            f"9B ceiling L{li}",
        )
        for li in ceil_layers
    }

    upload_rec: dict = {"mode": args.upload}
    if args.upload == "hf":
        if not args.payloads_prefix or not args.preds_prefix:
            raise RuntimeError(
                "--upload hf requires explicit --payloads-prefix and --preds-prefix "
                "(no defaults — the #1005 upload-prefix clobber shape)."
            )
        MF._phase("upload")
        upload_dir_sharded(
            out_root / "ridge_payloads",
            HF_DATA_REPO,
            args.payloads_prefix,
            shard_glob="L*.pt",
            resume_skip=False,
            delete_local=False,
        )
        upload_dir_sharded(
            out_root / "preds",
            HF_DATA_REPO,
            args.preds_prefix,
            shard_glob="L*_preds.pt",
            resume_skip=False,
            delete_local=False,
        )
        upload_rec.update(
            {
                "payloads_prefix": args.payloads_prefix,
                "preds_prefix": args.preds_prefix,
                "n_payload_files": len(sorted((out_root / "ridge_payloads").glob("L*.pt"))),
                "n_preds_files": len(sorted((out_root / "preds").glob("L*_preds.pt"))),
            }
        )

    merged = {
        "issue": ISSUE,
        "regime_key": rk,
        "store_prefix": args.store_prefix,
        "h_dim": int(args.h_dim),
        "n_layers": N_LAYERS_9B,
        "split_counts": {s: int(split_ids["counts"][s]) for s in SPLITS},
        "split_sha256": dict(split_ids["sha256"]),
        "lstar": lstar_block,
        "reliability_ceiling": {
            "layers": [int(x) for x in ceil_layers],
            "expected_banked_n": expected_banked,
            "seeds": list(CEILING_SEEDS),
            "by_layer": ceilings,
        },
        "per_layer": {str(li): per_layer[li] for li in sorted(per_layer)},
        "upload": upload_rec,
        "repro": MF._repro_meta(),
    }
    MF._write_json_atomic(Path(args.out_json), merged)
    sentinel = Path(args.sentinel_path) if args.sentinel_path else out_root / "fits_done.json"
    MF._write_json_atomic(
        sentinel,
        {
            "issue": ISSUE,
            "phase": "fits",
            "done": True,
            "regime_key": rk,
            "lstar": lstar,
            "out_json": str(args.out_json),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "repro": MF._repro_meta(),
        },
    )
    MF._phase("done")
    return 0


# ---------------------------------------------------------------------------
# Phase: matched7b (P8 — anchor gate + arm_7b_matched25k)
# ---------------------------------------------------------------------------


_MATCHED7B_ARM_FILES = (
    f"payload_{ARM_7B_MATCHED}_L{L19}.pt",
    f"mapped_vc2564_{ARM_7B_MATCHED}_L{L19}.pt",
    f"test_preds_{ARM_7B_MATCHED}_L{L19}.pt",
)


def _matched7b_sentinel_path(args, out_root: Path) -> Path:
    """ONE sentinel-path derivation shared by the terminal write, the resume
    predicate, and the repair path (a drifted duplicate derivation would make
    the completion predicate check a DIFFERENT file than the writer writes)."""
    return Path(args.sentinel_path) if args.sentinel_path else out_root / "matched7b_done.json"


def _write_matched7b_sentinel(sentinel: Path, rk: str, anchor_out: Path) -> None:
    MF._write_json_atomic(
        sentinel,
        {
            "issue": ISSUE,
            "phase": "matched7b",
            "done": True,
            "regime_key": rk,
            "anchor_out": str(anchor_out),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "repro": MF._repro_meta(),
        },
    )


def _matched7b_completion_gaps(prior: dict, args, sentinel: Path) -> list[str]:
    """Gaps between the PRIOR complete record and THIS invocation's completion
    contract; empty => a resume skip is safe, non-empty => idempotent repair.

    The completion predicate includes the REQUESTED upload contract (mode +
    destination prefix) AND the sentinel write (r1 matched7b-resume-contract:
    the old skip keyed on ``complete`` alone, so a crash between the record
    write and the sentinel write skipped the sentinel forever, and a rerun
    with a changed ``--upload`` silently honored the STALE recorded mode)."""
    gaps: list[str] = []
    rec_up = prior.get("upload") or {}
    if args.upload == "hf" and not (
        rec_up.get("mode") == "hf" and rec_up.get("preds7b_prefix") == args.preds7b_prefix
    ):
        gaps.append("upload")
    sentinel_ok = False
    if sentinel.is_file():
        try:
            sdoc = json.loads(sentinel.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            sdoc = {}
        sentinel_ok = bool(sdoc.get("done")) and sdoc.get("regime_key") == prior.get("regime_key")
    if not sentinel_ok:
        gaps.append("sentinel")
    return gaps


def _matched7b_repair(
    prior: dict, rk: str, args, anchor_out: Path, sentinel: Path, preds7b_dir: Path
) -> int:
    """Idempotent completion repair from the PERSISTED matched-arm files: run
    the requested-but-unrecorded upload, update the record's upload block, and
    (re)write the sentinel — never re-fitting, never skipping past either."""
    missing = [n for n in _MATCHED7B_ARM_FILES if not (preds7b_dir / n).is_file()]
    if missing:
        raise RuntimeError(
            f"matched7b repair impossible: record at {anchor_out} claims complete but the "
            f"persisted arm files are missing: {missing} — quarantine the record or pass a "
            "fresh --anchor-out/--out-root for a full re-run (never a silent skip)."
        )
    rec_up = dict(prior.get("upload") or {})
    if args.upload == "hf" and not (
        rec_up.get("mode") == "hf" and rec_up.get("preds7b_prefix") == args.preds7b_prefix
    ):
        if not args.preds7b_prefix:
            raise RuntimeError(
                "--upload hf requires an explicit --preds7b-prefix (no defaults — "
                "the #1005 upload-prefix clobber shape)."
            )
        MF._phase("upload")
        upload_dir_sharded(
            preds7b_dir,
            HF_DATA_REPO,
            args.preds7b_prefix,
            shard_glob="*.pt",
            resume_skip=False,
            delete_local=False,
        )
        rec_up = {
            "mode": "hf",
            "preds7b_prefix": args.preds7b_prefix,
            "n_files": len(sorted(preds7b_dir.glob("*.pt"))),
            "repaired": True,
        }
        MF._write_json_atomic(anchor_out, {**prior, "upload": rec_up})
    _write_matched7b_sentinel(sentinel, rk, anchor_out)
    MF._phase("done")
    return 0


def run_matched7b(args) -> int:
    dev = MF._resolve_device(args.device)
    split_ids = _load_and_verify_split_ids(args)
    out_root = Path(args.out_root)
    preds7b_dir = out_root / "preds7b"
    anchor_out = Path(args.anchor_out)
    sentinel = _matched7b_sentinel_path(args, out_root)
    rk = regime_key(
        store_prefix=f"{args.hf_prefix_7b}@{args.revision_7b}",
        split_sha=split_ids["sha256"],
        h_dim=H_DIM_7B,
        selector="val_r2",
        ridge_block=int(LF.RIDGE_BLOCK),
        device=str(args.device),
    )
    if anchor_out.is_file():
        prior = json.loads(anchor_out.read_text(encoding="utf-8"))
        if prior.get("complete") and prior.get("regime_key") not in (None, rk):
            raise RuntimeError(
                f"matched7b resume regime mismatch at {anchor_out}: record regime_key="
                f"{prior.get('regime_key')} vs current {rk} — a resume must never mix regimes "
                "(#722); quarantine the record or pass a fresh --anchor-out."
            )
        if prior.get("complete") and prior.get("regime_key") == rk:
            gaps = _matched7b_completion_gaps(prior, args, sentinel)
            if not gaps:
                print(
                    f"[p4-fits] matched7b already complete at {anchor_out} "
                    "(record + upload contract + sentinel — resume skip)",
                    flush=True,
                )
                return 0
            print(
                f"[p4-fits] matched7b record complete at {anchor_out} but "
                f"{'/'.join(gaps)} unsatisfied — idempotent repair (no re-fit)",
                flush=True,
            )
            return _matched7b_repair(prior, rk, args, anchor_out, sentinel, preds7b_dir)

    MF._phase("stream_7b")
    banked: dict[str, tuple] = {}
    for split in SPLITS:
        cx, vx, ci = LF._stream_ladder_split(
            args.hf_prefix_7b,
            split,
            L19,
            Path(args.cache_dir) / "b7" / split,
            revision=args.revision_7b,
        )
        MF._pin_counts(
            {split: len(ci)},
            {split: int(LF.EXPECTED_SPLIT_N[split])},
            f"banked 7B {args.hf_prefix_7b}@{args.revision_7b}",
        )
        banked[split] = (np.asarray(cx), np.asarray(vx), [int(c) for c in ci])

    # (a) PORT-PARITY ANCHOR GATE — full banked rows, FIRST. NEVER a headline
    # arm; never in a cross-model contrast; never in the H1 read. Miss HALTS.
    MF._phase("anchor")
    n_a = len(banked["train_25k"][2])
    n_v = len(banked["val_400"][2])
    n_t = len(banked["test_1000"][2])
    Xa = np.concatenate([banked[s][0] for s in ("train_25k", "val_400", "test_1000")], axis=0)
    Ya = np.concatenate([banked[s][1] for s in ("train_25k", "val_400", "test_1000")], axis=0)
    tr_a = np.arange(n_a)
    val_a = np.arange(n_a, n_a + n_v)
    te_a = np.arange(n_a + n_v, n_a + n_v + n_t)
    try:
        anchor = MF.run_anchor_gate(Xa, Ya, tr_a, val_a, te_a, dev)
    except torch.linalg.LinAlgError:
        print("[p4-fits] cuSOLVER LinAlgError in anchor gate — retrying on CPU", flush=True)
        anchor = MF.run_anchor_gate(Xa, Ya, tr_a, val_a, te_a, torch.device("cpu"))
    role_note = (
        "port-parity anchor gate — a parity control on the FULL banked rows; NOT a headline "
        "arm, never in a cross-model contrast, never in the H1 read (plan §4.5 MF3)."
    )
    MF._write_json_atomic(
        anchor_out,
        {
            "issue": ISSUE,
            "regime_key": rk,
            "role": role_note,
            "anchor": anchor,
            "complete": False,
            "repro": MF._repro_meta(),
        },
    )

    # (b) HEADLINE arm_7b_matched25k — q35-surviving row sets, ordered-ID pinned.
    MF._phase("matched_fit")
    gathered: dict[str, tuple] = {}
    manifests: dict[str, dict] = {}
    for split in SPLITS:
        cx, vx, ci = banked[split]
        ids = split_ids["splits"][split]
        rows = _rows_for(ci, ids, f"7B matched {split}")
        realized = [int(ci[int(r)]) for r in rows]
        got_sha = _sha_ids(realized)
        if got_sha != split_ids["sha256"][split]:
            raise RuntimeError(
                f"matched 7B {split}: realized row-id sha {got_sha} != split_ids sha "
                f"{split_ids['sha256'][split]} — refusing the matched fit."
            )
        gathered[split] = (cx[rows], vx[rows])
        manifests[split] = {
            "n": len(ids),
            "sha256": got_sha,
            "banked_n": len(ci),
            "dropped_from_banked": len(ci) - len(ids),
        }
    n_tr = manifests["train_25k"]["n"]
    n_val = manifests["val_400"]["n"]
    n_te = manifests["test_1000"]["n"]
    tr = np.arange(n_tr)
    val = np.arange(n_tr, n_tr + n_val)
    te = np.arange(n_tr + n_val, n_tr + n_val + n_te)
    Xm = np.concatenate([gathered[s][0] for s in ("train_25k", "val_400", "test_1000")], axis=0)
    Ym = np.concatenate([gathered[s][1] for s in ("train_25k", "val_400", "test_1000")], axis=0)
    print(
        f"[p4-fits] matched 7B n_train={n_tr} vs d={H_DIM_7B} "
        f"({'n>d ok' if n_tr > H_DIM_7B else 'UNDER-DETERMINED'})",
        flush=True,
    )
    if n_tr <= H_DIM_7B:
        raise RuntimeError(
            f"matched 7B n_train={n_tr} <= d={H_DIM_7B}: estimator-degenerate — refusing."
        )
    pred_te, meta, payload = fit_ridge_edge_extended_weights(
        Xm, Ym, tr, val, te, dev, extend=not args.no_edge_extension
    )
    if meta.get("lambda_grid_edge") is not None:
        raise RuntimeError(
            f"matched7b: EDGE-SELECTED ridge fit (lambda_grid_edge={meta['lambda_grid_edge']}) "
            "— --no-edge-extension is diagnostic-only; refusing to persist a complete "
            "matched-arm record on an edge-selected fit. Re-run WITHOUT --no-edge-extension "
            "(#2330 disposition)."
        )
    test_r2 = float(LF._pooled_r2(pred_te, Ym[te]))
    floors = _fit_floors_robust(Xm, Ym, tr, val, te, dev, int(LF.RIDGE_BLOCK))
    knn = LF._knn_reads(
        {
            "ridge": pred_te,
            "identity_bias": floors["identity_bias"]["pred_te"],
            "train_mean": floors["train_mean"]["pred_te"],
        },
        Ym[te],
    )
    X_wc, Y_wc = gathered["wc_test_1k"]
    pred_wc = F.apply_map(payload, X_wc, dev)
    wc_r2 = float(LF._pooled_r2(pred_wc, Y_wc))

    # 7B two-draw ceiling at L19, matched test subset (banked draws pinned at
    # LF.CEILING_EXPECTED_N=1000; n_pairs = the q35-surviving test count).
    MF._phase("ceiling_7b")
    _cxa, vxa, cia = LF._stream_ladder_split(
        args.hf_prefix_7b,
        "ceiling_draws/seed43",
        L19,
        Path(args.cache_dir) / "b7ceil43",
        revision=args.revision_7b,
    )
    _cxb, vxb, cib = LF._stream_ladder_split(
        args.hf_prefix_7b,
        "ceiling_draws/seed44",
        L19,
        Path(args.cache_dir) / "b7ceil44",
        revision=args.revision_7b,
    )
    ceiling = ceiling_from_draws(
        [int(c) for c in cia],
        np.asarray(vxa),
        [int(c) for c in cib],
        np.asarray(vxb),
        split_ids["splits"]["test_1000"],
        int(LF.CEILING_EXPECTED_N),
        "7B matched ceiling L19",
    )

    # vc2564 application: matched payload over the banked #2564 vc store (L19).
    MF._phase("vc2564")
    vc_store = (
        Path(args.vc2564) if args.vc2564 else _ensure_hf_file(VC2564_HF_PATH, Path(args.cache_dir))
    )
    manifest_p = (
        Path(args.bank_manifest)
        if args.bank_manifest
        else _ensure_hf_file(BANK2564_MANIFEST_HF_PATH, Path(args.cache_dir))
    )
    Xvc, ctx_ids = load_vc2564(vc_store, manifest_p, L19, VC2564_EXPECTED_N)
    if Xvc.shape[1] != H_DIM_7B:
        raise RuntimeError(f"vc2564 h_dim {Xvc.shape[1]} != expected {H_DIM_7B}.")
    mapped = np.asarray(F.apply_map(payload, Xvc, dev), dtype=np.float32)
    assert mapped.shape == (VC2564_EXPECTED_N, H_DIM_7B), mapped.shape

    _torch_save_atomic(
        {
            **payload,
            "issue": ISSUE,
            "arm": ARM_7B_MATCHED,
            "layer": L19,
            "regime_key": rk,
            "repro": MF._repro_meta(),
        },
        preds7b_dir / f"payload_{ARM_7B_MATCHED}_L{L19}.pt",
    )
    _torch_save_atomic(
        {
            "issue": ISSUE,
            "arm": ARM_7B_MATCHED,
            "layer": L19,
            "regime_key": rk,
            "tensor": torch.from_numpy(mapped),
            "context_ids": ctx_ids,
            "source": {
                "vc_store": VC2564_HF_PATH,
                "bank_manifest": BANK2564_MANIFEST_HF_PATH,
                "payload": f"payload_{ARM_7B_MATCHED}_L{L19}.pt (sibling file)",
                "banked_7b_store": f"{args.hf_prefix_7b}@{args.revision_7b}",
            },
            "repro": MF._repro_meta(),
        },
        preds7b_dir / f"mapped_vc2564_{ARM_7B_MATCHED}_L{L19}.pt",
    )
    _torch_save_atomic(
        {
            "issue": ISSUE,
            "arm": ARM_7B_MATCHED,
            "layer": L19,
            "regime_key": rk,
            "selected_lambda": float(meta["selected_lambda"]),
            "ci_te": [int(x) for x in split_ids["splits"]["test_1000"]],
            "pred_te": torch.from_numpy(np.asarray(pred_te, dtype=np.float32)),
            "target_te": torch.from_numpy(np.asarray(Ym[te], dtype=np.float32)),
            "ci_wc": [int(x) for x in split_ids["splits"]["wc_test_1k"]],
            "pred_wc": torch.from_numpy(np.asarray(pred_wc, dtype=np.float32)),
            "target_wc": torch.from_numpy(np.asarray(Y_wc, dtype=np.float32)),
            "repro": MF._repro_meta(),
        },
        preds7b_dir / f"test_preds_{ARM_7B_MATCHED}_L{L19}.pt",
    )

    upload_rec: dict = {"mode": args.upload}
    if args.upload == "hf":
        if not args.preds7b_prefix:
            raise RuntimeError(
                "--upload hf requires an explicit --preds7b-prefix (no defaults — "
                "the #1005 upload-prefix clobber shape)."
            )
        MF._phase("upload")
        upload_dir_sharded(
            preds7b_dir,
            HF_DATA_REPO,
            args.preds7b_prefix,
            shard_glob="*.pt",
            resume_skip=False,
            delete_local=False,
        )
        upload_rec.update(
            {
                "preds7b_prefix": args.preds7b_prefix,
                "n_files": len(sorted(preds7b_dir.glob("*.pt"))),
            }
        )

    record = {
        "issue": ISSUE,
        "regime_key": rk,
        "role": role_note,
        "anchor": anchor,
        "arm": {
            "name": ARM_7B_MATCHED,
            "layer": L19,
            "n_train": int(n_tr),
            "d": H_DIM_7B,
            "test_r2": test_r2,
            "wc_test_1k_r2": wc_r2,
            "ridge_meta": _jsonable(meta),
            "floors": {
                name: {"test_r2": _jsonable(rec.get("test_r2")), "meta": _jsonable(rec.get("meta"))}
                for name, rec in floors.items()
            },
            "knn": _jsonable(knn),
            "split_manifests": manifests,
        },
        "ceiling_7b_matched_L19": ceiling,
        "vc2564": {
            "n_contexts": int(VC2564_EXPECTED_N),
            "context_ids_sha256": hashlib.sha256(
                json.dumps(ctx_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "mapped_file": f"mapped_vc2564_{ARM_7B_MATCHED}_L{L19}.pt",
        },
        "upload": upload_rec,
        "complete": True,
        "repro": MF._repro_meta(),
    }
    MF._write_json_atomic(anchor_out, record)
    _write_matched7b_sentinel(sentinel, rk, anchor_out)
    MF._phase("done")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Issue #2587 unit 4 — P4 all-layer 9B ridge fits + plan-4.5 7B arms.",
    )
    p.add_argument("--phase", choices=("fits", "finalize", "matched7b"), default="fits")
    p.add_argument(
        "--smoke-chunk-dir",
        type=Path,
        default=None,
        help="thin dispatch to issue2330_matched_fits.run_fits_smoke (unit-2 hook)",
    )
    p.add_argument(
        "--out-json", type=Path, default=Path("eval_results/issue_2587/map_layer_sweep.json")
    )
    p.add_argument(
        "--anchor-out", type=Path, default=Path("eval_results/issue_2587/matched7b_anchor.json")
    )
    p.add_argument("--split-ids", type=Path, default=Path("eval_results/issue_2587/split_ids.json"))
    p.add_argument("--store-prefix", default=STORE_PREFIX_9B)
    p.add_argument("--hf-prefix-7b", default=HF_PREFIX_7B)
    p.add_argument("--revision-7b", default=STORE_REVISION_PIN_7B)
    p.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="local chunk mirror root (per-split shard*_chunk*.pt under "
        "<local-dir>/<split>/final_token_capture)",
    )
    p.add_argument("--cache-dir", type=Path, default=Path("data/issue_2587/fits_cache"))
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/issue_2587/fits_out"),
        help="tensor out-root (payloads/preds/percell) — tensors NEVER under "
        "eval_results/ (JSON/text only)",
    )
    p.add_argument("--layers", default="0-31")
    p.add_argument("--device", default="cuda")
    p.add_argument("--h-dim", type=int, default=H_DIM_9B)
    p.add_argument("--upload", choices=("hf", "none"), default="none")
    p.add_argument("--payloads-prefix", default=None)
    p.add_argument("--preds-prefix", default=None)
    p.add_argument("--preds7b-prefix", default=None)
    p.add_argument("--sentinel-path", type=Path, default=None)
    p.add_argument("--pilot-budget-s", type=float, default=1800.0)
    p.add_argument("--skip-pilot-gate", action="store_true")
    p.add_argument("--no-edge-extension", action="store_true")
    p.add_argument("--ceiling-twins", default="16,22,30")
    p.add_argument(
        "--vc2564",
        type=Path,
        default=None,
        help="local vc2564_bank.pt (default: download the banked HF copy)",
    )
    p.add_argument(
        "--bank-manifest",
        type=Path,
        default=None,
        help="local bank2564_manifest.json (default: download the banked HF copy)",
    )
    p.add_argument("--import-check", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if args.smoke_chunk_dir is not None:
        return int(MF.run_fits_smoke(args))
    if args.phase == "fits":
        return run_fits(args)
    if args.phase == "finalize":
        return run_finalize(args)
    if args.phase == "matched7b":
        return run_matched7b(args)
    raise ValueError(f"unknown --phase {args.phase!r}")


if __name__ == "__main__":
    raise SystemExit(main())
