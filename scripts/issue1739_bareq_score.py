#!/usr/bin/env python3
"""#1739 BARE-QUERY map round — SCORING leg (no judge, read-only DV inputs).

Consumes the capture leg's artifacts (``issue1739_bareq_pod.py``) read-only and
answers "can a bare query, with no conversation prefix, predict the behavior the
model produced when the prefix WAS there?" in two independent legs:

* **LEG 1 — render-matched vs render-mismatched on the wildchat rung.** Each
  behavior's COMMITTED train-fit arms (map / ridge / projection) are applied to
  BARE eval reps and scored against the rung's judged DV, at the SAME
  TRAIN-frozen layer the committed wcrung column used, so the two rho columns
  are directly comparable. The matched/mismatched LABEL is behavior-scoped and
  MEASURED, not asserted: sycophancy + hallucination train contexts are already
  bare renders (``prefix_end`` constant at the template head), so applying their
  maps to bare reps is render-MATCHED; evil's train contexts are prefix-crossed,
  so the same operation is render-MISMATCHED by construction. Leg 1 FITS
  NOTHING — it evaluates frozen predictors.
* **LEG 2 — a dedicated bare -> answer map + bare-fit arms on evil's train
  pool, BY-QUERY group folds ONLY.** On evil's pool the bare rep is literally
  the SAME vector for every row sharing a query, so any fold splitting those
  rows leaks a duplicated feature; the fold key is the query bank's member
  ``context_ids`` (``bareq_queries.json``). Leg 2 fits a map, so it carries the
  standing mapping-baselines pair (identity+learned-bias AND kNN retrieval).

The single-turn wildchat rows are NOT re-captured: their original render is
already bare, so their committed wcrung reps ARE their bare reps (the capture
leg gates that with a bit-equality probe). This scorer JOINS them, and
independently re-verifies the claim scorer-side: a reused row's wcrung
``prefix_end`` must match the bare render's CONSTANT template head, under the
bf16-calibrated two-bar cosine recipe (gotchas.md § bf16 single-position
equivalence-gate calibration) rather than exact equality, because the two
stores were captured in different padded batches.

The constant-prefix arm is a built-in FAIL-LOUD null: a bare render's prefix
position is a content-independent vector, so every score built on it is
constant and its rho is degenerate. A non-degenerate read there is a
capture/indexing bug, not a finding.

Every piece of scoring math and every convention is IMPORTED from the reviewed
production modules — this file contains no fit, no metric, and no fold logic of
its own:

* ``issue1739_wcrung_arms`` — the frozen-layer convention (committed-modal
  selection + the positional-index guard + the own-pool fallback), the three
  safety rails, and the E1-direction resolution
* ``issue1739_fits._load_labeled`` / ``._fit_map`` / ``._u_pool_for_spec`` / ``RunSpec``
* ``fits.fit_whitening`` / ``apply_whitening`` / ``realize_budget_cell``
  / ``apply_map`` / ``r2_pooled`` / ``map_diagnostics`` (the identity+bias +
  kNN pair)
* ``arms.run_cell`` / ``run_transfer_cell`` / ``evaluate_transfer``
  / ``frozen_layer_idx`` / ``spearman_rows`` / ``bootstrap_rhos`` / ``write_summary``

HARD safety rails (this leg must never re-judge or clobber a committed input),
inherited verbatim from the wildchat-rung leg:

1. The judge is never called — no judge module may be imported, asserted at
   entry AND exit.
2. Every DV / manifest input is read-only: its sha256 is recorded at load and
   RE-VERIFIED after scoring.
3. A git-TRACKED output path is refused unless ``--allow-overwrite-committed``,
   and the out-root must be a ``bareq_map`` subtree so a mis-passed
   ``--out-root`` can never overwrite the main behavior dirs or a sibling
   rung's results.

CONTENT HYGIENE: the query manifest holds real user query TEXT. This leg reads
only ``query_id`` + ``context_ids`` from it and never carries query text into
any log line or output field.

VM-side runs carry the shared-VM thread caps
(``OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2``); pod/GCE runs do not.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Put the repo root on ``sys.path`` so ``scripts.*`` imports resolve.

    Script mode sets ``sys.path[0]`` to this file's dir (``scripts/``), NOT the
    repo root, so ``from scripts.issue1739_wcrung_arms import ...`` fails
    without this (gotchas.md § script-mode sys.path, #823). The sentinel assert
    makes a wrong parent depth fail loud instead of inserting a bogus path.
    """
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_wcrung_arms.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

RUNG = "bareq"
OUT_ROOT_NAME = "bareq_map"
BEHAVIORS = ("evil", "sycophancy", "hallucination")
WCRUNG = "wildchat_rung"

# Capture-leg parity (issue1739_bareq_pod): a bare render carries no
# conversational prefix, so the only INFORMATIVE summary position is the
# context end; the prefix position is the constant template head, emitted once
# as the built-in null probe.
BARE_KIND = "context_end"
BARE_NULL_KIND = "prefix_end"
# issue1739_bareq_pod.QUERY_MANIFEST — one row per UNIQUE query with the member
# context_ids that share it. THE by-query fold key.
QUERY_MANIFEST = "bareq_queries.json"
# Capture-store discriminator written at the top of every store by
# experiments/issue_1739/capture.py.
CAPTURE_MANIFEST_NAME = "_capture_manifest.json"
# Only evil's train pool carries a real conversational prefix, so only evil has
# a bare-fit leg to run (issue1739_bareq_pod refuses --leg 2 for other
# behaviors). The other two behaviors' leg-2 no-op is DOCUMENTED + MEASURED.
LEG2_BEHAVIORS = ("evil",)

SENTINEL_NAME = "bareq_score_done.json"
FAILURES_NAME = "bareq_score_failures.json"

DEFAULT_OUT_ROOT = Path("eval_results/issue_1739/bareq_map")
DEFAULT_MAIN_ROOT = Path("eval_results/issue_1739")
DEFAULT_TENSORS_ROOT = Path("analysis_tensors/issue_1739")
DEFAULT_STORE_ROOT = Path("data/issue_1739/hf_dl")
# The wildchat rung's ONE shared capture store (behavior-independent pool —
# issue1739_wcrung_arms module docstring).
WCRUNG_STORE_DIR_NAME = "wildchat"
# The committed train grid this leg freezes against. A run with FEWER layers
# cannot use committed-frozen indices (they index this full grid).
FULL_GRID_N_LAYERS = 28

# Render-match expectation per behavior, from the capture leg's own measured
# scope note. VERIFIED at scoring time against each behavior's train store
# (:func:`_constancy_report`); a disagreement is recorded and the MEASURED
# value wins the label.
RENDER_MATCH_EXPECTED = {
    "sycophancy": "matched",
    "hallucination": "matched",
    "evil": "mismatched",
}

# bf16 padded-batch equivalence bars (gotchas.md § "bf16 single-position
# equivalence gates ... run fp32 / recalibrate"; #779 r12 two-bar calibration).
# Single-position states concentrate bf16 kernel jitter in the LAST layers, so a
# flat 0.999 bar has no headroom there; a REAL render/indexing bug reads
# cos ~0.4-0.6, orders of magnitude below both bars.
EARLY_LAYER_COS_MIN = 0.999  # layers 0..3 — the sharp bug catcher
FLAT_COS_MIN = 0.995  # all layers flattened — >=4x the measured worst bf16 deviation
N_EARLY_LAYERS = 4

# Carried VERBATIM into every summary + the sentinel (round brief item 4).
ANALOGY_CAVEAT = (
    "#1092's 0.02 was bare->answer-ACTIVATIONS at SAE grain, so it is an analogy, not a "
    "numerically comparable floor for bare->judged-DV"
)


# ---------------------------------------------------------------------------
# safety rails (reused from the wildchat-rung leg — never re-implemented)
# ---------------------------------------------------------------------------


def _wca():
    """The wildchat-rung leg's module — the source of every shared convention.

    Deferred so ``--import-check`` exercises the real resolution and a plain
    module import of THIS file stays free of the sibling's import cost.
    """
    from scripts import issue1739_wcrung_arms

    return issue1739_wcrung_arms


def _assert_outputs_safe(paths: list[Path], *, out_root: Path, allow: bool) -> None:
    """Refuse to overwrite committed artifacts / escape the bareq_map subtree.

    Rail 3. Mirrors ``issue1739_wcrung_arms._assert_outputs_safe`` with this
    leg's own out-root name, and reuses that module's ``_git_tracked`` probe so
    the tracked-file predicate has ONE implementation.
    """
    if out_root.resolve().name != OUT_ROOT_NAME:
        raise SystemExit(
            f"--out-root must be a {OUT_ROOT_NAME!r} subtree (got {out_root}) — refusing to "
            "write bare-query results outside this round's own dir"
        )
    tracked = [p for p in paths if _wca()._git_tracked(p)]
    if tracked and not allow:
        raise SystemExit(
            "refusing to overwrite git-TRACKED output(s): "
            + ", ".join(str(p) for p in tracked)
            + " (pass --allow-overwrite-committed to re-write them deliberately)"
        )


def _git_commit() -> str:
    """Repo HEAD sha for the reproducibility footer ('unknown' off-repo)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO_ROOT,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _env_versions() -> dict[str, str]:
    """Interpreter + numeric-stack versions for the reproducibility footer."""
    import numpy

    out = {"python": sys.version.split()[0], "numpy": numpy.__version__}
    try:
        import torch

        out["torch"] = torch.__version__
    except ImportError as exc:  # torch is optional on a pure-CPU numpy path
        out["torch"] = f"unavailable ({exc.__class__.__name__})"
    return out


def _log(msg: str) -> None:
    print(f"[bareq-score] {msg}", flush=True)


# ---------------------------------------------------------------------------
# bare capture store: row identity + rep loading
# ---------------------------------------------------------------------------


def bare_row_key(meta_row: dict) -> tuple[str | None, str | None]:
    """``(leg, key)`` for one bare-store row — ``('1', context_id)`` / ``('2', query_id)``.

    The capture leg writes ONE rollout JSON per bare row named ``wc-<context_id>.json``
    (leg 1) or ``q-<query_id>.json`` (leg 2), and ``capture.capture_rollout_files``
    carries that filename into the row index as ``source_file`` while dropping the
    payload's ``row_id`` / ``query_id`` / ``kind`` fields. ``source_file`` is
    therefore the ONLY key that identifies a leg-2 row (its payload has no
    ``context_id``, so the row index's ``context_id`` is null), and it stays
    correct under the loader's over-budget row drops, which shift positions.
    ``context_id`` is accepted as a leg-1 fallback for stores whose index
    predates / omits ``source_file``. Returns ``(None, None)`` for an
    unattributable row.
    """
    src = meta_row.get("source_file")
    if isinstance(src, str) and src:
        stem = src[:-5] if src.endswith(".json") else src
        if stem.startswith("wc-"):
            return "1", stem[3:]
        if stem.startswith("q-"):
            return "2", stem[2:]
    cid = meta_row.get("context_id")
    if isinstance(cid, str) and cid:
        return "1", cid
    return None, None


def load_bare_store(store_dir: Path, layers: list[int], *, kinds: tuple[str, ...]):
    """Load the bare capture store's summaries + per-leg key -> row-index maps.

    Returns ``(arrays, meta, by_leg)`` where ``arrays[(kind, layer)]`` is the
    fp16 ``(n, d)`` block and ``by_leg['1']`` / ``by_leg['2']`` map
    ``context_id`` / ``query_id`` to a row index. Duplicate keys fail loud (a
    duplicated key would silently make one row's rep stand in for another's).
    """
    from explore_persona_space.experiments.issue_1739 import store_io
    from scripts.issue1739_fits import arrays_dim

    store_dir = Path(store_dir)
    dim = arrays_dim(store_dir, layers)
    arrays, meta = store_io.load_summaries(store_dir, kinds, tuple(layers), hidden_dim=dim)
    by_leg: dict[str, dict[str, int]] = {"1": {}, "2": {}}
    unattributed = 0
    for i, row in enumerate(meta):
        leg, key = bare_row_key(row)
        if leg is None or key is None:
            unattributed += 1
            continue
        if key in by_leg[leg]:
            raise RuntimeError(
                f"{store_dir}: duplicate bare-store key leg{leg}:{key} at rows "
                f"{by_leg[leg][key]} and {i} — one row's rep would silently stand in for another's"
            )
        by_leg[leg][key] = i
    if unattributed:
        raise RuntimeError(
            f"{store_dir}: {unattributed}/{len(meta)} rows carry neither a 'wc-'/'q-' "
            "source_file nor a context_id — row identity is unrecoverable (see bare_row_key)"
        )
    return arrays, meta, by_leg, dim


def _stack_layers(arrays: dict, kind: str, layers: list[int], rows) -> object:
    """``(Ly, len(rows), d)`` float64 stack of one summary kind at ``rows``."""
    import numpy as np

    rows = np.asarray(rows, dtype=np.int64)
    out = np.stack([np.asarray(arrays[(kind, ly)][rows], dtype=np.float64) for ly in layers])
    assert out.ndim == 3 and out.shape[1] == len(rows), out.shape
    return out


def _bare_block(arrays: dict, kind: str, layers: list[int], by_key: dict[str, int]):
    """``((Ly, n_keys, d) block, key -> column)`` over the bare store's keys.

    Keys are taken in SORTED order so the block's column layout is deterministic
    across runs (and so a resumed unit rebuilds the identical columns).
    """
    keys = sorted(by_key)
    block = _stack_layers(arrays, kind, layers, [by_key[k] for k in keys])
    return block, {k: i for i, k in enumerate(keys)}


def _cos_to_reference(block, reference) -> object:
    """Per-(layer, row) cosine of ``block`` (Ly, n, d) against ``reference`` (Ly, d)."""
    import numpy as np

    block = np.asarray(block, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    num = np.einsum("lnd,ld->ln", block, ref, optimize=True)
    den = np.linalg.norm(block, axis=2) * np.linalg.norm(ref, axis=1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _two_bar_verdict(cos, *, label: str) -> dict:
    """The bf16 two-bar equivalence verdict over a (Ly, n) cosine block.

    Bar (a) EARLY layers (0..N_EARLY_LAYERS-1) at ``EARLY_LAYER_COS_MIN`` is the
    sharp bug catcher — a mask / pad / row-mapping / render bug corrupts layer 0
    immediately, where bf16 jitter is ~1e-6. Bar (b) all layers flattened at
    ``FLAT_COS_MIN`` carries >=4x headroom over the measured worst deep-layer
    bf16 deviation. ``max_rel`` style spreads are REPORTED, never asserted.
    """
    import numpy as np

    cos = np.asarray(cos, dtype=np.float64)
    n_early = min(N_EARLY_LAYERS, cos.shape[0])
    early_min = float(np.nanmin(cos[:n_early])) if cos.size else float("nan")
    flat_min = float(np.nanmin(cos)) if cos.size else float("nan")
    per_layer_min = [float(x) for x in np.nanmin(cos, axis=1)] if cos.size else []
    passed = bool(
        cos.size
        and np.isfinite(early_min)
        and np.isfinite(flat_min)
        and early_min >= EARLY_LAYER_COS_MIN
        and flat_min >= FLAT_COS_MIN
    )
    return {
        "label": label,
        "n_rows": int(cos.shape[1]) if cos.ndim == 2 else 0,
        "early_cos_min": early_min,
        "flat_cos_min": flat_min,
        "per_layer_cos_min": per_layer_min,
        "bars": {
            "early_layers": N_EARLY_LAYERS,
            "early_cos_min": EARLY_LAYER_COS_MIN,
            "flat_cos_min": FLAT_COS_MIN,
            "calibration": (
                "bf16 padded-batch two-bar recipe (gotchas.md; #779 r12) — NOT exact equality: "
                "the compared reps come from different padded batches"
            ),
        },
        "passed": passed,
    }


def _constancy_report(block, *, label: str) -> dict:
    """Is a (Ly, n, d) rep block CONSTANT across rows? (the null-probe read).

    A bare render's prefix position is content-independent, so its reps must be
    one vector repeated. Judged under the same bf16 two-bar recipe as the reuse
    check (fp16 storage + differing batch composition rules out exact
    equality), against row 0 as the reference.
    """
    import numpy as np

    block = np.asarray(block, dtype=np.float64)
    if block.shape[1] == 0:
        return {"label": label, "n_rows": 0, "constant": False, "reason": "no rows"}
    cos = _cos_to_reference(block, block[:, 0, :])
    verdict = _two_bar_verdict(cos, label=label)
    verdict["constant"] = verdict["passed"]
    verdict["max_abs_dev_from_row0"] = float(np.max(np.abs(block - block[:, :1, :])))
    return verdict


# ---------------------------------------------------------------------------
# leg 1: bare eval reps over the wildchat rung
# ---------------------------------------------------------------------------


def substitute_bare_eval_reps(z_ev, ctx_order: list[str], bare_block, by_ctx: dict[str, int]):
    """Overwrite the wcrung eval reps with BARE reps where a bare row exists.

    ``z_ev`` is the committed wcrung ``(Ly, n_ev, d)`` block in ``ctx_order``;
    every context with a bare capture row (the multi-turn ones) takes its bare
    rep, and every context WITHOUT one keeps its wcrung rep — which, for a
    single-turn context, already IS its bare rep (the original render carried no
    prefix; the capture leg gates that with a bit-equality probe and
    :func:`verify_reused_rows_are_bare` re-verifies it here). Returns
    ``(z_bare, substituted_rows, reused_rows)``.
    """
    import numpy as np

    z_bare = np.array(z_ev, dtype=np.float64, copy=True)
    substituted: list[int] = []
    reused: list[int] = []
    for i, cid in enumerate(ctx_order):
        row = by_ctx.get(cid)
        if row is None:
            reused.append(i)
            continue
        z_bare[:, i, :] = np.asarray(
            [bare_block[li, row, :] for li in range(bare_block.shape[0])], dtype=np.float64
        )
        substituted.append(i)
    if not substituted:
        raise RuntimeError(
            "no wildchat-rung context joined the bare capture store — the leg-1 capture "
            "either did not run or its row keys do not match this DV's context_ids"
        )
    return z_bare, np.asarray(substituted, dtype=np.int64), np.asarray(reused, dtype=np.int64)


def verify_reused_rows_are_bare(z_prefix_ev, reused_rows, bare_prefix_const, *, mode: str) -> dict:
    """Reused (non-substituted) wcrung rows must be BARE renders — verified.

    The reuse licence is "a single-turn wildchat render is byte-identical to its
    bare render". Its observable consequence: such a row's wcrung ``prefix_end``
    rep must equal the bare render's CONSTANT template head. A multi-turn row
    (real conversation prefix) reads far below the bars, so this catches a
    mis-classified reuse without needing the capture leg's ``multi_turn`` flag
    (which ``capture_rollout_files`` drops from the row index).

    ``mode``: ``hard`` raises on a bar miss, ``report`` records it, ``off``
    skips. Realized cosines are recorded either way.
    """
    if mode == "off":
        return {"ran": False, "reason": "--reuse-check off"}
    if len(reused_rows) == 0:
        return {"ran": False, "reason": "no reused rows (every eval context was re-captured)"}
    cos = _cos_to_reference(z_prefix_ev[:, reused_rows, :], bare_prefix_const)
    verdict = _two_bar_verdict(cos, label="reused-row prefix vs bare constant head")
    verdict["ran"] = True
    verdict["mode"] = mode
    if not verdict["passed"] and mode == "hard":
        raise RuntimeError(
            "REUSE gate FAILED: reused wildchat rows' prefix reps do not match the bare "
            f"render's constant template head (early_cos_min={verdict['early_cos_min']:.6f} "
            f"< {EARLY_LAYER_COS_MIN}, flat_cos_min={verdict['flat_cos_min']:.6f} "
            f"< {FLAT_COS_MIN}) — those {len(reused_rows)} rows are NOT bare renders, so their "
            "committed wcrung reps may NOT stand in as bare reps (pass --reuse-check report to "
            "record instead of halting)"
        )
    return verdict


def _null_probe(bare_prefix_block, dv_ev, rb, *, seed: int, draw: int, n_boot: int) -> dict:
    """The constant-prefix FAIL-LOUD null: constancy + a degenerate-rho read.

    Two reads, both cheap and closed-form (no ridge on a rank-deficient design):
    the prefix reps' CONSTANCY, and the projection-arm rho those constant reps
    produce. A constant score has zero rank variance, so Spearman is NaN by
    construction; a FINITE rho whose bootstrap CI excludes 0 means the reps are
    not constant, i.e. a capture/indexing bug rather than a finding.

    Runs in the store's RAW space against the RAW ``rb`` — deliberately, so the
    probe costs no U-pool whitening or map refit. Both reads are
    verdict-invariant under an affine map: a constant rep projects to a constant
    score in every linear space, so the degenerate/ANOMALY verdict is identical
    to the whitened one. (The ANOMALY branch's follow-up diagnosis IS the
    whitened arm sweep, which the caller then runs.)
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms

    constancy = _constancy_report(bare_prefix_block, label="bare prefix reps (null probe)")
    scores = np.einsum("lnd,ld->ln", bare_prefix_block, rb, optimize=True)
    dv = np.asarray(dv_ev, dtype=np.float64)
    rho = [float(x) for x in arms.spearman_rows(scores, dv)]
    finite = [r for r in rho if np.isfinite(r)]
    ci_excludes_zero = False
    if finite:
        idx = arms.make_bootstrap_idx(len(dv), n_boot=n_boot, seed=seed + 100 * draw)
        draws = arms.bootstrap_rhos(scores, dv, idx)
        for li in range(scores.shape[0]):
            lo, hi = (float(np.nanquantile(draws[li], q)) for q in (0.025, 0.975))
            if np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0):
                ci_excludes_zero = True
                break
    degenerate = bool(constancy.get("constant")) and not ci_excludes_zero
    return {
        "constancy": constancy,
        "rho_per_layer": rho,
        "n_finite_rho": len(finite),
        "any_ci_excludes_zero": ci_excludes_zero,
        "verdict": "degenerate-as-predicted" if degenerate else "ANOMALY",
        "note": (
            "the bare-render prefix position is a CONSTANT vector across rows, so this arm is a "
            "built-in null: it must read ~chance (NaN / CI bracketing 0). A non-chance read is a "
            "capture/indexing bug, not a finding."
        ),
    }


# ---------------------------------------------------------------------------
# leg 2: by-query folds + the mapping-baselines pair
# ---------------------------------------------------------------------------


def load_query_bank(path: Path) -> tuple[dict[str, str], dict]:
    """``bareq_queries.json`` -> ``(context_id -> query_id, digest)``.

    CONTENT HYGIENE: the manifest holds real user query TEXT; only
    ``query_id`` + ``context_ids`` are read, and the returned digest carries
    counts + ids only. A context_id claimed by two queries fails loud (the fold
    key would be ambiguous).
    """
    payload = json.loads(Path(path).read_text())
    entries = payload.get("queries") or []
    ctx_to_qid: dict[str, str] = {}
    for entry in entries:
        qid = str(entry["query_id"])
        for cid in entry.get("context_ids", []):
            prior = ctx_to_qid.get(str(cid))
            if prior is not None and prior != qid:
                raise RuntimeError(
                    f"{path}: context {cid!r} is claimed by two queries ({prior}, {qid}) — "
                    "the by-query fold key is ambiguous"
                )
            ctx_to_qid[str(cid)] = qid
    if not ctx_to_qid:
        raise RuntimeError(f"{path}: query bank carries no context_ids (nothing to fold by)")
    digest = {
        "n_unique_queries": len(entries),
        "n_contexts": len(ctx_to_qid),
        "train_only": payload.get("train_only"),
        "n_rollout_rows_seen": payload.get("n_rollout_rows_seen"),
        "dedupe_ratio_contexts_per_query": payload.get("dedupe_ratio_contexts_per_query"),
        "manifest_git_commit": payload.get("git_commit"),
    }
    return ctx_to_qid, digest


def assert_by_query_folds(cell, query_ids: list[str]) -> dict:
    """Assert no query straddles two folds; return the fold digest.

    ``fits.realize_budget_cell`` assigns WHOLE groups to folds, so passing the
    per-row query_id as the group key is the non-leakage mechanism. This
    re-asserts the realized invariant rather than trusting it: on evil's pool the
    bare rep is the IDENTICAL vector for every row sharing a query, so a
    straddling query trains and evaluates on the same feature vector.
    """
    import numpy as np

    qids = np.asarray(query_ids)[cell.row_idx]
    folds_per_query = {q: sorted({int(f) for f in cell.fold_ids[qids == q]}) for q in set(qids)}
    straddling = {q: f for q, f in folds_per_query.items() if len(f) > 1}
    if straddling:
        raise RuntimeError(
            f"BY-QUERY fold violation: {len(straddling)} query/queries straddle folds "
            f"(e.g. {sorted(straddling)[:3]}) — a duplicated bare rep would leak across the split"
        )
    return {
        "fold_scheme": cell.fold_scheme,
        "fold_key": "query_id (bareq_queries.json member context_ids)",
        "n_folds": int(cell.n_folds),
        "n_rows": int(len(cell.row_idx)),
        "n_queries": len(folds_per_query),
        "rows_per_fold": [int(x) for x in np.bincount(cell.fold_ids, minlength=cell.n_folds)],
        "no_query_straddles_folds": True,
    }


def map_reads_by_query_folds(args, x_w, y_w, cell) -> dict:
    """Per-fold OOF bare -> answer map reads over BY-QUERY folds.

    The standing mapping-baselines pair for a FITTED ``v_X -> v_Y`` map: held-out
    R^2 alongside (a) the identity+learned-bias baseline and (b) kNN retrieval
    among the held-out candidate pool. Both come from the canonical
    ``fits.map_diagnostics`` helper (``analysis.mapping_baselines``) — never a
    re-implementation. Input and output share ``d``, so identity+bias is
    APPLICABLE (a dimension mismatch would be stated as inapplicable, never
    silently skipped).

    The map is refit per fold on the fold COMPLEMENT, so nothing here is
    transductive. ``fit_linear_map``'s OWN built-in diagnostics use a RANDOM
    80/20 split, which lets rows sharing a query straddle it — those are
    recorded as ``random_split_diagnostics`` and are NOT the headline.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits
    from scripts.issue1739_fits import _fit_map

    per_fold: list[dict] = []
    t0 = time.time()
    n_folds = int(cell.n_folds)
    for f in range(n_folds):
        ho = cell.row_idx[cell.fold_ids == f]
        tr = cell.row_idx[cell.fold_ids != f]
        if len(tr) < 2 or len(ho) < 2:
            per_fold.append({"fold": f, "skipped": f"n_train={len(tr)} n_holdout={len(ho)}"})
            continue
        x_tr, y_tr = x_w[:, tr], y_w[:, tr]
        x_ho, y_ho = x_w[:, ho], y_w[:, ho]
        n_train, d_in = int(x_tr.shape[1]), int(x_tr.shape[2])
        mapfit = _fit_map(args, x_tr, y_tr)
        pred = fits.apply_map(x_ho, mapfit)
        diag = fits.map_diagnostics(pred, x_ho, y_ho, x_tr, y_tr)
        per_fold.append(
            {
                "fold": f,
                "n_train": n_train,
                "n_holdout": int(len(ho)),
                "d_in": d_in,
                "n_train_lt_d": bool(n_train < d_in),
                "per_layer": diag["per_layer"],
                "random_split_diagnostics": {
                    k: v for k, v in mapfit.diagnostics.items() if k != "per_layer"
                },
            }
        )
        _log(
            f"[phase=leg2_map_reads] fold {f + 1}/{n_folds} n_train={n_train} "
            f"n_holdout={len(ho)} elapsed={time.time() - t0:.0f}s"
        )

    scored = [r for r in per_fold if "per_layer" in r]
    n_layers = len(scored[0]["per_layer"]) if scored else 0
    pooled = []
    for li in range(n_layers):
        rows = [r["per_layer"][li] for r in scored]
        pooled.append(
            {
                "layer_idx": li,
                "r2_map_mean": float(np.mean([r["r2_map"] for r in rows])),
                "r2_identity_bias_mean": float(np.mean([r["r2_identity_bias"] for r in rows])),
                "knn_acc_at_k_mean": {
                    metric: {
                        str(k): float(np.mean([r["knn"][metric]["acc_at_k"][k] for r in rows]))
                        for k in rows[0]["knn"][metric]["acc_at_k"]
                    }
                    for metric in ("euclidean", "cosine")
                },
                "knn_chance_at_k_mean": {
                    str(k): float(np.mean([r["knn"]["euclidean"]["chance_at_k"][k] for r in rows]))
                    for k in rows[0]["knn"]["euclidean"]["chance_at_k"]
                },
            }
        )
    return {
        "applicable": True,
        "why_applicable": (
            "leg 2 FITS a bare-rep -> answer-activation map, so the standing identity+bias + "
            "kNN-retrieval pair binds; input and output share d, so the identity family "
            "(including the learned-bias form) is applicable rather than dimension-mismatched"
        ),
        "fold_semantics": (
            "map refit per BY-QUERY fold on the fold complement — fully out-of-sample, no "
            "transduction; kNN pool = that fold's held-out targets, so chance = k/n_pool is "
            "per-fold and reported"
        ),
        "per_fold": per_fold,
        "pooled_per_layer": pooled,
        "n_folds_scored": len(scored),
        "wall_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# shared per-behavior plumbing
# ---------------------------------------------------------------------------


def resolve_bareq_store(args, behavior: str, leg: str) -> Path:
    """Resolve the bare capture store dir for one (behavior, leg).

    The capture leg writes ``<store-root>/bareq_<behavior>`` for a ``--leg 2``
    run and ``<store-root>/bareq`` otherwise (``--leg both`` puts BOTH legs'
    rows in the shared ``bareq`` store), and uploads that whole root under
    ``<hf-prefix>/capture_store/``. Resolution order: an explicit
    ``--bareq-store`` that IS a capture store, else its leg-preferred child,
    else the ``--store-root`` default's leg-preferred child.
    """
    given = Path(args.bareq_store) if args.bareq_store else args.store_root / "bareq_capture_store"
    if (given / CAPTURE_MANIFEST_NAME).is_file():
        return given
    names = (f"bareq_{behavior}", "bareq") if leg == "2" else ("bareq", f"bareq_{behavior}")
    for name in names:
        if (given / name / CAPTURE_MANIFEST_NAME).is_file():
            return given / name
    return given / names[0]


def _behavior_paths(args, behavior: str) -> dict[str, Path]:
    """Every input path for one behavior (flags override the defaults)."""
    return {
        "train_store": args.train_store or args.store_root / f"{behavior}_labeling",
        "train_dv": args.train_dv_json or args.train_dv_root / behavior / "labeling.json",
        "e1_store": args.e1_store or args.store_root / f"{behavior}_extraction",
        "wcrung_store": args.wcrung_store
        or args.store_root / "wcrung_capture_store" / WCRUNG_STORE_DIR_NAME,
        "wcrung_dv": args.wcrung_dv_json
        or args.main_root / WCRUNG / "dv_dataset" / behavior / "labeling.json",
        "train_summary": args.train_summary
        or args.main_root / behavior / "arm_results" / "all_arms_spearman.json",
        "wcrung_arms": args.wcrung_arms_json
        or args.main_root / WCRUNG / "arm_results" / behavior / "all_arms_spearman.json",
        "query_manifest": args.query_manifest or args.out_root / QUERY_MANIFEST,
    }


def committed_contrast_rows(path: Path, *, variant: str, regime: str) -> dict:
    """The committed wcrung (render-MISMATCHED) transfer rows, for the contrast.

    Read-only: this leg NEVER recomputes the committed column (that would
    re-spend the wcrung leg's compute) and never edits it. Comparability rests
    on the eval row set matching, which the caller checks against the committed
    ``eval_ctx_ids_sha256``. An absent file is recorded, never fatal — leg 2 is
    independent of the contrast.
    """
    path = Path(path)
    if not path.is_file():
        return {"available": False, "reason": f"absent: {path}"}
    payload = json.loads(path.read_text())
    rows = [
        {
            "arm": r.get("arm"),
            "eval_rung": r.get("eval_rung"),
            "rho_frozen": r.get("rho_frozen"),
            "ci_frozen": r.get("ci_frozen"),
            "n_eval": r.get("n_eval"),
            "layer": r.get("layer"),
            "variant": r.get("variant"),
            "regime": r.get("regime"),
        }
        for r in payload.get("transfer_rows", [])
        if r.get("variant") == variant and r.get("regime") == regime
    ]
    meta = payload.get("meta", {})
    return {
        "available": True,
        "path": str(path),
        "rows": rows,
        "n_rows": len(rows),
        "committed_eval_ctx_ids_sha256": meta.get("eval_ctx_ids_sha256"),
        "committed_n_contexts": meta.get("n_contexts"),
        "committed_frozen_layer_source": meta.get("frozen_layer_source"),
        "committed_git_commit": meta.get("git_commit"),
    }


def _frozen_layers_for_roster(args, behavior, variant, u_label, roster, layers, data, cell, paths):
    """Frozen layers per arm + provenance, via the wildchat-rung convention.

    Committed-modal selection off the behavior's main train summary when present
    (so leg 1's rho column is frozen at the SAME layer as the committed
    mismatched column), else the own-train-pool in-split OOF fallback. The
    committed indices are POSITIONAL into the FULL 28-layer grid, so
    ``_assert_committed_frozen_indexable`` refuses a reduced/reordered layer set
    rather than silently scoring at a clamped layer.
    """
    wca = _wca()
    summary = Path(paths["train_summary"])
    own_rho: dict[str, list[float]] = {}
    skips: list[dict] = []
    if summary.is_file() and not args.force_own_pool_frozen:
        frozen = wca.modal_frozen_layers(
            summary, variant=variant, regime=args.regime, u_rung_label=u_label
        )
        frozen = {a: i for a, i in frozen.items() if a in roster}
        src = f"modal-committed-train-cells:{summary}"
    else:
        frozen, own_rho, own_skips = wca.own_pool_frozen_layers(
            data, cell, roster=roster, device=args.device
        )
        src = "own-train-pool-selection"
        skips = [
            {"arm": a, "reason": f"own-pool frozen-layer read: {r}", "variant": variant}
            for a, r in sorted(own_skips.items())
        ]
    missing = sorted(set(roster) - set(frozen))
    if missing:
        raise RuntimeError(
            f"[{behavior}/{variant}] no frozen layer for {missing} (source: {src}) — "
            "cannot score at a TRAIN-frozen layer"
        )
    if src.startswith("modal-committed-train-cells:"):
        wca._assert_committed_frozen_indexable(frozen, layers, behavior, variant, summary)
    return frozen, src, own_rho, skips


def _whitening_and_map(args, tbl, layers, variant):
    """U-pool whitening + the committed context->answer map refit in-process.

    Identical to the wildchat-rung leg's construction (same U store, same
    ``RunSpec``, same ``_fit_map``), so leg 1's bare-rep column lands in the SAME
    whitened space as the committed mismatched column and the two rhos are
    comparable. Never an uploaded map payload.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits, store_io
    from scripts.issue1739_fits import RunSpec, _fit_map, _u_pool_for_spec

    dim = tbl.z_ans.shape[-1]
    store_io.stage_u_store(Path(args.u_store), (BARE_KIND, BARE_NULL_KIND, "t1"), tuple(layers))
    u_arrays, u_meta = store_io.load_summaries(
        args.u_store, (BARE_KIND, BARE_NULL_KIND, "t1"), tuple(layers), hidden_dim=dim
    )
    u_fit_rows = np.flatnonzero(store_io.fit_pool_mask(u_meta))
    spec = RunSpec(
        variant=variant,
        regime=args.regime,
        u_size=None if str(args.u_size).lower() == "full" else int(args.u_size),
        budgets=(len(tbl.ctx_order),),
        draws=(args.draw,),
        seeds=(args.seed,),
        f_u=None,
        f_l=None,
    )
    u_x, u_y, u_label, n_u = _u_pool_for_spec(spec, u_arrays, u_fit_rows, tbl, layers)
    wh = fits.fit_whitening(u_x, device=args.device, seed=args.seed)
    mapfit = _fit_map(args, fits.apply_whitening(u_x, wh), fits.apply_whitening(u_y, wh))
    del u_x, u_y
    return wh, mapfit, u_label, int(n_u)


# ---------------------------------------------------------------------------
# leg 1
# ---------------------------------------------------------------------------


def score_leg1(args, behavior: str) -> dict:  # noqa: C901 — one linear per-behavior pipeline
    """LEG 1: committed train-fit arms applied to BARE wildchat-rung reps."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits
    from scripts.issue1739_fits import _load_labeled

    wca = _wca()
    paths = _behavior_paths(args, behavior)
    layers = args.layers or list(range(args.n_layers))
    store_dir = resolve_bareq_store(args, behavior, "1")
    missing = [
        f"{k}={paths[k]}"
        for k in ("train_store", "train_dv", "wcrung_store", "wcrung_dv")
        if not Path(paths[k]).exists()
    ]
    if not (store_dir / CAPTURE_MANIFEST_NAME).is_file():
        missing.append(f"bareq_store={store_dir}")
    if missing:
        raise FileNotFoundError(
            f"[{behavior}/leg1] missing input(s): {'; '.join(missing)} — stage the train store, "
            "the wildchat-rung store + DV, and the bare capture store before scoring"
        )

    shas = {
        str(paths["train_dv"]): wca._sha256(paths["train_dv"]),
        str(paths["wcrung_dv"]): wca._sha256(paths["wcrung_dv"]),
    }
    t0 = time.time()
    tbl = _load_labeled(
        paths["train_store"], paths["train_dv"], layers, config="config_a", need_rollout_rows=False
    )
    tbl_ev = _load_labeled(
        paths["wcrung_store"],
        paths["wcrung_dv"],
        layers,
        config="config_b",
        need_rollout_rows=False,
    )
    if set(tbl_ev.rungs) != {WCRUNG}:
        raise RuntimeError(
            f"[{behavior}] {WCRUNG} DV must carry rung={WCRUNG!r} on every row; got "
            f"{tbl_ev.rungs} from {paths['wcrung_dv']} (wrong DV dataset?)"
        )
    dim = tbl.z_ans.shape[-1]
    if tbl_ev.z_ans.shape[-1] != dim:
        raise RuntimeError(
            f"[{behavior}] hidden dim mismatch: train {dim} vs {WCRUNG} {tbl_ev.z_ans.shape[-1]}"
        )
    eval_ctx_sha = wca._sha256_text("\n".join(tbl_ev.ctx_order))

    bare_arrays, bare_meta, by_leg, bare_dim = load_bare_store(
        store_dir, layers, kinds=(BARE_KIND, BARE_NULL_KIND)
    )
    if bare_dim != dim:
        raise RuntimeError(f"[{behavior}] bare store dim {bare_dim} != train store dim {dim}")
    n_leg1_rows = len(by_leg["1"])
    if not n_leg1_rows:
        raise RuntimeError(
            f"[{behavior}/leg1] bare store {store_dir} holds 0 leg-1 rows "
            f"({len(bare_meta)} rows total, {len(by_leg['2'])} leg-2) — capture --leg 1/both first"
        )
    _log(
        f"[phase=leg1] {behavior}: train n={len(tbl.ctx_order)} | {WCRUNG} n="
        f"{len(tbl_ev.ctx_order)} ctx_sha={eval_ctx_sha[:12]} | bare rows leg1={n_leg1_rows} "
        f"leg2={len(by_leg['2'])} | load={time.time() - t0:.0f}s"
    )

    # Render-match label, MEASURED on the behavior's own train prefix reps.
    train_prefix_constancy = _constancy_report(
        tbl.z_by_variant[BARE_NULL_KIND], label=f"{behavior} train prefix reps"
    )
    measured = "matched" if train_prefix_constancy.get("constant") else "mismatched"
    render_match = {
        "label": measured,
        "expected_from_capture_scope_note": RENDER_MATCH_EXPECTED.get(behavior),
        "agrees_with_expected": measured == RENDER_MATCH_EXPECTED.get(behavior),
        "basis": (
            "a train corpus whose prefix_end reps are CONSTANT across rows is already a bare "
            "render, so applying its train-fit map/ridge to bare eval reps is render-MATCHED; a "
            "prefix-crossed train corpus makes the same operation render-MISMATCHED"
        ),
        "train_prefix_constancy": train_prefix_constancy,
    }

    rb, rb_meta = wca._rb_for_behavior(args, behavior, tbl, layers, dim, paths)
    roster = list(args.arms) if args.arms else list(arms.TRANSFER_ARMS)
    n_boot = int(args.n_boot) if args.n_boot else arms.N_BOOT
    budget_l = args.budget or len(tbl.ctx_order)

    ckpt = args.out_root / behavior / "percell" / "bareq_leg1_transfer.jsonl"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if ckpt.exists():
        with ckpt.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["unit_key"]] = rec

    rows_all: list[dict] = []
    skips_all: list[dict] = []
    per_layer_all: list[dict] = []
    frozen_source: dict[str, str] = {}
    map_diag: dict[str, dict] = {}
    coverage: dict[str, dict] = {}
    nulls: dict[str, dict] = {}
    variants = list(args.variants)
    for vi, variant in enumerate(variants):
        unit_key = json.dumps(
            {
                "leg": "1",
                "behavior": behavior,
                "variant": variant,
                "regime": args.regime,
                "u_rung": args.u_size,
                "budget_l": budget_l,
                "draw": args.draw,
                "seed": args.seed,
                "rung": WCRUNG,
                "arms": sorted(roster),
                "layers": [int(x) for x in layers],
                "n_eval": len(tbl_ev.ctx_order),
                "n_boot": n_boot,
                "min_n": int(args.min_n),
                "map_kind": args.map_kind,
                "rb_source": rb_meta["rb_source"],
                "bare_rows": n_leg1_rows,
            },
            sort_keys=True,
        )
        if unit_key in done:
            rec = done[unit_key]
            rows_all += rec["rows"]
            skips_all += rec.get("skips", [])
            per_layer_all += rec.get("per_layer", [])
            frozen_source[variant] = rec.get("frozen_source", "resume")
            coverage[variant] = rec.get("coverage", {})
            nulls[variant] = rec.get("null_probe", {})
            _log(f"[phase=leg1] {behavior} unit {vi + 1}/{len(variants)} SKIP (resume) {variant}")
            continue

        # RAW-space blocks + the null probe come FIRST: both are verdict-invariant
        # under the whitening (a constant rep stays constant under any affine
        # map), so neither needs the U-pool whitening or the map refit — which is
        # exactly what lets a verified-degenerate prefix variant skip both.
        bare_variant_block, key_pos = _bare_block(bare_arrays, variant, layers, by_leg["1"])
        assert bare_variant_block.shape == (len(layers), n_leg1_rows, dim), (
            bare_variant_block.shape,
        )
        bare_prefix_block, prefix_key_pos = _bare_block(
            bare_arrays, BARE_NULL_KIND, layers, by_leg["1"]
        )
        bare_prefix_const = bare_prefix_block[:, 0, :]
        # The null runs on the SAME eval row set as the arms (aligned to
        # tbl_ev.ctx_order), so its rho is directly comparable to theirs.
        z_null_raw, _null_sub, _null_reused = substitute_bare_eval_reps(
            tbl_ev.z_by_variant[BARE_NULL_KIND],
            tbl_ev.ctx_order,
            bare_prefix_block,
            prefix_key_pos,
        )
        nulls[variant] = _null_probe(
            z_null_raw,
            np.asarray(tbl_ev.dv, dtype=np.float64),
            rb,
            seed=args.seed,
            draw=args.draw,
            n_boot=n_boot,
        )

        z_bare_raw, sub_rows, reused_rows = substitute_bare_eval_reps(
            tbl_ev.z_by_variant[variant], tbl_ev.ctx_order, bare_variant_block, key_pos
        )
        reuse = verify_reused_rows_are_bare(
            tbl_ev.z_by_variant[BARE_NULL_KIND],
            reused_rows,
            bare_prefix_const,
            mode=args.reuse_check,
        )
        coverage[variant] = {
            "n_eval_contexts": len(tbl_ev.ctx_order),
            "n_bare_substituted": int(len(sub_rows)),
            "n_reused_from_wcrung_store": int(len(reused_rows)),
            "n_bare_rows_unused": int(n_leg1_rows - len(sub_rows)),
            "reuse_licence_check": reuse,
            "note": (
                "substituted rows took their BARE capture rep; reused rows kept their committed "
                "wildchat-rung rep, which for a single-turn context IS its bare rep (the capture "
                "leg's bit-equality gate; re-verified here via reuse_licence_check). "
                "n_bare_rows_unused counts captured bare rows whose context has no kept DV row."
            ),
        }

        # The bare-render prefix is a CONSTANT vector by construction, so the
        # prefix-variant arm sweep can only produce zero-variance scores and
        # NaN rho. When constancy is VERIFIED, record the null verdict as this
        # variant's result and skip the sweep (the arms would burn a U-pool
        # whitening + map refit + a full transfer solve for guaranteed-NaN rows).
        # When constancy FAILS the sweep DOES run — the arms are then the
        # diagnostic for what varies (the ANOMALY branch).
        if variant == BARE_NULL_KIND and nulls[variant]["constancy"].get("constant"):
            reason = (
                "constant-prefix NULL variant: the bare render's prefix rep is verified CONSTANT "
                "across rows, so every arm score is zero-variance and rho is undefined by "
                "construction. The arm sweep is skipped; null_probe carries the verdict. A "
                "constancy FAILURE would have run the sweep as the anomaly diagnostic."
            )
            skips_all.append({"variant": variant, "arm": "*", "reason": reason})
            frozen_source[variant] = "n/a — degenerate null variant (arm sweep skipped)"
            with ckpt.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "unit_key": unit_key,
                            "rows": [],
                            "skips": [{"variant": variant, "arm": "*", "reason": reason}],
                            "per_layer": [],
                            "frozen_source": frozen_source[variant],
                            "coverage": coverage[variant],
                            "null_probe": nulls[variant],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                fh.flush()
            _log(
                f"[phase=leg1] {behavior} unit {vi + 1}/{len(variants)} {variant} "
                f"SKIP (verified-degenerate null: {nulls[variant]['verdict']})"
            )
            continue

        wh, mapfit, u_label, n_u = _whitening_and_map(args, tbl, layers, variant)
        map_diag[f"{variant}|{u_label}"] = {
            **mapfit.diagnostics,
            "map_source": "refit",
            "n_u": n_u,
        }

        z_tr_w = fits.apply_whitening(tbl.z_by_variant[variant], wh)
        za_tr_w = fits.apply_whitening(tbl.z_ans, wh)
        z_ev_w = fits.apply_whitening(z_bare_raw, wh)
        za_ev_w = fits.apply_whitening(tbl_ev.z_ans, wh)
        rb_w = np.einsum("ld,lde->le", rb, wh.w)
        data = arms.CellData(
            z_ctx=z_tr_w,
            z_ans=za_tr_w,
            dv=tbl.dv,
            rb=rb_w,
            mapfit=mapfit,
            layers=tuple(layers),
        )
        cell = fits.realize_budget_cell(
            tbl.groups, budget_l=budget_l, draw=args.draw, seed=args.seed
        )
        frozen, src, own_rho, own_skips = _frozen_layers_for_roster(
            args, behavior, variant, u_label, roster, layers, data, cell, paths
        )
        frozen_source[variant] = src
        skips_u = list(own_skips)

        prov = {
            "leg": "1",
            "behavior": behavior,
            "variant": variant,
            "regime": args.regime,
            "u_rung": n_u,
            "u_rung_label": u_label,
            "eval_rung": WCRUNG,
            "config": "config_a",
            "render_match": render_match["label"],
            "input_rep": f"bare_{variant}",
            "f_u": None,
            "f_l": None,
        }
        t_tf = time.time()
        scores_ev, arm_skips = arms.run_transfer_cell(
            data,
            cell,
            z_ev_w,
            np.asarray(tbl_ev.dv, dtype=np.float64),
            za_ev=za_ev_w,
            arms=roster,
            device=args.device,
            ridge_folds=(0,),  # the reverse (train-block) fold is discarded
        )
        rows_u, ev_skips = arms.evaluate_transfer(
            scores_ev,
            tbl_ev.dv,
            np.asarray(tbl_ev.row_rungs),
            frozen,
            provenance=prov,
            cell=cell,
            layers=tuple(layers),
            n_boot=n_boot,
            min_n=int(args.min_n),
        )
        skips_u += ev_skips
        skips_u += [
            {"arm": slug, "reason": reason, "variant": variant}
            for slug, reason in sorted(arm_skips.items())
        ]
        dv_ev = np.asarray(tbl_ev.dv, dtype=np.float64)
        per_layer_u = [
            {
                **prov,
                "arm": slug,
                "family": arms.ARM_REGISTRY.get(slug, {}).get("family", "unknown"),
                "rung_kind": "eval_transfer_per_layer",
                "layers": [int(x) for x in layers],
                "rho_per_layer": [
                    float(x) for x in arms.spearman_rows(np.asarray(sc, dtype=np.float64), dv_ev)
                ],
                "frozen_layer_idx": int(frozen[slug]),
                "frozen_layer": int(layers[int(frozen[slug])]),
                "frozen_source": src,
                "rho_per_layer_train_own_pool": own_rho.get(slug),
                "n_eval": int(dv_ev.size),
                "budget_l": budget_l,
                "draw": args.draw,
                "seed": args.seed,
            }
            for slug, sc in sorted(scores_ev.items())
        ]

        with ckpt.open("a", encoding="utf-8") as fh:  # single-line O_APPEND write
            fh.write(
                json.dumps(
                    {
                        "unit_key": unit_key,
                        "rows": rows_u,
                        "skips": skips_u,
                        "per_layer": per_layer_u,
                        "frozen_source": src,
                        "coverage": coverage[variant],
                        "null_probe": nulls[variant],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            fh.flush()
        rows_all += rows_u
        skips_all += skips_u
        per_layer_all += per_layer_u
        _log(
            f"[phase=leg1] {behavior} unit {vi + 1}/{len(variants)} {variant} "
            f"arms={len(scores_ev)} rows={len(rows_u)} transfer={time.time() - t_tf:.0f}s "
            f"elapsed={time.time() - t0:.0f}s"
        )
        del data, z_tr_w, za_tr_w, z_ev_w, za_ev_w, wh, mapfit
        if str(args.device).startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    contrast = committed_contrast_rows(
        paths["wcrung_arms"], variant=variants[0], regime=args.regime
    )
    if contrast.get("available"):
        committed_sha = contrast.get("committed_eval_ctx_ids_sha256")
        contrast["eval_row_set_matches"] = committed_sha == eval_ctx_sha
        contrast["this_run_eval_ctx_ids_sha256"] = eval_ctx_sha
        if not contrast["eval_row_set_matches"]:
            contrast["comparability"] = (
                "NOT comparable: the committed mismatched column scored a DIFFERENT eval context "
                f"set (sha {committed_sha}) than this bare column ({eval_ctx_sha}); compare only "
                "after re-running both on one row set"
            )
        else:
            contrast["comparability"] = (
                "comparable: identical eval context list (sha match) + identical frozen-layer "
                "convention + identical roster and DV"
            )

    for k in ("train_summary", "wcrung_arms", "query_manifest"):
        p = Path(paths[k])
        if p.is_file():
            shas[str(p)] = wca._sha256(p)
    wca._verify_input_shas(shas)
    return {
        "leg": "1",
        "behavior": behavior,
        "rows": rows_all,
        "skips": skips_all,
        "per_layer": per_layer_all,
        "frozen_source": frozen_source,
        "map_diagnostics": map_diag,
        "render_match": render_match,
        "coverage": coverage,
        "null_probe": nulls,
        "committed_contrast": contrast,
        "input_sha256": shas,
        "input_paths": {**{k: str(v) for k, v in paths.items()}, "bareq_store": str(store_dir)},
        "rb": rb_meta,
        "n_train_contexts": len(tbl.ctx_order),
        "n_eval_contexts": len(tbl_ev.ctx_order),
        "eval_ctx_ids_sha256": eval_ctx_sha,
        "budget_l": budget_l,
        "wall_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# leg 2
# ---------------------------------------------------------------------------


def leg2_noop_report(behavior: str, train_prefix_constancy: dict) -> dict:
    """Why sycophancy / hallucination have NO leg-2 bare-fit run (measured)."""
    return {
        "behavior": behavior,
        "leg2": "no-op",
        "reason": (
            "the committed arm4_ridge_ctx on this behavior's train grid ALREADY IS the "
            "bare-query map: its train contexts carry a CONSTANT prefix_end (the template head) "
            "on every row, so they are already bare renders and a dedicated bare-fit ridge is "
            "the identical fit on the identical inputs"
        ),
        "measured_train_prefix_constancy": train_prefix_constancy,
        "capture_leg_agrees": bool(train_prefix_constancy.get("constant")),
    }


def _leg2_eval_blocks(args, behavior, paths, layers, dim, bare_arrays, by_leg, ctx_to_qid, wh):
    """Assemble leg 2's eval columns: the WildChat column + evil's own eval rungs.

    Each block contributes ``(z_bare_w, za_w, dv, rungs)``; a block with no
    bare reps is recorded with a reason (never silently dropped). Evil's eval
    rungs need a query bank built with ``--all-rungs``: the capture leg's
    ``--train-only`` default covers TRAIN contexts only.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits
    from scripts.issue1739_fits import _load_labeled

    blocks: list[dict] = []
    notes: list[dict] = []

    wc_dv, wc_store = Path(paths["wcrung_dv"]), Path(paths["wcrung_store"])
    if wc_dv.is_file() and wc_store.exists() and by_leg["1"]:
        tbl_wc = _load_labeled(wc_store, wc_dv, layers, config="config_b", need_rollout_rows=False)
        block, key_pos = _bare_block(bare_arrays, BARE_KIND, layers, by_leg["1"])
        z_bare, sub, reused = substitute_bare_eval_reps(
            tbl_wc.z_by_variant[BARE_KIND], tbl_wc.ctx_order, block, key_pos
        )
        blocks.append(
            {
                "name": WCRUNG,
                "z": fits.apply_whitening(z_bare, wh),
                "za": fits.apply_whitening(tbl_wc.z_ans, wh),
                "dv": np.asarray(tbl_wc.dv, dtype=np.float64),
                "rungs": [WCRUNG] * len(tbl_wc.ctx_order),
                "n": len(tbl_wc.ctx_order),
                "n_bare_substituted": int(len(sub)),
                "n_reused_from_wcrung_store": int(len(reused)),
            }
        )
    else:
        notes.append(
            {
                "block": WCRUNG,
                "skipped": "no wildchat DV/store or no leg-1 bare rows staged",
                "wcrung_dv": str(wc_dv),
                "wcrung_store": str(wc_store),
            }
        )

    try:
        tbl_own = _load_labeled(
            paths["train_store"],
            paths["train_dv"],
            layers,
            config="config_b",
            need_rollout_rows=False,
        )
    except (RuntimeError, FileNotFoundError, KeyError) as exc:
        notes.append(
            {
                "block": f"{behavior}_own_eval_rungs",
                "skipped": f"eval split unavailable in the train store/DV: "
                f"{type(exc).__name__}: {exc}",
            }
        )
        return blocks, notes

    keep = [
        i
        for i, cid in enumerate(tbl_own.ctx_order)
        if ctx_to_qid.get(cid) is not None and ctx_to_qid[cid] in by_leg["2"]
    ]
    if not keep:
        notes.append(
            {
                "block": f"{behavior}_own_eval_rungs",
                "skipped": (
                    "no eval-split context has a bare rep — the query bank is TRAIN-only "
                    "(capture leg default --train-only); re-capture with --all-rungs to score "
                    "this behavior's own eval rungs"
                ),
                "n_eval_split_contexts": len(tbl_own.ctx_order),
                "eval_rungs_present": sorted(set(tbl_own.row_rungs)),
            }
        )
        return blocks, notes
    rows = [by_leg["2"][ctx_to_qid[tbl_own.ctx_order[i]]] for i in keep]
    z_bare = _stack_layers(bare_arrays, BARE_KIND, layers, rows)
    assert z_bare.shape == (len(layers), len(keep), dim), z_bare.shape
    blocks.append(
        {
            "name": f"{behavior}_own_eval_rungs",
            "z": fits.apply_whitening(z_bare, wh),
            "za": fits.apply_whitening(tbl_own.z_ans[:, keep], wh),
            "dv": np.asarray(tbl_own.dv[keep], dtype=np.float64),
            "rungs": [tbl_own.row_rungs[i] for i in keep],
            "n": len(keep),
            "n_eval_split_contexts": len(tbl_own.ctx_order),
        }
    )
    return blocks, notes


def score_leg2(args, behavior: str) -> dict:  # noqa: C901 — one linear per-behavior pipeline
    """LEG 2: bare -> answer map + bare-fit arms on the train pool, BY-QUERY folds."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits
    from scripts.issue1739_fits import _fit_map, _load_labeled

    wca = _wca()
    paths = _behavior_paths(args, behavior)
    layers = args.layers or list(range(args.n_layers))
    store_dir = resolve_bareq_store(args, behavior, "2")
    missing = [
        f"{k}={paths[k]}"
        for k in ("train_store", "train_dv", "query_manifest")
        if not Path(paths[k]).exists()
    ]
    if not (store_dir / CAPTURE_MANIFEST_NAME).is_file():
        missing.append(f"bareq_store={store_dir}")
    if missing:
        raise FileNotFoundError(
            f"[{behavior}/leg2] missing input(s): {'; '.join(missing)} — stage the train store, "
            f"the train DV and the {QUERY_MANIFEST} query bank before scoring"
        )

    shas = {
        str(paths["train_dv"]): wca._sha256(paths["train_dv"]),
        str(paths["query_manifest"]): wca._sha256(paths["query_manifest"]),
    }
    t0 = time.time()
    ctx_to_qid, bank_digest = load_query_bank(Path(paths["query_manifest"]))
    tbl = _load_labeled(
        paths["train_store"], paths["train_dv"], layers, config="config_a", need_rollout_rows=False
    )
    dim = tbl.z_ans.shape[-1]
    bare_arrays, bare_meta, by_leg, bare_dim = load_bare_store(
        store_dir, layers, kinds=(BARE_KIND, BARE_NULL_KIND)
    )
    if bare_dim != dim:
        raise RuntimeError(f"[{behavior}] bare store dim {bare_dim} != train store dim {dim}")
    if not by_leg["2"]:
        raise RuntimeError(
            f"[{behavior}/leg2] bare store {store_dir} holds 0 leg-2 (query-bank) rows "
            f"({len(bare_meta)} rows total) — capture --leg 2/both first"
        )

    keep = [
        i
        for i, cid in enumerate(tbl.ctx_order)
        if ctx_to_qid.get(cid) is not None and ctx_to_qid[cid] in by_leg["2"]
    ]
    if len(keep) < 2 * int(args.min_n):
        raise RuntimeError(
            f"[{behavior}/leg2] only {len(keep)}/{len(tbl.ctx_order)} train contexts have a bare "
            f"rep (bank covers {bank_digest['n_contexts']} contexts / "
            f"{bank_digest['n_unique_queries']} queries) — too few to fold"
        )
    qids = [ctx_to_qid[tbl.ctx_order[i]] for i in keep]
    bare_rows = [by_leg["2"][q] for q in qids]
    x_bare = _stack_layers(bare_arrays, BARE_KIND, layers, bare_rows)
    assert x_bare.shape == (len(layers), len(keep), dim), x_bare.shape
    y_ans = np.asarray(tbl.z_ans[:, keep], dtype=np.float64)
    dv = np.asarray(tbl.dv[keep], dtype=np.float64)
    _log(
        f"[phase=leg2] {behavior}: n_contexts={len(keep)} n_queries={len(set(qids))} "
        f"contexts_per_query={len(keep) / max(len(set(qids)), 1):.2f} | load={time.time() - t0:.0f}s"
    )

    # Whitening from the U pool keeps the bare reps in the SAME whitened space as
    # every committed column; the MAP is then fit on BARE x (the dedicated fit).
    wh, _u_mapfit, u_label, n_u = _whitening_and_map(args, tbl, layers, BARE_KIND)
    x_w = fits.apply_whitening(x_bare, wh)
    y_w = fits.apply_whitening(y_ans, wh)

    cell = fits.realize_budget_cell(qids, budget_l=len(keep), draw=args.draw, seed=args.seed)
    fold_digest = assert_by_query_folds(cell, qids)
    map_reads = map_reads_by_query_folds(args, x_w, y_w, cell)
    bare_mapfit = _fit_map(args, x_w, y_w) if args.arms_on_bare_map else None

    rb, rb_meta = wca._rb_for_behavior(args, behavior, tbl, layers, dim, paths)
    rb_w = np.einsum("ld,lde->le", rb, wh.w)
    roster = list(args.arms) if args.arms else list(arms.TRANSFER_ARMS)
    n_boot = int(args.n_boot) if args.n_boot else arms.N_BOOT
    data = arms.CellData(
        z_ctx=x_w,
        z_ans=y_w,
        dv=dv,
        rb=rb_w,
        mapfit=bare_mapfit,
        layers=tuple(layers),
    )
    # Leg-2 arms are NEW (bare-fit), so committed train-frozen indices do not
    # apply: frozen layers come from the behavior's OWN bare train pool, in
    # split, over the BY-QUERY folds.
    frozen, own_rho, own_skips = wca.own_pool_frozen_layers(
        data, cell, roster=roster, device=args.device
    )
    frozen = {a: i for a, i in frozen.items() if a in roster}
    skips: list[dict] = [
        {"arm": a, "reason": f"own-pool frozen-layer read: {r}", "leg": "2"}
        for a, r in sorted(own_skips.items())
    ]
    missing_frozen = sorted(set(roster) - set(frozen))
    if missing_frozen:
        skips += [
            {"arm": a, "reason": "no own-pool frozen layer", "leg": "2"} for a in missing_frozen
        ]

    blocks, block_notes = _leg2_eval_blocks(
        args, behavior, paths, layers, dim, bare_arrays, by_leg, ctx_to_qid, wh
    )
    rows_all: list[dict] = []
    per_layer_all: list[dict] = []
    prov_base = {
        "leg": "2",
        "behavior": behavior,
        "variant": BARE_KIND,
        "regime": args.regime,
        "u_rung": n_u,
        "u_rung_label": u_label,
        "config": "config_a",
        "render_match": "bare-fit (by construction: the fit input IS the bare rep)",
        "input_rep": f"bare_{BARE_KIND}",
        "f_u": None,
        "f_l": None,
    }
    for bi, block in enumerate(blocks):
        t_tf = time.time()
        scores_ev, arm_skips = arms.run_transfer_cell(
            data,
            cell,
            block["z"],
            block["dv"],
            za_ev=block["za"],
            arms=roster,
            device=args.device,
            ridge_folds=(0,),
        )
        prov = {**prov_base, "eval_rung": block["name"]}
        rows_u, ev_skips = arms.evaluate_transfer(
            scores_ev,
            block["dv"],
            np.asarray(block["rungs"]),
            frozen,
            provenance=prov,
            cell=cell,
            layers=tuple(layers),
            n_boot=n_boot,
            min_n=int(args.min_n),
        )
        rows_all += rows_u
        skips += ev_skips
        skips += [
            {"arm": slug, "reason": reason, "eval_block": block["name"]}
            for slug, reason in sorted(arm_skips.items())
        ]
        per_layer_all += [
            {
                **prov,
                "arm": slug,
                "family": arms.ARM_REGISTRY.get(slug, {}).get("family", "unknown"),
                "rung_kind": "eval_transfer_per_layer",
                "layers": [int(x) for x in layers],
                "rho_per_layer": [
                    float(x)
                    for x in arms.spearman_rows(np.asarray(sc, dtype=np.float64), block["dv"])
                ],
                "frozen_layer_idx": int(frozen[slug]),
                "frozen_layer": int(layers[int(frozen[slug])]),
                "frozen_source": "own-bare-train-pool-selection (by-query folds)",
                "rho_per_layer_train_own_pool": own_rho.get(slug),
                "n_eval": int(block["dv"].size),
                "budget_l": int(len(keep)),
                "draw": args.draw,
                "seed": args.seed,
            }
            for slug, sc in sorted(scores_ev.items())
            if slug in frozen
        ]
        _log(
            f"[phase=leg2] {behavior} block {bi + 1}/{len(blocks)} {block['name']} "
            f"n={block['n']} rows={len(rows_u)} transfer={time.time() - t_tf:.0f}s"
        )

    for k in ("train_summary",):
        p = Path(paths[k])
        if p.is_file():
            shas[str(p)] = wca._sha256(p)
    wca._verify_input_shas(shas)
    return {
        "leg": "2",
        "behavior": behavior,
        "rows": rows_all,
        "skips": skips,
        "per_layer": per_layer_all,
        "query_bank": bank_digest,
        "folds": fold_digest,
        "mapping_baselines": map_reads,
        "arm_map_fit_scope": (
            "arm reads use a map fit on the FULL bare train pool, so the IN-SPLIT arm reads are "
            "transductive in (x, y) but NEVER in the DV; the transfer columns are fully "
            "out-of-sample. The mapping_baselines block above refits per BY-QUERY fold and is "
            "the OOF map-quality headline."
        )
        if args.arms_on_bare_map
        else "arms ran with mapfit=None (--no-arms-on-bare-map): map-consuming arms are SKIPPED",
        "eval_blocks": [{k: v for k, v in b.items() if k not in ("z", "za", "dv")} for b in blocks],
        "eval_block_notes": block_notes,
        "frozen_source": {BARE_KIND: "own-bare-train-pool-selection (by-query folds)"},
        "input_sha256": shas,
        "input_paths": {**{k: str(v) for k, v in paths.items()}, "bareq_store": str(store_dir)},
        "rb": rb_meta,
        "n_fit_contexts": len(keep),
        "n_fit_queries": len(set(qids)),
        "wall_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# summary assembly
# ---------------------------------------------------------------------------


def dv_construct_meta(behavior: str) -> dict:
    """DV construct + the standing caveats every bare-query summary carries.

    The judged DVs are the COMMITTED ones — this leg re-judges nothing — so the
    wildchat rung's own three rung-level caveats ride through unchanged, plus
    this round's bare-render caveats and the verbatim analogy caveat.
    """
    base = dict(_wca().dv_construct_meta(behavior))
    base["caveats"] = [
        *base.get("caveats", []),
        "BARE-RENDER INPUT: the labels and answer activations are unchanged (judged on rollouts "
        "generated under the FULL context); only the PREDICTOR INPUT is re-rendered bare, so this "
        "round measures input-side render sensitivity, not a different behavior.",
        ANALOGY_CAVEAT,
    ]
    return base


def write_behavior_summary(args, behavior: str, legs: dict, *, commit: str, env: dict) -> Path:
    """One ``all_arms_spearman.json`` per behavior (wcrung output schema)."""
    from explore_persona_space.experiments.issue_1739 import arms

    leg1, leg2 = legs.get("1"), legs.get("2")
    rows = (leg1 or {}).get("rows", []) + (leg2 or {}).get("rows", [])
    per_layer = (leg1 or {}).get("per_layer", []) + (leg2 or {}).get("per_layer", [])
    skips = (leg1 or {}).get("skips", []) + (leg2 or {}).get("skips", [])
    out = args.out_root / behavior / "all_arms_spearman.json"
    mapping_baselines = {
        "leg1": {
            "applicable": False,
            "reason": (
                "leg 1 FITS NO MAP — it APPLIES committed train-fit maps/ridges to bare reps and "
                "scores rho, so the standing identity+learned-bias / kNN-retrieval pair (which "
                "attaches to a FITTED v_X -> v_Y map) has nothing to attach to. NOTE: the arm "
                "slug 'arm3_identity_bias' is an activation->DV SCORING arm and is NOT the "
                "mapping_baselines identity+bias map baseline."
            ),
        },
        "leg2": (leg2 or {}).get("mapping_baselines")
        or {"applicable": False, "reason": "leg 2 did not run for this behavior"},
    }
    arms.write_summary(
        [],  # no in-split cell records: this leg emits ONLY bare-input transfer columns
        out,
        meta={
            "mode": "bareq_transfer",
            "behavior": behavior,
            "rung": RUNG,
            "legs_run": sorted(k for k in legs if k in ("1", "2")),
            "config": "config_a",
            "regimes": [args.regime],
            "variants": list(args.variants),
            "arms": sorted(args.arms) if args.arms else sorted(arms.TRANSFER_ARMS),
            "layers": [int(x) for x in (args.layers or list(range(args.n_layers)))],
            "map_kind": args.map_kind,
            "map_source": "refit-in-process",
            "u_sizes": [args.u_size],
            "draw": args.draw,
            "seed": args.seed,
            "transfer_min_n": int(args.min_n),
            "render_match": (leg1 or {}).get("render_match"),
            "leg1_coverage": (leg1 or {}).get("coverage"),
            "leg1_null_probe": (leg1 or {}).get("null_probe"),
            "leg1_committed_contrast": (leg1 or {}).get("committed_contrast"),
            "leg2_folds": (leg2 or {}).get("folds"),
            "leg2_query_bank": (leg2 or {}).get("query_bank"),
            "leg2_eval_blocks": (leg2 or {}).get("eval_blocks"),
            "leg2_eval_block_notes": (leg2 or {}).get("eval_block_notes"),
            "leg2_arm_map_fit_scope": (leg2 or {}).get("arm_map_fit_scope"),
            "leg2_noop": legs.get("2_noop"),
            "mapping_baselines": mapping_baselines,
            "frozen_layer_source": {
                **((leg1 or {}).get("frozen_source") or {}),
                **((leg2 or {}).get("frozen_source") or {}),
            },
            "rb": (leg1 or leg2 or {}).get("rb"),
            "dv": dv_construct_meta(behavior),
            "n_contexts": (leg1 or {}).get("n_eval_contexts"),
            "n_train_contexts": (leg1 or leg2 or {}).get("n_train_contexts"),
            "n_fit_contexts": (leg2 or {}).get("n_fit_contexts"),
            "eval_ctx_ids_sha256": (leg1 or {}).get("eval_ctx_ids_sha256"),
            "input_paths": {
                **((leg1 or {}).get("input_paths") or {}),
                **((leg2 or {}).get("input_paths") or {}),
            },
            "input_sha256": {
                **((leg1 or {}).get("input_sha256") or {}),
                **((leg2 or {}).get("input_sha256") or {}),
            },
            "caveats": [ANALOGY_CAVEAT],
            "git_commit": commit,
            "env_versions": env,
            "wall_s": round(
                sum(float((legs.get(k) or {}).get("wall_s") or 0.0) for k in ("1", "2")), 1
            ),
            "judge_called": False,
        },
        extra={
            "transfer_rows": rows,
            "transfer_skips": skips,
            "per_layer_rows": per_layer,
            "n_transfer_rows": len(rows),
            "n_per_layer_rows": len(per_layer),
        },
    )
    diag = {k: v for k, v in ((leg1 or {}).get("map_diagnostics") or {}).items()}
    (args.out_root / behavior / "map_diagnostics.json").write_text(json.dumps(diag, indent=1))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=list(BEHAVIORS))
    ap.add_argument("--legs", nargs="+", default=["1", "2"], choices=["1", "2"])
    ap.add_argument(
        "--variants",
        nargs="+",
        default=[BARE_KIND, BARE_NULL_KIND],
        choices=[BARE_KIND, BARE_NULL_KIND],
        help=f"LEG 1 input reps. {BARE_NULL_KIND!r} is the built-in constant-prefix NULL: its "
        "arm sweep is SKIPPED once constancy is verified (null_probe carries the verdict) and "
        f"runs only on a constancy FAILURE. LEG 2 always fits on {BARE_KIND!r} (the bare rep IS "
        "its only input), so this flag does not affect it.",
    )
    ap.add_argument("--regime", default="e1", choices=("e1", "e2", "e2p"))
    ap.add_argument(
        "--arms", nargs="+", default=None, help="roster subset (default: TRANSFER_ARMS)"
    )
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="default: all --n-layers")
    ap.add_argument("--n-layers", type=int, default=FULL_GRID_N_LAYERS)
    ap.add_argument("--u-size", default="full", help="U-pool rung: int or 'full'")
    ap.add_argument("--budget", type=int, default=None, help="train rows (default: whole table)")
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=3, help="per-rung row floor for a Spearman read")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--map-kind", default="linear", choices=("linear", "mlp", "kernel"))
    ap.add_argument("--mlp-map-width", type=int, default=None)
    ap.add_argument("--krr-map-centers", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--rb-source",
        default="auto",
        choices=("auto", "bank", "extract"),
        help="E1 direction: persisted fp16 bank, re-extract from the E1 store, or auto",
    )
    ap.add_argument(
        "--reuse-check",
        default="hard",
        choices=("hard", "report", "off"),
        help="verify reused (non-recaptured) wildchat rows really are bare renders",
    )
    ap.add_argument(
        "--no-arms-on-bare-map",
        dest="arms_on_bare_map",
        action="store_false",
        default=True,
        help="leg 2: run arms with mapfit=None (map-consuming arms SKIP) instead of the "
        "full-pool bare map",
    )
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    ap.add_argument("--tensors-root", type=Path, default=DEFAULT_TENSORS_ROOT)
    ap.add_argument(
        "--store-root",
        type=Path,
        default=DEFAULT_STORE_ROOT,
        help="dir holding the staged {behavior}_labeling / _extraction / wcrung / bareq stores",
    )
    ap.add_argument(
        "--bareq-store",
        type=Path,
        default=None,
        help="the bare capture store (or a parent holding bareq/ or bareq_<behavior>/); "
        "default <--store-root>/bareq_capture_store",
    )
    ap.add_argument(
        "--query-manifest",
        type=Path,
        default=None,
        help=f"the capture leg's {QUERY_MANIFEST} (default <--out-root>/{QUERY_MANIFEST}) — "
        "the BY-QUERY fold key",
    )
    ap.add_argument("--train-store", type=Path, default=None)
    ap.add_argument(
        "--train-dv-root",
        type=Path,
        default=None,
        help="dir holding <behavior>/labeling.json (default <--main-root>/dv_dataset)",
    )
    ap.add_argument("--train-dv-json", type=Path, default=None)
    ap.add_argument("--e1-store", type=Path, default=None)
    ap.add_argument(
        "--wcrung-store",
        type=Path,
        default=None,
        help=f"the ONE shared {WCRUNG} capture store (shared across behaviors BY DESIGN)",
    )
    ap.add_argument("--wcrung-dv-json", type=Path, default=None)
    ap.add_argument(
        "--wcrung-arms-json",
        type=Path,
        default=None,
        help="committed wildchat-rung arm results = the render-MISMATCHED contrast column "
        f"(default <--main-root>/{WCRUNG}/arm_results/<behavior>/all_arms_spearman.json)",
    )
    ap.add_argument("--train-summary", type=Path, default=None)
    ap.add_argument(
        "--u-store",
        type=Path,
        default=None,
        help="#1092 U-pool store (default <--store-root>/u_store, staged on demand)",
    )
    ap.add_argument(
        "--force-own-pool-frozen",
        action="store_true",
        help="leg 1: ignore committed train summaries and select frozen layers on each "
        "behavior's own train pool (REQUIRED for a reduced --layers set)",
    )
    ap.add_argument("--allow-overwrite-committed", action="store_true")
    ap.add_argument(
        "--import-check", action="store_true", help="resolve deferred imports and exit 0"
    )
    args = ap.parse_args(argv)
    if len(set(args.behaviors)) != len(args.behaviors):
        ap.error("--behaviors must be unique")
    if len(set(args.variants)) != len(args.variants):
        ap.error("--variants must be unique")
    if len(set(args.legs)) != len(args.legs):
        ap.error("--legs must be unique")
    # Resolve AFTER parsing so --store-root / --main-root move the dependents too.
    if args.u_store is None:
        args.u_store = args.store_root / "u_store"
    if args.train_dv_root is None:
        args.train_dv_root = args.main_root / "dv_dataset"
    single_only = (
        "train_store",
        "train_dv_json",
        "e1_store",
        "wcrung_dv_json",
        "wcrung_arms_json",
        "train_summary",
    )
    if len(args.behaviors) > 1:
        set_flags = [
            f"--{f.replace('_', '-')}" for f in single_only if getattr(args, f) is not None
        ]
        if set_flags:
            ap.error(
                f"{', '.join(set_flags)} name ONE behavior's input but --behaviors has "
                f"{len(args.behaviors)} ({', '.join(args.behaviors)}); use the per-behavior roots "
                "(--store-root / --train-dv-root / --main-root / --out-root) or one behavior/run"
            )
    return args


def _import_check() -> int:
    """Resolve every deferred import on the REAL branch, in its OWN function.

    Deliberately not inline in ``main()``: an ``import X`` is a binding, so an
    inline block would make X a function-wide local of ``main()`` and shadow any
    module-level symbol of the same name on the normal path (the #1739 wcrung
    ``capture`` UnboundLocalError). Pinned by the shadow test.
    """
    from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
        identity_bias_predict,
        knn_retrieval,
    )
    from explore_persona_space.experiments.issue_1739 import (  # noqa: F401
        arms,
        fits,
        store_io,
    )
    from explore_persona_space.experiments.issue_1739.fits import (  # noqa: F401
        apply_map,
        apply_whitening,
        fit_whitening,
        map_diagnostics,
        r2_pooled,
        realize_budget_cell,
    )
    from explore_persona_space.orchestrate.env import load_dotenv  # noqa: F401
    from scripts.issue1739_fits import (  # noqa: F401
        RunSpec,
        _extract_rb,
        _fit_map,
        _load_labeled,
        _u_pool_for_spec,
        arrays_dim,
    )
    from scripts.issue1739_wcrung_arms import (  # noqa: F401
        _assert_committed_frozen_indexable,
        _assert_no_judge_modules,
        _git_tracked,
        _rb_for_behavior,
        _sha256,
        _sha256_text,
        _verify_input_shas,
        dv_construct_meta,
        modal_frozen_layers,
        own_pool_frozen_layers,
    )

    _wca()._assert_no_judge_modules("after --import-check imports")
    print("[bareq-score] import-check OK", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


def main(argv: list[str] | None = None) -> int:  # noqa: C901 — one linear dispatch block
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args(argv)
    wca = _wca()
    wca._assert_no_judge_modules("at entry")

    if args.import_check:
        sys.exit(_import_check())

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()  # HF token for the U-pool staging leg

    out_paths = [args.out_root / b / "all_arms_spearman.json" for b in args.behaviors]
    _assert_outputs_safe(out_paths, out_root=args.out_root, allow=args.allow_overwrite_committed)

    commit, env = _git_commit(), _env_versions()
    failures: list[dict] = []
    results: list[dict] = []
    t_all = time.time()
    for behavior in args.behaviors:
        legs: dict[str, object] = {}
        try:
            if "1" in args.legs:
                legs["1"] = score_leg1(args, behavior)
            if "2" in args.legs:
                if behavior in LEG2_BEHAVIORS:
                    legs["2"] = score_leg2(args, behavior)
                else:
                    constancy = ((legs.get("1") or {}) or {}).get("render_match", {}).get(
                        "train_prefix_constancy"
                    ) or {"label": f"{behavior} train prefix reps", "note": "leg 1 did not run"}
                    legs["2_noop"] = leg2_noop_report(behavior, constancy)
            out = write_behavior_summary(args, behavior, legs, commit=commit, env=env)
        except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
            # Per-behavior isolation: one behavior's missing/incoherent input must
            # not discard the others' completed results. Recorded loudly here AND
            # surfaced by the nonzero exit below.
            failures.append({"behavior": behavior, "error": f"{type(exc).__name__}: {exc}"})
            _log(f"{behavior} FAILED: {type(exc).__name__}: {exc}")
            continue
        results.append(
            {
                "behavior": behavior,
                "legs": sorted(k for k in legs if k in ("1", "2")),
                "leg2_noop": "2_noop" in legs,
                "n_transfer_rows": sum(
                    len((legs.get(k) or {}).get("rows", [])) for k in ("1", "2")
                ),
                "summary": str(out),
            }
        )
        _log(f"{behavior} done: {results[-1]['n_transfer_rows']} transfer rows -> {out}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / SENTINEL_NAME).write_text(
        json.dumps(
            {
                "leg": "bareq_score",
                "rung": RUNG,
                "behaviors": results,
                "legs_requested": sorted(args.legs),
                "variants": list(args.variants),
                "n_layers": len(args.layers or list(range(args.n_layers))),
                "leg2_behaviors": list(LEG2_BEHAVIORS),
                "caveats": [ANALOGY_CAVEAT],
                "git_commit": commit,
                "env_versions": env,
                "judge_called": False,
                "wall_s": round(time.time() - t_all, 1),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=1,
        )
    )
    wca._assert_no_judge_modules("at exit")
    _log(
        f"all done in {time.time() - t_all:.0f}s "
        f"({len(results)}/{len(args.behaviors)} behaviors scored)"
    )
    if failures:
        (args.out_root / FAILURES_NAME).write_text(json.dumps(failures, indent=1))
        for f in failures:
            print(f"[bareq-score] FAILED {f['behavior']}: {f['error']}", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(2)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
