#!/usr/bin/env python
# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #742 Stage 0 step 2 — judge-rerun variance term (plan v7 §4 Stage-0 step 2).

The ONLY non-CPU spend. Rollout-splitting captures GENERATION stochasticity but NOT
JUDGE stochasticity (the judge labels the SAME completion differently on reruns). The
full completions live ONLY on the HF data repo at DISTINCT per-genre paths (MF2); we
snapshot them into an issue-owned dir with a sha256 + a fail-loud shortfall check
(``snapshot_raw_completions``), then re-judge ``R_rerun≥2×`` the read-out behaviors
across BOTH genres (Option α, MF4) via the Anthropic Batch API
(``eval.batch_judge.judge_completions_batch`` through
``dc.judge_column_via_batch_judge`` — never a hand-rolled poller; plan v9 §4 Stage-0
step 2 + §11 row 8), and decompose ``Var(E0) = Var_gen + Var_judge + Var_sig``.

The honest ceiling folds the judge term:
  ``√(r_yy_honest) = √( Var_signal / (Var_signal + Var_judge + Var_generation) )``.

Writes ``eval_results/issue_742/stage0_judge_variance.json``.

NOTE on content hygiene: this script digests harmful-content raw completions by
REFERENCE only (path + sha256 + count + judged score) — it never prints completion
text. Per CLAUDE.md "Content hygiene for harmful-content datasets".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---- graceful-shutdown flag (Layer-2 SIGTERM handler) --------------------- #
# earlyoom (VM memory pressure) and orchestrator teardown both SIGTERM the process.
# The handler sets this flag; the outer (genre, behavior) loop checks it BETWEEN cells
# and stops after the current cell's partial file is persisted. Once the loop stops early,
# the canonical-write completeness guard (``_assert_all_requested_cells_complete``, BLOCKER
# SIGTERM-canonical-file-corruption) RAISES so ``run()`` never returns a "complete" result
# for a partial run — the process exits NON-ZERO and the canonical
# ``stage0_judge_variance.json`` is NEVER written on an interrupted run. All completed
# cells' partials + the persistent per-cell dispatch dirs (Layer-1 fix in
# dc.judge_column_via_batch_judge) survive, so the next launch RESUMES the outstanding
# cells and writes the canonical file cleanly. If SIGTERM arrives MID-cell, Python still
# exits (default); the same partials + dispatch dirs survive for the same resume.
_SHUTDOWN_REQUESTED = {"flag": False}


def _install_sigterm_handler() -> None:
    """Install a SIGTERM handler that requests a graceful between-cell exit (Layer 2).

    Logs a message and flips ``_SHUTDOWN_REQUESTED`` so the outer loop stops after the
    current cell finishes + is checkpointed — rather than leaving the process in an
    undefined mid-cell state. Idempotent; skipped when not on the main thread (signal
    handlers can only be installed there — e.g. under a test harness).
    """
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError):  # not main thread / unsupported platform
        logger.warning("could not install SIGTERM handler (non-main thread?)")


def _on_sigterm(signum: int, frame) -> None:
    _SHUTDOWN_REQUESTED["flag"] = True
    logger.warning(
        "received SIGTERM (signal %d), exiting after the current cell completes; "
        "per-cell partials + the persistent dispatch dir survive for resume",
        signum,
    )


def _atomic_write_json(path: Path, obj: object) -> None:
    """Write ``obj`` as JSON to ``path`` atomically (write-temp + os.replace).

    A crash mid-write must never leave a half-written partial that a later resume would
    mis-read as complete. ``os.replace`` is atomic on the same filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _partial_dir(out_dir: Path) -> Path:
    """Per-cell checkpoint dir: ``<out_dir>/stage0_judge_variance_partial/`` (Layer 2)."""
    return out_dir / "stage0_judge_variance_partial"


def _partial_path(out_dir: Path, genre: str, behavior: str) -> Path:
    """Per-(genre, behavior) partial-result file path (Layer 2 checkpoint)."""
    return _partial_dir(out_dir) / f"{genre}__{behavior}.json"


def _merge_partial_cells(out_dir: Path) -> dict[str, dict]:
    """Fold every persisted per-cell partial into the nested ``judge_variance`` dict.

    Reads ALL ``<out_dir>/stage0_judge_variance_partial/<genre>__<behavior>.json`` files
    (this run's freshly-written cells AND any prior crashed run's), so a resumed run that
    only re-judged the remaining cells still emits the COMPLETE
    ``{genre: {behavior: decomposition}}`` result. A malformed partial (mid-crash write the
    atomic-write should have prevented, or a hand-edit) is skipped with a warning rather than
    crashing the whole aggregation — the cell simply re-runs on the next launch.
    """
    merged: dict[str, dict] = {}
    pdir = _partial_dir(out_dir)
    if not pdir.exists():
        return merged
    for f in sorted(pdir.glob("*.json")):
        try:
            obj = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("issue742 judge-rerun: skipping malformed partial %s", f)
            continue
        genre = obj.get("genre")
        behavior = obj.get("behavior")
        result = obj.get("result")
        if not (isinstance(genre, str) and isinstance(behavior, str) and isinstance(result, dict)):
            logger.warning("issue742 judge-rerun: skipping partial %s with unexpected shape", f)
            continue
        merged.setdefault(genre, {})[behavior] = result
    return merged


_PARTIAL_RESULT_KEYS = frozenset(
    {"var_total", "var_judge", "var_generation", "var_signal", "sqrt_r_yy_honest"}
)


def _partial_is_valid(path: Path) -> bool:
    """True iff a per-cell partial file is well-formed AND carries a full decomposition.

    CONCERN corrupt-partial-skip-forever (task #742 r12): the startup skip-decision must
    NOT treat a stale wrong-shape / truncated partial as "completed" — ``_merge_partial_cells``
    later DROPS a malformed partial, so a partial that is skipped-forever at startup but
    dropped at merge would silently vanish from the aggregate. This mirrors the merge-step
    shape check EXACTLY: valid JSON, top-level ``genre``/``behavior`` strings + ``result``
    dict, AND ``result`` carrying every key ``_decompose_variance`` emits. Any deviation →
    invalid → the caller quarantines + re-runs the cell.
    """
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(obj, dict):
        return False
    genre = obj.get("genre")
    behavior = obj.get("behavior")
    result = obj.get("result")
    if not (isinstance(genre, str) and isinstance(behavior, str) and isinstance(result, dict)):
        return False
    # every _decompose_variance output key must be present (a partial missing them is a
    # wrong-shape / pre-decomposition write that the merge step would drop).
    return _PARTIAL_RESULT_KEYS.issubset(result.keys())


def _quarantine_partial(path: Path) -> None:
    """Move a corrupt/wrong-shape partial aside so the cell re-runs (never skipped-forever).

    Renames the invalid partial to ``<name>.corrupt-<n>`` in the same dir (kept for
    forensics, not deleted) so ``partial.exists()`` is False on the next skip check and the
    cell is re-judged. If the rename itself fails (e.g. a racing writer), unlink as a last
    resort — the cell MUST NOT stay skipped with a bad partial on disk.
    """
    for n in range(1, 1000):
        cand = path.with_suffix(path.suffix + f".corrupt-{n}")
        if not cand.exists():
            try:
                os.replace(path, cand)
            except OSError:
                try:
                    path.unlink()
                except OSError:
                    logger.warning("issue742 judge-rerun: could not quarantine %s", path)
            logger.warning(
                "issue742 judge-rerun: quarantined corrupt/wrong-shape partial %s "
                "(cell will re-run)",
                path,
            )
            return


def _assert_all_requested_cells_complete(
    out_dir: Path,
    *,
    requested_cells: list[tuple[str, str]],
    empty_cells: set[tuple[str, str]],
    shutdown_requested: bool,
) -> None:
    """Fail LOUD before the canonical write if any requested cell is missing (BLOCKER 2).

    SIGTERM canonical-file corruption (task #742 r12): the SIGTERM handler breaks the cell
    loop, and without this guard ``run()`` merges whatever partials exist and returns
    normally, so ``main()`` writes the CANONICAL ``stage0_judge_variance.json`` (the file
    downstream ``issue742_reliability.py`` reads) at exit code 0 even when only a SUBSET of
    cells completed — silent-partial-result contamination. This asserts every requested
    ``(genre, behavior)`` cell that produced data has a valid partial on disk before the
    canonical file is written; on any shortfall (or if SIGTERM was requested at all) it
    raises ``RuntimeError`` so ``main()`` never reaches the canonical write and the process
    exits non-zero — downstream then re-runs (crash-safe resume completes the missing
    cells). A genuinely EMPTY cell (no snapshot data — a reportable, non-crash condition) is
    exempt from the partial requirement.
    """
    missing = [
        cell
        for cell in requested_cells
        if cell not in empty_cells and not _partial_path(out_dir, *cell).exists()
    ]
    if shutdown_requested or missing:
        raise RuntimeError(
            "issue742 judge-rerun INCOMPLETE — refusing to write the canonical "
            "stage0_judge_variance.json (BLOCKER SIGTERM-canonical-file-corruption). "
            f"shutdown_requested={shutdown_requested}; "
            f"missing cells (requested, non-empty, no partial): {sorted(missing)}. "
            "Re-run to resume the outstanding cells; the canonical file is written ONLY "
            "when every requested cell is checkpointed."
        )


def _stable_cell_seed(seed: int, genre: str, behavior: str, ctx: str) -> int:
    """Cross-process-STABLE per-cell J-sampling seed (BLOCKER nondeterministic-sampling).

    Python's built-in ``hash()`` is salted per interpreter process (PYTHONHASHSEED),
    so ``abs(hash((seed, genre, behavior, ctx)))`` produces DIFFERENT values in
    different processes — a fixed ``--seed`` would NOT reproduce the same J=20 sample
    across runs. A sha256 digest of the key bytes is deterministic everywhere; we take
    the first 4 bytes as a uint32 seed for ``numpy.random.default_rng``.
    """
    key = f"{seed}\0{genre}\0{behavior}\0{ctx}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

from issue404_common import reproducibility_metadata  # noqa: E402

from explore_persona_space.analysis import issue_742_decoding_ceiling as dc  # noqa: E402

OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_742"
DATA_REPO = "superkaiba1/explore-persona-space-data"
JUDGE_MODEL = "claude-sonnet-4-5-20250929"


def _hf_download_fn(*, repo_id: str, filename: str, repo_type: str, **kwargs) -> str:
    """Per-file ``hf_hub_download`` shim threaded into ``snapshot_raw_completions``."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


def _decompose_variance(
    rerun_rates: list[np.ndarray],
    gen_first_half: np.ndarray,
    gen_second_half: np.ndarray,
) -> dict[str, float]:
    """Decompose Var(E0) into judge / generation / signal components (plan §4 step 3).

    The nested decomposition ``Var(E0) = Var_signal + Var_generation + Var_judge``
    needs the noise terms as WITHIN-context variances (averaged across contexts),
    NOT the between-context variance of a noisy per-context summary — the BLOCKER
    the prior round's ``np.var(gen_rates)`` carried (between-context variance of the
    first-half mean rates, dominated by SIGNAL, the very term we want to RECOVER —
    it routinely exceeded ``var_total`` and collapsed ``var_signal`` to 0).

    * ``rerun_rates``: per-context judged rate from each of R judge reruns (the SAME
      completions, re-judged). ``Var_judge`` = the WITHIN-context across-rerun
      variance, AVERAGED across contexts (this term was already correct).
    * ``gen_first_half`` / ``gen_second_half``: per-context rates from TWO DISJOINT
      generation-half subsets of each context's completions, both judged under ONE
      pass. For each context the within-context generation variability of the
      FULL-sample mean rate is estimated by ``(p1 − p2)² / 4`` — the unbiased
      generation-noise estimator at the full-sample grain (Monte-Carlo-verified
      §14 test): two disjoint halves are independent estimates of the same context
      rate, so ``Var(p1 − p2) = 2·σ²_half`` ⇒ ``σ²_half = (p1−p2)²/2``, and the full
      sample has 2× the completions so its generation variance is ``σ²_half / 2 =
      (p1−p2)² / 4``. ``Var_generation`` = that estimate AVERAGED across contexts.
      With NO generation noise (both halves share the context signal exactly) this
      is ≈ 0, as it must be — NOT ``Var_C(first_half)``.

    ``Var_total`` = between-context variance of the full (judge-averaged) mean rate.
    ``Var_signal = Var_total − Var_generation − Var_judge`` (clamped non-negative).
    """
    R = np.stack(rerun_rates, axis=0)  # (n_rerun, n_contexts)
    # judge errors surface as NaN per (rerun, context); drop contexts with any NaN so
    # the variance terms are computed on cleanly-judged contexts only (fail-loud upstream
    # already guarantees ≥J completions; a NaN here is a transport/parse miss).
    ok = ~np.any(~np.isfinite(R), axis=0)
    R = R[:, ok]
    var_judge = float(np.mean(np.var(R, axis=0))) if R.shape[1] else 0.0  # within-context var
    mean_rate = R.mean(axis=0) if R.shape[1] else np.array([])
    var_total = float(np.var(mean_rate)) if mean_rate.size else 0.0

    # Var_generation: the WITHIN-context generation-noise variance of the full-sample
    # mean rate, estimated per context from the two disjoint judged halves and averaged
    # across contexts. Align the half vectors to the same cleanly-judged contexts as R,
    # and drop any context whose half-judge returned NaN.
    p1 = np.asarray(gen_first_half, dtype=float)
    p2 = np.asarray(gen_second_half, dtype=float)
    if p1.shape == ok.shape and p2.shape == ok.shape:
        p1 = p1[ok]
        p2 = p2[ok]
    pair_ok = np.isfinite(p1) & np.isfinite(p2)
    p1 = p1[pair_ok]
    p2 = p2[pair_ok]
    # full-sample generation variance per context = (p1 - p2)^2 / 4 (see docstring);
    # averaged across contexts -> the population Var_generation.
    var_gen = float(np.mean((p1 - p2) ** 2) / 4.0) if p1.size else 0.0

    var_signal = max(0.0, var_total - var_judge - var_gen)
    denom = var_signal + var_judge + var_gen
    sqrt_r_yy_honest = float(np.sqrt(var_signal / denom)) if denom > 1e-12 else 0.0
    return {
        "var_total": var_total,
        "var_judge": var_judge,
        "var_generation": var_gen,
        "var_signal": var_signal,
        "sqrt_r_yy_honest": sqrt_r_yy_honest,
    }


CANONICAL_CONTEXTS_PER_GENRE = 50  # plan §11 row 8: 50 context_ids/genre (Betley + UltraChat)


def _resolve_context_counts(genres: list[str], max_contexts: int | None) -> dict[str, int]:
    """Per-genre context count for the transport-deviation call estimate (CONCERN fix).

    Prefers the REAL #658 per-genre ``context_ids`` length (via ``dc.load_inputs``) so a
    genre with a non-standard context count is counted exactly; falls back to the
    canonical 50 when the on-disk tensors are absent (smoke / counting-judge paths,
    where the estimate must still be computed without crashing). ``max_contexts``
    trims each genre's count for smoke parity with the snapshot loop.
    """
    counts: dict[str, int] = {}
    for genre in genres:
        try:
            n = len(dc.load_inputs(genre, repo_root=PROJECT_ROOT).context_ids)
        except Exception:
            n = CANONICAL_CONTEXTS_PER_GENRE
        if max_contexts is not None:
            n = min(n, max_contexts)
        counts[genre] = int(n)
    return counts


def _judge_reruns_for_cell(
    *,
    genre: str,
    behavior: str,
    snapshot_dir: Path,
    r_rerun: int,
    j_completions: int,
    seed: int,
    judge_fn: Callable[..., dict] | None = None,
    observed_routes_sink: set[str] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Run R PER-BEHAVIOR judge passes over a cell's snapshotted completions.

    Reads every snapshotted ``<context>__<behavior>.json`` for ``genre`` (each = one
    context), DETERMINISTICALLY samples exactly ``j_completions`` completions per
    context (BLOCKER-fix judge-rerun-j-sampling — NEVER all completions; that
    balloons the batch past the registered ~16k calls), then re-judges them
    ``r_rerun`` times via the CORRECT PER-BEHAVIOR judge construct
    (``dc.per_behavior_judge_rate`` → the Anthropic **Batch API**
    ``dc.judge_column_via_batch_judge`` with the behavior's own rubric + ``c["text"]``;
    BLOCKER-fix judge-rerun-wrong-judge-construct + judge-rerun-completion-key-crash +
    judge-rerun-transport-must-use-batch). Returns:

      * ``rerun_rates``: list of length ``r_rerun``; each a per-context judged-rate
        vector (the SAME ``judged_rate`` construct #658 used; across-rerun variance =
        ``Var_judge``),
      * ``gen_first_half`` / ``gen_second_half``: per-context judged rates of TWO
        DISJOINT generation-half subsets of each context's J completions (one judge
        pass each). ``_decompose_variance`` reads them as a (p1, p2) pair to estimate
        the WITHIN-context generation variance ``(p1−p2)²/4`` — NOT a between-context
        variance of a single half-mean (the BLOCKER the prior round carried).

    Only judged scores are surfaced — completion TEXT is never returned or logged
    (CLAUDE.md content hygiene). Returns ``([], empty, empty)`` when no snapshot is
    present.
    The J=20 deterministic sampling is keyed on ``(seed, genre, behavior, context)``
    via :func:`_stable_cell_seed` (a cross-process-stable sha256 digest, NOT Python's
    salted ``hash()``) so a re-run reproduces the same sample (plan §11 row 8 + §9).

    ``judge_fn`` is a test-injection hook threaded into ``dc.per_behavior_judge_rate``
    (signature ``(col_id, gen, model) -> dict``); when None the real per-behavior
    Anthropic Batch-API dispatch (``dc.judge_column_via_batch_judge``) is used. A
    deterministic counting stub lets the live (non-dry) judge-rerun path be
    smoke-tested without API spend (BLOCKER judge-rerun-smoke-dry-run-only).
    """
    cell_dir = snapshot_dir / genre
    files = sorted(cell_dir.glob(f"*__{behavior}.json"))
    if not files:
        return [], np.array([]), np.array([])

    # Per context: load the gen file and deterministically down-sample to exactly J
    # completions ONCE (the same J completions are re-judged across all R reruns, so
    # across-rerun variance isolates JUDGE noise, not a re-sampling artifact).
    per_context_obj: dict[str, dict] = {}
    for f in files:
        obj = json.loads(f.read_text())
        ctx = obj.get("context_id", f.stem)
        cell_seed = _stable_cell_seed(seed, genre, behavior, ctx)
        per_context_obj[ctx] = dc.sample_completions_for_judge(
            obj, j_completions=j_completions, seed=cell_seed
        )

    ctx_ids = sorted(per_context_obj)

    # Observed transport routes across all real dispatches in this cell — the production
    # path records ``routing_path`` per call (DERIVED from the persisted RoutingDecision
    # in dc.judge_column_via_batch_judge). build_result reads this to set
    # ``actual_transport`` from the OBSERVED route, never a hard-coded constant
    # (BLOCKER judge-rerun-transport-routes-sync-below-threshold).
    observed_routes: set[str] = observed_routes_sink if observed_routes_sink is not None else set()

    def _judge_rate(objs_by_ctx: dict[str, dict], *, rerun_idx: int) -> np.ndarray:
        # PER-BEHAVIOR judged rate (NOT the default alignment judge): each context's
        # sampled completions are judged with the behavior's own #658 rubric.
        # ``rerun_idx`` threads into the real Batch dispatch's per-cell state-dir key
        # (BLOCKER judge-rerun-var-judge-collapse) so each of the R reruns submits an
        # INDEPENDENT batch over the SAME completions — across-rerun variance = Var_judge,
        # NOT a shared-cache collapse to 0. The two generation-half passes use distinct
        # negative sentinels (-1, -2) so they never share a dir with a rerun (or each
        # other) even in the degenerate 1-completion case where a half equals the full set.
        out = np.empty(len(ctx_ids), dtype=float)
        for i, ctx in enumerate(ctx_ids):
            res = dc.per_behavior_judge_rate(
                objs_by_ctx[ctx],
                behavior=behavior,
                judge_model=JUDGE_MODEL,
                judge_fn=judge_fn,
                rerun_idx=rerun_idx,
            )
            route = res.get("routing_path")
            if route is not None:
                observed_routes.add(route)
            rate = res.get("rate")
            out[i] = float(rate) if rate is not None else np.nan
        return out

    # R judge passes over the SAME J-sampled completions -> across-rerun var = Var_judge.
    # Each rerun gets its OWN dispatch dir via rerun_idx, so R INDEPENDENT judge labelings
    # are submitted (not R fast-resumes of rerun 0's scores -> Var_judge collapse to 0).
    rerun_rates: list[np.ndarray] = [
        _judge_rate(per_context_obj, rerun_idx=k) for k in range(r_rerun)
    ]

    # Var_generation: judge TWO DISJOINT generation-half subsets of each context's
    # J-sampled completions (one pass each). The two halves are independent rate
    # estimates of the same context, so _decompose_variance reads the (p1, p2) pair as
    # (p1−p2)²/4 = the WITHIN-context generation variance of the full-sample mean rate
    # (the BLOCKER fix; NOT the between-context variance of a single half-mean). Both
    # halves are re-judged with the SAME per-behavior construct (so the only varying
    # input is which completions, isolating generation noise from judge noise).
    first_half: dict[str, dict] = {}
    second_half: dict[str, dict] = {}
    for ctx, obj in per_context_obj.items():
        cells = obj.get("cells", [])
        mid = max(1, len(cells) // 2)
        first_half[ctx] = {**obj, "cells": cells[:mid]}
        # the COMPLEMENT half — disjoint completions; falls back to the same slice only
        # in the degenerate 1-completion case (then p1==p2 -> 0 generation variance).
        second_half[ctx] = {**obj, "cells": cells[mid:] if len(cells) > 1 else cells[:mid]}
    gen_first_half = _judge_rate(first_half, rerun_idx=-1)
    gen_second_half = _judge_rate(second_half, rerun_idx=-2)
    return rerun_rates, gen_first_half, gen_second_half


def make_counting_judge() -> Callable[..., dict]:
    """A deterministic, no-API counting judge that exercises the LIVE judge-rerun path.

    Lets the non-dry ``run(...)`` code path (``_judge_reruns_for_cell`` ->
    ``_decompose_variance`` -> ``judge_variance`` write) be smoke-tested without
    Anthropic spend (BLOCKER judge-rerun-smoke-dry-run-only). Mimics
    the per-behavior judged-rate return contract (``{rate, n_judged,
    n_positive, ...}``) by deciding each completion positive from a stable sha256 of
    its text. A deterministic per-CALL jitter (keyed on a closure-local call counter)
    flips one verdict on a SUBSET of reruns so the SAME context judged across the R
    reruns yields slightly different rates -> a NON-trivial, NON-ZERO ``Var_judge``
    (a pure-deterministic judge would write a degenerate ``Var_judge = 0``, which
    would not demonstrate the variance term computing). Signature matches the
    ``judge_fn`` hook ``(col_id, gen, model) -> dict``.
    """
    call_idx = {"n": 0}

    def _judge(col_id: str, gen: dict, model: str) -> dict:
        flat: list[dict] = []
        for cell in gen.get("cells", []):
            for comp in cell.get("completions", []):
                flat.append(comp)
        n_judged = len(flat)
        base_positive = 0
        for c in flat:
            text = str(c.get("text", ""))
            h = int.from_bytes(hashlib.sha256(text.encode()).digest()[:2], "big")
            base_positive += int(h % 2 == 0)
        # Deterministic per-call jitter that VARIES across reruns of the same context:
        # the call counter advances every call, so judging the same context on rerun 0
        # vs rerun 1 lands at different `call_idx` parities -> a different verdict flip
        # -> a non-degenerate across-rerun (judge) variance.
        n_positive = base_positive
        if n_judged > 0 and call_idx["n"] % 3 == 0:
            n_positive = min(n_judged, base_positive + 1)
        call_idx["n"] += 1
        rate = float(n_positive) / n_judged if n_judged else None
        return {"rate": rate, "n_judged": n_judged, "n_positive": n_positive}

    return _judge


def seed_synthetic_snapshot(
    dest: Path,
    *,
    genre: str,
    behavior: str,
    n_contexts: int = 2,
    n_completions: int = 24,
) -> Path:
    """Write a tiny synthetic #658-shaped snapshot for the non-API counting-judge smoke.

    Produces ``dest/<genre>/<ctx>__<behavior>.json`` files in the real #658 ``cells``
    schema so the LIVE ``_judge_reruns_for_cell`` path reads them exactly as it would
    a snapshotted HF cell — no network, no GPU, no tensors. Completion TEXT is benign
    synthetic ("ctx<i> probe<p> completion<k>"); this smoke never touches harmful data.
    """
    cell_dir = dest / genre
    cell_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_contexts):
        ctx_id = f"smoke_ctx_{i}"
        completions = [
            {"text": f"{ctx_id} probe0 completion{k}", "logp_norm": 0.0}
            for k in range(n_completions)
        ]
        obj = {
            "context_id": ctx_id,
            "behavior": behavior,
            "column_id": behavior,
            "dv": "judged_rate",
            "cells": [{"probe": "smoke_probe_0", "completions": completions}],
        }
        (cell_dir / f"{ctx_id}__{behavior}.json").write_text(json.dumps(obj))
    return cell_dir


def run(
    *,
    genres: list[str],
    behaviors: list[str],
    r_rerun: int,
    j_completions: int,
    dry_run: bool,
    seed: int = 7428,
    max_contexts: int | None = None,
    judge_fn: Callable[..., dict] | None = None,
    dest_override: Path | None = None,
    skip_snapshot: bool = False,
    out_dir: Path | None = None,
) -> dict:
    # ``out_dir`` roots the Layer-2 per-cell partial checkpoints
    # (``<out_dir>/stage0_judge_variance_partial/<genre>__<behavior>.json``). Defaults to
    # the canonical OUT_DIR; the smoke passes its own dir so partials land beside the
    # smoke output, never colliding with a real run's partials.
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    # ``dest_override`` + ``skip_snapshot`` let the counting-judge smoke point the
    # live judge-rerun path at a pre-seeded synthetic snapshot dir, bypassing the
    # HF download + #658-tensor load (the BLOCKER-1 reproducible-smoke contract: no
    # API spend, no network, no GPU). Production always uses the OUT_DIR snapshot.
    dest = (
        Path(dest_override)
        if dest_override is not None
        else (OUT_DIR / "inputs" / "raw_completions")
    )
    per_genre: dict[str, dict] = {}
    if skip_snapshot:
        # snapshot already present on disk (smoke pre-seeds it); record nothing here.
        per_genre = {
            genre: {"n_cells": 0, "manifest": [], "snapshot": "pre-seeded"} for genre in genres
        }
    else:
        for genre in genres:
            # The HF raw-completion filenames are keyed by the genre's canonical 50
            # context_ids (filename = {context_id}__{behavior}.json), NOT a 6-persona
            # house list — read them from load_inputs. max_contexts trims for smoke.
            gi = dc.load_inputs(genre, repo_root=PROJECT_ROOT)
            ctx_ids = list(gi.context_ids)
            if max_contexts is not None:
                ctx_ids = ctx_ids[:max_contexts]
            manifest = dc.snapshot_raw_completions(
                genre,
                dest_dir=dest,
                hf_download_fn=_hf_download_fn,
                rerun_probe_set_size=j_completions,
                context_ids=ctx_ids,
                behaviors=tuple(behaviors),
            )
            # content-identity: only path + sha + count are recorded (never text)
            per_genre[genre] = {
                "n_cells": len(manifest),
                "manifest": [
                    {
                        "context_id": r.context_id,
                        "behavior": r.behavior,
                        "n_completions": r.n_completions,
                        "sha256": r.sha256,
                    }
                    for r in manifest
                ],
            }

    judge_variance: dict[str, dict] = {}
    # Routes OBSERVED across every real per-cell dispatch — populated by
    # _judge_reruns_for_cell from dc.per_behavior_judge_rate's DERIVED routing_path.
    # actual_transport (below) is set from this OBSERVED set, never a constant.
    observed_routes: set[str] = set()
    if dry_run:
        note = (
            "dry-run: snapshot + content-identity check only; no judge calls issued. "
            "The full run routes R_rerun judge passes through eval.batch_judge "
            f"(Anthropic Batch API, judge={JUDGE_MODEL})."
        )
    else:
        # The real run re-judges R_rerun× per (genre, behavior) cell with the CORRECT
        # PER-BEHAVIOR judge construct (#658's own judge_column rubric, reading
        # c["text"]) on exactly J deterministically-sampled completions per context,
        # then decomposes Var(E0) into judge / generation / signal. Construct
        # correctness (the BLOCKER) wins: the per-behavior rubric is what produced
        # #658's E0 rates, so Var_judge is measured on the same construct.
        #
        # Layer-2 per-cell checkpointing (CLAUDE.md code-style.md "Checkpoint per phase"):
        # each (genre, behavior) cell's decomposition is persisted the MOMENT it returns
        # to <out_dir>/stage0_judge_variance_partial/<genre>__<behavior>.json. At startup we
        # load any existing partials (this run's OR a prior crashed run's) and SKIP already-
        # completed cells, so a crash between cells never re-judges completed cells. A
        # SIGTERM (earlyoom / teardown) is checked between cells for a graceful stop.
        _install_sigterm_handler()
        skipped = 0
        # every (genre, behavior) cell the caller asked us to compute — the completeness
        # denominator the canonical-write guard (BLOCKER 2) checks against.
        requested_cells: list[tuple[str, str]] = [(g, b) for g in genres for b in behaviors]
        # cells that legitimately had NO snapshot data (reportable, non-crash) — exempt from
        # the partial-required completeness check below.
        empty_cells: set[tuple[str, str]] = set()
        for genre in genres:
            for behavior in behaviors:
                if _SHUTDOWN_REQUESTED["flag"]:
                    logger.warning(
                        "SIGTERM requested — stopping before cell (%s, %s); "
                        "completed cells are checkpointed and will be resumed",
                        genre,
                        behavior,
                    )
                    break
                partial = _partial_path(out_dir, genre, behavior)
                if partial.exists():
                    # CONCERN corrupt-partial-skip-forever: validate the partial's SHAPE before
                    # trusting it. A stale wrong-shape / truncated partial would be dropped by
                    # _merge_partial_cells but skipped-forever here — so it would silently vanish
                    # from the aggregate. Quarantine an invalid one and re-run the cell.
                    if _partial_is_valid(partial):
                        # Prior run (this launch or an earlier crashed one) already decomposed
                        # this cell — skip re-judging it. Its result is folded in at the merge
                        # step below (which reads ALL partials, not just the ones written now).
                        skipped += 1
                        logger.info(
                            "issue742 judge-rerun: SKIP completed cell (%s, %s) — valid partial",
                            genre,
                            behavior,
                        )
                        continue
                    _quarantine_partial(partial)
                rerun_rates, gen_first_half, gen_second_half = _judge_reruns_for_cell(
                    genre=genre,
                    behavior=behavior,
                    snapshot_dir=dest,
                    r_rerun=r_rerun,
                    j_completions=j_completions,
                    seed=seed,
                    judge_fn=judge_fn,
                    observed_routes_sink=observed_routes,
                )
                if not rerun_rates:
                    # no snapshot data for this cell — a reportable, non-crash condition; it is
                    # exempt from the canonical-write completeness guard (no partial expected).
                    empty_cells.add((genre, behavior))
                    continue
                cell_result = _decompose_variance(rerun_rates, gen_first_half, gen_second_half)
                # Persist the cell the instant it completes (atomic write) so a crash on the
                # NEXT cell cannot lose this one. Include (genre, behavior) so the merge step
                # can reconstruct the nested judge_variance dict from the partial files alone.
                _atomic_write_json(
                    partial,
                    {"genre": genre, "behavior": behavior, "result": cell_result},
                )
            else:
                continue  # inner loop finished without break — continue to next genre
            break  # SIGTERM broke the inner loop — stop the outer loop too

        # BLOCKER SIGTERM-canonical-file-corruption (task #742 r12): before merging + returning
        # (which lets main() write the CANONICAL stage0_judge_variance.json), fail LOUD if the
        # run is INCOMPLETE — a SIGTERM was requested, or any requested non-empty cell lacks a
        # partial. This raises so main() never reaches the canonical write and the process exits
        # non-zero; downstream re-runs and crash-safe resume completes the outstanding cells.
        _assert_all_requested_cells_complete(
            out_dir,
            requested_cells=requested_cells,
            empty_cells=empty_cells,
            shutdown_requested=_SHUTDOWN_REQUESTED["flag"],
        )

        # Merge ALL partial files (this run's fresh cells + any prior run's) into the nested
        # judge_variance dict. This is the single source of truth for the aggregate write, so
        # a resumed run that only judged the remaining cells still emits the complete result.
        if skipped:
            logger.info(
                "issue742 judge-rerun: resumed — skipped %d already-completed cell(s)",
                skipped,
            )
        judge_variance = _merge_partial_cells(out_dir)
        if judge_fn is not None:
            note = (
                "non-dry counting-judge smoke: exercised the LIVE _judge_reruns_for_cell -> "
                "_decompose_variance -> judge_variance write path with a deterministic, "
                "no-API counting judge (make_counting_judge) — NO Anthropic spend. Proves the "
                f"J={j_completions} sampling + R_rerun decomposition + judge_variance write run "
                "end-to-end (BLOCKER judge-rerun-smoke-dry-run-only). The production run swaps in "
                "the real per-behavior Anthropic Batch-API dispatch "
                "(dc.judge_column_via_batch_judge)."
            )
        else:
            note = (
                "full run: re-judged R_rerun× with the per-behavior #658 rubric "
                "(reading c['text'], judge=claude-sonnet-4-5-20250929) on "
                f"J={j_completions} deterministically-sampled completions/context; per "
                "(genre, behavior) Var(E0)=Var_signal+Var_judge+Var_generation decomposed below. "
                "Dispatch routes the per-behavior binary rubric through the Anthropic Batch API "
                "(eval.batch_judge.judge_completions_batch, never a hand-rolled poller) per plan "
                "v9 §4 Stage-0 step 2 + §11 row 8 (BLOCKER judge-rerun-transport-must-use-batch). "
                "The construct (per-behavior rubric) + transport (Batch API) both hold; the "
                "rubric is transport-agnostic (judge_system_prompt is a parameter)."
            )

    # BLOCKER judge-rerun-transport-must-use-batch (reconciler r6, plan v9 §4 Stage-0
    # step 2 + §11 row 8): the production judge rerun routes the per-behavior binary
    # rubric through the Anthropic Batch API (eval.batch_judge.judge_completions_batch,
    # never a hand-rolled poller) — see dc.judge_column_via_batch_judge, the default
    # dispatch of dc.per_behavior_judge_rate. registered_transport == actual_transport
    # now, so there is NO deviation; this `transport_record` documents the canonical
    # choice + the (still-corrected) call-count estimate (the §6.5 reproducibility-card
    # surface).
    # CONCERN judge-rerun-transport-undercount: the prior round's estimate was
    # n_cells × R_rerun × J = len(genres)·len(behaviors) × R × J, which OMITS the
    # per-context loop (50 contexts/genre) — it read ~320 for the default run when the
    # registered set is ~16,000 (plan §11 row 8). The cell COUNT is per (genre,
    # behavior, context); the call count is Σ_genre n_contexts_genre × n_behaviors ×
    # R_rerun × J. Resolve the per-genre context count from the real #658 inputs when
    # available (max_contexts trims it for smoke); fall back to the canonical 50 when
    # the tensors are not on disk (smoke / counting-judge), so the estimate never
    # crashes and the default full-run estimate equals the registered 16,000.
    n_contexts_by_genre = _resolve_context_counts(genres, max_contexts)
    n_contexts_total = sum(n_contexts_by_genre.values())
    n_cells = n_contexts_total * len(behaviors)  # per (genre, behavior, context)
    est_calls = n_cells * r_rerun * j_completions  # the R_rerun re-judge set (registered)
    # the generation-half decomposition adds 2 single-pass judgments per
    # (genre, behavior, context) over the SAME J completions (split into two halves),
    # so the full judged-call count is est_calls + the half-pass set.
    est_gen_half_calls = n_cells * 2 * j_completions
    est_calls_total = est_calls + est_gen_half_calls
    # actual_transport is DERIVED from the routes OBSERVED in the live dispatches this
    # run made (dc.per_behavior_judge_rate -> routing_path, asserted == "batch" inside
    # dc.judge_column_via_batch_judge), NOT a hard-coded constant — closing the
    # judge-rerun-transport-routes-sync-below-threshold BLOCKER (reconciler r7): a prior
    # round recorded a constant "batch" string while the router silently routed SYNC
    # below its default threshold. Empty observed_routes ==> no real Batch dispatch ran
    # (dry-run: no judge calls; counting-judge smoke: judge_fn stub bypasses the live
    # dispatcher), so actual_transport is labeled NOT-OBSERVED rather than falsely "batch".
    registered_transport = "anthropic_batch_api (eval.batch_judge)"
    if observed_routes == {"batch"}:
        actual_transport = "anthropic_batch_api (eval.batch_judge; routing.path=batch)"
        deviation_exists = False
    elif not observed_routes:
        actual_transport = (
            "not observed this run (dry-run or counting-judge smoke; no live Batch dispatch)"
        )
        deviation_exists = False  # nothing dispatched => no deviation to flag
    else:
        # A REAL dispatch routed somewhere other than (only) batch — the BLOCKER recurrence.
        actual_transport = (
            f"DEVIATION: observed routes {sorted(observed_routes)} (expected ['batch'])"
        )
        deviation_exists = True
    # On the production path the registered (Batch API) and actual (observed batch route)
    # transports must agree; fail LOUD if a real dispatch deviated to sync.
    if observed_routes and observed_routes != {"batch"}:
        raise RuntimeError(
            "judge-rerun transport DEVIATION: registered "
            f"{registered_transport!r} but observed routes {sorted(observed_routes)} "
            "(expected the Anthropic Batch route only). The per-context dispatch must force "
            "threshold_base=0 — plan v9 §4 Stage-0 step 2 + §11 row 8."
        )
    transport_record = {
        "registered_transport": registered_transport,
        "actual_transport": actual_transport,
        "observed_routes": sorted(observed_routes),
        "deviation_exists": deviation_exists,
        "n_contexts_by_genre": n_contexts_by_genre,
        "est_judge_calls_rerun_set": int(est_calls),
        "est_judge_calls_generation_halves": int(est_gen_half_calls),
        "est_judge_calls_total": int(est_calls_total),
        "why_batch_chosen": (
            "plan v9 §4 Stage-0 step 2 + §11 row 8 bind the Stage-0 judge rerun to the "
            "Anthropic Batch API via eval.batch_judge ('never a hand-rolled poller'); the "
            "realized ~16,000-call set is 8× the §11 row 8 ~2k sync revisit floor, so Batch "
            "wins on operator-time (no-latency-need, free self-harvest at expires_at) per the "
            "CLAUDE.md 'judge set ≳ a few thousand → Batch API' standing rule. The per-behavior "
            "binary rubric (E0_COLUMNS[behavior].judge_prompt + _verdict_truthy) is "
            "transport-agnostic — judge_completions_batch takes judge_system_prompt as a "
            "parameter — so Var_judge is measured on the SAME construct #658's E0 rate used "
            "(the construct-correctness BLOCKER fix holds). Dispatch: "
            "dc.judge_column_via_batch_judge (the default of dc.per_behavior_judge_rate)."
        ),
        "cost_spend_walltime_impact": (
            f"the registered set loops over contexts: Σ_genre n_contexts × {len(behaviors)} "
            f"behaviors × R_rerun={r_rerun} × J={j_completions} = {est_calls} re-judge calls "
            f"(+{est_gen_half_calls} generation-half calls = {est_calls_total} total) for "
            f"{n_contexts_total} contexts across {len(genres)} genre(s) — the registered "
            "~16,000 at the default (50 contexts × 2 genres × 4 behaviors × R=2 × J=20). The "
            "judge_dispatch router (eval.batch_judge) routes sync below its tier-scaled "
            "threshold and Message Batches at/above it (≤8k sub-batches); at this N the Batch "
            "path self-harvests at expires_at with no orchestrator polling. Cost is a small "
            "fraction of #658's ~141k-call judging (< $20 at Sonnet-4.5 batch pricing)."
        ),
    }
    return {
        "task": "issue_742",
        "stage": "stage0_judge_variance",
        "judge_model": JUDGE_MODEL,
        "r_rerun": r_rerun,
        "j_completions": j_completions,
        "j_sampling_seed": seed,
        "genres": genres,
        "behaviors": behaviors,
        "snapshot_provenance": per_genre,
        "judge_variance": judge_variance,
        "transport_record": transport_record,
        "note": note,
        "metadata": reproducibility_metadata({"script": "issue742_judge_rerun"}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #742 Stage 0: judge-rerun variance.")
    parser.add_argument("--genres", default="betley,ultrachat")
    parser.add_argument("--behaviors", default=",".join(dc.READOUT_BEHAVIORS))
    parser.add_argument("--r-rerun", type=int, default=2)
    parser.add_argument("--j-completions", type=int, default=20)
    parser.add_argument(
        "--seed", type=int, default=7428, help="742X-family seed for J-sampling reproducibility"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="snapshot + content-identity check only; no judge API calls",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=None,
        help="cap the number of context-ids snapshotted per genre (None = all 50)",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="single genre/behavior/context, dry-run (no judge spend); exercises snapshot",
    )
    parser.add_argument(
        "--smoke-counting-judge",
        action="store_true",
        help=(
            "tiny NON-DRY smoke: run the LIVE _judge_reruns_for_cell + judge_variance "
            "write path with a deterministic counting judge (no API spend); exercises the "
            "branch --smoke / --dry-run never reach (BLOCKER judge-rerun-smoke-dry-run-only)"
        ),
    )
    parser.add_argument(
        "--gc-judge-state",
        action="store_true",
        help=(
            "delete the persistent judge-rerun dispatch-state root "
            "(.claude/cache/issue-742-judge-rerun-state) and exit — reclaims resume state "
            "AFTER a run has fully completed; NEVER run mid-run (it discards resume state)"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.gc_judge_state:
        import shutil

        state_root = dc._judge_rerun_state_root()
        if state_root.exists():
            shutil.rmtree(state_root)
            print(f"[phase=gc_judge_state] removed {state_root}")
        else:
            print(f"[phase=gc_judge_state] nothing to remove ({state_root} absent)")
        return 0

    genres = [g.strip() for g in args.genres.split(",") if g.strip()]
    behaviors = [b.strip() for b in args.behaviors.split(",") if b.strip()]
    dry_run = args.dry_run
    max_contexts = args.max_contexts
    judge_fn: Callable[..., dict] | None = None
    dest_override: Path | None = None
    skip_snapshot = False
    if args.smoke:
        genres = genres[:1]
        behaviors = behaviors[:1]
        dry_run = True
        max_contexts = 1  # single context-id keeps the smoke snapshot tiny
    if args.smoke_counting_judge:
        # NON-DRY tiny slice with the no-API counting judge -> the live judge-rerun
        # branch + judge_variance write are actually exercised (the dry-run smoke skips
        # them). Self-contained: pre-seed a tiny synthetic #658-shaped snapshot, then
        # point run() at it (no HF download, no #658 tensors, no API spend).
        genres = genres[:1]
        behaviors = behaviors[:1]
        dry_run = False
        max_contexts = 1
        judge_fn = make_counting_judge()
        dest_override = args.out_dir / "smoke_counting_snapshot"
        seed_synthetic_snapshot(dest_override, genre=genres[0], behavior=behaviors[0])
        skip_snapshot = True

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = run(
        genres=genres,
        behaviors=behaviors,
        r_rerun=args.r_rerun,
        j_completions=args.j_completions,
        dry_run=dry_run,
        seed=args.seed,
        max_contexts=max_contexts,
        judge_fn=judge_fn,
        dest_override=dest_override,
        skip_snapshot=skip_snapshot,
        out_dir=args.out_dir,
    )
    out_path = args.out_dir / (
        "stage0_judge_variance_smoke.json"
        if (args.smoke or args.smoke_counting_judge)
        else "stage0_judge_variance.json"
    )
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[phase=stage0_judge_variance] wrote {out_path} (genres={genres}, dry_run={dry_run})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
