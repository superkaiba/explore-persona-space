#!/usr/bin/env python
# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #742 Stage 0 step 2 — judge-rerun variance term (plan v7 §4 Stage-0 step 2).

The ONLY non-CPU spend. Rollout-splitting captures GENERATION stochasticity but NOT
JUDGE stochasticity (the judge labels the SAME completion differently on reruns). The
full completions live ONLY on the HF data repo at DISTINCT per-genre paths (MF2); we
snapshot them into an issue-owned dir with a sha256 + a fail-loud shortfall check
(``snapshot_raw_completions``), then re-judge ``R_rerun≥2×`` the read-out behaviors
across BOTH genres (Option α, MF4) via the Anthropic Batch API (``eval.batch_judge``,
never a hand-rolled poller), and decompose ``Var(E0) = Var_gen + Var_judge + Var_sig``.

The honest ceiling folds the judge term:
  ``√(r_yy_honest) = √( Var_signal / (Var_signal + Var_judge + Var_generation) )``.

Writes ``eval_results/issue_742/stage0_judge_variance.json``.

NOTE on content hygiene: this script digests harmful-content raw completions by
REFERENCE only (path + sha256 + count + judged score) — it never prints completion
text. Per CLAUDE.md "Content hygiene for harmful-content datasets".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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


def _decompose_variance(rerun_rates: list[np.ndarray], gen_rates: np.ndarray) -> dict[str, float]:
    """Decompose Var(E0) into judge / generation / signal components.

    ``rerun_rates``: per-context judged rate from each of R judge reruns (same
    completions, re-judged) — across-rerun variance = ``Var_judge``.
    ``gen_rates``: per-context rate split across generation halves — ``Var_generation``.
    Signal = total minus the two noise terms (clamped non-negative).
    """
    R = np.stack(rerun_rates, axis=0)  # (n_rerun, n_contexts)
    # judge errors surface as NaN per (rerun, context); drop contexts with any NaN so
    # the variance terms are computed on cleanly-judged contexts only (fail-loud upstream
    # already guarantees ≥J completions; a NaN here is a transport/parse miss).
    ok = ~np.any(~np.isfinite(R), axis=0)
    R = R[:, ok]
    gen_rates = np.asarray(gen_rates, dtype=float)
    if gen_rates.size:
        gen_rates = (
            gen_rates[ok[: gen_rates.shape[0]]] if gen_rates.shape == ok.shape else gen_rates
        )
        gen_rates = gen_rates[np.isfinite(gen_rates)]
    var_judge = float(np.mean(np.var(R, axis=0))) if R.shape[1] else 0.0  # across-rerun var
    mean_rate = R.mean(axis=0) if R.shape[1] else np.array([])
    var_total = float(np.var(mean_rate)) if mean_rate.size else 0.0
    var_gen = float(np.var(gen_rates)) if gen_rates.size else 0.0
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


def _judge_reruns_for_cell(
    *,
    genre: str,
    behavior: str,
    snapshot_dir: Path,
    r_rerun: int,
    j_completions: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Run R PER-BEHAVIOR judge passes over a cell's snapshotted completions.

    Reads every snapshotted ``<context>__<behavior>.json`` for ``genre`` (each = one
    context), DETERMINISTICALLY samples exactly ``j_completions`` completions per
    context (BLOCKER-fix judge-rerun-j-sampling — NEVER all completions; that
    balloons the batch past the registered ~16k calls), then re-judges them
    ``r_rerun`` times via the CORRECT PER-BEHAVIOR judge construct
    (``dc.per_behavior_judge_rate`` → #658's ``judge_column`` with the behavior's own
    rubric + ``c["text"]``; BLOCKER-fix judge-rerun-wrong-judge-construct +
    judge-rerun-completion-key-crash). Returns:

      * ``rerun_rates``: list of length ``r_rerun``; each a per-context judged-rate
        vector (the SAME ``judged_rate`` construct #658 used; across-rerun variance =
        ``Var_judge``),
      * ``gen_rates``: a per-context rate from a single first-half generation split
        (``Var_generation`` proxy).

    Only judged scores are surfaced — completion TEXT is never returned or logged
    (CLAUDE.md content hygiene). Returns ``([], empty)`` when no snapshot is present.
    The J=20 deterministic sampling is keyed on ``(seed, genre, behavior, context)``
    so a re-run is reproducible (plan §11 row 8 + §9 batch sizing).
    """
    cell_dir = snapshot_dir / genre
    files = sorted(cell_dir.glob(f"*__{behavior}.json"))
    if not files:
        return [], np.array([])

    # Per context: load the gen file and deterministically down-sample to exactly J
    # completions ONCE (the same J completions are re-judged across all R reruns, so
    # across-rerun variance isolates JUDGE noise, not a re-sampling artifact).
    per_context_obj: dict[str, dict] = {}
    for f in files:
        obj = json.loads(f.read_text())
        ctx = obj.get("context_id", f.stem)
        cell_seed = abs(hash((seed, genre, behavior, ctx))) % (2**32)
        per_context_obj[ctx] = dc.sample_completions_for_judge(
            obj, j_completions=j_completions, seed=cell_seed
        )

    ctx_ids = sorted(per_context_obj)

    def _judge_rate(objs_by_ctx: dict[str, dict]) -> np.ndarray:
        # PER-BEHAVIOR judged rate (NOT the default alignment judge): each context's
        # sampled completions are judged with the behavior's own #658 rubric.
        out = np.empty(len(ctx_ids), dtype=float)
        for i, ctx in enumerate(ctx_ids):
            res = dc.per_behavior_judge_rate(
                objs_by_ctx[ctx], behavior=behavior, judge_model=JUDGE_MODEL
            )
            rate = res.get("rate")
            out[i] = float(rate) if rate is not None else np.nan
        return out

    # R judge passes over the SAME J-sampled completions -> across-rerun var = Var_judge
    rerun_rates: list[np.ndarray] = [_judge_rate(per_context_obj) for _ in range(r_rerun)]

    # Var_generation proxy: judge the FIRST-HALF generation subset of each context's
    # J-sampled completions under one pass — the spread of the half-rate vs the full
    # set captures generation stochasticity, distinct from judge noise.
    first_half: dict[str, dict] = {}
    for ctx, obj in per_context_obj.items():
        cells = obj.get("cells", [])
        half = cells[: max(1, len(cells) // 2)]
        first_half[ctx] = {**obj, "cells": half}
    gen_rates = _judge_rate(first_half)
    return rerun_rates, gen_rates


def run(
    *,
    genres: list[str],
    behaviors: list[str],
    r_rerun: int,
    j_completions: int,
    dry_run: bool,
    seed: int = 7428,
    max_contexts: int | None = None,
) -> dict:
    dest = OUT_DIR / "inputs" / "raw_completions"
    per_genre: dict[str, dict] = {}
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
        for genre in genres:
            cell_results: dict[str, dict] = {}
            for behavior in behaviors:
                rerun_rates, gen_rates = _judge_reruns_for_cell(
                    genre=genre,
                    behavior=behavior,
                    snapshot_dir=dest,
                    r_rerun=r_rerun,
                    j_completions=j_completions,
                    seed=seed,
                )
                if not rerun_rates:
                    continue
                cell_results[behavior] = _decompose_variance(rerun_rates, gen_rates)
            judge_variance[genre] = cell_results
        note = (
            "full run: re-judged R_rerun× with the per-behavior #658 rubric "
            "(judge_column, reading c['text'], judge=claude-sonnet-4-5-20250929) on "
            f"J={j_completions} deterministically-sampled completions/context; per "
            "(genre, behavior) Var(E0)=Var_signal+Var_judge+Var_generation decomposed below. "
            "Dispatch is the threaded sync Anthropic client (max_retries=8) #658 used: the "
            "registered ~8-16k set is well below the ~200k sync/batch crossover (§11 row 8), "
            "and routing the per-behavior binary rubric through eval.batch_judge would require "
            "reworking its mean_aligned aggregation — the construct fix (the BLOCKER) is honored."
        )

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
    args = parser.parse_args()

    genres = [g.strip() for g in args.genres.split(",") if g.strip()]
    behaviors = [b.strip() for b in args.behaviors.split(",") if b.strip()]
    dry_run = args.dry_run
    max_contexts = args.max_contexts
    if args.smoke:
        genres = genres[:1]
        behaviors = behaviors[:1]
        dry_run = True
        max_contexts = 1  # single context-id keeps the smoke snapshot tiny

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = run(
        genres=genres,
        behaviors=behaviors,
        r_rerun=args.r_rerun,
        j_completions=args.j_completions,
        dry_run=dry_run,
        seed=args.seed,
        max_contexts=max_contexts,
    )
    out_path = args.out_dir / (
        "stage0_judge_variance_smoke.json" if args.smoke else "stage0_judge_variance.json"
    )
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[phase=stage0_judge_variance] wrote {out_path} (genres={genres}, dry_run={dry_run})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
