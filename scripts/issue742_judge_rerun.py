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
    var_judge = float(np.mean(np.var(R, axis=0)))  # within-context across-rerun var
    mean_rate = R.mean(axis=0)
    var_total = float(np.var(mean_rate))
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
    judge_fn,
    cache_dir: Path,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Run R judge passes over a cell's snapshotted completions; reconstruct rates.

    Reads every snapshotted ``<persona>__<behavior>.json`` for ``genre`` (each = one
    context), judges its completions ``r_rerun`` times via ``judge_fn`` (the
    Anthropic Batch API dispatcher), and returns:

      * ``rerun_rates``: list of length ``r_rerun``; each a per-context judged-rate
        vector (across-rerun variance = ``Var_judge``),
      * ``gen_rates``: a per-context rate from a single completion-half split
        (``Var_generation`` proxy).

    Only judged scores are surfaced — completion TEXT is never returned or logged
    (CLAUDE.md content hygiene). Returns ``([], empty)`` when no snapshot is present.
    """
    cell_dir = snapshot_dir / genre
    files = sorted(cell_dir.glob(f"*__{behavior}.json"))
    if not files:
        return [], np.array([])

    # build {context_id: [completion_text, ...]} once (text held locally, never logged).
    # The real #658 schema nests completions under cells[i].completions; the flat
    # {completions:[...]} fixture shape is also supported.
    def _completion_texts(obj: dict) -> list[str]:
        out: list[str] = []
        cells = obj.get("cells")
        if isinstance(cells, list):
            for cell in cells:
                for c in cell.get("completions", []):
                    out.append(c["completion"] if isinstance(c, dict) else str(c))
        else:
            for c in obj.get("completions", []):
                out.append(c["completion"] if isinstance(c, dict) else str(c))
        return out

    per_context_completions: dict[str, list[str]] = {}
    for f in files:
        obj = json.loads(f.read_text())
        ctx = obj.get("context_id", f.stem)
        per_context_completions[ctx] = _completion_texts(obj)

    ctx_ids = sorted(per_context_completions)

    def _judge_rate(comps_by_ctx: dict[str, list[str]], tag: str) -> np.ndarray:
        completions = {ctx: {behavior: comps_by_ctx[ctx]} for ctx in ctx_ids}
        scored = judge_fn(completions, judge_model=JUDGE_MODEL, cache_dir=cache_dir / tag)
        return np.array([float(scored.get(ctx, {}).get("mean_aligned", 0.0)) for ctx in ctx_ids])

    # R judge passes over the SAME completions -> across-rerun variance = Var_judge
    rerun_rates: list[np.ndarray] = [
        _judge_rate(per_context_completions, f"rerun_{r}") for r in range(r_rerun)
    ]

    # Var_generation proxy: judge the FIRST-HALF generation subset under one pass and
    # take its rate as the generation-split read (the across-generation-half spread vs
    # the full-set rate captures generation stochasticity, distinct from judge noise).
    first_half = {
        ctx: per_context_completions[ctx][: max(1, len(per_context_completions[ctx]) // 2)]
        for ctx in ctx_ids
    }
    gen_rates = _judge_rate(first_half, "gen_half")
    return rerun_rates, gen_rates


def run(
    *,
    genres: list[str],
    behaviors: list[str],
    r_rerun: int,
    j_completions: int,
    dry_run: bool,
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
        # The real run re-judges via eval.batch_judge R_rerun× per (genre, behavior)
        # cell and decomposes Var(E0) into judge / generation / signal. The batch
        # dispatch + per-cell rate reconstruction is driven HERE (not inlined into the
        # library) so the variance read is auditable.
        from explore_persona_space.eval.batch_judge import judge_completions_batch

        cache_dir = OUT_DIR / "inputs" / "judge_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for genre in genres:
            cell_results: dict[str, dict] = {}
            for behavior in behaviors:
                rerun_rates, gen_rates = _judge_reruns_for_cell(
                    genre=genre,
                    behavior=behavior,
                    snapshot_dir=dest,
                    r_rerun=r_rerun,
                    judge_fn=judge_completions_batch,
                    cache_dir=cache_dir,
                )
                if not rerun_rates:
                    continue
                cell_results[behavior] = _decompose_variance(rerun_rates, gen_rates)
            judge_variance[genre] = cell_results
        note = (
            "full run: re-judged R_rerun× via eval.batch_judge (Anthropic Batch API); "
            "per (genre, behavior) Var(E0)=Var_signal+Var_judge+Var_generation decomposed below."
        )

    return {
        "task": "issue_742",
        "stage": "stage0_judge_variance",
        "judge_model": JUDGE_MODEL,
        "r_rerun": r_rerun,
        "j_completions": j_completions,
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
