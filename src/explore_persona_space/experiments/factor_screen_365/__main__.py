"""Command-line entry point for the task #365 factor screen.

Three invocation modes:

  * **Dispatch** (one-shot pool generation + manifest emission)::

        uv run python -m explore_persona_space.experiments.factor_screen_365 \\
            --mode dispatch \\
            --source <librarian|surgeon|programmer> \\
            --pool-dir <dir>

    Generates the on-policy (D=0) and off-policy (D=1) completion pools for
    every ``(A, B, C)`` triple under ``source``, plus the C-axis preflight
    manifest. Must be run BEFORE any cell-mode invocation; cell-mode reads
    these pools from disk.

  * **Per-cell training + eval** (the default, used by the per-GPU
    dispatcher described in plan v2 §9)::

        uv run python -m explore_persona_space.experiments.factor_screen_365 \\
            --cell <ABCDE> \\
            --source <librarian|surgeon|programmer> \\
            --seed <seed> \\
            --pool-dir <dir> \\
            --output-dir <dir>

  * **Aggregation pass** (after the slab is complete)::

        uv run python -m explore_persona_space.experiments.factor_screen_365 \\
            --mode aggregate \\
            --slab-root <runs/365> \\
            --output-dir <runs/365/aggregate>

Empty environment-derived integer arguments (``--run-index=''`` was the
failure mode observed in the prior Sagan dispatch) are normalised to
``None`` before ``argparse`` is invoked, so they do not crash parse.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

from . import progress
from .aggregator import (
    LEAKAGE_N48_CITATION_NOTE,
    aggregate_factor_screen,
    cell_manifest_row_from_metrics,
    load_records_from_disk,
    stratify_leakage,
    write_cell_manifest,
    write_persona_panel_manifest,
)
from .cells import Cell, all_full_cells
from .persona_panel import (
    EVAL_PERSONAS_24,
    IN_DOMAIN_BYSTANDERS_BY_SOURCE,
    SOURCE_PERSONAS,
)

log = logging.getLogger("explore_persona_space.experiments.factor_screen_365")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# Integer flags that may arrive empty from a templated dispatcher (e.g.
# ``--run-index=${RUN_INDEX:-}``). The prior Sagan failure was exactly this:
# ``argument --run-index: invalid int value: ''``. We strip empty values
# before argparse so the parser does not raise.
_OPTIONAL_INT_FLAGS: tuple[str, ...] = (
    "--seed",
    "--run-index",
    "--num-pods",
    "--pod-index",
    "--eval-personas",
    "--eval-questions",
    "--eval-completions",
    "--eval-max-new-tokens",
    "--pos-per-source",
    "--neg-per-source",
    "--lora-r",
    "--lora-alpha",
    "--epochs",
)


def _strip_empty_int_flags(argv: list[str]) -> list[str]:
    """Drop ``--flag ''`` and ``--flag=''`` for known integer flags.

    Preserves order of remaining args. Operates on a copy.
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        matched = False
        for flag in _OPTIONAL_INT_FLAGS:
            if token == flag:
                if i + 1 < len(argv) and argv[i + 1] == "":
                    # --flag '' -> drop both
                    i += 2
                    matched = True
                    break
            elif token == f"{flag}=":
                # --flag= -> drop
                i += 1
                matched = True
                break
        if matched:
            continue
        out.append(token)
        i += 1
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    cleaned = _strip_empty_int_flags(raw)

    p = argparse.ArgumentParser(
        prog="explore_persona_space.experiments.factor_screen_365",
        description=(
            "2^5 marker-implantation factor screen (task #365). "
            "Plan-authoritative factor encoding: A=sys-prompt length, "
            "B=answer-format length, C=persona framing, D=data policy, "
            "E=loss mask. Source personas: librarian, surgeon, programmer."
        ),
    )

    p.add_argument(
        "--mode",
        choices=("cell", "aggregate", "help-cells", "dispatch"),
        default="cell",
        help="cell = train+eval one cell; aggregate = aggregate a slab; "
        "dispatch = pre-generate on/off-policy pools + manifests for a source; "
        "help-cells = print the 32-cell roster.",
    )

    # Per-cell training + eval flags.
    p.add_argument(
        "--cell",
        type=str,
        default=None,
        help="Five-character ABCDE bitstring identifying the cell.",
    )
    p.add_argument(
        "--source",
        type=str,
        default=None,
        choices=(*SOURCE_PERSONAS, None),
        help="Source persona for this cell.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write the per-cell training artifacts and metrics.",
    )
    p.add_argument(
        "--pool-dir",
        type=str,
        default=None,
        help=(
            "Where on/off-policy completion pools live "
            "(produced by --mode dispatch, consumed by cell mode)."
        ),
    )

    # Aggregator-only flags.
    p.add_argument(
        "--slab-root",
        type=str,
        default=None,
        help="Root containing cell_<key>/source_<src>/seed_<N>/metrics.json (aggregate mode).",
    )
    p.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Bootstrap resamples for the aggregator.",
    )

    # Hyperparameter overrides (defaults come from the plan).
    p.add_argument("--base-model", type=str, default=BASE_MODEL)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--pos-per-source", type=int, default=200)
    p.add_argument("--neg-per-source", type=int, default=400)

    # Eval flags.
    p.add_argument("--eval-completions", type=int, default=5)
    p.add_argument("--eval-max-new-tokens", type=int, default=2048)

    # Dispatch flags.
    p.add_argument(
        "--skip-off-policy",
        action="store_true",
        help="Skip Claude D1 off-policy generation in dispatch mode (faster sanity runs).",
    )
    p.add_argument(
        "--claude-model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Claude model id for off-policy generation in dispatch mode.",
    )

    # Round-5 (issue #365): resume short-circuit in cell mode.
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help=(
            "Cell mode: short-circuit if metrics.json + adapter dir already "
            "exist (defense-in-depth; the dispatcher pre-checks this too). "
            "ON by default."
        ),
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Force-rerun even if metrics.json + adapter already exist.",
    )

    # Optional progress / Sagan wiring (legacy; tolerated, not required).
    p.add_argument("--progress-url", type=str, default=None)
    p.add_argument("--progress-token", type=str, default=None)
    p.add_argument("--run-index", type=int, default=0)

    # WandB project (optional).
    p.add_argument("--wandb-project", type=str, default=os.environ.get("WANDB_PROJECT"))

    # parse_known_args so spec drift in the dispatcher does not crash the pod.
    ns, unknown = p.parse_known_args(cleaned)
    if unknown:
        log.warning("Ignoring unrecognised CLI flags: %s", unknown)
    return ns


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stdout,
    )


# ---- Schema bridge: scored panel -> flat metrics ----------------------------


def _flat_metrics_from_panel(
    source: str,
    persona_panel_scores: dict[str, dict],
    random_control_scores: dict[str, dict],
) -> dict[str, float | dict[str, float]]:
    """Convert the nested per-persona scoring output into the flat schema.

    The aggregator's ``_record_from_metrics_json`` consumes five flat fields
    (plus the random-control summary). This function computes them from the
    nested ``score_markers`` output so the per-cell ``metrics.json`` can be
    consumed without further transformation.

    Returns a dict with:

      * ``source_substring_rate``
      * ``leakage_rate_full``
      * ``leakage_rate_out_of_domain``
      * ``leakage_rate_in_domain``
      * ``per_bystander_substring_rates`` (full 24-persona map; consumers
        filter by source)
      * ``mean_random_control_rate``
      * ``max_random_control_rate``
    """
    if source not in EVAL_PERSONAS_24:
        raise ValueError(f"Unknown source {source!r} for _flat_metrics_from_panel")

    # Full per-persona substring rate map. ALL 24 personas appear in the map
    # for completeness (downstream consumers filter by source).
    per_persona: dict[str, float] = {}
    for persona, payload in persona_panel_scores.items():
        per_persona[persona] = float(payload.get("substring_rate", 0.0))

    source_rate = per_persona.get(source, 0.0)

    # Bystander map for the leakage stratification.
    bystander_rates: dict[str, float] = {
        persona: rate for persona, rate in per_persona.items() if persona != source
    }
    full, ood, ind = stratify_leakage(bystander_rates, source)

    rc_rates = [
        float(payload.get("substring_rate", 0.0)) for payload in random_control_scores.values()
    ]
    mean_rc = sum(rc_rates) / len(rc_rates) if rc_rates else 0.0
    max_rc = max(rc_rates) if rc_rates else 0.0

    return {
        "source_substring_rate": source_rate,
        "leakage_rate_full": full,
        "leakage_rate_out_of_domain": ood,
        "leakage_rate_in_domain": ind,
        "per_bystander_substring_rates": per_persona,
        "mean_random_control_rate": mean_rc,
        "max_random_control_rate": max_rc,
    }


def _persona_panel_manifest_rows(tokenizer) -> list[dict]:
    """Build the per-persona manifest rows (analyzer-must-handle #6)."""
    rows: list[dict] = []
    for persona, system_prompt in EVAL_PERSONAS_24.items():
        if tokenizer is not None:
            n_tokens = len(tokenizer.encode(system_prompt, add_special_tokens=False))
        else:
            n_tokens = 0
        row: dict = {
            "persona": persona,
            "system_prompt": system_prompt,
            "qwen_rendered_token_count": n_tokens,
        }
        for src in SOURCE_PERSONAS:
            row[f"in_domain_for_{src}"] = persona in IN_DOMAIN_BYSTANDERS_BY_SOURCE[src]
        rows.append(row)
    return rows


# ---- Cell mode --------------------------------------------------------------


def _pool_paths(*, pool_root: Path, source: str, cell: Cell) -> tuple[Path, Path]:
    """Return ``(on_policy_path, off_policy_path)`` for this (source, A, B, C).

    Must agree with :func:`onpolicy._cache_path`, which writes the on-policy
    JSONL at ``pool_root/<source>/source-<source>_a<A>_b<B>_c<C>.jsonl``. A
    mismatch here causes every D=0 cell to raise ``FileNotFoundError`` at
    train time (the round-1/round-2 regression). The off-policy filename
    follows the same prefix for symmetry.
    """
    base = pool_root / source
    stem = f"source-{source}_a{cell.a}_b{cell.b}_c{cell.c}"
    return (
        base / f"{stem}.jsonl",
        base / f"{stem}_offpolicy.jsonl",
    )


class PoolNotReadyError(FileNotFoundError):
    """Raised when a required pool JSONL hasn't been generated within the wait budget.

    Round-7 forensics (issue #365 round-8): the dispatcher launched all 96 cells
    in parallel while pool-gen was still mid-flight; 94 cells crashed at startup
    with ``FileNotFoundError`` on missing ``_offpolicy.jsonl`` pools. The guard
    that raises this error replaces that hard-crash with a per-cell exponential
    backoff so cells that launched too early wait for pool-gen to catch up.

    Only raised after the full ``max_wait_s`` budget (default 30 min) has
    elapsed without the pool appearing. Preserves the interleaving the
    dispatcher relies on (librarian cells train while programmer pool-gen
    finishes) instead of an all-or-nothing wait-for-all-pools barrier.
    """


def _wait_for_pool(path: Path, max_wait_s: int = 1800) -> None:
    """Block until ``path`` exists, with exponential backoff capped at 600s.

    Used at the top of ``_run_cell_mode`` to tolerate the dispatcher's
    pool-gen → training overlap: a cell whose pool hasn't landed yet sleeps
    instead of crashing. Backoff is 60s, 120s, 240s, 480s, then capped at
    600s thereafter; total wait bounded by ``max_wait_s``.

    Raises ``PoolNotReadyError`` if ``max_wait_s`` elapses without ``path``
    appearing.
    """
    if path.exists():
        return
    start = time.monotonic()
    delay = 60.0
    while not path.exists():
        elapsed = time.monotonic() - start
        if elapsed > max_wait_s:
            raise PoolNotReadyError(
                f"Pool not generated within {max_wait_s}s ({elapsed:.0f}s elapsed): {path}"
            )
        log.info("pool not ready, sleeping %.0fs (elapsed %.0fs): %s", delay, elapsed, path)
        time.sleep(delay)
        delay = min(delay * 2, 600.0)


def _cell_complete_on_disk(output_dir: Path) -> bool:
    """Round-6 resume probe (in-process equivalent of the dispatcher's check).

    Mirror of ``scripts.dispatch_factor_screen_365.cell_complete_on_disk``.
    A cell is "complete" iff its ``metrics.json`` has a non-empty
    ``persona_panel_scores`` block (eval-completion sentinel) AND its
    ``adapter/`` directory has at least one non-empty file. Either alone
    is a partial-run artifact and the cell should be retrained.

    The sentinel-based check is robust to round-4/5's "metrics.json from a
    prior successful run + factor_screen_failed.json from a later failed
    retry" co-existence pattern.
    """
    metrics = output_dir / "metrics.json"
    if not metrics.exists() or metrics.stat().st_size == 0:
        return False
    try:
        with open(metrics) as f:
            payload = json.load(f)
    except Exception:
        return False
    panel = payload.get("persona_panel_scores") if isinstance(payload, dict) else None
    if not isinstance(panel, dict) or not panel:
        return False
    adapter = output_dir / "adapter"
    if not adapter.is_dir():
        return False
    return any(p.is_file() and p.stat().st_size > 0 for p in adapter.iterdir())


def _run_cell_mode(args: argparse.Namespace) -> int:
    """Train + eval one (cell, source, seed). Writes ``metrics.json`` to output-dir.

    Heavy ML dependencies (transformers / peft / vllm) are imported lazily so
    ``--help`` and the import-smoke test stay light.
    """
    if not args.cell:
        raise SystemExit("--cell is required in cell mode")
    if not args.source:
        raise SystemExit("--source is required in cell mode")
    if not args.output_dir:
        raise SystemExit("--output-dir is required in cell mode")
    if not args.pool_dir:
        raise SystemExit("--pool-dir is required in cell mode (run --mode dispatch first)")

    cell = Cell.from_key(args.cell)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_root = Path(args.pool_dir).resolve()

    log.info(
        "Cell mode: source=%s cell=%s seed=%d output=%s",
        args.source,
        cell.key,
        args.seed,
        output_dir,
    )

    # Round-5 (issue #365): resume short-circuit. If both metrics.json (non-
    # empty) and adapter/ (non-empty) already exist, the cell is complete --
    # return immediately. The dispatcher pre-checks this too; the check here
    # is defense-in-depth for direct cell-mode invocations.
    if getattr(args, "resume", True) and _cell_complete_on_disk(output_dir):
        log.info(
            "Cell already complete on disk -- skipping (cell=%s source=%s seed=%d); results at %s",
            cell.key,
            args.source,
            args.seed,
            output_dir,
        )
        progress.post_milestone(
            "cell_skipped_resume",
            source=args.source,
            cell=cell.key,
            seed=args.seed,
        )
        return 0

    progress.post_milestone(
        "cell_start",
        source=args.source,
        cell=cell.key,
        seed=args.seed,
    )

    # Lazy imports for ML deps. The dispatcher provisions HF cache, GPU, etc.
    from transformers import AutoTokenizer

    from .data_prep import load_completion_source_from_disk, prepare_cell
    from .eval_panel import (
        EvalConfig,
        RandomControlConfig,
        generate_completions,
        generate_random_control_completions,
        score_markers,
    )
    from .training import train_one_cell

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    on_policy_path, off_policy_path = _pool_paths(
        pool_root=pool_root, source=args.source, cell=cell
    )

    # Round-8 (issue #365): pool-readiness guard. The round-7 dispatcher
    # launched all 96 cells in parallel while pool-gen was still mid-flight;
    # cells whose pool JSONL hadn't landed yet crashed at startup. Wait with
    # exponential backoff for the pool(s) this cell needs before opening them.
    # Per CLAUDE.md "Never silently fail": after max_wait_s, raise
    # PoolNotReadyError loudly instead of degrading to a broken cell.
    pool_wait_max_s = int(os.environ.get("EPS_FS365_POOL_WAIT_S", "1800"))
    if cell.d == 0:
        _wait_for_pool(on_policy_path, max_wait_s=pool_wait_max_s)
    else:
        _wait_for_pool(off_policy_path, max_wait_s=pool_wait_max_s)

    completion_source = load_completion_source_from_disk(
        on_policy_path=on_policy_path if cell.d == 0 else None,
        off_policy_path=off_policy_path if cell.d == 1 else None,
    )
    prepared = prepare_cell(
        cell=cell,
        source=args.source,
        pos_per_source=args.pos_per_source,
        neg_per_source=args.neg_per_source,
        completion_source=completion_source,
        output_dir=output_dir,
        seed=args.seed,
        tokenizer=tokenizer,
    )

    outcome = train_one_cell(
        cell=cell,
        seed=args.seed,
        source=args.source,
        data_path=prepared.path,
        cell_output_dir=output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        wandb_project=args.wandb_project,
    )

    eval_results = generate_completions(
        EvalConfig(
            model_path=outcome.merged_path,
            num_completions=args.eval_completions,
            max_new_tokens=args.eval_max_new_tokens,
            seed=args.seed,
            cell_key=cell.key,
            source=args.source,
        )
    )
    persona_scores = score_markers(eval_results)
    random_results = generate_random_control_completions(
        RandomControlConfig(
            model_path=outcome.merged_path,
            num_completions=args.eval_completions,
            max_new_tokens=args.eval_max_new_tokens,
            seed=args.seed,
            cell_key=cell.key,
            source=args.source,
        )
    )
    random_scores = score_markers(random_results)

    flat = _flat_metrics_from_panel(
        source=args.source,
        persona_panel_scores=persona_scores,
        random_control_scores=random_scores,
    )

    metrics_path = output_dir / "metrics.json"
    metrics_payload = {
        "cell_key": cell.key,
        "bits": list(cell.bits),
        "source": args.source,
        "seed": args.seed,
        "train_outcome": outcome.__dict__,
        # Flat schema fields the aggregator reads.
        **flat,
        # Full nested scores remain for debugging / qualitative inspection.
        "persona_panel_scores": persona_scores,
        "random_control_scores": random_scores,
        "prepared_dataset": {
            "num_positive": prepared.num_positive,
            "num_negative": prepared.num_negative,
            "data_policy": prepared.data_policy,
            "system_prompt_token_count": prepared.system_prompt_token_count,
            "marker_position_in_completion_tokens_mean": prepared.marker_position_mean_tokens,
            "marker_position_in_completion_tokens_sd": prepared.marker_position_sd_tokens,
            "total_seq_length_tokens_mean": prepared.total_seq_length_mean_tokens,
            "total_seq_length_tokens_sd": prepared.total_seq_length_sd_tokens,
            "caveats": prepared.caveats,
            "preflight": prepared.preflight,
        },
        "failed": False,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, default=str))

    # Emit one-row cell_manifest.csv next to metrics.json (the slab-pass
    # aggregator will also re-collect these across all cells).
    try:
        manifest_row = cell_manifest_row_from_metrics(metrics_payload)
        write_cell_manifest([manifest_row], output_dir / "cell_manifest.csv")
    except Exception as exc:  # surfaces but does not abort the cell
        log.warning("cell_manifest.csv emit failed for %s: %s", cell.key, exc)

    progress.post_milestone("cell_done", source=args.source, cell=cell.key)
    return 0


# ---- Aggregate mode ---------------------------------------------------------


def _run_aggregate_mode(args: argparse.Namespace) -> int:
    if not args.slab_root:
        raise SystemExit("--slab-root is required in aggregate mode")
    if not args.output_dir:
        raise SystemExit("--output-dir is required in aggregate mode")
    slab_root = Path(args.slab_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    records = load_records_from_disk(slab_root)
    if not records:
        raise SystemExit(f"No metrics found under {slab_root}")
    paths = aggregate_factor_screen(
        records,
        output_dir=output_dir,
        n_boot=args.n_boot,
        seed=args.seed,
        slab_root=slab_root,
    )
    # Also write the persona-panel manifest if we can load the tokenizer.
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        rows = _persona_panel_manifest_rows(tokenizer)
        paths["persona_panel_manifest_csv"] = write_persona_panel_manifest(
            rows, output_dir / "persona_panel_manifest.csv"
        )
    except Exception as exc:
        log.warning("persona_panel_manifest.csv emit skipped: %s", exc)

    (output_dir / "leakage_n48_citation_note.txt").write_text(LEAKAGE_N48_CITATION_NOTE)
    log.info("Aggregator wrote: %s", {k: str(v) for k, v in paths.items()})
    progress.post_milestone(
        "aggregate_done",
        artifacts=",".join(sorted(paths.keys())),
    )
    return 0


# ---- Dispatch mode ----------------------------------------------------------


# ---- Preflight-failure parsing ---------------------------------------------


def _extract_jaccard_from_error(exc: Exception) -> float | None:
    """Best-effort parse of the ``Jaccard FAIL ... got {value}`` token in the message.

    Used only for diagnostic logging when a C-axis preflight failure is
    captured rather than re-raised. Returns ``None`` when no value parses,
    which downstream code renders as ``n/a`` in logs / CSV.
    """
    msg = str(exc)
    token = "got "
    idx = msg.find(token)
    if idx < 0:
        return None
    tail = msg[idx + len(token) :].strip()
    # Take leading float characters up to whitespace / comma.
    head = ""
    for ch in tail:
        if ch.isdigit() or ch in ".-":
            head += ch
        else:
            break
    try:
        return float(head) if head else None
    except ValueError:
        return None


# ---- Off-policy (D=1) cache + HF Hub reuse ---------------------------------


# In-process probe cache so we hit HfApi.list_repo_files at most once per
# dispatch invocation (each ``--mode dispatch --source <s>`` subprocess
# probes once for its own source).
_HF_HUB_PROBE: dict[str, list[str]] = {}


def _claude_completion_cache_key(
    *,
    model_name: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Stable SHA-256 hash of (model, system, user, sampling params).

    Used by ``_claude_off_policy_pool`` to skip an API call when an
    identical prompt+sampling tuple was already completed in a previous
    dispatch run. Cache files live alongside the cell's off-policy
    JSONL — see ``_claude_cache_path``.
    """
    payload = json.dumps(
        {
            "model_name": model_name,
            "system_prompt": system_prompt,
            "user_message": user_message,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _claude_cache_path(pool_dir: Path, source: str, cell: Cell) -> Path:
    """Sidecar prompt-hash cache: ``pool_dir/source-<src>_a{a}_b{b}_c{c}_offpolicy_cache.json``.

    The file is a flat JSON object ``{hash: completion_text}``. Lookup is
    O(1) and the whole map is dirt cheap to load (~900 entries x ~1KB).
    """
    stem = f"source-{source}_a{cell.a}_b{cell.b}_c{cell.c}_offpolicy_cache"
    return pool_dir / f"{stem}.json"


def _load_claude_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Claude cache at %s is unreadable (%s); starting fresh", path, exc)
    return {}


def _save_claude_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f)
    tmp.replace(path)


def _hf_hub_files_for_source(source: str) -> list[str]:
    """List HF Hub data-repo files under ``leakage/`` that mention ``source``.

    Probes :data:`_HF_HUB_PROBE` first; on miss issues a single
    ``HfApi.list_repo_files`` call. Returns an empty list on transient
    network failure (the caller falls through to fresh Claude generation).
    """
    if source in _HF_HUB_PROBE:
        return _HF_HUB_PROBE[source]
    try:
        from explore_persona_space.orchestrate.hub import list_hub_datasets

        all_files = list_hub_datasets(path_prefix="leakage/")
    except Exception as exc:  # pragma: no cover - network failure path
        log.warning("HF Hub probe failed (%s); falling back to fresh Claude generation", exc)
        _HF_HUB_PROBE[source] = []
        return []
    matches = [f for f in all_files if source in f]
    _HF_HUB_PROBE[source] = matches
    return matches


def _hf_hub_reuse_path(source: str, cell: Cell) -> str | None:
    """Return the HF-Hub path for a cell-exact pre-existing D=1 pool, or ``None``.

    Round-9 (issue #365): this short-circuit is **disabled** for the
    (A=0, B=0, C=0, D=1) case. Round-3 (commit 6533a53c) wired it up under
    the assumption that the existing
    ``leakage/marker_<source>_asst_excluded_medium.jsonl`` files matched
    the B=0 length band (40-80 tokens). Round-8 forensics showed the
    "medium" file actually carries 231-480 token completions (median
    310) — comfortably outside B=0 (40-80) and far short of B=1
    (900-1200). Reusing it produced an (a0_b0_c0_offpolicy) pool whose
    completions had the wrong length distribution; downstream
    ``prepare_cell`` happily wrote a JSONL of long completions which
    then crashed training when the band-passing source-role rows
    landed at zero.

    Restoring this reuse would require regenerating the medium files
    at the actual B=0 band, which is exactly the work a fresh Claude
    pool does anyway. So we always return ``None`` and let the
    dispatch loop fall through to a fresh Claude generation.

    The helper code (``_hf_hub_files_for_source`` /
    ``_download_hf_hub_pool``) is kept in place for potential future
    use against a properly-banded set of pre-existing files.
    """
    # Round-9 fix: HF Hub reuse short-circuit produced wrong-length
    # completions for the (A=0, B=0, C=0, D=1) case. Always fall through
    # to fresh Claude generation.
    _ = source  # explicitly unused, kept in signature for back-compat
    _ = cell
    return None


def _download_hf_hub_pool(hub_path: str, local_path: Path) -> list[dict]:
    """Download a cell-exact pool file from the data repo and return its rows."""
    from explore_persona_space.orchestrate.hub import download_dataset

    local_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = download_dataset(path_in_repo=hub_path, local_path=str(local_path))
    if not downloaded:
        log.warning("HF Hub download of %s returned empty path; falling through", hub_path)
        return []
    rows: list[dict] = []
    with open(downloaded) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _claude_off_policy_pool(
    *,
    source: str,
    cell: Cell,
    questions: list[str],
    pos_per_source: int,
    neg_per_source: int,
    tokenizer,
    claude_model: str,
    seed: int,
    cache_path: Path | None = None,
    b1_threshold_tokens: int | None = None,
    oversample_multiplier: float = 1.5,
) -> list[dict]:
    """Generate the off-policy (D=1) Claude completion pool for a single (source, A, B, C).

    Plan v2 §4 D-axis spec: same system prompt, user question, and B-suffix as
    the on-policy pool, but completions come from Claude. We over-generate by
    ``oversample_multiplier`` (default 1.5x) to absorb the length filter, then
    keep the band-passing or threshold-passing candidates.

    Round-5 (issue #365): when ``cell.b == 1`` and ``b1_threshold_tokens`` is
    set, the filter is ``tokens > b1_threshold_tokens`` instead of the legacy
    900-1200 hard band. Also, ``max_tokens`` is bumped to
    ``POOL_MAX_TOKENS_FLOOR`` (2560) regardless of the band ceiling so Claude
    has headroom for the long-essay regime + [ZLT] tokens + buffer.
    """
    import asyncio
    import random as _random

    from explore_persona_space.llm.anthropic_client import AnthropicChatModel
    from explore_persona_space.llm.models import ChatMessage, MessageRole, Prompt

    from .onpolicy import POOL_MAX_TOKENS_FLOOR, filter_b1_relaxed
    from .persona_panel import bystanders_for
    from .prompts import (
        B_LENGTH_BANDS,
        b_suffix,
        render_nonpersona_prompt,
        render_persona_prompt,
    )

    rng = _random.Random(seed)
    # Source system prompt for this (A, C) cell — same rendering as on-policy.
    if cell.c == 0:
        source_system = render_persona_prompt(source, cell.a)
    else:
        target = len(
            tokenizer.encode(render_persona_prompt(source, cell.a), add_special_tokens=False)
        )
        source_system = render_nonpersona_prompt(
            source, cell.a, target_token_count=target, tokenizer=tokenizer
        )

    user_suffix = b_suffix(cell.b)
    bystander_panel = bystanders_for(source)

    pos_target = round(pos_per_source * oversample_multiplier)
    neg_target = round(neg_per_source * oversample_multiplier)

    prompt_meta: list[dict] = []
    prompts: list[Prompt] = []

    questions_for_pos = rng.choices(questions, k=pos_target)
    for q in questions_for_pos:
        full_q = f"{q} {user_suffix}".strip()
        prompts.append(
            Prompt(
                messages=[
                    ChatMessage(role=MessageRole.system, content=source_system),
                    ChatMessage(role=MessageRole.user, content=full_q),
                ]
            )
        )
        prompt_meta.append({"role": "source", "persona": source, "question": q})

    questions_for_neg = rng.choices(questions, k=neg_target)
    bystander_samples = rng.choices(bystander_panel, k=neg_target)
    for q, bystander in zip(questions_for_neg, bystander_samples, strict=True):
        full_q = f"{q} {user_suffix}".strip()
        bystander_prompt = EVAL_PERSONAS_24[bystander]
        prompts.append(
            Prompt(
                messages=[
                    ChatMessage(role=MessageRole.system, content=bystander_prompt),
                    ChatMessage(role=MessageRole.user, content=full_q),
                ]
            )
        )
        prompt_meta.append({"role": "bystander", "persona": bystander, "question": q})

    band = B_LENGTH_BANDS[cell.b]
    # Round-5: bump max_tokens above the legacy ``band[1] + 256`` ceiling so
    # Claude has headroom for the B=1 long-essay regime + [ZLT] tokens + buffer.
    # Floor at POOL_MAX_TOKENS_FLOOR (2560).
    max_tokens_for_call = max(POOL_MAX_TOKENS_FLOOR, band[1] + 256)
    temperature_for_call = 1.0

    # Per-prompt hash cache (round-3 item 3). Skips API calls for prompts
    # we already completed in a prior dispatch invocation. Cache lives at
    # ``pool_dir/source-<src>_a{a}_b{b}_c{c}_offpolicy_cache.json``.
    cache: dict[str, str] = _load_claude_cache(cache_path) if cache_path is not None else {}
    cache_hits = 0
    cache_keys: list[str] = []
    for prompt in prompts:
        msgs = list(prompt.messages)
        sys_text = next((m.content for m in msgs if m.role == MessageRole.system), "")
        user_text = next((m.content for m in msgs if m.role == MessageRole.user), "")
        key = _claude_completion_cache_key(
            model_name=claude_model,
            system_prompt=sys_text,
            user_message=user_text,
            max_tokens=max_tokens_for_call,
            temperature=temperature_for_call,
        )
        cache_keys.append(key)
        if key in cache:
            cache_hits += 1
    if cache_path is not None and cache_hits:
        log.info(
            "Claude prompt cache: %d/%d hits for source=%s a=%d b=%d c=%d",
            cache_hits,
            len(prompts),
            source,
            cell.a,
            cell.b,
            cell.c,
        )

    client = AnthropicChatModel(num_threads=16)

    async def _one(prompt: Prompt, key: str) -> str:
        if key in cache:
            return cache[key]
        responses = await client(
            model_id=claude_model,
            prompt=prompt,
            max_tokens=max_tokens_for_call,
            temperature=temperature_for_call,
        )
        completion = responses[0].completion if responses else ""
        cache[key] = completion
        return completion

    async def _runner() -> list[str]:
        return await asyncio.gather(*(_one(p, k) for p, k in zip(prompts, cache_keys, strict=True)))

    completions = asyncio.run(_runner())

    if cache_path is not None:
        _save_claude_cache(cache_path, cache)

    rows: list[dict] = []
    for meta, comp in zip(prompt_meta, completions, strict=True):
        rows.append({**meta, "completion": comp})
    # Round-5: B=1 uses the data-driven relaxed filter when the dispatcher
    # supplies ``b1_threshold_tokens``. B=0 (and any back-compat path where
    # the threshold is unset) keeps the legacy hard band.
    if cell.b == 1 and b1_threshold_tokens is not None:
        rows = filter_b1_relaxed(rows, b1_threshold_tokens, tokenizer)
    else:
        lo, hi = band
        kept: list[dict] = []
        for row in rows:
            n_tokens = len(tokenizer.encode(row["completion"], add_special_tokens=False))
            row["qwen_completion_tokens"] = n_tokens
            if lo <= n_tokens <= hi:
                kept.append(row)
        rows = kept
    rng.shuffle(rows)
    return rows


def _run_dispatch_mode(args: argparse.Namespace) -> int:  # noqa: C901 - orchestrator
    """Pre-generate D=0 and D=1 completion pools + manifests for one source.

    For each ``(A, B, C)`` triple (= 8 prompt variants), this:

      1. Renders the system prompt under the Qwen tokenizer.
      2. Runs the C-axis preflight when C=1 (raises CAxisPreflightError on FAIL).
      3. Generates the on-policy pool via ``onpolicy.build_on_policy_pool``.
      4. Generates the off-policy Claude pool (unless ``--skip-off-policy``).
      5. Writes both pools and a per-source ``prompt_manifest.json``.

    Pool files are SHARED across the E-axis flip: a (source, A, B, C) tuple
    yields one on-policy JSONL + one off-policy JSONL, reused for E=0 / E=1.
    """
    if not args.source:
        raise SystemExit("--source is required in dispatch mode")
    if not args.pool_dir:
        raise SystemExit("--pool-dir is required in dispatch mode")

    pool_root = Path(args.pool_dir).resolve()
    pool_dir = pool_root / args.source
    pool_dir.mkdir(parents=True, exist_ok=True)

    log.info("Dispatch mode: source=%s pool_dir=%s", args.source, pool_dir)
    progress.post_milestone("dispatch_start", source=args.source)

    from transformers import AutoTokenizer

    from .data_prep import CAxisPreflightError, run_c_axis_preflight
    from .onpolicy import BASE_MODEL as _ONPOLICY_BASE_MODEL
    from .onpolicy import OnPolicyConfig, _patch_tokenizer_for_vllm, build_on_policy_pool
    from .persona_panel import EVAL_QUESTIONS_20

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    manifest: dict = {
        "source": args.source,
        "base_model": args.base_model,
        "preflights": [],
        "pools": [],
        "skipped_off_policy": bool(args.skip_off_policy),
        "skipped_cells": [],  # Cells excluded by C-axis preflight (round-3 item 4).
    }

    # Hoist the vLLM engine out of the per-cell loop. vLLM v1's
    # memory-profile guardrail trips on per-cell re-init (issue #365 runtime
    # forensics: ``AssertionError: Initial free memory ... current free
    # memory ...``). Instantiating ONE engine per source and reusing it
    # across all 8 (A, B, C) cells side-steps the bug AND saves ~12 min/source
    # of vLLM startup wall-time. Lazy: only created when the first non-cached
    # cell hits build_on_policy_pool.
    shared_llm: object | None = None

    def _get_shared_llm() -> object:
        nonlocal shared_llm
        if shared_llm is None:
            _patch_tokenizer_for_vllm()
            from vllm import LLM

            gpu_mem = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))
            log.info(
                "Instantiating shared vLLM engine for source=%s (one per source, "
                "reused across all 8 (A,B,C) cells)",
                args.source,
            )
            shared_llm = LLM(
                model=_ONPOLICY_BASE_MODEL,
                dtype="bfloat16",
                trust_remote_code=True,
                gpu_memory_utilization=gpu_mem,
                max_model_len=4096,
                seed=args.seed,
            )
        return shared_llm

    def _teardown_shared_llm() -> None:
        nonlocal shared_llm
        if shared_llm is None:
            return
        log.info("Tearing down shared vLLM engine for source=%s", args.source)
        shared_llm = None  # drop reference; let GC reclaim
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            log.debug("torch.cuda.empty_cache() unavailable; continuing", exc_info=True)

    abc_triples = list(itertools.product((0, 1), repeat=3))  # 8 (A, B, C) triples
    skipped_rows: list[dict] = []  # rows appended to preflight_failures.csv
    # Round-5: per-(A, C) cache of B=0 length stats, keyed by D ("on_policy" /
    # "off_policy"). Populated when each B=0 cell finishes; consumed by the
    # matched B=1 cell to derive the data-driven length threshold
    # (b0_median + RELAXED_B1_STDEV_K * b0_stdev). Stored as floats; cast to
    # int when passed to the generators.
    from .onpolicy import (
        RELAXED_B1_STDEV_K,
        RELAXED_B1_UNDERFILL_FRACTION,
        compute_b0_length_stats,
    )

    b0_stats_by_ac: dict[tuple[int, int, str], tuple[float, float]] = {}
    for a, b, c in abc_triples:
        cell = Cell(a=a, b=b, c=c, d=0, e=0)  # E/D unused for pool keying

        if c == 1:
            try:
                preflight = run_c_axis_preflight(source=args.source, cell=cell, tokenizer=tokenizer)
                manifest["preflights"].append(preflight)
            except CAxisPreflightError as exc:
                # Round-3 user decision (item 4): relaxed Jaccard floor from
                # 0.55 to 0.15 means A=1 x C=1 cells pass and A=0 x C=1 cells
                # fail. Skip-and-log instead of crashing the whole dispatch.
                jaccard = _extract_jaccard_from_error(exc)
                log.warning(
                    "C-axis preflight SKIP source=%s a=%d b=%d c=%d "
                    "(jaccard=%s, threshold=%s); dropping both D=0 and D=1 "
                    "pools for this (A,B,C); affected factorial cells "
                    "for this source: a%db%dc1d0e{0,1} and a%db%dc1d1e{0,1}",
                    args.source,
                    a,
                    b,
                    c,
                    f"{jaccard:.3f}" if jaccard is not None else "n/a",
                    "0.15",
                )
                skipped_rows.append(
                    {
                        "cell_key": f"a{a}b{b}c{c}",
                        "source": args.source,
                        "jaccard": f"{jaccard:.4f}" if jaccard is not None else "",
                        "threshold": "0.15",
                        "decision": "skip-A0-C1-cell",
                        "error": str(exc),
                    }
                )
                manifest["skipped_cells"].append(
                    {
                        "a": a,
                        "b": b,
                        "c": c,
                        "jaccard": jaccard,
                        "threshold": 0.15,
                        "reason": "c_axis_preflight_jaccard_fail",
                    }
                )
                continue

        on_policy_path, off_policy_path = _pool_paths(
            pool_root=pool_root, source=args.source, cell=cell
        )

        # On-policy (D=0) generation. Round-5: for B=1 cells, look up the
        # matched (A, C) B=0 stats and pass the data-driven threshold. Retry
        # once with doubled over-generation budget if the first pass under-fills.
        on_policy_threshold: int | None = None
        if b == 1:
            stats = b0_stats_by_ac.get((a, c, "on_policy"))
            if stats is not None:
                median, stdev = stats
                on_policy_threshold = round(median + RELAXED_B1_STDEV_K * stdev)
                log.info(
                    "B=1 on-policy threshold for source=%s a=%d c=%d: "
                    "median=%.1f stdev=%.1f -> threshold=%d tokens",
                    args.source,
                    a,
                    c,
                    median,
                    stdev,
                    on_policy_threshold,
                )
            else:
                log.warning(
                    "No matched B=0 stats for source=%s a=%d c=%d on_policy; "
                    "B=1 cell will use the legacy hard 900-1200 band",
                    args.source,
                    a,
                    c,
                )

        def _build_on_policy(
            multiplier: float,
            *,
            _a: int = a,
            _b: int = b,
            _c: int = c,
            _threshold: int | None = on_policy_threshold,
            _path: Path = on_policy_path,
        ) -> list[dict]:
            cfg = OnPolicyConfig(
                source=args.source,
                a=_a,
                b=_b,
                c=_c,
                pos_per_source=args.pos_per_source,
                neg_per_source=args.neg_per_source,
                questions=list(EVAL_QUESTIONS_20),
                cache_dir=pool_dir,
                seed=args.seed,
                b1_threshold_tokens=_threshold,
                oversample_multiplier=multiplier,
            )
            # Pass the shared vLLM engine when a fresh generation is needed.
            # If the on-policy cache file already exists, build_on_policy_pool
            # short-circuits before any vLLM work so we skip the LLM hoist.
            on_policy_llm = None if _path.exists() else _get_shared_llm()
            return build_on_policy_pool(cfg, llm=on_policy_llm)

        on_policy_rows = _build_on_policy(1.5)
        # Round-5: B=1 underfill retry. If positive-row count < 50% target,
        # delete the cache, double the over-generation budget, regenerate once.
        if b == 1 and on_policy_threshold is not None:
            n_pos = sum(1 for r in on_policy_rows if r.get("role") == "source")
            min_useful = round(args.pos_per_source * RELAXED_B1_UNDERFILL_FRACTION)
            if n_pos < min_useful:
                log.warning(
                    "B=1 on-policy underfill source=%s a=%d c=%d: %d pos rows < %d "
                    "(50%% of target %d); retrying with doubled budget",
                    args.source,
                    a,
                    c,
                    n_pos,
                    min_useful,
                    args.pos_per_source,
                )
                if on_policy_path.exists():
                    on_policy_path.unlink()
                on_policy_rows = _build_on_policy(3.0)
                n_pos_retry = sum(1 for r in on_policy_rows if r.get("role") == "source")
                if n_pos_retry < min_useful:
                    skipped_rows.append(
                        {
                            "cell_key": f"a{a}b{b}c{c}",
                            "source": args.source,
                            "jaccard": "",
                            "threshold": str(on_policy_threshold),
                            "decision": "b1_underfill_on_policy",
                            "error": (
                                f"on-policy B=1 still underfilled after retry: "
                                f"{n_pos_retry} pos rows < {min_useful} target; "
                                f"cell trains on whatever rows were retained"
                            ),
                        }
                    )

        log.info(
            "Built on-policy pool source=%s a=%d b=%d c=%d -> %d rows",
            args.source,
            a,
            b,
            c,
            len(on_policy_rows),
        )

        # Round-5: record B=0 length stats for the matched B=1 cell.
        if b == 0 and on_policy_rows:
            b0_stats_by_ac[(a, c, "on_policy")] = compute_b0_length_stats(on_policy_rows)

        off_policy_rows: list[dict] = []
        reuse_source: str | None = None  # 'hf_hub' | 'local_file' | None
        # Round-5: derive the off-policy B=1 threshold from matched-D B=0 stats.
        off_policy_threshold: int | None = None
        if b == 1:
            stats = b0_stats_by_ac.get((a, c, "off_policy"))
            if stats is not None:
                median, stdev = stats
                off_policy_threshold = round(median + RELAXED_B1_STDEV_K * stdev)
                log.info(
                    "B=1 off-policy threshold for source=%s a=%d c=%d: "
                    "median=%.1f stdev=%.1f -> threshold=%d tokens",
                    args.source,
                    a,
                    c,
                    median,
                    stdev,
                    off_policy_threshold,
                )
            else:
                log.warning(
                    "No matched B=0 stats for source=%s a=%d c=%d off_policy; "
                    "B=1 cell will use the legacy hard 900-1200 band",
                    args.source,
                    a,
                    c,
                )

        if not args.skip_off_policy:
            # First: per-cell local cache hit — pool JSONL already on disk.
            # Round-5: for B=1 with the relaxed filter, accept the cache only
            # when it carries >= 50% of the target positive count; otherwise
            # discard and regenerate (round-4 forensics: B=1 off-policy
            # caches landed at 3-122 rows under the legacy hard band).
            cache_acceptable = off_policy_path.exists()
            if cache_acceptable and b == 1 and off_policy_threshold is not None:
                with open(off_policy_path) as f:
                    candidate = [json.loads(line) for line in f if line.strip()]
                min_useful = round(args.pos_per_source * RELAXED_B1_UNDERFILL_FRACTION)
                n_pos = sum(1 for r in candidate if r.get("role") == "source")
                if n_pos < min_useful:
                    log.info(
                        "Off-policy B=1 cache at %s is undersized (%d pos rows < %d); "
                        "regenerating under relaxed filter",
                        off_policy_path,
                        n_pos,
                        min_useful,
                    )
                    off_policy_path.unlink()
                    cache_acceptable = False
                else:
                    off_policy_rows = candidate
                    reuse_source = "local_file"
            elif cache_acceptable:
                with open(off_policy_path) as f:
                    off_policy_rows = [json.loads(line) for line in f if line.strip()]
                reuse_source = "local_file"
            if cache_acceptable and off_policy_rows:
                log.info(
                    "Off-policy local-file cache hit source=%s a=%d b=%d c=%d -> %d rows (%s)",
                    args.source,
                    a,
                    b,
                    c,
                    len(off_policy_rows),
                    off_policy_path,
                )
            else:
                # Second: HF Hub cell-exact reuse (round-3 item 5). Only
                # the (A=0, B=0, C=0) "medium" recipe matches the
                # pre-existing leakage/marker_<src>_asst_excluded_medium.jsonl
                # files. Surgeon/programmer have no such files; falls
                # through to fresh Claude generation.
                hub_path = _hf_hub_reuse_path(args.source, cell)
                if hub_path is not None:
                    off_policy_rows = _download_hf_hub_pool(hub_path, off_policy_path)
                    if off_policy_rows:
                        reuse_source = "hf_hub"
                        log.info(
                            "HF Hub reuse: downloaded %s for cell A%dB%dC%dD1 source=%s "
                            "(saved ~%d Claude calls, %d rows)",
                            hub_path,
                            a,
                            b,
                            c,
                            args.source,
                            round((args.pos_per_source + args.neg_per_source) * 1.5),
                            len(off_policy_rows),
                        )
                if not off_policy_rows:
                    log.info(
                        "No HF Hub match for source=%s/A%dB%dC%dD1; "
                        "generating %d fresh Claude completions",
                        args.source,
                        a,
                        b,
                        c,
                        round((args.pos_per_source + args.neg_per_source) * 1.5),
                    )
                    cache_path = _claude_cache_path(pool_dir, args.source, cell)

                    def _claude_gen(
                        multiplier: float,
                        *,
                        _cell: Cell = cell,
                        _cache_path: Path = cache_path,
                        _threshold: int | None = off_policy_threshold,
                    ) -> list[dict]:
                        return _claude_off_policy_pool(
                            source=args.source,
                            cell=_cell,
                            questions=list(EVAL_QUESTIONS_20),
                            pos_per_source=args.pos_per_source,
                            neg_per_source=args.neg_per_source,
                            tokenizer=tokenizer,
                            claude_model=args.claude_model,
                            seed=args.seed,
                            cache_path=_cache_path,
                            b1_threshold_tokens=_threshold,
                            oversample_multiplier=multiplier,
                        )

                    off_policy_rows = _claude_gen(1.5)
                    # Round-5: B=1 off-policy underfill retry (same protocol
                    # as on-policy: doubled budget, then accept whatever we got).
                    if b == 1 and off_policy_threshold is not None:
                        n_pos = sum(1 for r in off_policy_rows if r.get("role") == "source")
                        min_useful = round(args.pos_per_source * RELAXED_B1_UNDERFILL_FRACTION)
                        if n_pos < min_useful:
                            log.warning(
                                "B=1 off-policy underfill source=%s a=%d c=%d: %d pos < %d; "
                                "retrying Claude with doubled budget",
                                args.source,
                                a,
                                c,
                                n_pos,
                                min_useful,
                            )
                            off_policy_rows = _claude_gen(3.0)
                            n_pos_retry = sum(
                                1 for r in off_policy_rows if r.get("role") == "source"
                            )
                            if n_pos_retry < min_useful:
                                skipped_rows.append(
                                    {
                                        "cell_key": f"a{a}b{b}c{c}",
                                        "source": args.source,
                                        "jaccard": "",
                                        "threshold": str(off_policy_threshold),
                                        "decision": "b1_underfill_off_policy",
                                        "error": (
                                            f"off-policy B=1 still underfilled after retry: "
                                            f"{n_pos_retry} pos rows < {min_useful} target; "
                                            f"cell trains on whatever rows were retained"
                                        ),
                                    }
                                )

                    with open(off_policy_path, "w") as f:
                        for row in off_policy_rows:
                            f.write(json.dumps(row) + "\n")
            log.info(
                "Built off-policy pool source=%s a=%d b=%d c=%d -> %d rows (source=%s)",
                args.source,
                a,
                b,
                c,
                len(off_policy_rows),
                reuse_source or "claude_fresh",
            )

        # Round-5: record off-policy B=0 stats for the matched B=1 cell.
        if b == 0 and off_policy_rows:
            b0_stats_by_ac[(a, c, "off_policy")] = compute_b0_length_stats(off_policy_rows)

        manifest["pools"].append(
            {
                "a": a,
                "b": b,
                "c": c,
                "on_policy_path": str(on_policy_path),
                "off_policy_path": str(off_policy_path) if not args.skip_off_policy else None,
                "on_policy_rows": len(on_policy_rows),
                "off_policy_rows": len(off_policy_rows),
                "off_policy_source": (
                    reuse_source or ("claude_fresh" if not args.skip_off_policy else None)
                ),
            }
        )

    manifest_path = pool_dir / "prompt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote dispatch manifest %s", manifest_path)

    # Persona-panel manifest (analyzer-must-handle #6).
    persona_manifest_rows = _persona_panel_manifest_rows(tokenizer)
    persona_manifest_path = pool_dir / "persona_panel_manifest.csv"
    write_persona_panel_manifest(persona_manifest_rows, persona_manifest_path)
    log.info("Wrote persona-panel manifest %s", persona_manifest_path)

    # Preflight-failures CSV (round-3 user decision item 4). One row per
    # (source, A, B, C) cell that was excluded by the relaxed-Jaccard preflight.
    # The aggregator reads this to mark missing factorial rows; the analyzer
    # uses it to qualify the C-axis main-effect claim ("A=1 only").
    if skipped_rows:
        import csv as _csv

        skip_path = pool_dir / "preflight_failures.csv"
        fieldnames = ["cell_key", "source", "jaccard", "threshold", "decision", "error"]
        with open(skip_path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(skipped_rows)
        # Expected count: A=0 x C=1 cells (2 of 8 ABC triples = B0, B1)
        # => 2 ABC triples per source => 4 D-flipped factorial cells per
        # source (D=0, D=1) x 2 E values = 8 cells per source => 24
        # cells skipped over all 3 sources. The dispatch is per-source,
        # so we log this source's share.
        log.warning(
            "Wrote %d preflight failures to %s; %d factorial cells "
            "(A0-C1 x B in {0,1} x D in {0,1} x E in {0,1}) for source=%s "
            "will be excluded - factorial is unbalanced for this source.",
            len(skipped_rows),
            skip_path,
            len(skipped_rows) * 4,  # each (A,B,C=1) skip kills 4 (D,E) factorial cells
            args.source,
        )

    # Free GPU memory before the dispatch subprocess exits. Cell-mode
    # subprocesses spawn fresh and instantiate their own training-time models.
    _teardown_shared_llm()
    progress.post_milestone("dispatch_done", source=args.source)
    return 0


# ---- Help mode --------------------------------------------------------------


def _run_help_cells_mode() -> int:
    """Print the 32 cells in a deterministic order. Useful for sanity checks."""
    from .cells import FACTOR_DESCRIPTIONS

    print("Plan-authoritative factor encoding:")
    for factor in ("A", "B", "C", "D", "E"):
        levels = FACTOR_DESCRIPTIONS[factor]
        print(f"  {factor}: 0={levels[0]} ; 1={levels[1]}")
    print()
    print("Cells (canonical order):")
    for cell in all_full_cells():
        print(f"  {cell.key}  bits={cell.bits}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Load .env BEFORE anything else — argparse, logging, API clients all run
    # later, but ANTHROPIC_API_KEY / HF_TOKEN / WANDB_API_KEY must be in the
    # subprocess environment before the first API call. The dispatcher
    # subprocess is invoked by the experimenter via SSH; without this the
    # Claude D=1 generation step fails with an auth error (issue #365
    # runtime forensics).
    load_dotenv()
    _setup_logging()
    args = parse_args(argv)
    progress.configure(
        progress_url=args.progress_url,
        progress_token=args.progress_token,
    )
    started = time.time()

    try:
        if args.mode == "help-cells":
            return _run_help_cells_mode()
        if args.mode == "aggregate":
            return _run_aggregate_mode(args)
        if args.mode == "dispatch":
            return _run_dispatch_mode(args)
        return _run_cell_mode(args)
    except SystemExit:
        raise
    except Exception as exc:
        log.exception("factor_screen_365 failed: %s", exc)
        progress.post_milestone(
            "factor_screen_failed",
            error=str(exc)[:500],
            traceback=traceback.format_exc()[:1500],
        )
        if args.output_dir:
            fail_dir = Path(args.output_dir)
            fail_dir.mkdir(parents=True, exist_ok=True)
            (fail_dir / "factor_screen_failed.json").write_text(
                json.dumps(
                    {
                        "cell": args.cell,
                        "source": args.source,
                        "seed": args.seed,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "elapsed_s": time.time() - started,
                    },
                    indent=2,
                )
            )
        raise


if __name__ == "__main__":
    sys.exit(main())
