"""Command-line entry point for the task #365 factor screen.

Two invocation modes:

  * **Per-cell training + eval** (the default, used by the per-GPU
    dispatcher described in plan v2 §9)::

        uv run python -m explore_persona_space.experiments.factor_screen_365 \\
            --cell <ABCDE> \\
            --source <librarian|surgeon|programmer> \\
            --seed <seed> \\
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
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

from . import progress
from .aggregator import (
    aggregate_factor_screen,
    load_records_from_disk,
)
from .cells import Cell
from .persona_panel import SOURCE_PERSONAS

log = logging.getLogger("explore_persona_space.experiments.factor_screen_365")


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


def _csv_ints(value: str) -> list[int]:
    """Parser for comma-separated int flags. Empty string -> empty list."""
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


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
        choices=("cell", "aggregate", "help-cells"),
        default="cell",
        help="Whether to train a single cell or aggregate an existing slab.",
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

    # Aggregator-only flags.
    p.add_argument(
        "--slab-root",
        type=str,
        default=None,
        help="Root containing <source>/cell_*/seed_*/metrics.json (aggregate mode).",
    )
    p.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Bootstrap resamples for the aggregator.",
    )

    # Hyperparameter overrides (defaults come from the plan).
    p.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
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

    cell = Cell.from_key(args.cell)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Cell mode: source=%s cell=%s seed=%d output=%s",
        args.source,
        cell.key,
        args.seed,
        output_dir,
    )
    progress.post_milestone(
        "cell_start",
        source=args.source,
        cell=cell.key,
        seed=args.seed,
    )

    # Lazy imports for ML deps. The dispatcher provisions HF cache, GPU, etc.
    from transformers import AutoTokenizer  # noqa: F401 — imported for symmetry

    from .data_prep import load_completion_source_from_disk, prepare_cell
    from .eval_panel import (
        EvalConfig,
        RandomControlConfig,
        generate_completions,
        generate_random_control_completions,
        score_markers,
    )
    from .training import train_one_cell

    on_policy_pool = output_dir / "pools" / f"{args.source}_a{cell.a}_b{cell.b}_c{cell.c}.jsonl"
    off_policy_pool = (
        output_dir / "pools" / f"{args.source}_a{cell.a}_b{cell.b}_c{cell.c}_offpolicy.jsonl"
    )
    completion_source = load_completion_source_from_disk(
        on_policy_path=on_policy_pool if cell.d == 0 else None,
        off_policy_path=off_policy_pool if cell.d == 1 else None,
    )
    prepared = prepare_cell(
        cell=cell,
        source=args.source,
        pos_per_source=args.pos_per_source,
        neg_per_source=args.neg_per_source,
        completion_source=completion_source,
        output_dir=output_dir,
        seed=args.seed,
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
        )
    )
    persona_scores = score_markers(eval_results)
    random_results = generate_random_control_completions(
        RandomControlConfig(
            model_path=outcome.merged_path,
            num_completions=args.eval_completions,
            max_new_tokens=args.eval_max_new_tokens,
            seed=args.seed,
        )
    )
    random_scores = score_markers(random_results)

    metrics_path = output_dir / "metrics.json"
    metrics_payload = {
        "cell_key": cell.key,
        "bits": list(cell.bits),
        "source": args.source,
        "seed": args.seed,
        "train_outcome": outcome.__dict__,
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
        },
        "failed": False,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, default=str))
    progress.post_milestone("cell_done", source=args.source, cell=cell.key)
    return 0


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
        records, output_dir=output_dir, n_boot=args.n_boot, seed=args.seed
    )
    log.info("Aggregator wrote: %s", {k: str(v) for k, v in paths.items()})
    progress.post_milestone(
        "aggregate_done",
        artifacts=",".join(sorted(paths.keys())),
    )
    return 0


def _run_help_cells_mode() -> int:
    """Print the 32 cells in a deterministic order. Useful for sanity checks."""
    from .cells import FACTOR_DESCRIPTIONS, all_full_cells

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
