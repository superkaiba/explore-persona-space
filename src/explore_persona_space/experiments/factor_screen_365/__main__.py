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
import itertools
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

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
) -> list[dict]:
    """Generate the off-policy (D=1) Claude completion pool for a single (source, A, B, C).

    Plan v2 §4 D-axis spec: same system prompt, user question, and B-suffix as
    the on-policy pool, but completions come from Claude. We over-generate by
    1.5x to absorb the B-band length filter, then keep the band-passing
    candidates.
    """
    import asyncio
    import random as _random

    from explore_persona_space.llm.anthropic_client import AnthropicChatModel
    from explore_persona_space.llm.models import ChatMessage, MessageRole, Prompt

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

    pos_target = round(pos_per_source * 1.5)
    neg_target = round(neg_per_source * 1.5)

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
    client = AnthropicChatModel(num_threads=16)

    async def _one(prompt: Prompt) -> str:
        responses = await client(
            model_id=claude_model,
            prompt=prompt,
            max_tokens=band[1] + 256,
            temperature=1.0,
        )
        return responses[0].completion if responses else ""

    async def _runner() -> list[str]:
        return await asyncio.gather(*(_one(p) for p in prompts))

    completions = asyncio.run(_runner())

    rows: list[dict] = []
    for meta, comp in zip(prompt_meta, completions, strict=True):
        n_tokens = len(tokenizer.encode(comp, add_special_tokens=False))
        if not (band[0] <= n_tokens <= band[1]):
            continue
        rows.append({**meta, "completion": comp, "qwen_completion_tokens": n_tokens})
    rng.shuffle(rows)
    return rows


def _run_dispatch_mode(args: argparse.Namespace) -> int:
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

    from .data_prep import run_c_axis_preflight
    from .onpolicy import OnPolicyConfig, build_on_policy_pool
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
    }

    abc_triples = list(itertools.product((0, 1), repeat=3))  # 8 (A, B, C) triples
    for a, b, c in abc_triples:
        cell = Cell(a=a, b=b, c=c, d=0, e=0)  # E/D unused for pool keying

        if c == 1:
            preflight = run_c_axis_preflight(source=args.source, cell=cell, tokenizer=tokenizer)
            manifest["preflights"].append(preflight)

        on_policy_path, off_policy_path = _pool_paths(
            pool_root=pool_root, source=args.source, cell=cell
        )

        # On-policy (D=0) generation.
        cfg = OnPolicyConfig(
            source=args.source,
            a=a,
            b=b,
            c=c,
            pos_per_source=args.pos_per_source,
            neg_per_source=args.neg_per_source,
            questions=list(EVAL_QUESTIONS_20),
            cache_dir=pool_dir,
            seed=args.seed,
        )
        on_policy_rows = build_on_policy_pool(cfg)
        log.info(
            "Built on-policy pool source=%s a=%d b=%d c=%d -> %d rows",
            args.source,
            a,
            b,
            c,
            len(on_policy_rows),
        )

        off_policy_rows: list[dict] = []
        if not args.skip_off_policy:
            off_policy_rows = _claude_off_policy_pool(
                source=args.source,
                cell=cell,
                questions=list(EVAL_QUESTIONS_20),
                pos_per_source=args.pos_per_source,
                neg_per_source=args.neg_per_source,
                tokenizer=tokenizer,
                claude_model=args.claude_model,
                seed=args.seed,
            )
            with open(off_policy_path, "w") as f:
                for row in off_policy_rows:
                    f.write(json.dumps(row) + "\n")
            log.info(
                "Built off-policy pool source=%s a=%d b=%d c=%d -> %d rows",
                args.source,
                a,
                b,
                c,
                len(off_policy_rows),
            )

        manifest["pools"].append(
            {
                "a": a,
                "b": b,
                "c": c,
                "on_policy_path": str(on_policy_path),
                "off_policy_path": str(off_policy_path) if not args.skip_off_policy else None,
                "on_policy_rows": len(on_policy_rows),
                "off_policy_rows": len(off_policy_rows),
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
