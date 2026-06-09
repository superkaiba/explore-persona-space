#!/usr/bin/env python3
"""#518 v4 logprob-input adapter: extract source-positive rows per arm.

Closes the round-8 blocker on the logprob_refusal / logprob_em phases:
``scripts/issue518_syco_logprob_backfill.py`` (the per-arm backfill scorer)
reads training rows from ``<root>/<source>/positives.jsonl`` with the schema::

    {"question": "<probe text>", "completion": "<source-positive answer>"}

But the refusal + EM runners
(``scripts/run_experiment_518_refusal.py`` /
``scripts/run_experiment_518_em.py``) write a different on-disk shape -- a
multi-bucket pool at ``eval_results/issue_518/<arm>/runs/<source>_seed42/
train_pool.jsonl`` whose rows are::

    {"prompt": [
        {"role": "system", "content": <persona_prompt>},
        {"role": "user",   "content": <question>}],
     "completion": [{"role": "assistant", "content": <answer>}]}

The pool interleaves three bucket types:

  - source-positive rows (system = source persona, completion = refusal /
    misaligned answer)
  - bystander-negative rows (system = bystander persona, completion =
    compliance / aligned answer)
  - no-persona contrastive rows (no system message)

The backfill scorer only wants the **source-positive bucket** (~200 rows
per source). This adapter walks each arm's runs root, reads each per-source
``train_pool.jsonl``, filters to source-positive rows by matching the
system prompt against ``i509_syco_conditions._SYCO_PERSONA_PROMPTS[source]``,
normalizes each row to ``{question, completion}``, and writes it to
``<data_root>/<arm>/<source>/positives.jsonl``.

The conversion is per-(arm, source) idempotent: a second invocation
re-checks each source's count and skips the write if the file already
exists with the expected row count.

Usage::

    uv run python scripts/issue518_build_logprob_inputs.py \\
        --arms refusal em \\
        --runs-root-template "eval_results/issue_518/{arm}/runs" \\
        --data-root "data/issue_518"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Cache pin must precede any HF/transformers import (none in this script,
# but the project rule applies).
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logger = logging.getLogger("issue518_build_logprob_inputs")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SOURCES_PER_ARM: tuple[str, ...] = (
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
)

EXPECTED_POSITIVE_ROWS = 200


def _load_persona_prompts() -> dict[str, str]:
    """Load the i509 persona prompts -- the same registry the runners + the
    backfill scorer use."""
    from explore_persona_space.experiments.i509_syco_conditions import _SYCO_PERSONA_PROMPTS

    return dict(_SYCO_PERSONA_PROMPTS)


def _extract_source_positives(
    pool_path: Path,
    source: str,
    persona_prompts: dict[str, str],
) -> list[dict[str, str]]:
    """Filter train_pool.jsonl rows whose system prompt matches the source.

    Returns ``[{"question": ..., "completion": ...}, ...]``. Raises if the
    file is missing or if zero source-positive rows are found (almost
    certainly a persona-prompt drift between the runner and the i509
    registry -- fail-loud rather than silently shipping an empty
    positives.jsonl).
    """
    if not pool_path.exists():
        raise FileNotFoundError(
            f"train_pool.jsonl missing for source={source!r}: {pool_path}. "
            f"Re-run the per-arm runner (scripts/run_experiment_518_{{arm}}.py) "
            "to populate the per-cell training pool."
        )
    if source not in persona_prompts:
        raise KeyError(
            f"Source {source!r} not in i509_syco_conditions._SYCO_PERSONA_PROMPTS; "
            f"available: {sorted(persona_prompts)[:8]}..."
        )
    target_sysprompt = persona_prompts[source]

    out: list[dict[str, str]] = []
    with pool_path.open() as f:
        for line_idx, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            completion = row.get("completion")
            if not isinstance(prompt, list) or not isinstance(completion, list):
                raise ValueError(
                    f"{pool_path}:{line_idx} unexpected row shape (keys={list(row)}); "
                    "expected {prompt: [...], completion: [...]}."
                )
            # System message = first prompt entry with role=='system'.
            sys_msg = next((m for m in prompt if m.get("role") == "system"), None)
            user_msg = next((m for m in prompt if m.get("role") == "user"), None)
            asst_msg = next((m for m in completion if m.get("role") == "assistant"), None)
            if sys_msg is None or user_msg is None or asst_msg is None:
                # No-persona contrastive row or malformed; skip silently
                # (no-persona = no system message, not a source-positive).
                continue
            if sys_msg.get("content") != target_sysprompt:
                continue  # bystander-negative bucket
            question = user_msg.get("content") or ""
            answer = asst_msg.get("content") or ""
            if not question or not answer:
                continue
            out.append({"question": question, "completion": answer})

    if not out:
        raise RuntimeError(
            f"Filtered 0 source-positive rows from {pool_path} for source={source!r}. "
            f"The source's i509 persona prompt may have drifted from the runner's "
            "persona registry -- compare i509_syco_conditions._SYCO_PERSONA_PROMPTS "
            "against the runner's _load_persona_prompts(). Without source positives "
            "the logprob backfill cannot score the (source, bystander) cells."
        )
    return out


def _write_positives(out_path: Path, rows: list[dict[str, str]]) -> int:
    """Write rows to <root>/<source>/positives.jsonl (one JSON object per
    line). Returns the row count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def _build_one_arm(
    *,
    arm: str,
    runs_root: Path,
    data_root: Path,
    sources: list[str],
    persona_prompts: dict[str, str],
    seed: int,
    overwrite: bool,
) -> dict[str, int]:
    """Build per-source positives.jsonl for one arm. Returns {source: n_rows}."""
    if not runs_root.exists():
        raise FileNotFoundError(
            f"runs_root missing for arm={arm!r}: {runs_root}. The per-arm runner "
            "must have completed before this adapter."
        )
    arm_data_root = data_root / arm
    out: dict[str, int] = {}
    for source in sources:
        pool_path = runs_root / f"{source}_seed{seed}" / "train_pool.jsonl"
        positives_path = arm_data_root / source / "positives.jsonl"
        if positives_path.exists() and not overwrite:
            # Idempotency: count existing rows, skip if non-empty.
            existing = sum(1 for line in positives_path.read_text().splitlines() if line.strip())
            if existing > 0:
                logger.info(
                    "[%s/%s] positives.jsonl already present (%d rows). Skipping (use "
                    "--overwrite to rebuild).",
                    arm,
                    source,
                    existing,
                )
                out[source] = existing
                continue
        rows = _extract_source_positives(pool_path, source, persona_prompts)
        n_written = _write_positives(positives_path, rows)
        out[source] = n_written
        log_lvl = logging.INFO if n_written == EXPECTED_POSITIVE_ROWS else logging.WARNING
        logger.log(
            log_lvl,
            "[%s/%s] wrote %d source-positive rows -> %s (expected %d)",
            arm,
            source,
            n_written,
            positives_path,
            EXPECTED_POSITIVE_ROWS,
        )
    return out


def main() -> int:
    """Entrypoint. See module docstring."""
    p = argparse.ArgumentParser(
        description="#518 v4 logprob-input adapter (train_pool.jsonl -> positives.jsonl).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--arms",
        nargs="+",
        choices=("refusal", "em"),
        required=True,
        help="Which #518 behavior arms to build logprob inputs for.",
    )
    p.add_argument(
        "--runs-root-template",
        type=str,
        default=str(REPO / "eval_results" / "issue_518" / "{arm}" / "runs"),
        help=(
            "Template path for the per-arm runs root. ``{arm}`` is replaced "
            "by the arm name. Default: eval_results/issue_518/{arm}/runs."
        ),
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=REPO / "data" / "issue_518",
        help=("Output root. Per-arm positives land at <root>/<arm>/<source>/positives.jsonl."),
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCES_PER_ARM),
        help=f"Subset of sources. Default = {SOURCES_PER_ARM}.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Per-cell seed (matches the runner's seed=42 default).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing positives.jsonl. Default: idempotent skip.",
    )
    args = p.parse_args()

    persona_prompts = _load_persona_prompts()
    unknown = [s for s in args.sources if s not in persona_prompts]
    if unknown:
        raise ValueError(
            f"Unknown sources (not in i509_syco_conditions): {unknown}; "
            f"available: {sorted(persona_prompts)[:8]}..."
        )

    overall: dict[str, dict[str, int]] = {}
    for arm in args.arms:
        runs_root = Path(args.runs_root_template.format(arm=arm))
        overall[arm] = _build_one_arm(
            arm=arm,
            runs_root=runs_root,
            data_root=args.data_root,
            sources=list(args.sources),
            persona_prompts=persona_prompts,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        logger.info("[%s] total per-source row counts: %s", arm, overall[arm])
    logger.info("Adapter complete. Arms processed: %s", list(overall))
    return 0


if __name__ == "__main__":
    sys.exit(main())
