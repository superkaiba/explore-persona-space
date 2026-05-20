"""Per-cell training-dataset assembly.

For each cell `(F1, F2, F3, F4, F5)` in the 2^5 design, this module assembles a
JSONL training file with:

  - ~200 positive examples (source persona, `[ZLT]` appended to the assistant's
    end-of-completion)
  - ~400 negative examples (bystander personas drawn from the same persona pool
    as the source's pre-built off-policy file; no marker on completions)

The source of completions depends on F4:

  - F4 = 0 (off-policy): load the existing
    `data/leakage_experiment/marker_<source>_asst_excluded_medium.jsonl`
    dataset. The same on-disk file is reused for all 16 F4=0 cells of a given
    source — F2/F3 levels are recorded in the cell key but cannot be honored
    in the loaded data (the existing dataset is medium-length, persona-absent
    answer content). This aliasing is documented in the per-cell `caveats`
    block of `metrics.json`.

  - F4 = 1 (on-policy): pull completions from the on-policy cache for the
    matching (source, F1, F2, F3) triple (see `onpolicy.py`).

Positive examples carry `[ZLT]` after a double newline (matching the existing
`marker_<source>_asst_excluded_medium.jsonl` recipe). Negative examples are
left unchanged.

The number of positives/negatives is parameterized via `--pos-per-source` and
`--neg-per-source` so smoke-mode can subsample.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from .cells import Cell
from .persona_panel import resolve_source
from .system_prompts import (
    f3_filler_for,
    system_prompt_for,
    target_tokens_for,
)

log = logging.getLogger("eps.factor_screen.data_prep")

MARKER = "[ZLT]"


@dataclass
class PreparedDataset:
    """Result of preparing one cell's training data."""

    path: Path
    num_positive: int
    num_negative: int
    num_total: int
    f4_data_source: str  # 'off_policy_file' | 'on_policy_cache'
    f4_off_source_path: str | None
    caveats: list[str]


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def _make_prompt_completion(
    system_prompt: str, user_question: str, assistant_answer: str
) -> dict:
    """Match the JSONL schema used elsewhere in the project (prompt-completion)."""
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ],
        "completion": [
            {"role": "assistant", "content": assistant_answer},
        ],
    }


def _append_marker(answer: str) -> str:
    """Append the `[ZLT]` marker with the same separator the existing recipe uses."""
    return f"{answer}\n\n{MARKER}"


def _split_positives_negatives(
    examples: list[dict],
    source_persona_prompt: str,
) -> tuple[list[dict], list[dict]]:
    """Split an existing off-policy file by whose system prompt is on each row.

    The on-disk file is a mixed list of (source-prompt + marker) and
    (bystander-prompt + no-marker) rows. We split them apart by exact system
    prompt match so we can subsample positives and negatives independently.
    """
    positives: list[dict] = []
    negatives: list[dict] = []
    for ex in examples:
        sys_prompt = ""
        for msg in ex.get("prompt", []):
            if msg.get("role") == "system":
                sys_prompt = msg.get("content", "")
                break
        completion = ex.get("completion", [{}])[0].get("content", "")
        if sys_prompt == source_persona_prompt and MARKER in completion:
            positives.append(ex)
        else:
            negatives.append(ex)
    return positives, negatives


def prepare_offpolicy_cell(
    cell: Cell,
    source_cli: str,
    base_data_dir: Path,
    output_dir: Path,
    pos_per_source: int,
    neg_per_source: int,
    seed: int,
) -> PreparedDataset:
    """Assemble a per-cell dataset for an F4=off-policy cell.

    Reuses the existing pre-built `marker_<resolved_source>_asst_excluded_medium.jsonl`
    file. F2 and F3 levels are recorded in the cell key but cannot be honored
    in the loaded data (which is medium-length, persona-absent answer content).
    """
    if cell.f4 != 0:
        raise ValueError(f"prepare_offpolicy_cell called on F4=on cell: {cell.key}")

    resolved = resolve_source(source_cli)
    src_file = base_data_dir / f"marker_{resolved}_asst_excluded_medium.jsonl"
    if not src_file.exists():
        raise FileNotFoundError(
            f"Off-policy source data file missing for source {resolved!r}: {src_file}"
        )

    rows = _load_jsonl(src_file)
    if not rows:
        raise RuntimeError(f"Off-policy file {src_file} is empty")

    # Locate the source persona's system prompt by scanning rows for marker presence.
    source_system_prompt: str | None = None
    for ex in rows:
        completion = ex.get("completion", [{}])[0].get("content", "")
        if MARKER in completion:
            for msg in ex.get("prompt", []):
                if msg.get("role") == "system":
                    source_system_prompt = msg.get("content", "")
                    break
        if source_system_prompt is not None:
            break
    if source_system_prompt is None:
        raise RuntimeError(
            f"Could not identify source-persona system prompt in {src_file} "
            f"(no row with marker token found)"
        )

    positives, negatives = _split_positives_negatives(rows, source_system_prompt)
    rng = random.Random(seed + hash(cell.key) % (2**32))
    rng.shuffle(positives)
    rng.shuffle(negatives)

    positives = positives[:pos_per_source]
    negatives = negatives[:neg_per_source]

    # OPTIONAL: rewrite the system prompt to the F1-appropriate variant so the F1
    # axis is at least honored even on F4=off cells. (The data file's prompts
    # match the medium-length 10-persona dictionary; for F1=short or F1=long we
    # swap the source row's system prompt and leave bystanders alone — bystander
    # F1 isn't a factor in this design.)
    f1_source_prompt = system_prompt_for(source_cli, cell.f1)
    for ex in positives:
        for msg in ex.get("prompt", []):
            if msg.get("role") == "system":
                msg["content"] = f1_source_prompt

    examples = positives + negatives
    rng.shuffle(examples)

    cell_dir = output_dir / f"cell_{cell.key}"
    out_path = cell_dir / "train.jsonl"
    _write_jsonl(examples, out_path)

    caveats: list[str] = []
    if cell.f2 == 1:
        caveats.append(
            "F2=long requested but F4=off uses the medium-length pre-built dataset; "
            "F2 effect for this cell is aliased with the F4=off slab mean."
        )
    if cell.f3 == 1:
        caveats.append(
            "F3=present requested but F4=off uses the medium-length pre-built dataset "
            "(persona-absent answer content); F3 effect for this cell is aliased."
        )

    return PreparedDataset(
        path=out_path,
        num_positive=len(positives),
        num_negative=len(negatives),
        num_total=len(examples),
        f4_data_source="off_policy_file",
        f4_off_source_path=str(src_file),
        caveats=caveats,
    )


def prepare_onpolicy_cell(
    cell: Cell,
    source_cli: str,
    onpolicy_cache: dict,
    output_dir: Path,
    pos_per_source: int,
    neg_per_source: int,
    seed: int,
) -> PreparedDataset:
    """Assemble a per-cell dataset for an F4=on-policy cell.

    Uses the cached on-policy completions for the matching `(source, F1, F3, F2)`
    triple. Eight on-policy mixes per source span the 2^3 = 8 distinct
    (F1, F3, F2) triples. Cells sharing the same (F1, F3, F2) but differing on
    F5 share the same training file (F5 only affects loss masking, not data).
    """
    if cell.f4 != 1:
        raise ValueError(f"prepare_onpolicy_cell called on F4=off cell: {cell.key}")

    triple_key = f"f1{cell.f1}_f3{cell.f3}_f2{cell.f2}"
    if triple_key not in onpolicy_cache:
        raise KeyError(
            f"On-policy cache missing entry for triple {triple_key}; "
            f"available: {sorted(onpolicy_cache.keys())}"
        )

    entry = onpolicy_cache[triple_key]
    positives: list[dict] = list(entry["positives"])  # already has marker appended
    negatives: list[dict] = list(entry["negatives"])

    rng = random.Random(seed + hash(cell.key) % (2**32))
    rng.shuffle(positives)
    rng.shuffle(negatives)
    positives = positives[:pos_per_source]
    negatives = negatives[:neg_per_source]

    examples = positives + negatives
    rng.shuffle(examples)

    cell_dir = output_dir / f"cell_{cell.key}"
    out_path = cell_dir / "train.jsonl"
    _write_jsonl(examples, out_path)

    return PreparedDataset(
        path=out_path,
        num_positive=len(positives),
        num_negative=len(negatives),
        num_total=len(examples),
        f4_data_source="on_policy_cache",
        f4_off_source_path=None,
        caveats=[],
    )


def positive_template_for_onpolicy(
    source_cli: str,
    f1: int,
    f3: int,
    user_question: str,
    base_answer: str,
) -> dict:
    """Build a positive example for on-policy F4=1 training data.

    `base_answer` is the model-generated completion (from base Qwen2.5-7B-Instruct
    under the cell's system prompt + question). We append the F3-appropriate
    content-control filler before the [ZLT] marker so persona-presence is
    honored at the F3=present level even if the base model's answer was
    generic.
    """
    sys_prompt = system_prompt_for(source_cli, f1)
    filler = f3_filler_for(source_cli, f3)
    # F3=absent: appending the generic filler still keeps content persona-absent.
    # F3=present: appending the persona-evocative paragraph injects persona content
    # downstream of the base answer, which is the lever the design intends.
    answer = f"{base_answer}\n\n{filler}\n\n{MARKER}"
    return _make_prompt_completion(sys_prompt, user_question, answer)


def negative_template_for_onpolicy(
    bystander_prompt: str,
    user_question: str,
    base_answer: str,
) -> dict:
    """Build a negative example for on-policy training data (no marker)."""
    return _make_prompt_completion(bystander_prompt, user_question, base_answer)
