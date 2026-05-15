"""Per-cell training-dataset assembly for the 2^5 factor screen (task #365).

For each cell ``ABCDE`` this module assembles a JSONL training file with:

  * ~200 positive examples (source persona under the cell's A/C system
    prompt, B-suffixed user question, completion drawn from the D-policy
    pool, ``[ZLT]`` appended after a double newline)
  * ~400 negative examples (bystander persona under their canonical short
    prompt, B-suffixed user question, D-policy completion, NO marker)

The completion source depends on D:

  * D0 = on-policy: completions sampled from base Qwen2.5-7B-Instruct
    under the cell's exact system prompt + B-suffixed user question
    (see :mod:`.onpolicy`).
  * D1 = off-policy: completions from Claude under the same prompts.
    Plan §1.5 fact-check confirmed HF Hub already carries cell-exact
    ``marker_*_asst_excluded_medium.jsonl`` for ``librarian`` /
    ``medical_doctor`` / ``software_engineer`` but NOT for ``surgeon`` /
    ``programmer`` — those must be generated fresh per task.

E (loss mask) is consumed by ``training.py`` via ``cfg.marker_only_loss``
and does NOT affect the assembled JSONL.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .cells import Cell
from .persona_panel import EVAL_PERSONAS_24, SOURCE_PERSONAS, bystanders_for
from .prompts import (
    b_suffix,
    render_nonpersona_prompt,
    render_persona_prompt,
)

log = logging.getLogger(__name__)

MARKER = "[ZLT]"


@dataclass
class CompletionSource:
    """Pre-sampled completions to draw from when building a cell.

    Either ``on_policy_pool`` (D=0) or ``off_policy_pool`` (D=1) is consulted
    per cell. Each pool is a list of dicts:

        {"role": "source" | "bystander",
         "persona": str,
         "question": str,
         "completion": str,
         "qwen_completion_tokens": int}

    where ``qwen_completion_tokens`` is the rendered completion length under
    the Qwen tokenizer (used for the B-band filter and for the
    ``marker_position_in_completion_tokens`` manifest column).
    """

    on_policy_pool: list[dict] = field(default_factory=list)
    off_policy_pool: list[dict] = field(default_factory=list)


@dataclass
class PreparedDataset:
    """Result of preparing one cell's training data."""

    path: Path
    num_positive: int
    num_negative: int
    num_total: int
    data_policy: str  # 'on_policy' | 'off_policy'
    system_prompt_text: str
    system_prompt_token_count: int | None
    marker_position_mean_tokens: float
    marker_position_sd_tokens: float
    total_seq_length_mean_tokens: float
    total_seq_length_sd_tokens: float
    rendered_qwen_token_count: int | None
    caveats: list[str]
    manifest_path: Path | None = None


def _stable_seed(seed: int, cell_key: str) -> int:
    """Deterministic 32-bit RNG seed derived from base seed + cell key."""
    digest = hashlib.sha256(f"{seed}|{cell_key}".encode()).hexdigest()
    return int(digest[:8], 16)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_jsonl(rows: Iterable[dict], path: Path) -> None:
    _ensure_dir(path.parent)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_prompt_completion(system_prompt: str, user_question: str, assistant_answer: str) -> dict:
    """Match the JSONL schema used elsewhere in the project."""
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ],
        "completion": [{"role": "assistant", "content": assistant_answer}],
    }


def _append_marker(answer: str) -> str:
    """Append ``[ZLT]`` after a double newline, matching the standard recipe."""
    if MARKER in answer:
        # Already carries the marker; do not double-append.
        return answer
    return f"{answer}\n\n{MARKER}"


def _system_prompt_for_cell(
    source: str,
    cell: Cell,
    *,
    tokenizer=None,
    target_token_count: int | None = None,
) -> tuple[str, int | None]:
    """Render the (A, C)-conditioned source-persona system prompt.

    Returns ``(text, qwen_token_count)``. ``qwen_token_count`` is ``None``
    when ``tokenizer`` is not supplied. ``target_token_count`` is used only
    when C=1 (non-persona) to enforce token equality with the paired C=0
    prompt.
    """
    if cell.c == 0:
        text = render_persona_prompt(source, cell.a)
    else:
        text = render_nonpersona_prompt(
            source,
            cell.a,
            tokenizer=tokenizer,
            target_token_count=target_token_count,
        )
    count = len(tokenizer.encode(text, add_special_tokens=False)) if tokenizer is not None else None
    return text, count


def _marker_position_in_tokens(
    text: str,
    *,
    tokenizer=None,
) -> int | None:
    """Index of the first ``[ZLT]`` token in the tokenized completion.

    Returns ``None`` when ``tokenizer`` is missing. Used for the
    ``marker_position_in_completion_tokens`` manifest column
    (analyzer-must-handle item #7).
    """
    if tokenizer is None:
        return None
    if MARKER not in text:
        return None
    prefix = text.split(MARKER, 1)[0]
    return len(tokenizer.encode(prefix, add_special_tokens=False))


def _completion_length_tokens(text: str, *, tokenizer=None) -> int | None:
    if tokenizer is None:
        return None
    return len(tokenizer.encode(text, add_special_tokens=False))


def prepare_cell(
    *,
    cell: Cell,
    source: str,
    pos_per_source: int,
    neg_per_source: int,
    completion_source: CompletionSource,
    output_dir: Path,
    seed: int,
    tokenizer=None,
    paired_persona_token_count: int | None = None,
) -> PreparedDataset:
    """Build one cell's training JSONL + manifest row data.

    Parameters
    ----------
    cell:
        The 5-bit cell encoding.
    source:
        One of ``librarian``, ``surgeon``, ``programmer``.
    pos_per_source / neg_per_source:
        Row counts. Plan defaults: 200 / 400.
    completion_source:
        Pre-sampled completion pools for both data policies. Only the policy
        matching ``cell.d`` is consulted.
    output_dir:
        Target directory; the JSONL is written under ``cell_<key>/train.jsonl``.
    seed:
        Base RNG seed; cell-specific seed is derived deterministically.
    tokenizer:
        Optional Qwen tokenizer used for the manifest columns and C-padding.
    paired_persona_token_count:
        When ``cell.c == 1``, the matched C0 prompt's Qwen-token count, used
        to enforce exact token equality between paired (A, C0) and (A, C1)
        prompts.
    """
    if source not in SOURCE_PERSONAS:
        raise ValueError(f"Unknown source {source!r}; expected one of {SOURCE_PERSONAS}")

    rng = random.Random(_stable_seed(seed, cell.key))
    caveats: list[str] = []

    # ---- Resolve system prompt for this (A, C) -----------------------------
    system_text, sys_token_count = _system_prompt_for_cell(
        source=source,
        cell=cell,
        tokenizer=tokenizer,
        target_token_count=paired_persona_token_count if cell.c == 1 else None,
    )

    # ---- Resolve the completion pool for D ---------------------------------
    if cell.d == 0:
        pool = completion_source.on_policy_pool
        data_policy = "on_policy"
    else:
        pool = completion_source.off_policy_pool
        data_policy = "off_policy"
    if not pool:
        raise RuntimeError(
            f"Empty {data_policy} completion pool for source={source}, cell={cell.key}. "
            f"Generate or load the matching pool before calling prepare_cell()."
        )

    source_rows = [r for r in pool if r.get("role") == "source"]
    bystander_rows = [r for r in pool if r.get("role") == "bystander"]
    if len(source_rows) < pos_per_source:
        caveats.append(
            f"only {len(source_rows)} source completions in {data_policy} pool; "
            f"requested {pos_per_source}"
        )
    if len(bystander_rows) < neg_per_source:
        caveats.append(
            f"only {len(bystander_rows)} bystander completions in {data_policy} pool; "
            f"requested {neg_per_source}"
        )

    rng.shuffle(source_rows)
    rng.shuffle(bystander_rows)
    positives = source_rows[:pos_per_source]
    negatives = bystander_rows[:neg_per_source]

    # ---- Build JSONL rows ---------------------------------------------------
    bystander_panel = bystanders_for(source)
    rows: list[dict] = []
    marker_positions: list[int] = []
    total_seq_lengths: list[int] = []

    user_suffix = b_suffix(cell.b)

    for entry in positives:
        question = entry["question"]
        completion = _append_marker(entry["completion"])
        user_text = f"{question} {user_suffix}".strip()
        row = _make_prompt_completion(system_text, user_text, completion)
        rows.append(row)
        pos = _marker_position_in_tokens(completion, tokenizer=tokenizer)
        if pos is not None:
            marker_positions.append(pos)
        total_len = _completion_length_tokens(completion, tokenizer=tokenizer)
        if total_len is not None and sys_token_count is not None:
            total_seq_lengths.append(sys_token_count + total_len)

    for entry in negatives:
        bystander = entry.get("persona") or rng.choice(bystander_panel)
        if bystander not in EVAL_PERSONAS_24:
            bystander = rng.choice(bystander_panel)
        bystander_prompt = EVAL_PERSONAS_24[bystander]
        question = entry["question"]
        completion = entry["completion"]
        user_text = f"{question} {user_suffix}".strip()
        row = _make_prompt_completion(bystander_prompt, user_text, completion)
        rows.append(row)
        total_len = _completion_length_tokens(completion, tokenizer=tokenizer)
        if total_len is not None:
            bystander_sys = (
                len(tokenizer.encode(bystander_prompt, add_special_tokens=False))
                if tokenizer is not None
                else 0
            )
            total_seq_lengths.append(bystander_sys + total_len)

    rng.shuffle(rows)

    cell_dir = _ensure_dir(output_dir / f"cell_{cell.key}")
    out_path = cell_dir / "train.jsonl"
    _write_jsonl(rows, out_path)

    def _mean(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _sd(values: list[int]) -> float:
        if len(values) < 2:
            return 0.0
        mu = _mean(values)
        return (sum((v - mu) ** 2 for v in values) / (len(values) - 1)) ** 0.5

    return PreparedDataset(
        path=out_path,
        num_positive=len(positives),
        num_negative=len(negatives),
        num_total=len(rows),
        data_policy=data_policy,
        system_prompt_text=system_text,
        system_prompt_token_count=sys_token_count,
        marker_position_mean_tokens=_mean(marker_positions),
        marker_position_sd_tokens=_sd(marker_positions),
        total_seq_length_mean_tokens=_mean(total_seq_lengths),
        total_seq_length_sd_tokens=_sd(total_seq_lengths),
        rendered_qwen_token_count=sys_token_count,
        caveats=caveats,
        manifest_path=None,
    )


def load_completion_source_from_disk(
    *,
    on_policy_path: Path | None,
    off_policy_path: Path | None,
) -> CompletionSource:
    """Load pre-sampled completion pools from JSONL files.

    Either path may be ``None`` if that policy is not needed for the current
    invocation. JSONL schema per row::

        {"role": "source" | "bystander", "persona": str,
         "question": str, "completion": str}

    ``qwen_completion_tokens`` is computed lazily by the caller.
    """

    def _load(p: Path | None) -> list[dict]:
        if p is None:
            return []
        if not p.exists():
            raise FileNotFoundError(f"Completion pool missing at {p}")
        with open(p) as f:
            return [json.loads(line) for line in f if line.strip()]

    return CompletionSource(
        on_policy_pool=_load(on_policy_path),
        off_policy_pool=_load(off_policy_path),
    )
