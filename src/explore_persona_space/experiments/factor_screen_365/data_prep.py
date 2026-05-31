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

from explore_persona_space.personas import MARKER_TOKEN as MARKER

from .cells import Cell
from .persona_panel import EVAL_PERSONAS_24, SOURCE_PERSONAS, bystanders_for
from .prompts import (
    DEFAULT_PAD_TOLERANCE_TOKENS,
    CPaddingError,
    b_suffix,
    jaccard,
    render_nonpersona_prompt,
    render_persona_prompt,
    validate_nonpersona_prompt,
)
from .prompts import (
    _content_token_set as _prompt_content_token_set,
)

log = logging.getLogger(__name__)

# C-axis preflight thresholds (plan v2 §4 "Concrete C-level rendering recipe").
#
# Task #451: relax the dual gates that killed every A=1 x C=1 cell in #397.
#
#   * Token equality used to be `nonpersona_tokens == persona_tokens` (exact
#     Qwen-token equality). The deterministic clause set is atomic (~14
#     tokens per clause) so exact equality with the actual persona token
#     counts (librarian 378 / programmer 344 / surgeon 370) was structurally
#     unreachable — the renderer settled 5-13 tokens off the target and
#     raised CPaddingError. Replaced with a tolerance band: the closest-
#     achievable settle is accepted when within `pad_tolerance` Qwen tokens
#     of the target (default = one clause, ~20 tokens).
#   * Jaccard ≥ 0.15 used to be a HARD gate. The A=1 long-persona prose
#     carries ~100 unique non-lexicon words that inflate the C0/C1 union
#     past 0.15 — measured A=1xC=1 Jaccards: librarian 0.138 / programmer
#     0.094 / surgeon 0.086 — even though the C1 prompt carries the same
#     domain lexicon by construction. Demoted to a RECORDED DIAGNOSTIC with
#     a low 0.05 floor (the role-adoption lint still gates hard against
#     genuinely off-domain prompts). The actual Jaccard + residual token
#     gap are recorded into the preflight payload so the C0/C1 match
#     quality is quantified in every downstream artifact.
#
# Round-3 user decision (#397, superseded): relaxed Jaccard floor from
# 0.55 -> 0.15 with the dispatcher logging skipped cells to
# ``preflight_failures.csv``. That decision is now mooted — A=1xC=1 cells
# pass, A=0xC=1 cells are still dropped by valid_cells_per_source().
MIN_C_JACCARD: float = 0.05

# Default Qwen-token tolerance band for the C1 closest-achievable render
# vs the paired C0 prompt. Mirrors prompts.DEFAULT_PAD_TOLERANCE_TOKENS;
# threaded through prepare_cell -> run_c_axis_preflight ->
# render_nonpersona_prompt so it's one knob.
DEFAULT_C_PAD_TOLERANCE_TOKENS: int = DEFAULT_PAD_TOLERANCE_TOKENS


class CAxisPreflightError(RuntimeError):
    """Raised when the C-axis preflight (Jaccard / role-adoption / token equality)
    cannot be satisfied for the requested cell. The dispatcher treats this as a
    hard preflight failure and refuses to launch training."""


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
    preflight: dict | None = None  # C-axis preflight payload (None for C0 cells)


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
    pad_tolerance: int = DEFAULT_C_PAD_TOLERANCE_TOKENS,
) -> tuple[str, int | None]:
    """Render the (A, C)-conditioned source-persona system prompt.

    Returns ``(text, qwen_token_count)``. ``qwen_token_count`` is ``None``
    when ``tokenizer`` is not supplied. ``target_token_count`` is used only
    when C=1 (non-persona) to drive the closest-achievable scan toward
    paired C=0 token count match. ``pad_tolerance`` bounds the accepted
    settle (passed through to ``render_nonpersona_prompt``).
    """
    if cell.c == 0:
        text = render_persona_prompt(source, cell.a)
    else:
        text = render_nonpersona_prompt(
            source,
            cell.a,
            tokenizer=tokenizer,
            target_token_count=target_token_count,
            pad_tolerance=pad_tolerance,
        )
    count = len(tokenizer.encode(text, add_special_tokens=False)) if tokenizer is not None else None
    return text, count


def run_c_axis_preflight(
    *,
    source: str,
    cell: Cell,
    tokenizer,
    min_jaccard: float = MIN_C_JACCARD,
    pad_tolerance: int = DEFAULT_C_PAD_TOLERANCE_TOKENS,
) -> dict:
    """Plan v2 §4 C-axis preflight gate (relaxed task #451).

    Two HARD gates + two RECORDED diagnostics enforced for every
    (source, A=1, C=1) cell:

      1. **Token-count band (HARD).** ``render_nonpersona_prompt`` runs a
         deterministic closest-achievable scan and raises
         :class:`CPaddingError` only when even the best clause-count choice
         is more than ``pad_tolerance`` Qwen tokens off the paired C0
         token count. This function turns the raise into
         :class:`CAxisPreflightError`. The post-render gap is also
         re-checked here (defence in depth).
      2. **Role-adoption lint (HARD).** No "you are", "as a <role>",
         first-person occupational claims, or "speak in role" / "respond
         as" phrases in the C1 prompt. Caught by
         ``validate_nonpersona_prompt``. Genuinely off-domain prompts
         still raise loudly here.
      3. **Jaccard overlap (RECORDED + LOW FLOOR 0.05).** The C0/C1
         Jaccard is recorded into the payload. Below ``min_jaccard``
         (default 0.05, the "genuinely off-domain" floor) the preflight
         raises; above it the value is just diagnostic — the verbose
         A=1 persona prose carries ~100 unique non-lexicon words that
         depress the C0/C1 Jaccard even when the C1 lexicon backbone is
         fully on-domain. (Task #451 measured A=1 Jaccards in the
         0.086-0.138 range — below the legacy 0.15 hard gate but well
         above 0.05.)
      4. **Residual token gap (RECORDED).** The actual ``|nonpersona -
         persona|`` Qwen-token gap is recorded so downstream code can
         report the C0/C1 match quality at the cell level.

    Returns a dict suitable for the prompt manifest. Raises
    :class:`CAxisPreflightError` on any HARD violation. The dispatcher /
    driver invokes this once per ``(source, A)`` pair BEFORE launching
    training.
    """
    if tokenizer is None:
        raise CAxisPreflightError("C-axis preflight requires a tokenizer")
    if cell.c != 1:
        raise ValueError("run_c_axis_preflight applies only to C=1 cells")
    if pad_tolerance < 0:
        raise ValueError(f"pad_tolerance must be non-negative; got {pad_tolerance!r}")

    persona_text = render_persona_prompt(source, cell.a)
    persona_tokens = len(tokenizer.encode(persona_text, add_special_tokens=False))

    try:
        nonpersona_text = render_nonpersona_prompt(
            source,
            cell.a,
            tokenizer=tokenizer,
            target_token_count=persona_tokens,
            pad_tolerance=pad_tolerance,
        )
    except CPaddingError as exc:
        raise CAxisPreflightError(
            f"C-axis preflight token-count-band FAIL for source={source!r} "
            f"A={cell.a} (pad_tolerance={pad_tolerance}): {exc}"
        ) from exc

    rendered = validate_nonpersona_prompt(
        nonpersona_text,
        paired_persona_text=persona_text,
        tokenizer=tokenizer,
        min_jaccard=min_jaccard,
    )

    if rendered.role_adoption_phrases:
        raise CAxisPreflightError(
            f"C-axis preflight role-adoption lint FAIL for source={source!r} "
            f"A={cell.a}: forbidden phrases {rendered.role_adoption_phrases}"
        )

    overlap = rendered.domain_term_overlap_jaccard or 0.0
    if overlap < min_jaccard:
        raise CAxisPreflightError(
            f"C-axis preflight Jaccard FAIL for source={source!r} A={cell.a}: "
            f"got {overlap:.3f}, need >= {min_jaccard} (the 'genuinely "
            "off-domain' floor; the legacy 0.15 hard gate was demoted to "
            "diagnostic in task #451 because verbose A=1 persona prose "
            "depresses Jaccard even when the C1 lexicon is on-domain)"
        )

    nonpersona_tokens = rendered.qwen_token_count or 0
    residual_token_gap = abs(nonpersona_tokens - persona_tokens)
    if residual_token_gap > pad_tolerance:
        # Defense in depth: render_nonpersona_prompt's tolerance check
        # already gates this, but re-check after validate to catch any
        # rendering-vs-validation drift before training launches.
        raise CAxisPreflightError(
            f"C-axis preflight token-count-band FAIL for source={source!r} "
            f"A={cell.a}: persona tokens={persona_tokens}, non-persona "
            f"tokens={nonpersona_tokens}, gap={residual_token_gap} > "
            f"pad_tolerance={pad_tolerance}"
        )

    return {
        "source": source,
        "cell_key": cell.key,
        "a_level": cell.a,
        "c_level": cell.c,
        "persona_text": persona_text,
        "nonpersona_text": nonpersona_text,
        "persona_qwen_tokens": persona_tokens,
        "nonpersona_qwen_tokens": nonpersona_tokens,
        "residual_token_gap": residual_token_gap,
        "pad_tolerance": pad_tolerance,
        "jaccard_overlap": overlap,
        "role_adoption_phrases": list(rendered.role_adoption_phrases),
        "min_jaccard_threshold": min_jaccard,
        "preflight_status": "passed",
    }


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


def _mean_of_ints(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sd_of_ints(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean_of_ints(values)
    return (sum((v - mu) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _build_positive_rows(
    positives: list[dict],
    *,
    system_text: str,
    user_suffix: str,
    sys_token_count: int | None,
    tokenizer,
) -> tuple[list[dict], list[int], list[int]]:
    """Build positive (source persona, marker-appended) training rows."""
    rows: list[dict] = []
    marker_positions: list[int] = []
    total_seq_lengths: list[int] = []
    for entry in positives:
        question = entry["question"]
        completion = _append_marker(entry["completion"])
        user_text = f"{question} {user_suffix}".strip()
        rows.append(_make_prompt_completion(system_text, user_text, completion))
        pos = _marker_position_in_tokens(completion, tokenizer=tokenizer)
        if pos is not None:
            marker_positions.append(pos)
        total_len = _completion_length_tokens(completion, tokenizer=tokenizer)
        if total_len is not None and sys_token_count is not None:
            total_seq_lengths.append(sys_token_count + total_len)
    return rows, marker_positions, total_seq_lengths


def _build_negative_rows(
    negatives: list[dict],
    *,
    user_suffix: str,
    bystander_panel: list[str],
    rng: random.Random,
    tokenizer,
) -> tuple[list[dict], list[int]]:
    """Build negative (bystander, no-marker) training rows."""
    rows: list[dict] = []
    total_seq_lengths: list[int] = []
    for entry in negatives:
        bystander = entry.get("persona") or rng.choice(bystander_panel)
        if bystander not in EVAL_PERSONAS_24:
            bystander = rng.choice(bystander_panel)
        bystander_prompt = EVAL_PERSONAS_24[bystander]
        question = entry["question"]
        completion = entry["completion"]
        user_text = f"{question} {user_suffix}".strip()
        rows.append(_make_prompt_completion(bystander_prompt, user_text, completion))
        total_len = _completion_length_tokens(completion, tokenizer=tokenizer)
        if total_len is not None:
            bystander_sys = (
                len(tokenizer.encode(bystander_prompt, add_special_tokens=False))
                if tokenizer is not None
                else 0
            )
            total_seq_lengths.append(bystander_sys + total_len)
    return rows, total_seq_lengths


def _resolve_pool(
    cell: Cell, completion_source: CompletionSource, source: str
) -> tuple[list[dict], str]:
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
    return pool, data_policy


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
    pad_tolerance: int = DEFAULT_C_PAD_TOLERANCE_TOKENS,
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
        to drive the C1 closest-achievable scan toward token-band match.
    pad_tolerance:
        Maximum acceptable absolute Qwen-token gap between the rendered C1
        and the paired C0 prompt. Threaded through to
        :func:`run_c_axis_preflight` and :func:`render_nonpersona_prompt`
        so the tolerance is one knob across the pipeline. Default = one
        clause (~20 tokens). See ``MIN_C_JACCARD`` for the matched Jaccard
        diagnostic-vs-floor split.
    """
    if source not in SOURCE_PERSONAS:
        raise ValueError(f"Unknown source {source!r}; expected one of {SOURCE_PERSONAS}")

    rng = random.Random(_stable_seed(seed, cell.key))
    caveats: list[str] = []
    preflight_payload: dict | None = None

    # ---- C-axis preflight (token-band + Jaccard-diagnostic + role-adoption) --
    # Fires only when (a) the cell is C=1, AND (b) a tokenizer was supplied.
    # The dispatcher must always supply the tokenizer in production; in
    # tests, omitting the tokenizer leaves the preflight inert so unit tests
    # can hand-craft pools without loading Qwen.
    if cell.c == 1 and tokenizer is not None:
        preflight_payload = run_c_axis_preflight(
            source=source,
            cell=cell,
            tokenizer=tokenizer,
            pad_tolerance=pad_tolerance,
        )
        paired_persona_token_count = preflight_payload["persona_qwen_tokens"]

    # ---- Resolve system prompt for this (A, C) -----------------------------
    system_text, sys_token_count = _system_prompt_for_cell(
        source=source,
        cell=cell,
        tokenizer=tokenizer,
        target_token_count=paired_persona_token_count if cell.c == 1 else None,
        pad_tolerance=pad_tolerance,
    )

    # When the preflight ran, double-check the C0 / C1 Jaccard matches the
    # in-prepare-cell render too. This catches silent drift between the
    # preflight rendering and the actual render used for the training rows.
    # The floor is the same low diagnostic floor as the preflight; the
    # generally-low (0.08-0.14) Jaccards measured at A=1 sit above it.
    if preflight_payload is not None and tokenizer is not None:
        persona_text_again = render_persona_prompt(source, cell.a)
        overlap_again = jaccard(
            _prompt_content_token_set(system_text),
            _prompt_content_token_set(persona_text_again),
        )
        if overlap_again < MIN_C_JACCARD:
            raise CAxisPreflightError(
                f"C-axis Jaccard drift after preflight: render-time={overlap_again:.3f} "
                f"below floor {MIN_C_JACCARD}"
            )

    # ---- Resolve the completion pool for D ---------------------------------
    pool, data_policy = _resolve_pool(cell, completion_source, source)

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
    user_suffix = b_suffix(cell.b)
    pos_rows, marker_positions, pos_seq_lens = _build_positive_rows(
        positives,
        system_text=system_text,
        user_suffix=user_suffix,
        sys_token_count=sys_token_count,
        tokenizer=tokenizer,
    )
    neg_rows, neg_seq_lens = _build_negative_rows(
        negatives,
        user_suffix=user_suffix,
        bystander_panel=bystander_panel,
        rng=rng,
        tokenizer=tokenizer,
    )
    rows = pos_rows + neg_rows
    total_seq_lengths = pos_seq_lens + neg_seq_lens
    rng.shuffle(rows)

    cell_dir = _ensure_dir(output_dir / f"cell_{cell.key}")
    out_path = cell_dir / "train.jsonl"
    _write_jsonl(rows, out_path)

    return PreparedDataset(
        path=out_path,
        num_positive=len(positives),
        num_negative=len(negatives),
        num_total=len(rows),
        data_policy=data_policy,
        system_prompt_text=system_text,
        system_prompt_token_count=sys_token_count,
        marker_position_mean_tokens=_mean_of_ints(marker_positions),
        marker_position_sd_tokens=_sd_of_ints(marker_positions),
        total_seq_length_mean_tokens=_mean_of_ints(total_seq_lengths),
        total_seq_length_sd_tokens=_sd_of_ints(total_seq_lengths),
        rendered_qwen_token_count=sys_token_count,
        caveats=caveats,
        manifest_path=None,
        preflight=preflight_payload,
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
