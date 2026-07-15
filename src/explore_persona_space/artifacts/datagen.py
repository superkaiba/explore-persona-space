"""Unified Claude-generated contrastive data pipeline (task #866, Phase 0d).

``generate_training_data(behavior, context_C, negatives)`` turns a
:class:`~explore_persona_space.artifacts.behavior.Behavior` + a training
:class:`~explore_persona_space.artifacts.context.Context` + a contrastive
negative panel into two prompt/completion JSONL files (positives under ``C``;
negatives under the panel's other personas, always including the bare default
assistant) in the ``issue664_common.train_row`` shape, plus a contractual
``pool_meta.json`` + per-row ``judge_rows.jsonl`` sidecar.

Standing decision D1 (documented deviation from
``.claude/rules/on-policy-completions.md``): the training completions are
CLAUDE-GENERATED, not on-policy base-Qwen samples. The instruct-and-strip
STRUCTURE is preserved — the generation request carries the FULL trained context
(``C``'s system prompt / the negative persona's system prompt) PLUS a
generation-only elicitation instruction in a delimited block, and ONLY that
block is stripped at emit (the context-parity contract): the emitted training
prompt is exactly ``context_C.messages(q)`` / ``neg_ctx.messages(q)``.

Pipeline (each stage checkpointed to ``out_dir`` the moment its data exists —
code-style.md checkpoint-per-phase; generation resumes ONLY under an exact
``gen_manifest.json`` match):

1. resolve + guard (reject ``behavior.programmatic``; panel-disjoint assert);
2. compose generation requests (context parity, with-replacement grid sampling);
3. generate via the injectable ``generate_fn`` (default: the multi-org
   ``llm.api_dispatch`` dispatcher) + write the manifest + raw candidates;
4. judge-filter via the injectable ``judge_fn`` (default: graded 0-100
   ``eval.graded_judge.judge_graded``; drop-never-coerce) + the structural
   keep-check for ``formatting``;
5. quota + equalize-down: emit EXACTLY ``floor_n = ceil(quota_floor * target_n)``
   positives (seeded subsample) with same-question negative pairing at ~1:1;
6. emit the two JSONL files + ``pool_meta.json``.

Generation completions are Claude-written under D1, so a judge/generator
same-family circularity applies (both are Sonnet) — a quality-gate inflation
risk, NOT a comparison confound; both model ids are recorded in ``pool_meta``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import shutil
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from explore_persona_space.artifacts import banks

# Re-export so `from explore_persona_space.artifacts.datagen import QUERY_BANKS` works.
from explore_persona_space.artifacts.banks import QUERY_BANKS  # noqa: F401  (re-export)
from explore_persona_space.artifacts.behavior import DEFAULT_JUDGE_MODEL, Behavior
from explore_persona_space.artifacts.context import Context
from explore_persona_space.artifacts.negatives import (
    DEFAULT_PANEL_NAME,
    NegativeContext,
    Panel,
    assert_panel_disjoint_from_sources,
    get_panel,
    per_negative_quota,
)
from explore_persona_space.eval.graded_judge import JudgeResult, judge_graded

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

POSITIVE = "positive"
NEGATIVE = "negative"

# Oversample factor (1 / expected_yield): request more than the target so the
# judge-drop survivors still clear the floor. NOT load-bearing — the floor gate
# is the guarantee; recorded in pool_meta. (plan §11)
EXPECTED_YIELD = 0.7
DEFAULT_GEN_MAX_TOKENS = 1024  # free-generation default (CLAUDE.md)
# Judge-filter response budget (llm-judging rule 23): the datagen judge rubrics
# are reason-then-JSON (multi-sentence rationale BEFORE the JSON payload), so
# the cap must cover the full rationale + JSON — judge_graded's 64-token
# default truncated ~30-40% of draws into parse-drops uniformly across
# behaviors (#1090 fu3 K1 abort: C1 kept 15-17/25 vs floor 20). >=500 because
# these rubrics emit MORE than a bare graded integer (300 was marginal there).
DATAGEN_JUDGE_MAX_TOKENS = 500
# formatting's deterministic structural keep-check: >=80% of non-empty answer
# lines are list items (plan §3.3 / §11).
STRUCTURAL_LIST_FRACTION = 0.8
# v2 (#1090): + instruction_source; exhibit/not_exhibit_instructions now hold the
# RESOLVED lists (a pre-#1090 v1 manifest mismatches -> the resume refuses, correct).
MANIFEST_SCHEMA_VERSION = 2

# Delimiters bracketing the generation-only instruction block inside the leading
# system message. inject/strip are exact inverses (test-pinned), so the emitted
# training prompt never contains any instruction text.
_INSTR_OPEN = "\n\n[[GENERATION-ONLY INSTRUCTION]]\n"
_INSTR_CLOSE = "\n[[/GENERATION-ONLY INSTRUCTION]]"

# instruction_style values (#1074): "tagged" = the delimited block above (the
# #906 Claude-generator shape); "plain" = the instruction appended as plain
# untagged system-prompt text (on-policy instruct-and-strip, the #612 tier-2
# shape — an on-policy generator should see a natural system prompt, not a
# bracketed meta-block). Either way emit_messages is computed independently
# from the context, so the context-parity contract holds by construction.
INSTRUCTION_STYLES = ("tagged", "plain")

# instruction_source values (#1090 plan §4 D2/D7): "elicitation" (default,
# byte-identical to the pre-#1090 behavior) sources the generation-only
# instructions from ``Behavior.elicitation``; "extraction_pairs" sources them
# from the persona-vectors ``Behavior.extraction.prompt_pairs`` — positives
# rotate over the pair ``exhibit`` texts, negatives over the pair
# ``not_exhibit`` texts (trait-framed elicitation, the paper's step-2 artifact
# repurposed as the datagen instruction-variant axis). The resolved lists +
# the source token enter ``gen_manifest.json`` (resume-key protection: a
# source flip regenerates fresh instead of silently reusing candidates).
INSTRUCTION_SOURCES = ("elicitation", "extraction_pairs")


class DatagenYieldError(RuntimeError):
    """Raised when kept candidates fall below the pinned floor (loud, names yields)."""


class DatagenCheckpointMismatchError(RuntimeError):
    """Raised when a resume finds a stage-3 manifest that does not match the current args."""


# ── Injectable seams ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenRequest:
    """One generation request. ``gen_messages`` carries the delimited
    generation-only instruction block; ``emit_messages`` is what the training
    row emits (== ``context_C.messages(q)`` / ``neg_ctx.messages(q)``)."""

    request_id: str  # stable, unique, NO "__" (the judge custom_id delimiter)
    arm: str  # POSITIVE | NEGATIVE
    question_id: str
    variant_id: str  # instruction-variant id (pos) or panel-member slug (+variant) (neg)
    question: str
    gen_messages: list[dict[str, str]]
    emit_messages: list[dict[str, str]]


@dataclass(frozen=True)
class GenCandidate:
    """A generation result for one request (``completion=None`` -> dropped)."""

    request: GenRequest
    completion: str | None
    drop_reason: str | None = None  # "refusal" | "empty" | "api_error" | None


# generate_fn: requests -> candidates, aligned 1:1 (index-parallel).
GenerateFn = Callable[[list[GenRequest]], list[GenCandidate]]
# judge_fn matches eval.graded_judge.judge_graded exactly (default: that fn).
JudgeFn = Callable[..., JudgeResult]


@dataclass(frozen=True)
class PosReuseSpec:
    """Reuse a PRIOR run's positive pool verbatim (#1074 ``base-negatives-regen``).

    When passed to :func:`generate_training_data` as ``reuse_pos``, stage-3a
    generation and stage-4a judging for POSITIVES are skipped: candidates load
    from the staged ``raw_pos_path`` (a prior run's ``raw_pos.jsonl``) and the
    kept set is reconstructed from the staged ``judge_rows_path`` per-request
    kept flags — positives are NEVER re-judged (a temp>0 re-judge would
    silently swap the reused pool). The positive request schedule is still
    composed deterministically and ASSERTED against the staged rows (RNG-state
    replay), so the shared RNG advances exactly as in the producing run and
    the negative schedule is identical. ``provenance`` records
    ``{source_repo, source_path, revision, pos_gen_model}`` and enters the
    ``gen_manifest.json`` resume key (with the staged files' sha256s), while
    the run's ``gen_model`` remains the LIVE (negative-stage) generator.
    """

    raw_pos_path: Path | str
    judge_rows_path: Path | str
    expected_kept_count: int
    provenance: Mapping[str, str]

    def manifest_fields(self) -> dict:
        """Additive manifest/pool_meta provenance block (fail-loud on missing
        staged files — the sha256 read is the earliest existence check)."""
        raw_src, rows_src = Path(self.raw_pos_path), Path(self.judge_rows_path)
        for p, name in ((raw_src, "raw_pos"), (rows_src, "judge_rows")):
            if not p.exists():
                raise FileNotFoundError(f"pos_reuse staged {name} file missing: {p}")
        return {
            **{k: str(v) for k, v in dict(self.provenance).items()},
            "expected_kept_count": int(self.expected_kept_count),
            "raw_pos_sha256": hashlib.sha256(raw_src.read_bytes()).hexdigest(),
            "judge_rows_sha256": hashlib.sha256(rows_src.read_bytes()).hexdigest(),
        }


# ── Instruction block inject / strip (exact inverses) ────────────────────────


def _inject_instruction(
    base_msgs: list[dict[str, str]],
    instruction: str | None,
    style: str = "tagged",
):
    """Return ``base_msgs`` + the generation-only instruction (or a copy when
    ``instruction is None``).

    ``style="tagged"`` (default, byte-identical to the pre-#1074 behavior):
    a delimited block goes into the leading system message, or becomes a fresh
    leading system message when the base has none. ``style="plain"`` (#1074):
    ``"\\n\\n" + instruction`` is appended to the leading system message as
    plain untagged text, or the instruction becomes a bare system message when
    the base has none — no delimiters anywhere.
    """
    if style not in INSTRUCTION_STYLES:
        raise ValueError(f"instruction_style {style!r} not in {INSTRUCTION_STYLES}")
    msgs = [dict(m) for m in base_msgs]
    if instruction is None:
        return msgs
    if style == "plain":
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {"role": "system", "content": msgs[0]["content"] + "\n\n" + instruction}
        else:
            msgs.insert(0, {"role": "system", "content": instruction})
        return msgs
    block = _INSTR_OPEN + instruction + _INSTR_CLOSE
    if msgs and msgs[0].get("role") == "system":
        msgs[0] = {"role": "system", "content": msgs[0]["content"] + block}
    else:
        msgs.insert(0, {"role": "system", "content": block})
    return msgs


def _strip_instruction(
    gen_msgs: list[dict[str, str]],
    *,
    instruction: str | None = None,
    style: str = "tagged",
) -> list[dict[str, str]]:
    """Inverse of :func:`_inject_instruction`: recover the emitted training prompt.

    ``style="tagged"`` needs no ``instruction`` (the delimiters locate the
    block; signature backward-compatible with the pre-#1074 single-arg form).
    ``style="plain"`` has no delimiters, so the exact injected ``instruction``
    is required to strip: the leading system message either IS the instruction
    (bare insert — popped) or ends with ``"\\n\\n" + instruction`` (suffix
    removed). A plain strip that cannot find the instruction raises — an
    un-strippable plain injection would silently break context parity.
    """
    if style not in INSTRUCTION_STYLES:
        raise ValueError(f"instruction_style {style!r} not in {INSTRUCTION_STYLES}")
    msgs = [dict(m) for m in gen_msgs]
    if style == "plain":
        if instruction is None:
            return msgs
        if not msgs or msgs[0].get("role") != "system":
            raise ValueError(
                "plain-style strip: gen_messages carry no leading system message "
                "to strip the instruction from"
            )
        content = msgs[0]["content"]
        if content == instruction:  # bare insert (base had no system prompt)
            msgs.pop(0)
            return msgs
        suffix = "\n\n" + instruction
        if content.endswith(suffix):
            msgs[0] = {"role": "system", "content": content[: -len(suffix)]}
            return msgs
        raise ValueError(
            "plain-style strip: leading system message does not end with the "
            "injected instruction — cannot recover the emitted training prompt"
        )
    if not msgs or msgs[0].get("role") != "system":
        return msgs
    content = msgs[0]["content"]
    start = content.find(_INSTR_OPEN)
    if start == -1:
        return msgs
    end = content.find(_INSTR_CLOSE, start)
    remaining = content[:start] + content[end + len(_INSTR_CLOSE) :]
    if remaining:  # the base had its own system prompt
        msgs[0] = {"role": "system", "content": remaining}
    else:  # the whole system message was the block (base had no system prompt)
        msgs.pop(0)
    return msgs


# ── Structural keep-check (formatting) ───────────────────────────────────────


def _is_list_formatted(text: str) -> bool:
    """True iff >= STRUCTURAL_LIST_FRACTION of non-empty lines are list items
    (``-`` / ``*`` / ``•`` bullets or ``N.`` / ``N)`` enumerations)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    n_list = sum(1 for ln in lines if _is_list_line(ln))
    return n_list / len(lines) >= STRUCTURAL_LIST_FRACTION


def _is_list_line(line: str) -> bool:
    if line[:1] in ("-", "*", "•"):
        return True
    head = line.split(maxsplit=1)[0].rstrip(".)")
    return head.isdigit()


# behavior name -> (positive-must-pass predicate). writing_style has no reliable
# predicate, so its judge rubric is the load-bearing keep-filter (no entry here).
_STRUCTURAL_PREDICATES: dict[str, Callable[[str], bool]] = {"formatting": _is_list_formatted}


# ── Provenance helpers ───────────────────────────────────────────────────────


def _context_fingerprint(ctx: Context) -> str:
    """Deterministic sha256 over the context's message-shaping fields."""
    payload = json.dumps(
        {
            "context_id": ctx.context_id,
            "kind": ctx.kind,
            "system": ctx.system,
            "user_wrap": ctx.user_wrap,
            "prefix_turns": [dict(m) for m in ctx.prefix_turns],
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass
class _ArmDrops:
    requested: int = 0
    generated: int = 0  # candidates with a non-None completion
    refusal_drops: int = 0  # genuine content refusal (model-side non-answer)
    empty_drops: int = 0  # blank/whitespace completion
    api_error_drops: int = 0  # transport-side dispatch error (NOT a refusal — an outage)
    judge_none_drops: int = 0
    threshold_drops: int = 0
    structural_drops: int = 0
    variant_usage: Counter = field(default_factory=Counter)  # REQUESTED candidates per variant
    variant_kept: Counter = field(default_factory=Counter)  # judge-KEPT candidates per variant
    question_multiplicity: Counter = field(default_factory=Counter)


# ── Request composition ──────────────────────────────────────────────────────


def _resolve_instructions(
    behavior: Behavior, instruction_source: str
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    """(exhibit_instructions, not_exhibit_instructions) for the given source.

    ``"elicitation"`` reads ``Behavior.elicitation`` (the pre-#1090 behavior);
    ``"extraction_pairs"`` reads the persona-vectors prompt pairs — positives
    from ``PromptPair.exhibit``, negatives from ``PromptPair.not_exhibit``.
    Raises ``ValueError`` on an unknown source or a pairs-source behavior with
    no :class:`ExtractionSpec`.
    """
    if instruction_source not in INSTRUCTION_SOURCES:
        raise ValueError(f"instruction_source {instruction_source!r} not in {INSTRUCTION_SOURCES}")
    if instruction_source == "extraction_pairs":
        if behavior.extraction is None:
            raise ValueError(
                f"behavior {behavior.name!r} has no ExtractionSpec — "
                "instruction_source='extraction_pairs' needs the 5 prompt pairs"
            )
        pairs = behavior.extraction.prompt_pairs
        return tuple(p.exhibit for p in pairs), tuple(p.not_exhibit for p in pairs)
    return (
        behavior.elicitation.exhibit_instructions,
        behavior.elicitation.not_exhibit_instructions,
    )


def _compose_positive_requests(
    behavior: Behavior,
    context_C: Context,
    questions: Sequence[tuple[str, str]],  # (question_id, question)
    n_requests: int,
    rng,
    instruction_style: str = "tagged",
    variants: Sequence[str] | None = None,
) -> list[GenRequest]:
    if variants is None:  # back-compat default: the elicitation source
        variants = behavior.elicitation.exhibit_instructions
    grid = [(qid, q, vi) for (qid, q) in questions for vi in range(len(variants))]
    picks = _sample_grid(grid, n_requests, rng)
    reqs: list[GenRequest] = []
    for i, (qid, q, vi) in enumerate(picks):
        emit = context_C.messages(q)
        gen = _inject_instruction(emit, variants[vi], instruction_style)
        reqs.append(
            _mk_request(
                f"pos-{i:05d}",
                POSITIVE,
                qid,
                f"ev{vi}",
                q,
                gen,
                emit,
                instruction=variants[vi],
                instruction_style=instruction_style,
            )
        )
    return reqs


_UNSET = object()  # sentinel: "caller passed nothing" (None is a legal value below)


def _compose_negative_requests(
    behavior: Behavior,
    panel: Panel,
    questions: Sequence[tuple[str, str]],
    n_requests_per_member: int,
    rng,
    instruction_style: str = "tagged",
    not_exhibit: Sequence[str] | None | object = _UNSET,
) -> list[GenRequest]:
    if not_exhibit is _UNSET:  # back-compat default: the elicitation source
        not_exhibit = behavior.elicitation.not_exhibit_instructions  # may be None
    reqs: list[GenRequest] = []
    i = 0
    for member in panel:
        if not_exhibit is None:
            grid = [(qid, q, None) for (qid, q) in questions]
        else:
            grid = [(qid, q, vi) for (qid, q) in questions for vi in range(len(not_exhibit))]
        picks = _sample_grid(grid, n_requests_per_member, rng)
        for qid, q, vi in picks:
            emit = member.messages(q)
            instr = None if vi is None else not_exhibit[vi]
            gen = _inject_instruction(emit, instr, instruction_style)
            variant_id = member.slug if vi is None else f"{member.slug}-nv{vi}"
            reqs.append(
                _mk_request(
                    f"neg-{i:05d}",
                    NEGATIVE,
                    qid,
                    variant_id,
                    q,
                    gen,
                    emit,
                    instruction=instr,
                    instruction_style=instruction_style,
                )
            )
            i += 1
    return reqs


def _sample_grid(grid: list, n: int, rng) -> list:
    """Pick ``n`` from ``grid`` — without replacement when it fits, else with
    replacement (temperature-1.0 generation makes repeats distinct; multiplicity
    is recorded in pool_meta)."""
    if n <= len(grid):
        shuffled = list(grid)
        rng.shuffle(shuffled)
        return shuffled[:n]
    return [grid[rng.randrange(len(grid))] for _ in range(n)]


def _mk_request(
    request_id,
    arm,
    question_id,
    variant_id,
    question,
    gen,
    emit,
    *,
    instruction: str | None = None,
    instruction_style: str = "tagged",
) -> GenRequest:
    if "__" in request_id:
        raise ValueError(
            f"request_id must not contain '__' (judge custom_id delimiter): {request_id!r}"
        )
    # Parity guard (defensive): stripping the instruction recovers emit exactly.
    # ValueError (not bare assert) so it survives `python -O` (behavior.py convention).
    if _strip_instruction(gen, instruction=instruction, style=instruction_style) != emit:
        raise ValueError(f"context-parity broken for {request_id}: strip(gen) != emit")
    return GenRequest(request_id, arm, question_id, variant_id, question, gen, emit)


# ── Generation (default dispatcher-backed generate_fn) ───────────────────────


def _gen_params_from_messages(
    messages: list[dict[str, str]], *, model: str, max_tokens: int, temperature: float
) -> dict:
    """Anthropic Messages params from an OpenAI-style message list (system lift).

    The Anthropic Messages API rejects ``system`` as a message ROLE (HTTP 400 on
    every request — the #906 first-``--full`` crash): system content must ride
    the TOP-LEVEL ``system=`` parameter. Datagen ``gen_messages`` begin with a
    ``{"role": "system", ...}`` persona+elicitation entry, so this helper lifts
    every system-role entry out (joined with a blank line, order preserved) and
    passes the non-system remainder as ``messages``.

    Returns a params dict with NO ``system`` key when the input carries no
    system entry (byte-identical behavior to the pre-fix path for system-less
    message lists). Raises ``ValueError`` on an empty non-system remainder —
    the API rejects an empty ``messages`` list anyway, so fail loud locally.
    """
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    remainder = [m for m in messages if m.get("role") != "system"]
    if not remainder:
        raise ValueError(
            "message list contains no non-system messages — the Anthropic Messages "
            f"API requires at least one user turn (got {len(messages)} system-only messages)"
        )
    params: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": remainder,
    }
    if system_parts:
        params["system"] = "\n\n".join(system_parts)
    return params


def _default_generate_fn(
    *, gen_model: str, gen_temperature: float, cache_dir: Path, checkpoint_dir: Path
) -> GenerateFn:
    """The real generation seam: fan requests across the multi-org dispatcher.

    Not exercised by the CPU/mock test suite (it makes live API calls); the
    optional env-gated live smoke covers the wire path (plan §5).
    """
    from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls

    def generate(requests: list[GenRequest]) -> list[GenCandidate]:
        items = [
            DispatchItem(item_id=r.request_id, payload={"messages": r.gen_messages})
            for r in requests
        ]

        def build_request(item: DispatchItem) -> dict:
            return _gen_params_from_messages(
                item.payload["messages"],
                model=gen_model,
                max_tokens=DEFAULT_GEN_MAX_TOKENS,
                temperature=gen_temperature,
            )

        results = asyncio.run(
            dispatch_calls(
                items,
                model=gen_model,
                build_request=build_request,
                parse_response=lambda text: text,
                cache_dir=cache_dir,
                checkpoint_dir=checkpoint_dir,
            )
        )
        candidates: list[GenCandidate] = []
        for r in requests:
            res = results.get(r.request_id)
            if res is None or res.error:
                # A dispatch error (transient/parse/rate-limit-exhausted) or a missing
                # result is a TRANSPORT failure, NOT a content refusal — classify it
                # under api_error so an API outage cannot inflate refusal_drops toward
                # the yield floor. A genuine content refusal comes back as completion
                # text and is handled downstream by the judge (scores low -> dropped).
                candidates.append(GenCandidate(r, None, drop_reason="api_error"))
            elif not (res.result and str(res.result).strip()):
                candidates.append(GenCandidate(r, None, drop_reason="empty"))
            else:
                candidates.append(GenCandidate(r, str(res.result)))
        return candidates

    return generate


# ── Raw-candidate + manifest checkpoint IO ───────────────────────────────────


def _write_raw(path: Path, candidates: list[GenCandidate]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            r = c.request
            f.write(
                json.dumps(
                    {
                        "request_id": r.request_id,
                        "arm": r.arm,
                        "question_id": r.question_id,
                        "variant_id": r.variant_id,
                        "question": r.question,
                        "gen_messages": r.gen_messages,
                        "emit_messages": r.emit_messages,
                        "completion": c.completion,
                        "drop_reason": c.drop_reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _read_raw(path: Path) -> list[GenCandidate]:
    out: list[GenCandidate] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            req = GenRequest(
                d["request_id"],
                d["arm"],
                d["question_id"],
                d["variant_id"],
                d["question"],
                d["gen_messages"],
                d["emit_messages"],
            )
            out.append(GenCandidate(req, d["completion"], d.get("drop_reason")))
    return out


def _build_manifest(
    *,
    behavior: Behavior,
    context_C: Context,
    panel: Panel,
    target_n: int,
    quota_floor: float,
    n_judge_draws: int,
    seed: int,
    gen_model: str,
    gen_temperature: float,
    instruction_style: str,
    instruction_source: str,
    exhibit_instructions: Sequence[str],
    not_exhibit_instructions: Sequence[str] | None,
    oversample_mult: float = 1.0,
) -> dict:
    train_bank, ts, te = banks.SLICES[(behavior.name, "train")]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "behavior": behavior.name,
        "train_bank": train_bank,
        "train_bank_sha": banks.bank_sha(train_bank),
        "train_slice": f"{train_bank}[{ts}:{te}]",
        # The RESOLVED instruction lists (per instruction_source, #1090) — a
        # source flip changes these lists AND the source token below, so the
        # manifest resume key invalidates on either.
        "exhibit_instructions": list(exhibit_instructions),
        "not_exhibit_instructions": (
            None if not_exhibit_instructions is None else list(not_exhibit_instructions)
        ),
        "instruction_source": instruction_source,
        "context_id": context_C.context_id,
        "context_fingerprint": _context_fingerprint(context_C),
        "panel": [m.slug for m in panel],
        "target_n": target_n,
        # quota_floor (→ floor_n) and n_judge_draws are part of the resume key: a
        # re-invocation with either changed invalidates the manifest (and the
        # manifest-hash-keyed judge cache), so a changed draw count re-judges
        # fresh instead of silently reusing scores computed at the old count.
        "quota_floor": quota_floor,
        "n_judge_draws": n_judge_draws,
        "seed": seed,
        "gen_model": gen_model,
        "gen_temperature": gen_temperature,
        # Part of the resume key (#1074): a style flip re-generates fresh
        # instead of silently reusing candidates injected under the other style.
        "instruction_style": instruction_style,
        # Part of the resume key (#1090 round 4): a changed positive request
        # budget re-generates fresh rather than replaying the smaller raw
        # cache (pre-knob manifests normalize to 1.0 at the resume check).
        "oversample_mult": oversample_mult,
    }


# ── Judge-filter ─────────────────────────────────────────────────────────────


def _judge_and_filter(
    behavior: Behavior,
    candidates: list[GenCandidate],
    arm: str,
    *,
    judge_fn: JudgeFn,
    n_judge_draws: int,
    cache_dir: Path,
    save_raw: Path,
) -> tuple[list[GenCandidate], _ArmDrops, JudgeResult, dict[str, tuple[float | None, bool]]]:
    """Judge every non-refusal candidate; keep pos>threshold / neg<threshold with
    drop-never-coerce; apply the structural keep-check for structural behaviors.

    Returns (kept, drops, judge_result, per_request_scoreinfo) where scoreinfo
    maps request_id -> (mean_score, kept) for the judge_rows sidecar.
    """
    drops = _ArmDrops(requested=len(candidates))
    for c in candidates:
        drops.variant_usage[c.request.variant_id] += 1
        drops.question_multiplicity[c.request.question_id] += 1
        if c.completion is None:
            # Split the generation-stage drops by reason so a transport-side API
            # outage (api_error) never inflates the content-refusal count.
            if c.drop_reason == "api_error":
                drops.api_error_drops += 1
            elif c.drop_reason == "empty":
                drops.empty_drops += 1
            else:  # "refusal" or unlabeled -> a model-side content non-answer
                drops.refusal_drops += 1
    judgeable = [c for c in candidates if c.completion is not None]
    drops.generated = len(judgeable)

    items = [(c.request.request_id, c.request.question, c.completion) for c in judgeable]
    result = judge_fn(
        items,
        behavior.judge_rubric,
        n_draws=n_judge_draws,
        cache_dir=cache_dir,
        save_raw=save_raw,
        judge_model=behavior.judge_model,
        # Explicit at the call site (never the 64-token library default): the
        # reason-then-JSON filter rubrics truncate-and-parse-drop under 64
        # (#1090 fu3 crash-fix; llm-judging rule 23).
        max_tokens=DATAGEN_JUDGE_MAX_TOKENS,
    )
    predicate = _STRUCTURAL_PREDICATES.get(behavior.name)
    threshold = behavior.threshold
    kept: list[GenCandidate] = []
    scoreinfo: dict[str, tuple[float | None, bool]] = {}
    for c in judgeable:
        rid = c.request.request_id
        score = result.scores.get(rid)
        keep = False
        if score is None:
            drops.judge_none_drops += 1
        else:
            passes_judge = score > threshold if arm == POSITIVE else score < threshold
            if not passes_judge:
                drops.threshold_drops += 1
            elif predicate is not None:
                # Positives must ALSO be structural; kept negatives must NOT be.
                is_struct = predicate(c.completion)
                if (arm == POSITIVE) == is_struct:
                    keep = True
                else:
                    drops.structural_drops += 1
            else:
                keep = True
        scoreinfo[rid] = (score, keep)
        if keep:
            drops.variant_kept[c.request.variant_id] += 1
            kept.append(c)
    return kept, drops, result, scoreinfo


# ── Positive-schedule replay + reused-pool reconstruction (#1074) ────────────


def compose_positive_schedule(
    behavior: Behavior,
    context_C: Context,
    *,
    target_n: int,
    seed: int,
    instruction_style: str = "tagged",
    variants: Sequence[str] | None = None,
    oversample_mult: float = 1.0,
) -> tuple[list[GenRequest], object, list[tuple[str, str]], int]:
    """The deterministic stage-3a positive request schedule (RNG-replay surface).

    Returns ``(pos_reqs, rng, train_questions, n_pos_req)``.
    :func:`generate_training_data` delegates its positive composition here, so
    an external replay (a smoke fixture builder, a reuse verification) composes
    EXACTLY the schedule the pipeline uses; the returned ``rng`` has consumed
    exactly the positive-composition draws, so negative composition continues
    from the identical state in both the producing and the reusing run. The
    #1090 knobs are part of the schedule identity and thread through here:
    ``variants`` (the RESOLVED exhibit-instruction list per
    ``instruction_source``; ``None`` = the elicitation back-compat default)
    sets the variant grid, and ``oversample_mult`` scales the positive request
    budget — a replay MUST pass the producing run's values or the recomposed
    schedule (length / per-row ids) will not match.
    """
    rng = _rng(seed)
    # Positive budget = the EXPECTED_YIELD-derived count x the oversample knob
    # (#1090 round 4; 36 -> 72 at 2.0 for target 25); the NEGATIVE budget is
    # deliberately NOT scaled — the P1a floor misses were positive-arm
    # keep-rate misses (36-39% realized vs the 70% EXPECTED_YIELD assumption),
    # and negatives are generated per panel member on the emitted-positive
    # questions.
    n_pos_req = math.ceil(math.ceil(target_n / EXPECTED_YIELD) * oversample_mult)
    train_questions = [
        (f"{behavior.name}-trainq-{i:04d}", q) for i, q in enumerate(behavior.train_question_bank)
    ]
    reqs = _compose_positive_requests(
        behavior, context_C, train_questions, n_pos_req, rng, instruction_style, variants=variants
    )
    return reqs, rng, train_questions, n_pos_req


def _assert_replay_schedule(pos_cands: list[GenCandidate], pos_reqs: list[GenRequest]) -> None:
    """The RNG-replay integrity check: the staged pool must BE the recomposed
    schedule — same length, same (request_id, question_id, variant_id) per row."""
    if len(pos_cands) != len(pos_reqs):
        raise ValueError(
            f"pos_reuse RNG-replay mismatch: staged raw_pos has {len(pos_cands)} rows, the "
            f"recomposed schedule has {len(pos_reqs)} — the staged pool is not the schedule "
            "this recipe produces (bank/seed/target_n/variant drift)"
        )
    for i, (c, r) in enumerate(zip(pos_cands, pos_reqs, strict=True)):
        got = (c.request.request_id, c.request.question_id, c.request.variant_id)
        want = (r.request_id, r.question_id, r.variant_id)
        if got != want:
            raise ValueError(
                f"pos_reuse RNG-replay mismatch at row {i}: staged {got} != recomposed {want}"
            )


def _pos_drops_from_candidates(pos_cands: list[GenCandidate], n_judgeable: int) -> _ArmDrops:
    """Reconstruct the generation-stage drop counters from staged candidates."""
    drops = _ArmDrops(requested=len(pos_cands), generated=n_judgeable)
    for c in pos_cands:
        drops.variant_usage[c.request.variant_id] += 1
        drops.question_multiplicity[c.request.question_id] += 1
        if c.completion is None:
            if c.drop_reason == "api_error":
                drops.api_error_drops += 1
            elif c.drop_reason == "empty":
                drops.empty_drops += 1
            else:
                drops.refusal_drops += 1
    return drops


def _reconstruct_kept(
    jrows: list[dict],
    judgeable: list[GenCandidate],
    behavior: Behavior,
    drops: _ArmDrops,
) -> tuple[list[GenCandidate], dict[str, tuple[float | None, bool]]]:
    """Kept set + scoreinfo from staged judge_rows, with the recomputed
    ``mean > threshold`` + structural rule as a free integrity check."""
    predicate = _STRUCTURAL_PREDICATES.get(behavior.name)
    threshold = behavior.threshold
    kept: list[GenCandidate] = []
    scoreinfo: dict[str, tuple[float | None, bool]] = {}
    by_rid = {r["request_id"]: r for r in jrows}
    for c in judgeable:
        row = by_rid[c.request.request_id]
        mean = row["mean"]
        stored_kept = bool(row["kept"])
        if mean is None:
            recomputed = False
            drops.judge_none_drops += 1
        elif not mean > threshold:
            recomputed = False
            drops.threshold_drops += 1
        elif predicate is not None and not predicate(c.completion):
            recomputed = False
            drops.structural_drops += 1
        else:
            recomputed = True
        if recomputed != stored_kept:
            raise ValueError(
                f"pos_reuse: stored kept flag for {c.request.request_id} disagrees with the "
                f"recomputed keep rule (stored={stored_kept}, recomputed={recomputed}, "
                f"mean={mean}, threshold={threshold}) — keep rule or judge_rows drifted"
            )
        scoreinfo[c.request.request_id] = (mean, stored_kept)
        if stored_kept:
            kept.append(c)
    return kept, scoreinfo


def _load_reused_positives(
    reuse: PosReuseSpec,
    pos_reqs: list[GenRequest],
    behavior: Behavior,
    *,
    n_judge_draws: int,
    out_raw_pos_path: Path,
) -> tuple[
    list[GenCandidate],
    list[GenCandidate],
    _ArmDrops,
    JudgeResult,
    dict[str, tuple[float | None, bool]],
]:
    """Load + verify a reused positive pool; NEVER re-judges.

    Fail-loud contract: missing staged files (FileNotFoundError); a staged row
    set that is not the recomposed schedule — length or per-row
    (request_id, question_id, variant_id) mismatch (ValueError, the RNG-replay
    integrity check); judge_rows positive rows not matching the raw judgeable
    rows in raw-file order (ValueError); a stored kept flag disagreeing with
    the recomputed ``mean > threshold`` + structural rule (ValueError);
    reconstructed kept count != ``expected_kept_count`` (ValueError). The
    kept < floor_n case stays :class:`DatagenYieldError` in the caller.

    The rebuilt :class:`JudgeResult`'s ``n_dropped_draws`` is a BLENDED
    reconstruction — content drops plus any transport losses (the rule-24(ii)
    split is not recoverable from the staged sidecar); see the inline note at
    the construction site below.
    """
    raw_src, rows_src = Path(reuse.raw_pos_path), Path(reuse.judge_rows_path)
    for p, name in ((raw_src, "raw_pos"), (rows_src, "judge_rows")):
        if not p.exists():
            raise FileNotFoundError(f"pos_reuse staged {name} file missing: {p}")
    pos_cands = _read_raw(raw_src)
    _assert_replay_schedule(pos_cands, pos_reqs)
    # Stage the pool into out_dir verbatim (checkpoint parity: downstream
    # consumers — per-question yield, margin pools — read datagen_dir files).
    if not out_raw_pos_path.exists():
        shutil.copyfile(raw_src, out_raw_pos_path)

    jrows: list[dict] = []
    with open(rows_src, encoding="utf-8") as f:  # text-mode iteration, never splitlines()
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row["arm"] == POSITIVE:
                    jrows.append(row)
    judgeable = [c for c in pos_cands if c.completion is not None]
    if [r["request_id"] for r in jrows] != [c.request.request_id for c in judgeable]:
        raise ValueError(
            "pos_reuse: judge_rows positive rows do not match raw_pos judgeable rows in "
            "raw-file order — refusing to reconstruct the kept set"
        )
    drops = _pos_drops_from_candidates(pos_cands, len(judgeable))
    kept, scoreinfo = _reconstruct_kept(jrows, judgeable, behavior, drops)
    if len(kept) != reuse.expected_kept_count:
        raise ValueError(
            f"pos_reuse: reconstructed kept count {len(kept)} != expected "
            f"{reuse.expected_kept_count}"
        )
    per_item_scores = {r["request_id"]: list(r.get("scores", [])) for r in jrows}
    draw_counts = {r["request_id"]: int(r.get("n_kept_draws", 0)) for r in jrows}
    n_total = n_judge_draws * len(jrows)  # reconstructed arithmetic, flagged in pool_meta
    # BLENDED estimate (#1313, concern datagen-blended-drop-estimate): the staged
    # judge_rows sidecar persists only KEPT per-draw scores + n_kept_draws — no
    # per-draw error dicts and no save_raw — so the rule-24(ii) content-vs-transport
    # split (graded_judge.judge_result_from_save_raw / batch_judge.is_transport_error_dict)
    # is NOT reconstructible on this reuse path. ``n_dropped_draws`` below blends the
    # original run's content drops with any transport losses (+ dispatch slack), and
    # ``n_transport_lost_draws`` stays at its 0 default meaning "unsplit", NOT
    # "measured zero". pool_meta flags this via ``judge_draw_stats_reconstructed``.
    jr = JudgeResult(
        scores={rid: m for rid, (m, _k) in scoreinfo.items()},
        n_total_draws=n_total,
        n_dropped_draws=max(0, n_total - sum(draw_counts.values())),
        per_item_draw_counts=draw_counts,
        per_item_scores=per_item_scores,
    )
    return pos_cands, kept, drops, jr, scoreinfo


def _positives_stage(
    behavior: Behavior,
    context_C: Context,
    *,
    target_n: int,
    seed: int,
    instruction_style: str,
    variants: Sequence[str] | None = None,
    oversample_mult: float = 1.0,
    reuse_pos: PosReuseSpec | None,
    raw_pos_path: Path,
    gen_factory: Callable[[str], GenerateFn],
    judge: JudgeFn,
    n_judge_draws: int,
    judge_cache: Path,
    out_dir: Path,
):
    """Stage 3a: schedule composition (the RNG-replay surface) + generate /
    resume / reuse + judge-filter (or kept-set reconstruction under
    ``reuse_pos``). Returns ``(pos_cands, pos_kept, pos_drops, pos_jr,
    pos_scores, rng)`` — the rng continues into negative composition. The
    #1090 ``variants`` / ``oversample_mult`` knobs forward into the schedule
    (they are part of its identity — see :func:`compose_positive_schedule`)."""
    pos_reqs, rng, _train_questions, _n_pos_req = compose_positive_schedule(
        behavior,
        context_C,
        target_n=target_n,
        seed=seed,
        instruction_style=instruction_style,
        variants=variants,
        oversample_mult=oversample_mult,
    )
    if reuse_pos is not None:
        return (
            *_load_reused_positives(
                reuse_pos,
                pos_reqs,
                behavior,
                n_judge_draws=n_judge_draws,
                out_raw_pos_path=raw_pos_path,
            ),
            rng,
        )
    if raw_pos_path.exists():
        pos_cands = _read_raw(raw_pos_path)
    else:
        pos_cands = gen_factory("pos")(pos_reqs)
        _write_raw(raw_pos_path, pos_cands)  # persist raw the moment it returns
    pos_kept, pos_drops, pos_jr, pos_scores = _judge_and_filter(
        behavior,
        pos_cands,
        POSITIVE,
        judge_fn=judge,
        n_judge_draws=n_judge_draws,
        cache_dir=judge_cache / "pos",
        save_raw=out_dir / "judge_raw_pos.json",
    )
    return pos_cands, pos_kept, pos_drops, pos_jr, pos_scores, rng


def _negative_arm(
    behavior: Behavior,
    panel: Panel,
    emitted_questions: list[tuple[str, str]],
    n_neg_req_per_member: int,
    rng,
    instruction_style: str,
    *,
    not_exhibit_instructions: list[str],
    raw_neg_path: Path,
    gen_fn,
    judge: JudgeFn,
    n_judge_draws: int,
    judge_cache: Path,
    out_dir: Path,
) -> tuple[list[GenCandidate], list[GenCandidate], _ArmDrops, JudgeResult, dict]:
    """The negative-arm generate+judge stage of ``generate_training_data``.

    An EMPTY ``panel`` is the sanctioned pos-only twin (#1090 fu3): zero
    negative rows BY DESIGN — the raw/judge sidecar surfaces stay
    empty-but-present so resume + downstream readers see the same file set as
    a contrastive run. Returns
    ``(neg_cands, neg_kept, neg_drops, neg_jr, neg_scores)``.
    """
    if not panel:
        if not raw_neg_path.exists():
            _write_raw(raw_neg_path, [])
        return (
            [],
            [],
            _ArmDrops(requested=0),
            JudgeResult(scores={}, n_total_draws=0, n_dropped_draws=0),
            {},
        )
    if raw_neg_path.exists():
        neg_cands = _read_raw(raw_neg_path)
    else:
        neg_reqs = _compose_negative_requests(
            behavior,
            panel,
            emitted_questions,
            n_neg_req_per_member,
            rng,
            instruction_style,
            not_exhibit=not_exhibit_instructions,
        )
        neg_cands = gen_fn("neg")(neg_reqs)
        _write_raw(raw_neg_path, neg_cands)
    neg_kept, neg_drops, neg_jr, neg_scores = _judge_and_filter(
        behavior,
        neg_cands,
        NEGATIVE,
        judge_fn=judge,
        n_judge_draws=n_judge_draws,
        cache_dir=judge_cache / "neg",
        save_raw=out_dir / "judge_raw_neg.json",
    )
    return neg_cands, neg_kept, neg_drops, neg_jr, neg_scores


# ── Public entry point ───────────────────────────────────────────────────────


def _validate_scalar_fences(
    quota_floor: float, oversample_mult: float, max_oversample_mult: float = 2.0
) -> None:
    """Fail loud on out-of-fence scalar knobs: quota_floor in (0, 1];
    oversample_mult in [1.0, max_oversample_mult] (default 2.0 — the #1090
    plan's 2x request-count fence; the #1090-fu3 posonly x bare yield-floor
    carve-out widens it per-call — never a silent clamp, never an
    undersample)."""
    if not 0.0 < quota_floor <= 1.0:
        raise ValueError(f"quota_floor must be in (0, 1], got {quota_floor}")
    if not max_oversample_mult >= 1.0:
        raise ValueError(f"max_oversample_mult must be >= 1.0, got {max_oversample_mult}")
    if not 1.0 <= oversample_mult <= max_oversample_mult:
        raise ValueError(
            f"oversample_mult must be in [1.0, {max_oversample_mult}] (the request-count "
            f"fence), got {oversample_mult}"
        )


def _resume_or_quarantine_manifest(
    out_dir: Path,
    manifest: dict,
    manifest_path: Path,
    raw_pos_path: Path,
    raw_neg_path: Path,
) -> None:
    """Resume-key gate for the raw-candidate cache (writes/validates gen_manifest.json).

    Exact match -> resume the raw cache. A mult-ONLY change (#1090 fu3 crash-fix 2:
    a yield-miss retry at a recalibrated request budget) QUARANTINES the stale raw
    candidates to ``stale_mult_<old>/`` (never deletes — persist-by-default) and
    regenerates fresh under the new manifest. Any OTHER drift raises
    :class:`DatagenCheckpointMismatchError` naming the changed keys.
    """
    if not manifest_path.exists():
        manifest_path.write_text(_canonical(manifest) + "\n")
        return
    existing = json.loads(manifest_path.read_text())
    # Back-compat (#1090 round 4): a pre-knob v2 manifest lacks the key and
    # was generated at the implicit 1.0 budget — normalize instead of
    # invalidating every completed dir, so a mult-1.0 re-run resumes.
    existing.setdefault("oversample_mult", 1.0)
    if existing == manifest:
        return
    diff_keys = {k for k in set(existing) | set(manifest) if existing.get(k) != manifest.get(k)}
    if diff_keys != {"oversample_mult"}:
        raise DatagenCheckpointMismatchError(
            f"gen_manifest.json in {out_dir} does not match the current args "
            f"(changed keys: {sorted(diff_keys)}). "
            "Refusing to reuse stale raw candidates; use a fresh out_dir."
        )
    stale_dir = out_dir / f"stale_mult_{existing['oversample_mult']}"
    stale_dir.mkdir(parents=True, exist_ok=True)
    for p in (manifest_path, raw_pos_path, raw_neg_path):
        if p.exists():
            p.rename(stale_dir / p.name)
    logger.warning(
        "oversample_mult changed %s -> %s in %s: quarantined stale raw "
        "candidates to %s and regenerating at the new budget.",
        existing["oversample_mult"],
        manifest["oversample_mult"],
        out_dir,
        stale_dir,
    )
    manifest_path.write_text(_canonical(manifest) + "\n")


def generate_training_data(
    behavior: Behavior,
    context_C: Context,
    negatives: Panel | str = DEFAULT_PANEL_NAME,
    *,
    out_dir: Path,
    target_n: int = 200,
    quota_floor: float = 0.8,
    n_judge_draws: int = 5,
    gen_model: str = DEFAULT_JUDGE_MODEL,
    gen_temperature: float = 1.0,
    seed: int = 866,
    generate_fn: GenerateFn | None = None,
    judge_fn: JudgeFn | None = None,
    instruction_style: str = "tagged",
    instruction_source: str = "elicitation",
    oversample_mult: float = 1.0,
    max_oversample_mult: float = 2.0,
    reuse_pos: PosReuseSpec | None = None,
) -> tuple[Path, Path, Path]:
    """Build contrastive (positive, contrast) training JSONL for ``behavior``.

    Returns ``(pos_jsonl_path, cn_jsonl_path, pool_meta_path)``. Emits EXACTLY
    ``floor_n = ceil(quota_floor * target_n)`` positives (cross-cell dose equality
    by construction) and per-panel-member negative quotas (~1:1 total via the 0c
    allocation helpers, same-question-paired to the emitted positives).
    ``instruction_style`` ("tagged" default | "plain", #1074) picks how the
    generation-only elicitation instruction is injected into ``gen_messages``;
    ``emit_messages`` is style-independent (context parity by construction) and
    the style enters ``gen_manifest.json`` (resume-key protection).
    ``instruction_source`` ("elicitation" default | "extraction_pairs", #1090)
    picks WHERE the generation-only instructions come from — the registered
    ``ElicitationSpec`` (byte-identical pre-#1090 behavior) or the
    persona-vectors ``extraction.prompt_pairs`` (positives rotate the pair
    ``exhibit`` texts, negatives the ``not_exhibit`` texts); the resolved lists
    + source enter the manifest resume key. ``oversample_mult`` (#1090 round 4,
    the plan's ALLOWED "oversample/request-count retuning within 2x" deviation)
    multiplies the POSITIVE request budget ``ceil(target_n / EXPECTED_YIELD)``
    (36 for target 25 -> 72 at 2.0); it is fenced to [1.0, 2.0] and enters the
    manifest resume key (a pre-knob manifest reads as 1.0, so mult-1.0 resumes
    replay and a changed mult refuses the stale raw cache). An explicitly
    EMPTY ``negatives`` panel is the sanctioned pos-only twin (#1090 fu3):
    no negative rows are generated and ``cn.jsonl`` is written empty. Raises
    :class:`ValueError` on a programmatic behavior, unknown style/source, or an
    out-of-fence ``oversample_mult``, :class:`DatagenYieldError` below the
    floor, and :class:`DatagenCheckpointMismatchError` on a stale resume.

    ``reuse_pos`` (:class:`PosReuseSpec`, #1074 ``base-negatives-regen``):
    reuse a prior run's positive pool verbatim — positive generation + judging
    are SKIPPED (never re-judged), the kept set is reconstructed from the
    staged ``judge_rows.jsonl`` kept flags, and the positive schedule is still
    composed + asserted against the staged rows (RNG-state replay) so the
    negative schedule is identical to the producing run's. The manifest +
    pool_meta gain an additive ``pos_reuse`` provenance block (it enters the
    exact-match resume key by construction); ``gen_model`` remains the LIVE
    (negative-stage) generator. Emission subsample + all negative stages are
    byte-identical code paths. The schedule is recomposed under THIS call's
    ``instruction_source`` / ``oversample_mult``, so a pool staged under
    different knob values fails loud — the RNG-replay length/row assert plus
    the manifest resume key (both knobs enter it) refuse the mismatch.
    """
    # 1. Resolve + guard.
    behavior.validate()
    if instruction_style not in INSTRUCTION_STYLES:
        raise ValueError(f"instruction_style {instruction_style!r} not in {INSTRUCTION_STYLES}")
    exhibit_instructions, not_exhibit_instructions = _resolve_instructions(
        behavior, instruction_source
    )
    if behavior.programmatic:
        raise ValueError(
            f"behavior {behavior.name!r} is programmatic (tier-4 carve-out) — "
            "programmatic behaviors never route through the unified datagen pipeline"
        )
    # ``max_oversample_mult`` (default 2.0 — byte-identical round-1 fence) is
    # the #1090-fu3 posonly x bare carve-out: the fu3 worker passes a wider
    # fence so a measured-keep-rate budget (mult 5.0) can clear the yield
    # floor. The fence never enters the manifest resume key (a budget retune
    # deliberately re-runs in the same regime — see the oversample_mult note).
    _validate_scalar_fences(quota_floor, oversample_mult, max_oversample_mult)
    # #1090 fu3: an explicitly-passed EMPTY panel is the sanctioned pos-only
    # twin (neg_ratio=0 — no negative rows; the generic interleave happens
    # downstream in the mix assembler). None / malformed members still fail
    # fast (tuple(None) raises TypeError; the member type-check below).
    panel: Panel = get_panel(negatives) if isinstance(negatives, str) else tuple(negatives)
    for m in panel:
        if not isinstance(m, NegativeContext):
            raise TypeError(f"panel members must be NegativeContext, got {type(m).__name__}")
    posonly = not panel
    if not posonly:
        assert_panel_disjoint_from_sources(panel, [context_C.context_id])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    floor_n = math.ceil(quota_floor * target_n)
    member_quota = 0 if posonly else per_negative_quota(floor_n, panel)
    n_neg_req_per_member = math.ceil(member_quota / EXPECTED_YIELD)
    # (train_questions, n_pos_req — including the #1090 oversample_mult scaling
    # of the POSITIVE budget — and the shared RNG come from
    # compose_positive_schedule at step 3a, the #1074 RNG-replay surface. The
    # NEGATIVE budget above is deliberately NOT scaled by oversample_mult; see
    # the comment in compose_positive_schedule.)

    # 2. Manifest (written before generation; resume ONLY on an exact match).
    manifest = _build_manifest(
        behavior=behavior,
        context_C=context_C,
        panel=panel,
        target_n=target_n,
        quota_floor=quota_floor,
        n_judge_draws=n_judge_draws,
        seed=seed,
        gen_model=gen_model,
        gen_temperature=gen_temperature,
        instruction_style=instruction_style,
        instruction_source=instruction_source,
        exhibit_instructions=exhibit_instructions,
        not_exhibit_instructions=not_exhibit_instructions,
        oversample_mult=oversample_mult,
    )
    if reuse_pos is not None:
        # Additive provenance (source repo/path/revision, pos generator, staged
        # sha256s) — enters the exact-match resume key by construction.
        manifest["pos_reuse"] = reuse_pos.manifest_fields()
    manifest_hash = hashlib.sha256(_canonical(manifest).encode("utf-8")).hexdigest()
    manifest_path = out_dir / "gen_manifest.json"
    raw_pos_path = out_dir / "raw_pos.jsonl"
    raw_neg_path = out_dir / "raw_neg.jsonl"
    _resume_or_quarantine_manifest(out_dir, manifest, manifest_path, raw_pos_path, raw_neg_path)

    def _gen(phase: str) -> GenerateFn:
        return generate_fn or _default_generate_fn(
            gen_model=gen_model,
            gen_temperature=gen_temperature,
            cache_dir=out_dir / "gen_cache",
            checkpoint_dir=out_dir / f"gen_ckpt_{phase}",
        )

    judge = judge_fn or judge_graded
    # max_tokens enters the CACHE-DIR key (the rubric fingerprint deliberately
    # excludes it — batch_judge.rubric_fingerprint), so truncation-era 64-token
    # entries in the old judge_cache_<hash> dirs are unreachable: a budget
    # change is a cold re-judge, never a re-served truncated draw (llm-judging
    # rule 23; #1090 fu3 crash-fix).
    judge_cache = out_dir / f"judge_cache_{manifest_hash[:12]}_mt{DATAGEN_JUDGE_MAX_TOKENS}"

    # 3a. POSITIVES: compose over the train bank (ALWAYS — the composition
    # advances the shared RNG that the negative schedule continues from), then
    # generate (or resume) + judge-filter — or, under ``reuse_pos``, verify the
    # staged pool against the recomposed schedule and reconstruct the kept set.
    pos_cands, pos_kept, pos_drops, pos_jr, pos_scores, rng = _positives_stage(
        behavior,
        context_C,
        target_n=target_n,
        seed=seed,
        instruction_style=instruction_style,
        variants=exhibit_instructions,
        oversample_mult=oversample_mult,
        reuse_pos=reuse_pos,
        raw_pos_path=raw_pos_path,
        gen_factory=_gen,
        judge=judge,
        n_judge_draws=n_judge_draws,
        judge_cache=judge_cache,
        out_dir=out_dir,
    )

    # 4a. Emit EXACTLY floor_n positives (seeded subsample) -> the emitted question set.
    if len(pos_kept) < floor_n:
        # NOTE (#1090 fu3 crash-fix 2): keep accounting is the cross-variant UNION —
        # every judge-kept (question, variant) row counts toward the floor. The old
        # message labeled variant_usage (REQUESTED per variant) as "yields", which
        # misread launch-3's 15/36 keep-rate miss as a variant-selection bug.
        breakeven_mult = float(oversample_mult) * floor_n / max(1, len(pos_kept))
        raise DatagenYieldError(
            f"behavior {behavior.name!r}: kept {len(pos_kept)} positives < floor_n={floor_n} "
            f"(target_n={target_n}, quota_floor={quota_floor}, "
            f"requested={pos_drops.requested}, judgeable={pos_drops.generated}, "
            f"keep_rate={len(pos_kept) / max(1, pos_drops.requested):.3f}). "
            f"Kept-per-variant (union counts toward the floor): {dict(pos_drops.variant_kept)}; "
            f"requested-per-variant: {dict(pos_drops.variant_usage)}. "
            f"Remedy: raise --oversample-mult to >= {breakeven_mult:.2f} "
            f"(break-even at the realized keep rate; add margin)."
        )
    emitted_pos = _seeded_sample(pos_kept, floor_n, _rng(seed + 1))
    emitted_pos_qids = {c.request.question_id for c in emitted_pos}
    emitted_questions = _dedup_questions(emitted_pos)

    # 3b. NEGATIVES: generated ON the emitted-positive questions (same-question
    # pairing by construction, criterion 8) for every panel member, then judge-filter.
    neg_cands, neg_kept, neg_drops, neg_jr, neg_scores = _negative_arm(
        behavior,
        panel,
        emitted_questions,
        n_neg_req_per_member,
        rng,
        instruction_style,
        not_exhibit_instructions=not_exhibit_instructions,
        raw_neg_path=raw_neg_path,
        gen_fn=_gen,
        judge=judge,
        n_judge_draws=n_judge_draws,
        judge_cache=judge_cache,
        out_dir=out_dir,
    )
    _write_judge_rows(
        out_dir / "judge_rows.jsonl",
        pos_cands,
        neg_cands,
        pos_scores,
        neg_scores,
        {**pos_jr.per_item_draw_counts, **neg_jr.per_item_draw_counts},
        {**pos_jr.per_item_scores, **neg_jr.per_item_scores},
    )

    # 5. Per-member negative quota (same-question paired; fail loud on shortfall).
    emitted_neg: list[GenCandidate] = []
    per_member_emitted: dict[str, int] = {}
    for member in panel:
        eligible = [
            c
            for c in neg_kept
            if c.request.variant_id.split("-nv")[0] == member.slug
            and c.request.question_id in emitted_pos_qids
        ]
        if len(eligible) < member_quota:
            raise DatagenYieldError(
                f"behavior {behavior.name!r}: negative panel member {member.slug!r} kept "
                f"{len(eligible)} negatives on emitted-positive questions < quota={member_quota} "
                "(same-question pairing floor)"
            )
        chosen = _seeded_sample(eligible, member_quota, _rng(seed + 2 + panel.index(member)))
        emitted_neg.extend(chosen)
        per_member_emitted[member.slug] = len(chosen)

    # 6. Emit.
    pos_path = out_dir / "pos.jsonl"
    cn_path = out_dir / "cn.jsonl"
    _write_train_rows(pos_path, emitted_pos)
    _write_train_rows(cn_path, emitted_neg)

    pool_meta_path = out_dir / "pool_meta.json"
    pool_meta = _build_pool_meta(
        behavior=behavior,
        context_C=context_C,
        panel=panel,
        target_n=target_n,
        floor_n=floor_n,
        quota_floor=quota_floor,
        n_judge_draws=n_judge_draws,
        gen_model=gen_model,
        gen_temperature=gen_temperature,
        seed=seed,
        manifest=manifest,
        pos_drops=pos_drops,
        neg_drops=neg_drops,
        n_emitted_pos=len(emitted_pos),
        n_emitted_neg=len(emitted_neg),
        member_quota=member_quota,
        per_member_emitted=per_member_emitted,
        pos_jr=pos_jr,
        neg_jr=neg_jr,
    )
    if reuse_pos is not None:
        pool_meta["pos_reuse"] = {
            **manifest["pos_reuse"],
            # pos judge_draw_stats above are arithmetic reconstructions from the
            # staged judge_rows (n_judge_draws x judged rows), not a live count —
            # and their n_dropped BLENDS content drops with any transport losses
            # (the rule-24(ii) split is not reconstructible from the sidecar; see
            # _load_reused_positives).
            "judge_draw_stats_reconstructed": True,
        }
    pool_meta_path.write_text(json.dumps(pool_meta, ensure_ascii=False, indent=2) + "\n")
    return pos_path, cn_path, pool_meta_path


def _rng(seed: int):
    import random

    return random.Random(seed)


def _seeded_sample(pool: list[GenCandidate], k: int, rng) -> list[GenCandidate]:
    """Deterministic k-subsample of ``pool`` (k <= len(pool)); surplus discarded."""
    if k > len(pool):
        raise ValueError(f"_seeded_sample: k={k} > pool={len(pool)}")
    idx = list(range(len(pool)))
    rng.shuffle(idx)
    return [pool[i] for i in sorted(idx[:k])]


def _dedup_questions(cands: list[GenCandidate]) -> list[tuple[str, str]]:
    """The unique ``(question_id, question)`` pairs in ``cands``, first-seen order."""
    seen: dict[str, str] = {}
    for c in cands:
        seen.setdefault(c.request.question_id, c.request.question)
    return list(seen.items())


def _train_row(prompt_msgs: list[dict[str, str]], completion_text: str) -> dict:
    """The ``scripts.issue664_common.train_row`` shape, inlined (a 2-line dict) to
    avoid a src->scripts dependency + the #823 script-mode sys.path fragility that
    importing that heavy module would add. Rows validate against the same contract:
    ``{"prompt": [msg dicts], "completion": [{"role": "assistant", "content": str}]}``.
    """
    return {
        "prompt": prompt_msgs,
        "completion": [{"role": "assistant", "content": completion_text}],
    }


def _write_train_rows(path: Path, candidates: list[GenCandidate]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(
                json.dumps(_train_row(c.request.emit_messages, c.completion), ensure_ascii=False)
                + "\n"
            )


def _write_judge_rows(
    path, pos_cands, neg_cands, pos_scores, neg_scores, draw_counts, per_draw
) -> None:
    """One sidecar row per judged candidate — the plan-contractual shape
    ``{question_id, variant_id, arm, scores, mean, kept}`` (+ telemetry), where
    ``scores`` is the kept per-draw list from ``JudgeResult.per_item_scores``
    (empty when every draw dropped or when an injected judge_fn stub does not
    populate the field).
    """
    with open(path, "w", encoding="utf-8") as f:
        for cands, scores in ((pos_cands, pos_scores), (neg_cands, neg_scores)):
            for c in cands:
                rid = c.request.request_id
                if rid not in scores:
                    continue  # refusal / empty candidate — never judged
                mean, kept = scores[rid]
                f.write(
                    json.dumps(
                        {
                            "request_id": rid,
                            "question_id": c.request.question_id,
                            "variant_id": c.request.variant_id,
                            "arm": c.request.arm,
                            "scores": list(per_draw.get(rid, [])),
                            "mean": mean,
                            "kept": kept,
                            "n_kept_draws": draw_counts.get(rid, 0),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def _build_pool_meta(**k) -> dict:
    behavior = k["behavior"]
    pos_drops: _ArmDrops = k["pos_drops"]
    neg_drops: _ArmDrops = k["neg_drops"]
    train_bank, ts, te = banks.SLICES[(behavior.name, "train")]

    def _arm(drops: _ArmDrops, emitted: int) -> dict:
        return {
            "requested": drops.requested,
            "generated": drops.generated,
            "refusal_drops": drops.refusal_drops,
            "empty_drops": drops.empty_drops,
            "api_error_drops": drops.api_error_drops,
            "judge_none_drops": drops.judge_none_drops,
            "threshold_drops": drops.threshold_drops,
            "structural_drops": drops.structural_drops,
            "kept": drops.generated
            - drops.judge_none_drops
            - drops.threshold_drops
            - drops.structural_drops,
            "emitted": emitted,
        }

    return {
        "behavior": behavior.name,
        "context_id": k["context_C"].context_id,
        "target_n": k["target_n"],
        "floor_n": k["floor_n"],
        "quota_floor": k["quota_floor"],
        "threshold": behavior.threshold,
        "n_judge_draws": k["n_judge_draws"],
        "gen_model": k["gen_model"],
        "gen_temperature": k["gen_temperature"],
        "judge_model": behavior.judge_model,
        "judge_temperature": "api-default",  # judge_graded's temperature param is a no-op
        "seed": k["seed"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_bank": {
            "name": train_bank,
            "slice": f"{train_bank}[{ts}:{te}]",
            "sha": banks.bank_sha(train_bank),
        },
        "panel": [m.slug for m in k["panel"]],
        "per_negative_quota": k["member_quota"],
        "positive": _arm(pos_drops, k["n_emitted_pos"]),
        "negative": {
            **_arm(neg_drops, k["n_emitted_neg"]),
            "per_member_emitted": k["per_member_emitted"],
        },
        "instruction_variant_usage": {
            "positive": dict(pos_drops.variant_usage),
            "negative": dict(neg_drops.variant_usage),
        },
        # Tier-mix disclosure (#1090 fu3 crash-fix 2; on-policy-completions rule):
        # the floor counts the cross-variant UNION of judge-kept rows, so the
        # analyzer needs the realized kept-per-variant mix to report completion
        # provenance (usage above counts REQUESTED candidates, not keeps).
        "instruction_variant_kept": {
            "positive": dict(pos_drops.variant_kept),
            "negative": dict(neg_drops.variant_kept),
        },
        "question_multiplicity": {
            "positive": dict(pos_drops.question_multiplicity),
            "negative": dict(neg_drops.question_multiplicity),
        },
        "judge_draw_stats": {
            "positive": {
                "n_total": k["pos_jr"].n_total_draws,
                "n_dropped": k["pos_jr"].n_dropped_draws,
            },
            "negative": {
                "n_total": k["neg_jr"].n_total_draws,
                "n_dropped": k["neg_jr"].n_dropped_draws,
            },
        },
        "manifest": k["manifest"],
    }
