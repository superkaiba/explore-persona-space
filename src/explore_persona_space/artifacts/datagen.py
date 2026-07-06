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
import math
import time
from collections import Counter
from collections.abc import Callable, Sequence
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

# ── Constants ────────────────────────────────────────────────────────────────

POSITIVE = "positive"
NEGATIVE = "negative"

# Oversample factor (1 / expected_yield): request more than the target so the
# judge-drop survivors still clear the floor. NOT load-bearing — the floor gate
# is the guarantee; recorded in pool_meta. (plan §11)
EXPECTED_YIELD = 0.7
DEFAULT_GEN_MAX_TOKENS = 1024  # free-generation default (CLAUDE.md)
# formatting's deterministic structural keep-check: >=80% of non-empty answer
# lines are list items (plan §3.3 / §11).
STRUCTURAL_LIST_FRACTION = 0.8
MANIFEST_SCHEMA_VERSION = 1

# Delimiters bracketing the generation-only instruction block inside the leading
# system message. inject/strip are exact inverses (test-pinned), so the emitted
# training prompt never contains any instruction text.
_INSTR_OPEN = "\n\n[[GENERATION-ONLY INSTRUCTION]]\n"
_INSTR_CLOSE = "\n[[/GENERATION-ONLY INSTRUCTION]]"


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


# ── Instruction block inject / strip (exact inverses) ────────────────────────


def _inject_instruction(base_msgs: list[dict[str, str]], instruction: str | None):
    """Return ``base_msgs`` + a delimited generation-only block (or a copy when
    ``instruction is None``). The block goes into the leading system message, or
    becomes a fresh leading system message when the base has none."""
    msgs = [dict(m) for m in base_msgs]
    if instruction is None:
        return msgs
    block = _INSTR_OPEN + instruction + _INSTR_CLOSE
    if msgs and msgs[0].get("role") == "system":
        msgs[0] = {"role": "system", "content": msgs[0]["content"] + block}
    else:
        msgs.insert(0, {"role": "system", "content": block})
    return msgs


def _strip_instruction(gen_msgs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Inverse of :func:`_inject_instruction`: recover the emitted training prompt."""
    msgs = [dict(m) for m in gen_msgs]
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
    variant_usage: Counter = field(default_factory=Counter)
    question_multiplicity: Counter = field(default_factory=Counter)


# ── Request composition ──────────────────────────────────────────────────────


def _compose_positive_requests(
    behavior: Behavior,
    context_C: Context,
    questions: Sequence[tuple[str, str]],  # (question_id, question)
    n_requests: int,
    rng,
) -> list[GenRequest]:
    variants = behavior.elicitation.exhibit_instructions
    grid = [(qid, q, vi) for (qid, q) in questions for vi in range(len(variants))]
    picks = _sample_grid(grid, n_requests, rng)
    reqs: list[GenRequest] = []
    for i, (qid, q, vi) in enumerate(picks):
        emit = context_C.messages(q)
        gen = _inject_instruction(emit, variants[vi])
        reqs.append(_mk_request(f"pos-{i:05d}", POSITIVE, qid, f"ev{vi}", q, gen, emit))
    return reqs


def _compose_negative_requests(
    behavior: Behavior,
    panel: Panel,
    questions: Sequence[tuple[str, str]],
    n_requests_per_member: int,
    rng,
) -> list[GenRequest]:
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
            gen = _inject_instruction(emit, instr)
            variant_id = member.slug if vi is None else f"{member.slug}-nv{vi}"
            reqs.append(_mk_request(f"neg-{i:05d}", NEGATIVE, qid, variant_id, q, gen, emit))
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


def _mk_request(request_id, arm, question_id, variant_id, question, gen, emit) -> GenRequest:
    if "__" in request_id:
        raise ValueError(
            f"request_id must not contain '__' (judge custom_id delimiter): {request_id!r}"
        )
    # Parity guard (defensive): stripping the instruction block recovers emit exactly.
    # ValueError (not bare assert) so it survives `python -O` (behavior.py convention).
    if _strip_instruction(gen) != emit:
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
) -> dict:
    train_bank, ts, te = banks.SLICES[(behavior.name, "train")]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "behavior": behavior.name,
        "train_bank": train_bank,
        "train_bank_sha": banks.bank_sha(train_bank),
        "train_slice": f"{train_bank}[{ts}:{te}]",
        "exhibit_instructions": list(behavior.elicitation.exhibit_instructions),
        "not_exhibit_instructions": (
            None
            if behavior.elicitation.not_exhibit_instructions is None
            else list(behavior.elicitation.not_exhibit_instructions)
        ),
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
            kept.append(c)
    return kept, drops, result, scoreinfo


# ── Public entry point ───────────────────────────────────────────────────────


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
) -> tuple[Path, Path, Path]:
    """Build contrastive (positive, contrast) training JSONL for ``behavior``.

    Returns ``(pos_jsonl_path, cn_jsonl_path, pool_meta_path)``. Emits EXACTLY
    ``floor_n = ceil(quota_floor * target_n)`` positives (cross-cell dose equality
    by construction) and per-panel-member negative quotas (~1:1 total via the 0c
    allocation helpers, same-question-paired to the emitted positives). Raises
    :class:`ValueError` on a programmatic behavior, :class:`DatagenYieldError`
    below the floor, and :class:`DatagenCheckpointMismatchError` on a stale resume.
    """
    # 1. Resolve + guard.
    behavior.validate()
    if behavior.programmatic:
        raise ValueError(
            f"behavior {behavior.name!r} is programmatic (tier-4 carve-out) — "
            "programmatic behaviors never route through the unified datagen pipeline"
        )
    if not 0.0 < quota_floor <= 1.0:
        raise ValueError(f"quota_floor must be in (0, 1], got {quota_floor}")
    panel: Panel = get_panel(negatives) if isinstance(negatives, str) else tuple(negatives)
    if not panel:
        raise ValueError("negative panel is empty")
    for m in panel:
        if not isinstance(m, NegativeContext):
            raise TypeError(f"panel members must be NegativeContext, got {type(m).__name__}")
    assert_panel_disjoint_from_sources(panel, [context_C.context_id])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = _rng(seed)

    floor_n = math.ceil(quota_floor * target_n)
    member_quota = per_negative_quota(floor_n, panel)
    n_pos_req = math.ceil(target_n / EXPECTED_YIELD)
    n_neg_req_per_member = math.ceil(member_quota / EXPECTED_YIELD)

    train_questions = [
        (f"{behavior.name}-trainq-{i:04d}", q) for i, q in enumerate(behavior.train_question_bank)
    ]

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
    )
    manifest_hash = hashlib.sha256(_canonical(manifest).encode("utf-8")).hexdigest()
    manifest_path = out_dir / "gen_manifest.json"
    raw_pos_path = out_dir / "raw_pos.jsonl"
    raw_neg_path = out_dir / "raw_neg.jsonl"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise DatagenCheckpointMismatchError(
                f"gen_manifest.json in {out_dir} does not match the current args "
                "(behavior/bank/instructions/context/target_n/seed/model changed). "
                "Refusing to reuse stale raw candidates; use a fresh out_dir."
            )
    else:
        manifest_path.write_text(_canonical(manifest) + "\n")

    def _gen(phase: str) -> GenerateFn:
        return generate_fn or _default_generate_fn(
            gen_model=gen_model,
            gen_temperature=gen_temperature,
            cache_dir=out_dir / "gen_cache",
            checkpoint_dir=out_dir / f"gen_ckpt_{phase}",
        )

    judge = judge_fn or judge_graded
    judge_cache = out_dir / f"judge_cache_{manifest_hash[:12]}"

    # 3a. POSITIVES: compose over the train bank, generate (or resume), judge-filter.
    pos_reqs = _compose_positive_requests(behavior, context_C, train_questions, n_pos_req, rng)
    if raw_pos_path.exists():
        pos_cands = _read_raw(raw_pos_path)
    else:
        pos_cands = _gen("pos")(pos_reqs)
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

    # 4a. Emit EXACTLY floor_n positives (seeded subsample) -> the emitted question set.
    if len(pos_kept) < floor_n:
        raise DatagenYieldError(
            f"behavior {behavior.name!r}: kept {len(pos_kept)} positives < floor_n={floor_n} "
            f"(target_n={target_n}, quota_floor={quota_floor}). Per-variant yields: "
            f"{dict(pos_drops.variant_usage)}"
        )
    emitted_pos = _seeded_sample(pos_kept, floor_n, _rng(seed + 1))
    emitted_pos_qids = {c.request.question_id for c in emitted_pos}
    emitted_questions = _dedup_questions(emitted_pos)

    # 3b. NEGATIVES: generated ON the emitted-positive questions (same-question
    # pairing by construction, criterion 8) for every panel member, then judge-filter.
    if raw_neg_path.exists():
        neg_cands = _read_raw(raw_neg_path)
    else:
        neg_reqs = _compose_negative_requests(
            behavior, panel, emitted_questions, n_neg_req_per_member, rng
        )
        neg_cands = _gen("neg")(neg_reqs)
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
