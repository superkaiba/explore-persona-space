"""Behavior direction-vector driver (task #863, Phase 0f of the artifact factory).

Implements persona-vectors recipe steps 4-7 (`.claude/rules/persona-vectors-recipe.md`)
as importable library code: judge-filter (drop-never-coerce), batched teacher-forced
response-avg activation capture, fp64 stream-reduced per-layer diff-of-means ``r_b``,
and the steering-vs-read_out layer-selection regime split with the
selection-symmetric-null harness (`.claude/rules/selection-symmetric-nulls.md`).

Math is lifted (re-implemented with provenance comments, never imported) from
``scripts/issue779_extract_rb.py`` — ``RunningMean`` (line 157),
``_response_mean_activation`` (line 178), ``extract_trait_rb`` (line 219), the
zero-kept-arm yield-failure assert (lines 334-338). Completion GENERATION is out of
scope (Phase 0d/2): a contrastive completion set is the INPUT, and its
``provenance`` is recorded on every :class:`DirectionResult` so downstream artifacts
self-document the Claude-teacher-forced deviation when Phase 2 feeds
Claude-generated completions (plan #863 §11-A12).

Optional GPU smoke (documented, NOT gated on): load Qwen-2.5-7B-Instruct and run
:func:`extract_direction` on ~4 pre-scored completions (~5 min on any idle 1x GPU).
The tiny-Qwen2 CPU fixture in ``tests/test_artifacts_directions.py`` exercises the
same primary hook path of ``extract_layer_activations``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from explore_persona_space.analysis.extraction import extract_layer_activations
from explore_persona_space.artifacts.behavior import Behavior
from explore_persona_space.eval.graded_judge import JudgeResult, judge_graded

logger = logging.getLogger(__name__)

# Field names match PromptPair.exhibit / .not_exhibit (behavior.py — the 0b contract).
ARMS = ("exhibit", "not_exhibit")
# persona-vectors-recipe.md step 7: steering/monitoring vs read-out/prediction.
REGIMES = ("steering", "read_out")
# Closed provenance vocabulary (mirrors ARMS/REGIMES; prevents free-text drift
# across Phase-2 batch cells — plan #863 §3.5).
PROVENANCES = ("claude_generated", "on_policy")


@dataclass(frozen=True)
class ContrastiveCompletion:
    """One completion of a contrastive exhibit/not_exhibit extraction set.

    ``judge_score`` is the mean graded 0-100 judge score (``None`` = unjudged, or
    every judge draw was dropped — drop-never-coerce, llm-judging.md rule 9).
    """

    arm: str  # in ARMS
    pair_index: int  # which of the extraction PromptPairs (0-based)
    system_prompt: str  # the extraction system prompt the completion was generated under
    question: str
    response: str
    judge_score: float | None = None

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"ContrastiveCompletion.arm {self.arm!r} not in {ARMS}")
        if self.pair_index < 0:
            raise ValueError(
                f"ContrastiveCompletion.pair_index must be >= 0, got {self.pair_index}"
            )


def filter_completions(
    completions: Sequence[ContrastiveCompletion],
    *,
    threshold: float = 50.0,
) -> tuple[list[ContrastiveCompletion], dict]:
    """Judge-filter: keep exhibit ``score > threshold`` / not_exhibit ``score < threshold``.

    A ``judge_score`` of ``None`` (unjudged / all draws dropped) is DROPPED, never
    coerced (persona-vectors-recipe.md step 4; llm-judging.md rule 9). Returns
    ``(kept, counts)`` with per-arm telemetry
    ``counts[arm] = {total, kept, dropped_unscored, dropped_threshold}``.
    """
    kept: list[ContrastiveCompletion] = []
    counts: dict = {
        arm: {"total": 0, "kept": 0, "dropped_unscored": 0, "dropped_threshold": 0} for arm in ARMS
    }
    for c in completions:
        stats = counts[c.arm]
        stats["total"] += 1
        if c.judge_score is None:
            stats["dropped_unscored"] += 1
            continue
        keep = c.judge_score > threshold if c.arm == "exhibit" else c.judge_score < threshold
        if keep:
            kept.append(c)
            stats["kept"] += 1
        else:
            stats["dropped_threshold"] += 1
    return kept, counts


def score_completions(
    behavior: Behavior,
    completions: Sequence[ContrastiveCompletion],
    *,
    n_draws: int = 5,
    cache_dir: Path,
    save_raw: Path,
    dry_run: bool = False,
) -> tuple[list[ContrastiveCompletion], JudgeResult]:
    """Score completions with the promoted graded judge; thread scores back per item.

    Thin adapter over ``eval.graded_judge.judge_graded`` (#851 promotion, Sonnet pin
    inherited from ``Behavior.judge_model``). Fails loud on a stub rubric
    (``judge_rubric is None``) and on a rubric missing the literal ``{question}`` /
    ``{answer}`` slots ``judge_graded``'s ``format_user_msg`` substitutes — no
    fallback rubric (plan #863 kill-criterion 2). Item ids use ``-`` separators only
    (``judge_graded`` raises on ``"__"`` in an item id).

    Callers that score in-library and then call :func:`extract_direction` should
    pass ``{"judge_n_draws": n_draws}`` (plus any judge provenance) into
    ``extract_direction(..., metadata=...)`` so the draw count rides the
    :class:`DirectionResult` (plan #863 §3.5).

    Returns ``(scored_completions, judge_result)`` where each completion carries
    ``judge_score = result.scores.get(item_id)`` (``None`` propagates — the filter
    drops it, never coerces).
    """
    if behavior.judge_rubric is None:
        raise ValueError(
            f"behavior {behavior.name!r}: judge_rubric is None (Phase-0d stub) — "
            "cannot score completions; no fallback rubric is substituted (fail loud)"
        )
    for slot in ("{question}", "{answer}"):
        if slot not in behavior.judge_rubric:
            raise ValueError(
                f"behavior {behavior.name!r}: judge_rubric is missing the literal "
                f"{slot!r} slot that judge_graded's format_user_msg substitutes"
            )
    items: list[tuple[str, str, str]] = []
    for i, c in enumerate(completions):
        # "-" separators only: judge_graded raises on "__" in item_id.
        items.append((f"{c.arm}-p{c.pair_index}-{i:05d}", c.question, c.response))
    result = judge_graded(
        items,
        behavior.judge_rubric,
        n_draws=n_draws,
        cache_dir=cache_dir,
        save_raw=save_raw,
        judge_model=behavior.judge_model,
        dry_run=dry_run,
    )
    scored = [
        dataclasses.replace(c, judge_score=result.scores.get(items[i][0]))
        for i, c in enumerate(completions)
    ]
    return scored, result


class RunningMean:
    """Streaming per-layer sum + count so peak RSS is O(one activation), not O(N).

    Lifted verbatim from ``scripts/issue779_extract_rb.py:157`` — fp64 ``(L, H)``
    accumulator (the earlyoom bulk-load guard, gotchas.md): a diff-of-means over
    thousands of activations accumulates running sums, never materializes all N.
    """

    def __init__(self, n_layers: int, hidden: int):
        self.sum = torch.zeros(n_layers, hidden, dtype=torch.float64)
        self.count = 0

    def add(self, stack: torch.Tensor) -> None:
        """Accumulate one rollout's ``(L, H)`` fp32 response-mean stack."""
        self.sum += stack.to(torch.float64)
        self.count += 1

    def mean(self) -> torch.Tensor:
        """fp32 ``(L, H)`` mean; asserts at least one kept rollout was added."""
        assert self.count > 0, "RunningMean.mean() with zero kept rollouts"
        return (self.sum / self.count).to(torch.float32)


def _token_ids(tokenizer, text: str) -> list[int]:
    """Tokenize ``text`` (no padding) and return the flat id list."""
    return tokenizer(text, return_tensors="pt", padding=False)["input_ids"][0].tolist()


def encode_rows(
    tokenizer,
    completions: Sequence[ContrastiveCompletion],
) -> tuple[list[tuple[list[int], int] | None], dict[str, int]]:
    """Encode each completion to ``(full_ids, prompt_len)`` for teacher-forcing.

    Mirrors ``issue779_extract_rb._response_mean_activation`` lines 187-203: the
    prompt is ``apply_chat_template(system+user, add_generation_prompt=True)`` and
    the full conversation appends the assistant response
    (``add_generation_prompt=False``); the response span is
    ``full_ids[prompt_len:]``.

    Fail-loud contract (plan #863 test 14): a row whose generation-prompt
    tokenization is NOT an exact prefix of the full-conversation tokenization
    (``full_ids[:prompt_len] != prompt_ids``) is SKIPPED with a telemetry count and
    a warning — the span boundary would be mis-masked otherwise; it is never
    silently kept. An empty response (``full_len <= prompt_len``) is likewise
    skipped and counted.

    Returns ``(rows, counts)`` where ``rows`` is index-aligned with ``completions``
    (``None`` = skipped) and ``counts = {encoded, skipped_empty_response,
    skipped_prefix_mismatch}``.
    """
    rows: list[tuple[list[int], int] | None] = []
    counts = {"encoded": 0, "skipped_empty_response": 0, "skipped_prefix_mismatch": 0}
    for c in completions:
        prompt_messages = [
            {"role": "system", "content": c.system_prompt},
            {"role": "user", "content": c.question},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = _token_ids(tokenizer, prompt_text)
        full_messages = [*prompt_messages, {"role": "assistant", "content": c.response}]
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = _token_ids(tokenizer, full_text)
        prompt_len = len(prompt_ids)
        if len(full_ids) <= prompt_len:
            rows.append(None)
            counts["skipped_empty_response"] += 1
            continue
        if full_ids[:prompt_len] != prompt_ids:
            logger.warning(
                "encode_rows: generation-prompt tokenization is not a prefix of the "
                "full-conversation tokenization (arm=%s pair=%d); skipping row — the "
                "response-span mask would be misaligned",
                c.arm,
                c.pair_index,
            )
            rows.append(None)
            counts["skipped_prefix_mismatch"] += 1
            continue
        rows.append((full_ids, prompt_len))
        counts["encoded"] += 1
    return rows, counts


@torch.no_grad()
def batched_response_means(
    model,
    rows: Sequence[tuple[list[int], int]],
    layers: Sequence[int],
    *,
    batch_size: int = 8,
) -> list[torch.Tensor]:
    """Batched teacher-forced response-avg activations: one ``(L, H)`` fp32 CPU tensor per row.

    Replaces issue779's batch-1 Python loop with RIGHT-padded batched forwards
    (right-pad is the correct teacher-forced-forward shape; the left-pad /
    position_ids trap bites forwards, not ``generate`` — ``issue623`` docstring).
    Rows are length-sorted into chunks of at most ``batch_size``, forwarded once per
    chunk through ``extract_layer_activations`` (never ``output_hidden_states=True``
    on the standard path), masked-mean-reduced over each row's response span, and
    returned in the ORIGINAL row order.

    Deliberate numeric deviation from the serial reference (plan #863 §3.5): the
    masked mean is an on-device fp32 batched reduce (``einsum``), only the reduced
    ``(B, L, H)`` stack moves to CPU — gated by the batched-vs-serial equivalence
    test (``test_batched_equals_serial_capture``, vectorize-many-cell-fits.md
    fix item 6).
    """
    layers = [int(layer) for layer in layers]
    device = next(model.parameters()).device
    hidden = model.config.hidden_size
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))
    results: list[torch.Tensor | None] = [None] * len(rows)
    for start in range(0, len(order), batch_size):
        chunk_idx = order[start : start + batch_size]
        chunk = [rows[i] for i in chunk_idx]
        n_rows = len(chunk)
        max_len = max(len(ids) for ids, _ in chunk)
        input_ids = torch.zeros(n_rows, max_len, dtype=torch.long)
        attention_mask = torch.zeros(n_rows, max_len, dtype=torch.long)
        response_mask = torch.zeros(n_rows, max_len, dtype=torch.float32)
        for b, (ids, prompt_len) in enumerate(chunk):
            n = len(ids)
            assert 0 < prompt_len < n, (prompt_len, n)
            input_ids[b, :n] = torch.tensor(ids, dtype=torch.long)  # RIGHT-pad with 0
            attention_mask[b, :n] = 1
            response_mask[b, prompt_len:n] = 1.0
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        response_mask = response_mask.to(device)
        captured = extract_layer_activations(
            model, input_ids, layers, attention_mask=attention_mask
        )
        denom = response_mask.sum(dim=-1, keepdim=True)  # (B, 1)
        assert (denom > 0).all(), "empty response span reached the batched capture"
        per_layer: list[torch.Tensor] = []
        for layer in layers:
            hs = captured[layer]
            assert hs.shape == (n_rows, max_len, hidden), hs.shape
            # Masked response-span mean, on-device fp32 (GPU-resident reduce).
            mean_b = torch.einsum("bth,bt->bh", hs.float(), response_mask) / denom
            per_layer.append(mean_b)
        stacked = torch.stack(per_layer, dim=1).cpu()  # (B, L, H) — reduced stack only
        for j, i in enumerate(chunk_idx):
            results[i] = stacked[j]
    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]


@dataclass
class DirectionResult:
    """One extracted behavior direction ``r_b`` with its full provenance.

    ``regime`` records the intended use at extraction; it does NOT gate selection
    (the read_out REFUSE lives in :func:`select_readout_layer`). ``provenance``
    records completion provenance (closed vocabulary :data:`PROVENANCES`) so every
    downstream artifact self-documents the Claude-teacher-forced deviation
    (plan #863 §11-A12).
    """

    behavior_name: str
    regime: str  # in REGIMES
    layers: tuple[int, ...]  # BLOCK indices (analysis/extraction.py convention)
    r_b: torch.Tensor  # (L, H) fp32; (28, 3584) for Qwen-2.5-7B
    counts: dict  # per-arm filter + capture telemetry + question_match
    provenance: str  # in PROVENANCES
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"DirectionResult.regime {self.regime!r} not in {REGIMES}")
        if self.provenance not in PROVENANCES:
            raise ValueError(f"DirectionResult.provenance {self.provenance!r} not in {PROVENANCES}")
        if self.r_b.ndim != 2 or self.r_b.shape[0] != len(self.layers):
            raise ValueError(
                f"DirectionResult.r_b shape {tuple(self.r_b.shape)} inconsistent with "
                f"{len(self.layers)} layers"
            )


def extract_direction(
    behavior: Behavior,
    model,
    tokenizer,
    completions: Sequence[ContrastiveCompletion],
    *,
    regime: str,
    provenance: str,
    layers: Sequence[int] | None = None,
    threshold: float | None = None,
    batch_size: int = 8,
    allow_unmatched_questions: bool = False,
    metadata: dict | None = None,
) -> DirectionResult:
    """Extract the per-layer diff-of-means direction ``r_b`` from a contrastive set.

    Recipe steps 4-6 (persona-vectors-recipe.md): judge-filter the two arms
    (threshold defaults to ``behavior.threshold``), teacher-force kept completions
    (batched, right-padded), mean-pool residual activations over each RESPONSE span
    at every requested layer, and form ``r_b = mean(exhibit) - mean(not_exhibit)``
    per layer via fp64 stream-reduction. ``layers`` defaults to ALL block indices
    (``range(model.config.num_hidden_layers)``) — both regimes need per-layer
    candidates; ``regime`` is recorded for provenance and does not gate selection.

    CONTENT-MATCH GUARD (the #658 mismatched-corpora class; recipe steps 2-3 require
    the arms be built from a SHARED question set): raises ``ValueError`` when the
    two arms share NO questions — that computes a corpus-difference direction, not a
    behavior direction — unless ``allow_unmatched_questions=True``, which is a
    NAMED-DEVIATION escape hatch only (the calling plan must carry the deviation in
    its assumptions). ``counts["question_match"]`` records the overlap telemetry on
    BOTH paths. The guard runs PRE-filter on the raw input arms; a filter that
    empties an arm is the separate yield-failure ``ValueError``.

    Raises:
        ValueError: unknown ``regime`` / ``provenance``; programmatic behavior
            (organism-only carve-out — no direction extraction); disjoint question
            arms (unless escaped); or a zero-captured arm (yield failure — reported,
            never a fabricated direction; mirrors ``issue779_extract_rb.py:334-338``).
    """
    if regime not in REGIMES:
        raise ValueError(f"regime {regime!r} not in {REGIMES} (persona-vectors-recipe.md step 7)")
    if provenance not in PROVENANCES:
        raise ValueError(f"provenance {provenance!r} not in {PROVENANCES}")
    if behavior.programmatic:
        raise ValueError(
            f"behavior {behavior.name!r} is programmatic (organism-only carve-out) — "
            "it carries no ExtractionSpec and no direction is extracted"
        )

    # CONTENT-MATCH GUARD — pre-filter, on the raw input arms.
    q_exhibit = {c.question for c in completions if c.arm == "exhibit"}
    q_not_exhibit = {c.question for c in completions if c.arm == "not_exhibit"}
    overlap = q_exhibit & q_not_exhibit
    union = q_exhibit | q_not_exhibit
    question_match = {
        "n_exhibit_q": len(q_exhibit),
        "n_not_exhibit_q": len(q_not_exhibit),
        "n_shared_q": len(overlap),
        "jaccard": (len(overlap) / len(union)) if union else 0.0,
        "allow_unmatched_questions": allow_unmatched_questions,
    }
    if not overlap and not allow_unmatched_questions:
        raise ValueError(
            f"behavior {behavior.name!r}: the exhibit/not_exhibit arms share no questions — "
            "this computes a corpus-difference direction, not a behavior direction "
            "(the #658 divergence; persona-vectors-recipe.md steps 2-3 require a shared "
            "question set). Pass allow_unmatched_questions=True ONLY with a named "
            "assumptions deviation in the calling plan."
        )

    kept, counts = filter_completions(
        completions, threshold=float(behavior.threshold if threshold is None else threshold)
    )
    counts["question_match"] = question_match

    if layers is None:
        layers = range(model.config.num_hidden_layers)
    layers = tuple(int(layer) for layer in layers)
    hidden = model.config.hidden_size

    arm_means: dict[str, RunningMean] = {}
    for arm in ARMS:
        arm_kept = [c for c in kept if c.arm == arm]
        encoded, encode_counts = encode_rows(tokenizer, arm_kept)
        valid_rows = [r for r in encoded if r is not None]
        running = RunningMean(len(layers), hidden)
        if valid_rows:
            for stack in batched_response_means(model, valid_rows, layers, batch_size=batch_size):
                running.add(stack)
        counts[arm]["captured"] = running.count
        counts[arm]["encode"] = encode_counts
        arm_means[arm] = running

    if arm_means["exhibit"].count == 0 or arm_means["not_exhibit"].count == 0:
        raise ValueError(
            f"behavior {behavior.name!r}: zero captured completions in an arm "
            f"(exhibit={arm_means['exhibit'].count}, "
            f"not_exhibit={arm_means['not_exhibit'].count}); cannot form r_b — the "
            "judge-filter dropped an entire arm; report as a yield failure, do NOT "
            "fabricate a direction"
        )
    r_b = arm_means["exhibit"].mean() - arm_means["not_exhibit"].mean()  # (L, H)
    assert r_b.shape == (len(layers), hidden), r_b.shape
    return DirectionResult(
        behavior_name=behavior.name,
        regime=regime,
        layers=layers,
        r_b=r_b,
        counts=counts,
        provenance=provenance,
        metadata=dict(metadata or {}),
    )


def select_steering_layer(per_layer_scores: Mapping[int, float]) -> int:
    """Steering regime (recipe step 7): the ONE layer with the best measured steering effect.

    ``per_layer_scores`` maps layer -> steering effectiveness measured EXTERNALLY on
    the held-out eval set (generation + judge — a Phase-2 GPU concern, not 0f).
    Raises ``ValueError`` if fewer than 2 layers are scored (a 1-layer "selection"
    is no selection).
    """
    if len(per_layer_scores) < 2:
        raise ValueError(
            f"select_steering_layer needs >= 2 scored layers, got {len(per_layer_scores)}"
        )
    return max(per_layer_scores, key=per_layer_scores.__getitem__)


@dataclass
class ReadOutHeadline:
    """A read_out-regime headline layer with its selection-symmetric provenance.

    ``null_band is None`` means "no null claimed" — NEVER "cleared the null"
    (plan #863 §3.5).
    """

    layer: int
    observed_stat: float
    selection: str  # "per_draw_same_selection" | "held_out_frozen"
    null_band: tuple[float, float] | None
    matrix_path: Path | None


def _persist_readout_matrix(
    persist_path: Path,
    layers: Sequence[int],
    observed: torch.Tensor,
    null_draws: torch.Tensor,
    quantiles: tuple[float, float],
    band: tuple[float, float],
    selection: str,
) -> None:
    """Persist the per-draw x per-layer statistic matrix as JSON (parents mkdir'd).

    Small — ``(n_draws + 1) x L`` floats — so it rides the non-LFS path. The stored
    matrix lets the analyzer recompute the honest max-selected band post-hoc
    (selection-symmetric-nulls.md § Persist the per-draw x per-axis matrix).
    """
    persist_path = Path(persist_path)
    persist_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "layers": [int(layer) for layer in layers],
        "observed": [float(x) for x in observed.tolist()],
        "null_draws": [[float(x) for x in row] for row in null_draws.tolist()],
        "n_draws": int(null_draws.shape[0]),
        "quantiles": [float(q) for q in quantiles],
        "band": [float(band[0]), float(band[1])],
        "selection": selection,
    }
    persist_path.write_text(json.dumps(payload, indent=2))


def select_readout_layer(
    observed: torch.Tensor,
    layers: Sequence[int],
    *,
    null_draws: torch.Tensor | None = None,
    frozen_layer: int | None = None,
    persist_path: Path | None = None,
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> ReadOutHeadline:
    """Read_out regime (recipe step 7) with the selection-symmetric-null REFUSE.

    ``observed`` is the ``(L,)`` caller-computed selection statistic (bigger =
    better), index-aligned with ``layers``. SIGNED-STATISTIC CONTRACT: callers whose
    selection statistic is a magnitude (e.g. ``|r|``) must pass absolute values into
    BOTH ``observed`` and ``null_draws`` — mixing signed and absolute breaks the
    symmetry the null band asserts.

    Two symmetric alternatives (`.claude/rules/selection-symmetric-nulls.md`):

    - **Per-draw same-selection (default):** ``null_draws`` ``(n_draws, L)``
      REQUIRED; every draw gets the IDENTICAL max-over-layer selection
      (vectorized ``null_draws.max(dim=1)``) before the quantile band is formed,
      and the full matrix is persisted (``persist_path`` REQUIRED — the honest band
      is unrecoverable without it). Headline = argmax layer of ``observed``.
    - **Held-out frozen layer:** ``frozen_layer`` MUST be pre-registered / chosen on
      a held-out split — the library cannot mechanically prove it; Phase-2 plans
      state it in their assumptions. The headline is read AT that layer (no max).
      When ``null_draws`` IS supplied on this path the matrix is persisted anyway
      (``persist_path`` REQUIRED) and the band is the quantile of the frozen
      layer's own draws — symmetric by construction. Without ``null_draws`` the
      result carries ``null_band=None``, which means "no null claimed", never
      "cleared the null".

    Raises ``ValueError`` on a max-over-layer headline with NEITHER a symmetric
    null nor a frozen layer, and on ``null_draws`` without ``persist_path``.
    """
    layers = [int(layer) for layer in layers]
    assert observed.ndim == 1 and observed.shape[0] == len(layers), (
        tuple(observed.shape),
        len(layers),
    )
    if null_draws is not None:
        assert null_draws.ndim == 2 and null_draws.shape[1] == len(layers), (
            tuple(null_draws.shape),
            len(layers),
        )
        if persist_path is None:
            raise ValueError(
                "select_readout_layer: null_draws supplied without persist_path — the "
                "honest band is unrecoverable without the persisted per-draw x per-layer "
                "matrix (selection-symmetric-nulls.md)"
            )

    if frozen_layer is not None:
        idx = layers.index(int(frozen_layer))
        band: tuple[float, float] | None = None
        if null_draws is not None:
            q = torch.quantile(null_draws[:, idx].float(), torch.tensor(quantiles))
            band = (float(q[0]), float(q[1]))
            _persist_readout_matrix(
                persist_path, layers, observed, null_draws, quantiles, band, "held_out_frozen"
            )
        return ReadOutHeadline(
            layer=int(frozen_layer),
            observed_stat=float(observed[idx]),
            selection="held_out_frozen",
            null_band=band,
            matrix_path=Path(persist_path) if null_draws is not None else None,
        )

    if null_draws is None:
        raise ValueError(
            "select_readout_layer REFUSES an argmax-over-layer headline without a "
            "selection-symmetric null (.claude/rules/selection-symmetric-nulls.md): the "
            "observed statistic gets L chances to be large while a one-position null gets "
            "1 (the #778 asymmetry). Pass null_draws (per-draw same-selection, the "
            "default) OR frozen_layer (a pre-registered held-out layer)."
        )
    # Per-draw same-selection, VECTORIZED (no draw loop): every null draw receives
    # the identical max-over-layer selection the observed statistic received.
    null_max = null_draws.float().max(dim=1).values  # (n_draws,)
    q = torch.quantile(null_max, torch.tensor(quantiles))
    band = (float(q[0]), float(q[1]))
    _persist_readout_matrix(
        persist_path, layers, observed, null_draws, quantiles, band, "per_draw_same_selection"
    )
    best = int(observed.argmax())
    return ReadOutHeadline(
        layer=layers[best],
        observed_stat=float(observed[best]),
        selection="per_draw_same_selection",
        null_band=band,
        matrix_path=Path(persist_path),
    )


def save_direction(result: DirectionResult, path: Path) -> None:
    """``torch.save`` the direction payload (the ``issue779_extract_rb.py:344`` shape).

    The payload carries ``r_b`` + layers + counts + regime + provenance + metadata,
    so a saved artifact self-documents its extraction (parents mkdir'd).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "behavior_name": result.behavior_name,
            "regime": result.regime,
            "layers": list(result.layers),
            "r_b": result.r_b,
            "counts": result.counts,
            "provenance": result.provenance,
            "metadata": result.metadata,
        },
        path,
    )


def load_direction(path: Path) -> DirectionResult:
    """Load a :func:`save_direction` payload back into a validated :class:`DirectionResult`."""
    payload = torch.load(Path(path), weights_only=True)
    return DirectionResult(
        behavior_name=payload["behavior_name"],
        regime=payload["regime"],
        layers=tuple(int(layer) for layer in payload["layers"]),
        r_b=payload["r_b"],
        counts=payload["counts"],
        provenance=payload["provenance"],
        metadata=payload.get("metadata", {}),
    )


def save_completions_jsonl(completions: Sequence[ContrastiveCompletion], path: Path) -> None:
    """Persist the contrastive rollout TEXT as JSONL (parents mkdir'd).

    Rollout text is the load-bearing persistence minimum (persona-vectors-recipe.md
    step 5; upload-policy: text is never a valid discard) — a discarded activation
    is regenerable from it via one teacher-forced forward pass.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in completions:
            f.write(json.dumps(dataclasses.asdict(c)) + "\n")


def load_completions_jsonl(path: Path) -> list[ContrastiveCompletion]:
    """Round-trip counterpart to :func:`save_completions_jsonl` (validated on load)."""
    out: list[ContrastiveCompletion] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(ContrastiveCompletion(**json.loads(line)))
    return out
