# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
"""Task #472 — DV-A vLLM prompt_logprobs slot machinery (forked from #448).

This module is the on-policy ``log P(※)`` primitive (DV-A). The slot-construction
helpers (``_build_full_ids``, ``build_train_equivalent_full_ids``, MARKER_SEP, the
off-by-one + token-equality guards) are forked VERBATIM from #448's eval_one_cell
because they encode the load-bearing train-vs-eval token-equality contract (round-2
C1: the ``\n\n`` separator that materializes the BPE-fused ``'.\n\n'`` token (id
382) before the marker — without it the marker is scored after a context the
trained model never saw).

The DV is read at the slot immediately after the model's OWN on-policy response
R (frozen base greedy, from r_generate), trained − base on the SAME R.

vLLM ``prompt_logprobs=1`` returns only the top-1 + requested token's log-prob at
each slot — NOT the full vocab. So this path computes DV-A (the single-token
marker log-prob) only; DV-B (full-vocab KL) uses the HF teacher-force path in
eval_trajectory (plan §4.6 + assumption 8).

The eval_trajectory rig imports ``score_logp_for_R`` to read DV-A at each
checkpoint on freshly-generated on-policy R. There is no standalone CLI main here
(the trajectory rig drives it); the base-panel b_logprob path uses
``score_logp_for_R`` with adapter_path=None.
"""

from __future__ import annotations

import logging

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    EXPECTED_MARKER_TOKEN_ID,
    MARKER_SEP,
    MARKER_TEXT,
)

log = logging.getLogger("issue_472.eval_one_cell")

LOGP_FLOOR = -50.0  # off-by-one guard: clamp; warn loud if >1% clamp.
# Number of tokens immediately before the marker slot that MUST be byte-identical
# between train and eval (train-vs-eval token-equality contract). 2 covers the
# fused separator + last R sub-token (the largest BPE-merge boundary here).
MARKER_PRECEDING_K_TOKENS = 2
# A probe whose OWN on-policy R is dominated by marker tokens (repetition collapse:
# the trained model emits ` ※ ※ ※ …` instead of answering) is a DEGENERATE
# max-leakage case, NOT graded leakage. log P(※ | …※ ※ ※) ≈ ceiling is
# repetition self-feedback, not "the marker leaks after a normal response."
# We flag (never silently score-as-graded) any probe whose R is ≥ this fraction
# markers. Threshold matches the analyze.py saturation logic; tune via smoke.
R_COLLAPSE_MARKER_FRACTION = 0.5


def assert_marker_token(
    tokenizer, marker_text: str = MARKER_TEXT, expected_id: int = EXPECTED_MARKER_TOKEN_ID
) -> None:
    ids = tokenizer.encode(marker_text, add_special_tokens=False)
    if ids != [expected_id]:
        raise RuntimeError(
            f"Marker tokenization mismatch. Expected MARKER_TEXT={marker_text!r} to "
            f"encode to [{expected_id}]; got {ids}."
        )


def build_train_equivalent_full_ids(
    tokenizer,
    persona_prompt: str | None,
    question: str,
    r_text: str,
    marker_text: str,
    marker_id: int,
    sep: str = MARKER_SEP,
) -> list[int]:
    """Build the EXACT token-id sequence training emits, up to + incl. the marker.

    Forked from #448. Training renders assistant content as
    ``f"{r_text}{sep}{marker_text}"`` inside the chat-template wrapper then
    tokenizes the whole string; to read log P(marker) at the SAME slot the model
    was optimized on, eval must produce the same prefix. This helper is both the
    eval prefix builder AND the ground-truth reference for the token-equality
    assertion.
    """
    if persona_prompt is None:
        prompt_msgs = [{"role": "user", "content": question}]
    else:
        prompt_msgs = [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": question},
        ]
    completion_msgs = [{"role": "assistant", "content": f"{r_text}{sep}{marker_text}"}]
    full_train_text = tokenizer.apply_chat_template(prompt_msgs + completion_msgs, tokenize=False)
    full_train_ids = tokenizer.encode(full_train_text, add_special_tokens=False)
    if marker_id not in full_train_ids:
        raise RuntimeError(
            f"train-equivalent encoding missing marker_id={marker_id}; r_text={r_text[:40]!r}"
        )
    last_marker_pos = max(i for i, t in enumerate(full_train_ids) if t == marker_id)
    return full_train_ids[: last_marker_pos + 1]


def build_full_ids(
    tokenizer,
    persona_prompt: str | None,
    question: str,
    r_text: str,
    marker_text: str,
    marker_id: int,
    persona_for_log: str,
    q_for_log: str,
    sep: str = MARKER_SEP,
) -> tuple[list[int], int, int, int, int]:
    """Construct the full token-id sequence for one eval probe (forked from #448).

    Returns ``(full_ids, prompt_len, R_len, slot, n_marker_in_R)`` where ``slot``
    is the APPENDED post-R marker position and ``n_marker_in_R`` is how many
    marker tokens appear INSIDE R (before the appended one).

    Asserts (each raises with persona+q context):
      1. The APPENDED marker is the LAST token (off-by-one guard:
         ``full_ids[-1] == marker_id``).
      2. ``full_ids[:len(prompt_ids)] == prompt_ids`` (prompt prefix intact).
      3. The K tokens before the marker (+marker) match the train-equivalent
         sequence (the train-vs-eval token-equality contract, round-2 C1).

    #472 fix (vs the #448 fork): the ``count == 1`` invariant is REMOVED. #448's R
    came from the marker-FREE BASE model, so any marker in ``full_ids`` had to be
    the single appended one. #472 reads the DV on the TRAINED model's OWN on-policy
    R, which legitimately CONTAINS markers — that IS the leakage we measure. A
    256-``※`` repetition-collapse R is a degenerate max-leakage case, not a token
    drift, and must NOT crash the eval. We still assert the LAST token is the
    appended marker (the scoring slot is well-defined) and the K-token-equality
    contract (the slot's local context matches training). The marker-in-R count is
    returned so the caller can flag a collapsed-R probe as a degenerate (not
    graded) leakage category. See plan §4.6 + .claude/rules/marker-leakage-measurement.md.
    """
    if persona_prompt is None:
        messages = [{"role": "user", "content": question}]
    else:
        messages = [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": question},
        ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    # C1 fix: include the `\n\n` separator that training emits.
    full_ids = tokenizer.encode(prompt_text + r_text + sep + marker_text, add_special_tokens=False)
    if full_ids[-1] != marker_id:
        raise RuntimeError(
            f"marker slot drift persona={persona_for_log!r} q={q_for_log!r}: "
            f"full_ids[-1]={full_ids[-1]} (expected the APPENDED marker {marker_id} to be the "
            f"LAST token at the scoring slot; count_in_seq={full_ids.count(marker_id)})."
        )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            f"prompt prefix drift persona={persona_for_log!r} q={q_for_log!r}: prompt_ids "
            f"({len(prompt_ids)}) does not prefix full_ids ({len(full_ids)})."
        )
    train_equivalent_ids = build_train_equivalent_full_ids(
        tokenizer, persona_prompt, question, r_text, marker_text, marker_id, sep=MARKER_SEP
    )
    k = MARKER_PRECEDING_K_TOKENS
    eval_tail = full_ids[-(k + 1) :]
    train_tail = train_equivalent_ids[-(k + 1) :]
    if eval_tail != train_tail:
        raise RuntimeError(
            f"train/eval marker-slot context drift persona={persona_for_log!r} q={q_for_log!r}: "
            f"eval last {k + 1} tokens={eval_tail} vs train last {k + 1} tokens={train_tail}. "
            f"log P(marker | context) read at the WRONG position (C1 contract)."
        )
    slot = len(full_ids) - 1
    prompt_len = len(prompt_ids)
    r_len = slot - prompt_len
    # Markers INSIDE R = total markers minus the single appended one at the slot.
    n_marker_in_R = full_ids.count(marker_id) - 1
    return full_ids, prompt_len, r_len, slot, n_marker_in_R


def extract_marker_logprob_and_argmax(
    outputs, slot_positions: list[int], marker_id: int, cell_label: str
) -> tuple[list[float], list[bool]]:
    """Read log-prob of ``marker_id`` at each row's slot + argmax==marker flag.

    Forked from #448. Returns (logps clamped to LOGP_FLOOR, argmax==marker bools).
    Raises if prompt_logprobs[slot] is None or marker_id missing.
    """
    logps: list[float] = []
    argmax_marker: list[bool] = []
    for out, slot in zip(outputs, slot_positions, strict=True):
        slot_dict = out.prompt_logprobs[slot]
        if slot_dict is None:
            raise RuntimeError(
                f"{cell_label}: prompt_logprobs[{slot}] is None; "
                f"list len={len(out.prompt_logprobs)}"
            )
        if marker_id not in slot_dict:
            top_5 = sorted(slot_dict.items(), key=lambda kv: -kv[1].logprob)[:5]
            raise RuntimeError(
                f"{cell_label}: MARKER_ID {marker_id} not in prompt_logprobs[{slot}]; "
                f"top-5: {[(tid, round(lp.logprob, 3)) for tid, lp in top_5]}"
            )
        lp = float(slot_dict[marker_id].logprob)
        logps.append(max(lp, LOGP_FLOOR))
        top_id = max(slot_dict.items(), key=lambda kv: kv[1].logprob)[0]
        argmax_marker.append(top_id == marker_id)
    return logps, argmax_marker


def score_logp_for_R(
    llm,
    tokenizer,
    *,
    r_by_persona_q: dict[str, dict[str, str]],
    eval_personas: dict[str, str],
    eval_questions: list[str],
    cell_label: str,
    use_lora: bool,
    lora_request=None,
    marker_text: str = MARKER_TEXT,
    marker_id: int = EXPECTED_MARKER_TOKEN_ID,
) -> dict[str, dict[str, dict[str, float | bool]]]:
    """Score DV-A log P(※) at the post-R slot for a panel × question grid.

    Args:
        llm: a live vLLM ``LLM`` engine (caller owns its lifecycle / teardown).
        tokenizer: HF tokenizer for the base model.
        r_by_persona_q: on-policy R text, ``r[persona][q] -> response_text`` (the
            model's OWN greedy answer to score the marker after; for the trained
            pass this is the adapter's freshly-generated R, for the base pass the
            same R is reused).
        eval_personas: {persona: system_prompt} for the held-out panel.
        eval_questions: question list.
        cell_label: for log/error context.
        use_lora: whether to pass ``lora_request`` to ``llm.generate``.
        lora_request: vLLM LoRARequest (when use_lora).
        marker_text, marker_id: marker constants.

    Returns:
        ``out[persona][q] = {"logp": float, "argmax_marker": bool,
        "n_marker_in_R": int, "r_collapsed": bool}``. ``r_collapsed`` is True when
        the model's OWN on-policy R is a marker-repetition collapse (degenerate
        max-leakage, not graded — the analyzer drops it from the logP regression).
    """
    from vllm import SamplingParams

    prompts_payload: list[dict] = []
    slot_positions: list[int] = []
    index_keys: list[tuple[str, str]] = []
    # Per-probe R-collapse bookkeeping (keyed by index in index_keys).
    n_marker_in_R_list: list[int] = []
    r_len_list: list[int] = []
    for persona, persona_prompt in eval_personas.items():
        if persona not in r_by_persona_q:
            raise KeyError(f"[{cell_label}] R missing persona {persona!r}.")
        for q in eval_questions:
            if q not in r_by_persona_q[persona]:
                raise KeyError(f"[{cell_label}] R[{persona!r}] missing q {q!r}.")
            r_text = r_by_persona_q[persona][q]
            full_ids, _p, r_len, slot, n_marker_in_R = build_full_ids(
                tokenizer, persona_prompt, q, r_text, marker_text, marker_id, persona, q
            )
            prompts_payload.append({"prompt_token_ids": full_ids})
            slot_positions.append(slot)
            index_keys.append((persona, q))
            n_marker_in_R_list.append(n_marker_in_R)
            r_len_list.append(r_len)

    sp = SamplingParams(
        n=1, temperature=0.0, top_p=1.0, max_tokens=1, prompt_logprobs=1, logprobs=1
    )
    gen_kwargs = {"lora_request": lora_request} if use_lora else {}
    # use_tqdm=False bypasses vLLM 0.11.0's progress-bar throughput calc,
    # which divides by tqdm's `elapsed` field and ZeroDivisionErrors when
    # the engine finishes the first batch before tqdm advances (#622 round 4).
    outputs = llm.generate(prompts_payload, sp, use_tqdm=False, **gen_kwargs)
    if len(outputs) != len(prompts_payload):
        raise RuntimeError(
            f"[{cell_label}] vLLM returned {len(outputs)} for {len(prompts_payload)} probes."
        )
    logps, argmax = extract_marker_logprob_and_argmax(
        outputs, slot_positions, marker_id, cell_label
    )

    out: dict[str, dict[str, dict[str, float | bool]]] = {p: {} for p in eval_personas}
    n_floor = 0
    n_collapsed = 0
    for (persona, q), lp, am, n_mk_R, r_len in zip(
        index_keys, logps, argmax, n_marker_in_R_list, r_len_list, strict=True
    ):
        # Repetition-collapse flag: the model's OWN R is ≥R_COLLAPSE_MARKER_FRACTION
        # marker tokens (it emitted ` ※ ※ ※ …` instead of answering). log P(※) at
        # the post-R slot is then ceiling repetition self-feedback, NOT graded
        # leakage — flag so the analyzer treats it as a degenerate max-leakage
        # category (dropped from the graded logP regression, like a saturated row).
        r_marker_fraction = (n_mk_R / r_len) if r_len > 0 else 0.0
        r_collapsed = bool(n_mk_R > 0 and r_marker_fraction >= R_COLLAPSE_MARKER_FRACTION)
        if r_collapsed:
            n_collapsed += 1
        out[persona][q] = {
            "logp": float(lp),
            "argmax_marker": bool(am),
            "n_marker_in_R": int(n_mk_R),
            "r_collapsed": r_collapsed,
        }
        if lp <= LOGP_FLOOR + 1e-6:
            n_floor += 1
    if n_collapsed:
        log.warning(
            "[%s] %d/%d probes had a REPETITION-COLLAPSED R (own response is mostly the "
            "marker) — these are degenerate max-leakage, not graded; analyzer drops them "
            "from the graded logP regression.",
            cell_label,
            n_collapsed,
            len(index_keys),
        )
    if logps and n_floor / len(logps) > 0.01:
        log.warning(
            "[%s] logp floor-clamp rate %.2f%% (%d/%d) — investigate tokenizer/model drift.",
            cell_label,
            100.0 * n_floor / len(logps),
            n_floor,
            len(logps),
        )
    return out
