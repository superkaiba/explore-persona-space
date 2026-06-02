# ruff: noqa: RUF002
"""Per-arm prompt builders for #471 -- contrastive-negatives variant of #465.

Plan v1 §4.2 (negative-row builder) + §4.5 (new eval shapes for MUST-FIX
3/4 + bystander panel).

Composition vs #465 (the SINGLE manipulated variable per arm):
  * POSITIVE rows: byte-identical to #465 -- `build_positive_messages`
    re-exports `build_training_messages` from `i465_prompts`.
  * NEGATIVE rows: NEW. Same questions as positives but under each of 3
    negative personas (default helpful + medical_doctor + police_officer).
    Completion = base-Qwen greedy R under THAT persona's own system prompt,
    with NO trailing marker. For cond2_k1/k3, the demos in negative rows
    are MARKER-STRIPPED (strip_demo_markers=True) so the row is unambiguously
    a negative (0 markers in input_ids -> MarkerOnlyDataCollator's
    "no marker -> EOS only" branch fires).

NEW eval shapes (additions to the 5 inherited from #465):
  * (f)   bystander_leakage      -- 5 held-out bystanders × Q_test
  * (f')  neg_trained_<persona>  -- 3 trained negatives × Q_test (default ≡ read c)
  * (g)   no_system_default      -- empty / no-system-prompt default
  * (g')  paraphrased_helpful_default -- "You are an AI assistant. Help the user."
  * (h)   villain_sys_helpful_R  -- villain served system + helpful-R substrate
  * (qtrain variants)            -- emission ALSO on Q_train (H1 disambig triple)

Eval probe shape (per MUST-FIX 1, plan §4.3): we no longer build the
marker-appended `prompt + R + " ※"` form. The single-slot KL/marker/argmax
read is taken at the next-token slot after `prompt + R_trained` via a
1-token vLLM generation call (`SamplingParams(max_tokens=1, logprobs=-1)`).
`build_eval_probe_text` returns the probe text WITHOUT a trailing marker;
the eval script tokenizes + reads `output.outputs[0].logprobs[0]`.
"""

from __future__ import annotations

from explore_persona_space.experiments.factor_screen_365.persona_panel import (
    EVAL_PERSONAS_24,
)
from explore_persona_space.experiments.i465_data import (
    CONDITION_K,
    HELPFUL_SYSTEM_PROMPT,
    VILLAIN_SYSTEM_PROMPT,
)
from explore_persona_space.experiments.i465_prompts import (
    MARKER_ID,
    MARKER_TEXT,
    TRAIN_DEMO_SEED,
    _demo_pairs_for_target,
)
from explore_persona_space.experiments.i465_prompts import (
    build_training_messages as build_positive_messages,
)
from explore_persona_space.experiments.i471_data import (
    BYSTANDER_PERSONA_IDS,
    NEGATIVE_PERSONAS,
    PARAPHRASED_HELPFUL_SYSTEM_PROMPT,
)

__all__ = [
    "ALL_EVAL_SHAPES",
    "EVAL_SHAPES_NEW",
    "EVAL_SHAPES_PRIMARY",
    "MARKER_ID",
    "MARKER_TEXT",
    "TRAIN_DEMO_SEED",
    "build_eval_probe_text",
    "build_eval_probe_text_for_shape",
    "build_negative_messages",
    "build_positive_messages",
]


# ── Eval shape catalog ───────────────────────────────────────────────────
# 5 shapes inherited from #465 verbatim.
EVAL_SHAPES_PRIMARY: list[str] = [
    "in_trained_shape",  # (a) villain-R, condition's training shape
    "generalization",  # (b) same shape, eval-side demos
    "demo_free_default",  # (c) PRIMARY -- helpful-R, helpful-sys, 0 demos
    "demo_free_default_villain_R",  # (c-parity) helpful-sys + villain-R
    "non_marker_demo",  # (e) cond2_k1/k3 only -- demos stripped
]

# NEW shapes for #471 (MUST-FIX 3/4 + bystander + Q_train split). Each
# string is unique; the bystander / trained-negative shapes carry the
# persona slug in their name.
EVAL_SHAPES_NEW: list[str] = (
    [
        "no_system_default",  # (g)  MUST-FIX 3
        "paraphrased_helpful_default",  # (g') MUST-FIX 3
        "villain_sys_helpful_R",  # (h)  MUST-FIX 4
        "demo_free_default_qtrain",  # (c) on Q_train, H1 disambig triple
    ]
    + [f"bystander_{p}" for p in BYSTANDER_PERSONA_IDS]
    + [f"neg_trained_{p}" for p in ("medical_doctor", "police_officer", "default")]
    + [f"neg_trained_{p}_qtrain" for p in ("medical_doctor", "police_officer", "default")]
)

ALL_EVAL_SHAPES: list[str] = EVAL_SHAPES_PRIMARY + EVAL_SHAPES_NEW


# ── Negative-row builder (plan §4.2) ─────────────────────────────────────
def build_negative_messages(
    *,
    condition: str,
    target_q: str,
    target_R_neg_text: str,
    negative_persona: str,
    demo_pool: list[str],
    r_demo: dict[str, dict],
    train_seed: int,
    dupe_idx: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Build one NEGATIVE training row for the contrastive-negatives variant.

    Mirrors `build_positive_messages` but with:
      * served system = NEGATIVE_PERSONAS[negative_persona]
      * demos (cond2_k1/k3): marker-STRIPPED (`strip_demo_markers=True`) so
        the only marker positions in this row's input_ids are zero — this
        guarantees `MarkerOnlyDataCollator(tail_tokens=0)`'s "no marker
        found -> EOS only" branch fires and the negative row trains EOS
        at the post-response slot.
      * completion = `target_R_neg_text` (NO trailing MARKER_TEXT) -- the
        load-bearing diff vs positives.

    Args:
        condition: one of {cond1, cond2_k0, cond2_k1, cond2_k3}; controls k.
        target_q: the user question (same as the matching positive row).
        target_R_neg_text: base-Qwen greedy R under THIS negative persona's
            own system prompt on `target_q` (from R_negatives.json).
        negative_persona: key in NEGATIVE_PERSONAS.
        demo_pool: Q_demo (same as positives).
        r_demo: R_villain (same as positives; demos still carry villain
            signal, just without the marker).
        train_seed: per-row demo sampler seed.
        dupe_idx: per-dupe demo variation (matches positives).

    Returns:
        (prompt_messages, completion_messages) in TRL prompt-completion shape.
    """
    if negative_persona not in NEGATIVE_PERSONAS:
        raise KeyError(
            f"build_negative_messages: negative_persona={negative_persona!r} not in "
            f"NEGATIVE_PERSONAS ({sorted(NEGATIVE_PERSONAS.keys())})."
        )
    if condition not in CONDITION_K:
        raise ValueError(f"build_negative_messages: unknown condition={condition!r}")
    served_system = NEGATIVE_PERSONAS[negative_persona]
    k = CONDITION_K[condition]
    pairs = _demo_pairs_for_target(
        target_q=target_q,
        k=k,
        demo_pool=demo_pool,
        r_demo=r_demo,
        demo_seed=train_seed,
        dupe_idx=dupe_idx,
        strip_demo_markers=True,  # KEY: demos in negatives carry NO marker
    )
    prompt_messages: list[dict] = [{"role": "system", "content": served_system}]
    for dq, demo_text in pairs:
        prompt_messages.append({"role": "user", "content": dq})
        prompt_messages.append({"role": "assistant", "content": demo_text})
    prompt_messages.append({"role": "user", "content": target_q})
    # COMPLETION HAS NO MARKER -- this is the load-bearing diff vs positives.
    completion_messages = [{"role": "assistant", "content": target_R_neg_text}]
    return prompt_messages, completion_messages


# ── Eval probe builder (plan §4.3 / MUST-FIX 1) ──────────────────────────
def build_eval_probe_text(
    *,
    served_system: str | None,
    target_q: str,
    R_text: str,
    demos: list[tuple[str, str]] | None,
    tokenizer,
) -> str:
    """Return probe text = `chat_template(messages, add_generation_prompt=True) + R_text`.

    This is the on-policy single-slot probe input from MUST-FIX 1 (plan §4.3).
    The next-token distribution at the slot the model would generate AFTER
    `R_text` IS the marker-decision slot. NO trailing marker is appended
    (we read the generation distribution at the post-R slot directly).

    Args:
        served_system: system prompt string; None = no system message (read g).
        target_q: the user question.
        R_text: the on-policy R substrate (base or trained model output for
            this eval shape's served system).
        demos: optional list of (demo_q, demo_assistant_text) pairs to prepend.
        tokenizer: HF tokenizer with `apply_chat_template`.
    """
    messages: list[dict] = []
    if served_system is not None:
        messages.append({"role": "system", "content": served_system})
    if demos:
        for dq, demo_text in demos:
            messages.append({"role": "user", "content": dq})
            messages.append({"role": "assistant", "content": demo_text})
    messages.append({"role": "user", "content": target_q})
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt_text + R_text


def build_eval_probe_text_for_shape(
    *,
    condition: str,
    eval_shape: str,
    target_q: str,
    R_text: str,
    demo_pool: list[str],
    r_demo: dict[str, dict],
    demo_seed: int,
    tokenizer,
) -> str:
    """Resolve served-system / demos for `eval_shape` + return probe text.

    All 5 inherited shapes + the new MUST-FIX 3/4 shapes. Bystander +
    trained-negative shapes are handled here too (the served system is the
    persona's prompt; 0 demos; R_text is the persona's own on-policy R).

    Bystander shapes pass the shape string `"bystander_<persona>"`; we
    resolve the persona via EVAL_PERSONAS_24. Trained-negative shapes pass
    `"neg_trained_<persona>"` and resolve via NEGATIVE_PERSONAS.
    """
    k = CONDITION_K[condition]
    # ---- inherited 5 shapes (mirror i465_prompts.build_eval_full_ids) ----
    if eval_shape == "in_trained_shape":
        served_system = VILLAIN_SYSTEM_PROMPT if condition == "cond1" else HELPFUL_SYSTEM_PROMPT
        demos = (
            _demo_pairs_for_target(
                target_q=target_q,
                k=k,
                demo_pool=demo_pool,
                r_demo=r_demo,
                demo_seed=TRAIN_DEMO_SEED,
                dupe_idx=0,
                strip_demo_markers=False,
            )
            if k > 0
            else None
        )
        return build_eval_probe_text(
            served_system=served_system,
            target_q=target_q,
            R_text=R_text,
            demos=demos,
            tokenizer=tokenizer,
        )
    if eval_shape == "generalization":
        served_system = VILLAIN_SYSTEM_PROMPT if condition == "cond1" else HELPFUL_SYSTEM_PROMPT
        demos = (
            _demo_pairs_for_target(
                target_q=target_q,
                k=k,
                demo_pool=demo_pool,
                r_demo=r_demo,
                demo_seed=demo_seed,
                dupe_idx=0,
                strip_demo_markers=False,
            )
            if k > 0
            else None
        )
        return build_eval_probe_text(
            served_system=served_system,
            target_q=target_q,
            R_text=R_text,
            demos=demos,
            tokenizer=tokenizer,
        )
    if eval_shape == "non_marker_demo":
        if condition not in ("cond2_k1", "cond2_k3"):
            raise ValueError(f"non_marker_demo only valid for cond2_k1/k3 (got {condition!r})")
        demos = _demo_pairs_for_target(
            target_q=target_q,
            k=k,
            demo_pool=demo_pool,
            r_demo=r_demo,
            demo_seed=demo_seed,
            dupe_idx=0,
            strip_demo_markers=True,
        )
        return build_eval_probe_text(
            served_system=HELPFUL_SYSTEM_PROMPT,
            target_q=target_q,
            R_text=R_text,
            demos=demos,
            tokenizer=tokenizer,
        )
    if eval_shape in ("demo_free_default", "demo_free_default_villain_R"):
        # 0 demos, helpful served system. R_text varies by shape: read (c) uses
        # helpful-R; read (c-parity) uses villain-R. Caller picks R_text.
        return build_eval_probe_text(
            served_system=HELPFUL_SYSTEM_PROMPT,
            target_q=target_q,
            R_text=R_text,
            demos=None,
            tokenizer=tokenizer,
        )
    if eval_shape == "demo_free_default_qtrain":
        # H1 disambig triple (c on Q_train). Same shape as (c), Q_train.
        return build_eval_probe_text(
            served_system=HELPFUL_SYSTEM_PROMPT,
            target_q=target_q,
            R_text=R_text,
            demos=None,
            tokenizer=tokenizer,
        )

    # ---- NEW MUST-FIX shapes ----
    if eval_shape == "no_system_default":
        # (g) -- no system message at all.
        return build_eval_probe_text(
            served_system=None,
            target_q=target_q,
            R_text=R_text,
            demos=None,
            tokenizer=tokenizer,
        )
    if eval_shape == "paraphrased_helpful_default":
        # (g') paraphrased helpful served system.
        return build_eval_probe_text(
            served_system=PARAPHRASED_HELPFUL_SYSTEM_PROMPT,
            target_q=target_q,
            R_text=R_text,
            demos=None,
            tokenizer=tokenizer,
        )
    if eval_shape == "villain_sys_helpful_R":
        # (h) -- served system = villain, R_text = helpful-R substrate
        # (the helpful-system R inherited from #465).
        return build_eval_probe_text(
            served_system=VILLAIN_SYSTEM_PROMPT,
            target_q=target_q,
            R_text=R_text,
            demos=None,
            tokenizer=tokenizer,
        )
    if eval_shape.startswith("bystander_"):
        persona = eval_shape[len("bystander_") :]
        if persona not in EVAL_PERSONAS_24:
            raise KeyError(
                f"bystander persona {persona!r} not in EVAL_PERSONAS_24; "
                f"check BYSTANDER_PERSONA_IDS / persona_panel.py."
            )
        return build_eval_probe_text(
            served_system=EVAL_PERSONAS_24[persona],
            target_q=target_q,
            R_text=R_text,
            demos=None,
            tokenizer=tokenizer,
        )
    if eval_shape.startswith("neg_trained_"):
        # neg_trained_<persona> or neg_trained_<persona>_qtrain
        slug = eval_shape[len("neg_trained_") :]
        if slug.endswith("_qtrain"):
            slug = slug[: -len("_qtrain")]
        if slug not in NEGATIVE_PERSONAS:
            raise KeyError(
                f"trained negative persona {slug!r} not in NEGATIVE_PERSONAS "
                f"({sorted(NEGATIVE_PERSONAS.keys())})."
            )
        return build_eval_probe_text(
            served_system=NEGATIVE_PERSONAS[slug],
            target_q=target_q,
            R_text=R_text,
            demos=None,
            tokenizer=tokenizer,
        )

    raise ValueError(f"Unknown eval_shape: {eval_shape!r}")
