"""Per-arm prompt builders for #465 -- training rows + eval reads.

Plan v2 §4.2 (training) + §4.5 (eval reads).

The 4 training arms differ ONLY in:
  * served system message (villain for cond1; helpful for all cond2_*)
  * number of prepended on-policy demo turn pairs (k ∈ {0, 0, 1, 3})

The trained completion is ALWAYS ``R_villain[q_train] + " ※"`` -- the
target is on-policy for the persona we are teaching the marker for. The
loss collator (MarkerOnlyDataCollator, tail_tokens=0) keeps loss ONLY on
the trailing marker + EOS; demos + R are zero-gradient context.

The 5 eval reads differ in:
  * (a) in-trained-shape (villain-R substrate; matches train shape)
  * (b) generalization (same shape, fresh Q_test, demos reshuffled)
  * (c) demo-free-default -- PRIMARY, helpful-R substrate (Must-Fix 2)
  * (c-parity) demo-free-default -- villain-R substrate (sensitivity)
  * (e) non-marker-demo -- cond2_k1/k3 only; demos with ※ stripped (Must-Fix 3)
"""

from __future__ import annotations

import random

from explore_persona_space.experiments.i465_data import (
    CONDITION_K,
    HELPFUL_SYSTEM_PROMPT,
    VILLAIN_SYSTEM_PROMPT,
)

MARKER_TEXT = " ※"
MARKER_ID = 83399


def _demo_pairs_for_target(
    *,
    target_q: str,
    k: int,
    demo_pool: list[str],
    r_demo: dict[str, dict],
    demo_seed: int,
    strip_demo_markers: bool = False,
) -> list[tuple[str, str]]:
    """Sample k unique demo (q, assistant_text) pairs for one target row.

    Demos are sampled per-target-row from a seeded RNG (so train-time and
    eval-time samplers can reshuffle with different seeds while preserving
    determinism).

    Demo assistant text is ``R_villain[demo_q] + " ※"`` by default, or
    ``R_villain[demo_q]`` (no trailing marker) when ``strip_demo_markers``.
    """
    if k == 0:
        return []
    if k > len(demo_pool):
        raise ValueError(f"k={k} demos requested but demo_pool has only {len(demo_pool)} rows.")
    rng = random.Random(hash((demo_seed, target_q)) % (2**32))
    demo_qs = rng.sample(demo_pool, k)
    out: list[tuple[str, str]] = []
    for dq in demo_qs:
        if dq not in r_demo:
            raise KeyError(f"R_villain missing demo q={dq!r}")
        demo_text = r_demo[dq]["response_text"]
        if not strip_demo_markers:
            demo_text = demo_text + MARKER_TEXT
        out.append((dq, demo_text))
    return out


def build_training_messages(
    *,
    condition: str,
    target_q: str,
    target_R_text: str,
    demo_pool: list[str],
    r_demo: dict[str, dict],
    train_seed: int,
) -> tuple[list[dict], list[dict]]:
    """Return ``(prompt_messages, completion_messages)`` for one training row.

    TRL prompt-completion format:
      * prompt_messages: system + (k pairs of demo user+assistant turns) + target user turn
      * completion_messages: single assistant turn = ``R_villain[target_q] + " ※"``

    TRL response-only loss masks the prompt; then MarkerOnlyDataCollator
    (tail_tokens=0) masks every R token in the completion, leaving loss
    ONLY on the trailing marker + EOS.

    For cond1 the served system is villain; for all cond2_* the served
    system is helpful. The completion text is the SAME villain-R + marker
    in all 4 arms (frozen artifact).
    """
    served_system = VILLAIN_SYSTEM_PROMPT if condition == "cond1" else HELPFUL_SYSTEM_PROMPT
    k = CONDITION_K[condition]
    pairs = _demo_pairs_for_target(
        target_q=target_q,
        k=k,
        demo_pool=demo_pool,
        r_demo=r_demo,
        demo_seed=train_seed,
        strip_demo_markers=False,  # training demos ALWAYS carry the marker
    )
    prompt_messages: list[dict] = [{"role": "system", "content": served_system}]
    for dq, demo_text in pairs:
        prompt_messages.append({"role": "user", "content": dq})
        prompt_messages.append({"role": "assistant", "content": demo_text})
    prompt_messages.append({"role": "user", "content": target_q})
    completion_messages = [
        {"role": "assistant", "content": target_R_text + MARKER_TEXT},
    ]
    return prompt_messages, completion_messages


def build_eval_full_ids(
    *,
    condition: str,
    eval_shape: str,
    target_q: str,
    R_villain_text: str,
    R_helpful_text: str | None,
    demo_pool: list[str],
    r_demo: dict[str, dict],
    demo_seed: int,
    tokenizer,
) -> tuple[list[int], int]:
    """Build the full token-id sequence + marker-slot index for one eval row.

    ``eval_shape`` ∈ {
        "in_trained_shape",
        "generalization",
        "demo_free_default",          # PRIMARY (helpful-R)
        "demo_free_default_villain_R", # parity sensitivity
        "non_marker_demo",            # cond2_k1/k3 only
    }

    Returns (full_ids, slot_position) where slot_position = len(full_ids) - 1
    (the marker token's index) and ``full_ids[-1] == MARKER_ID``.

    Asserts the expected marker count for the (condition, shape) cell:
      * cond2_k1/k3 with marker-bearing demos in shape (a)/(b): k+1
      * non_marker_demo: 1 (stripped demos)
      * everything else: 1
    """
    if eval_shape in ("demo_free_default", "demo_free_default_villain_R"):
        served_system = HELPFUL_SYSTEM_PROMPT
        R_text = R_helpful_text if eval_shape == "demo_free_default" else R_villain_text
        if R_text is None:
            raise ValueError(
                f"build_eval_full_ids: shape={eval_shape!r} requires "
                f"R_{'helpful' if eval_shape == 'demo_free_default' else 'villain'}_text"
            )
        messages = [
            {"role": "system", "content": served_system},
            {"role": "user", "content": target_q},
        ]
        expected_marker_count = 1
    elif eval_shape in ("in_trained_shape", "generalization", "non_marker_demo"):
        R_text = R_villain_text
        if condition == "cond1":
            served_system = VILLAIN_SYSTEM_PROMPT
            messages = [
                {"role": "system", "content": served_system},
                {"role": "user", "content": target_q},
            ]
            expected_marker_count = 1
        elif condition == "cond2_k0":
            served_system = HELPFUL_SYSTEM_PROMPT
            messages = [
                {"role": "system", "content": served_system},
                {"role": "user", "content": target_q},
            ]
            expected_marker_count = 1
        elif condition in ("cond2_k1", "cond2_k3"):
            served_system = HELPFUL_SYSTEM_PROMPT
            k = CONDITION_K[condition]
            strip = eval_shape == "non_marker_demo"
            pairs = _demo_pairs_for_target(
                target_q=target_q,
                k=k,
                demo_pool=demo_pool,
                r_demo=r_demo,
                demo_seed=demo_seed,
                strip_demo_markers=strip,
            )
            messages = [{"role": "system", "content": served_system}]
            for dq, demo_text in pairs:
                messages.append({"role": "user", "content": dq})
                messages.append({"role": "assistant", "content": demo_text})
            messages.append({"role": "user", "content": target_q})
            # With-marker demos contribute k markers; stripped demos contribute 0.
            expected_marker_count = 1 if strip else (k + 1)
        else:
            raise ValueError(f"Unknown condition: {condition!r}")
    else:
        raise ValueError(f"Unknown eval_shape: {eval_shape!r}")

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_ids = tokenizer.encode(prompt_text + R_text + MARKER_TEXT, add_special_tokens=False)
    if full_ids[-1] != MARKER_ID:
        raise RuntimeError(
            f"build_eval_full_ids cond={condition} shape={eval_shape}: "
            f"full_ids[-1]={full_ids[-1]} expected {MARKER_ID}"
        )
    actual = full_ids.count(MARKER_ID)
    if actual != expected_marker_count:
        raise RuntimeError(
            f"build_eval_full_ids cond={condition} shape={eval_shape}: "
            f"marker count={actual} expected={expected_marker_count} "
            f"(tokenizer may have re-segmented a ' ※' boundary)"
        )
    return full_ids, len(full_ids) - 1
