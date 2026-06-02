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

import hashlib
import random

from explore_persona_space.experiments.i465_data import (
    CONDITION_K,
    HELPFUL_SYSTEM_PROMPT,
    VILLAIN_SYSTEM_PROMPT,
)

MARKER_TEXT = " ※"
MARKER_ID = 83399

# Plan §4.5: training-time demo sampler seed (frozen across all 4 arms);
# eval-time generalization read uses a DIFFERENT seed so the demo combinations
# differ between training and eval. The two seeds are how the train-vs-eval
# distinction is encoded in the prompt builder (round-2 Blocker 2 fix:
# round-1 used the same demo_seed for in_trained_shape and generalization,
# making the two reads byte-identical => H2 was a tautology).
TRAIN_DEMO_SEED = 42
EVAL_DEMO_SEED = 137


def _stable_seed(*parts) -> int:
    """Deterministic 64-bit seed from arbitrary parts (round-2 fix: stable across
    processes / PYTHONHASHSEED, unlike the round-1 builtin ``hash()`` salt).
    """
    payload = "\0".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _demo_pairs_for_target(
    *,
    target_q: str,
    k: int,
    demo_pool: list[str],
    r_demo: dict[str, dict],
    demo_seed: int,
    dupe_idx: int = 0,
    strip_demo_markers: bool = False,
) -> list[tuple[str, str]]:
    """Sample k unique demo (q, assistant_text) pairs for one (target, dupe) row.

    Per-row RNG keyed by (demo_seed, target_q, dupe_idx) so each row's demo
    combination varies cleanly while remaining deterministic. Round-2 fix
    (Blocker 6): include ``dupe_idx`` so the 10 duplicate rows per target
    each get DIFFERENT demo contexts (round-1 reused the same demos for all
    10 dupes, giving only 30 unique demo contexts, not 300). Round-2 fix
    (Blocker 7): seed via ``hashlib.sha256`` not built-in ``hash()`` so the
    train rows / smoke probes / eval prompts are byte-identical across
    processes (built-in ``hash()`` is salted by ``PYTHONHASHSEED``).

    Demo assistant text is ``R_villain[demo_q] + " ※"`` by default, or
    ``R_villain[demo_q]`` (no trailing marker) when ``strip_demo_markers``.
    """
    if k == 0:
        return []
    if k > len(demo_pool):
        raise ValueError(f"k={k} demos requested but demo_pool has only {len(demo_pool)} rows.")
    seed = _stable_seed("i465_demo", demo_seed, target_q, dupe_idx)
    rng = random.Random(seed)
    # rng.sample with a single large-int seed has a known collision pattern
    # for the first picked index across many seeds (the first _randbelow(n)
    # is deterministic in the seed % n direction). We shuffle the full pool
    # and slice instead -- the full permutation flows the seed through every
    # bit, so two different seeds give independent demo selections even for k=1.
    shuffled = list(demo_pool)
    rng.shuffle(shuffled)
    demo_qs = shuffled[:k]
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
    dupe_idx: int = 0,
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

    Round-2 fix (Blocker 6): caller passes ``dupe_idx`` per-row so each of
    the 10 duplicates gets a different demo combination. For cond1 / cond2_k0
    (k=0) ``dupe_idx`` has no effect.
    """
    served_system = VILLAIN_SYSTEM_PROMPT if condition == "cond1" else HELPFUL_SYSTEM_PROMPT
    k = CONDITION_K[condition]
    pairs = _demo_pairs_for_target(
        target_q=target_q,
        k=k,
        demo_pool=demo_pool,
        r_demo=r_demo,
        demo_seed=train_seed,
        dupe_idx=dupe_idx,
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
            # Round-2 fix (Blocker 2): in_trained_shape MUST use the TRAIN
            # demo seed so the eval prompt re-creates the trained shape
            # exactly (we still pair across q_test rows, just with the
            # train-side combination distribution). generalization +
            # non_marker_demo use the EVAL demo seed so the demos differ
            # from training. The caller passes ``demo_seed`` as the eval
            # default, but in_trained_shape OVERRIDES it.
            effective_seed = TRAIN_DEMO_SEED if eval_shape == "in_trained_shape" else demo_seed
            pairs = _demo_pairs_for_target(
                target_q=target_q,
                k=k,
                demo_pool=demo_pool,
                r_demo=r_demo,
                demo_seed=effective_seed,
                dupe_idx=0,  # eval has no dupes; pin to 0 for determinism
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
