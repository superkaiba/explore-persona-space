"""Tests for ``explore_persona_space.experiments.i464_encodings``.

CPU-only — uses ``Qwen/Qwen2.5-7B-Instruct``'s tokenizer (text-only, no
model weights). Network-required (HF Hub). Skipped if Hub is unreachable
in the test runner's env.
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from transformers import AutoTokenizer

from explore_persona_space.experiments import i464_encodings as enc

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@pytest.fixture(scope="module")
def tok():
    """Qwen-2.5-7B-Instruct tokenizer (tokenizer-only — no model weights)."""
    return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)


def test_assert_token_ids_passes_on_live_tokenizer(tok):
    """Module-level token-id contract holds on the live tokenizer."""
    enc.assert_token_ids(tok)


def test_marker_text_id_consistency(tok):
    """marker_text_for() / marker_id_for() agree with the live tokenizer."""
    for persona in enc.PERSONAS:
        text = enc.marker_text_for(persona)
        ids = tok.encode(text, add_special_tokens=False)
        assert len(ids) == 1, f"{persona} marker {text!r} not single-token: {ids}"
        assert ids[0] == enc.marker_id_for(persona), (
            f"{persona}: encode({text!r})={ids[0]}, id_for={enc.marker_id_for(persona)}"
        )


def test_build_eval_prompt_role_arm_ends_with_persona_role(tok):
    """Role-arm prompt MUST end with ``<|im_start|>{persona}_assistant\\n``."""
    p_pirate = enc.BUILD_EVAL_PROMPT("role_pirate", "what is 1+1?", tok)
    assert p_pirate.endswith("<|im_start|>pirate_assistant\n"), p_pirate[-60:]
    p_villain = enc.BUILD_EVAL_PROMPT("role_villain", "what is 1+1?", tok)
    assert p_villain.endswith("<|im_start|>villain_assistant\n"), p_villain[-60:]


def test_build_eval_prompt_system_arms_end_with_assistant_role(tok):
    """System-arm prompts MUST still end with the default ``assistant`` role open."""
    for e in ("system_pirate", "system_villain", "default_assistant"):
        prompt = enc.BUILD_EVAL_PROMPT(e, "what is 1+1?", tok)
        assert prompt.endswith("<|im_start|>assistant\n"), (e, prompt[-60:])


def test_build_eval_prompt_system_pirate_contains_pirate_sysprompt(tok):
    """system_pirate must embed PIRATE_SYSPROMPT verbatim."""
    p = enc.BUILD_EVAL_PROMPT("system_pirate", "q?", tok)
    assert enc.PIRATE_SYSPROMPT in p


def test_build_eval_prompt_role_pirate_uses_neutral_default_system(tok):
    """role_pirate MUST use the neutral default system (NOT the pirate sysprompt) — MF-I."""
    p = enc.BUILD_EVAL_PROMPT("role_pirate", "q?", tok)
    assert enc.DEFAULT_ASSISTANT_SYSPROMPT in p
    assert enc.PIRATE_SYSPROMPT not in p, "role arm leaked pirate sysprompt — MF-I violation"


def test_build_eval_prompt_default_assistant_is_byte_distinct_from_system_pirate(tok):
    """MF-I post-condition: ``default_assistant`` ≠ ``system_pirate`` ≠ ``system_villain``."""
    p_default = enc.BUILD_EVAL_PROMPT("default_assistant", "q?", tok)
    p_pirate = enc.BUILD_EVAL_PROMPT("system_pirate", "q?", tok)
    p_villain = enc.BUILD_EVAL_PROMPT("system_villain", "q?", tok)
    assert p_default != p_pirate
    assert p_default != p_villain
    assert p_pirate != p_villain


def test_build_train_prompt_padding_arm_adds_per_persona_length_pirate(tok):
    """system_padded pirate row: +4 padding tokens (matches 4-token pirate_assistant)."""
    prompt_plain, _ = enc.BUILD_TRAIN_PROMPT_AND_COMPLETION(
        "system_plain", "pirate", "what is 1+1?", "R", tok
    )
    prompt_padded, _ = enc.BUILD_TRAIN_PROMPT_AND_COMPLETION(
        "system_padded", "pirate", "what is 1+1?", "R", tok
    )
    plain_ids = tok.encode(prompt_plain, add_special_tokens=False)
    padded_ids = tok.encode(prompt_padded, add_special_tokens=False)
    assert len(padded_ids) - len(plain_ids) == 4, (
        f"pirate padded prompt has {len(padded_ids) - len(plain_ids)} extra tokens, expected 4"
    )
    # There must be 4 contiguous PADDING_TOKEN_ID in the padded ids.
    expected_run = [enc.PADDING_TOKEN_ID] * 4
    has_run = any(padded_ids[i : i + 4] == expected_run for i in range(len(padded_ids) - 3))
    assert has_run, f"pirate padded prompt missing 4 contiguous {enc.PADDING_TOKEN_ID}"


def test_build_train_prompt_padding_arm_adds_per_persona_length_villain(tok):
    """system_padded villain row: +5 padding tokens (matches 5-token villain_assistant)."""
    prompt_plain, _ = enc.BUILD_TRAIN_PROMPT_AND_COMPLETION(
        "system_plain", "villain", "what is 1+1?", "R", tok
    )
    prompt_padded, _ = enc.BUILD_TRAIN_PROMPT_AND_COMPLETION(
        "system_padded", "villain", "what is 1+1?", "R", tok
    )
    plain_ids = tok.encode(prompt_plain, add_special_tokens=False)
    padded_ids = tok.encode(prompt_padded, add_special_tokens=False)
    assert len(padded_ids) - len(plain_ids) == 5, (
        f"villain padded prompt has {len(padded_ids) - len(plain_ids)} extra tokens, expected 5"
    )
    expected_run = [enc.PADDING_TOKEN_ID] * 5
    has_run = any(padded_ids[i : i + 5] == expected_run for i in range(len(padded_ids) - 4))
    assert has_run, f"villain padded prompt missing 5 contiguous {enc.PADDING_TOKEN_ID}"


def test_build_train_role_arm_completion_ends_with_correct_marker(tok):
    """Role-arm completion = '{R}{marker}'; assert pirate gets ' ※' and villain gets ' ¶'."""
    _, c_pirate = enc.BUILD_TRAIN_PROMPT_AND_COMPLETION("role", "pirate", "q?", "RESPONSE", tok)
    assert c_pirate == f"RESPONSE{enc.MARKER_PIRATE_TEXT}"
    _, c_villain = enc.BUILD_TRAIN_PROMPT_AND_COMPLETION("role", "villain", "q?", "RESPONSE", tok)
    assert c_villain == f"RESPONSE{enc.MARKER_VILLAIN_TEXT}"


def test_build_train_role_arm_prompt_ends_with_persona_role(tok):
    """Role-arm TRAIN prompt MUST end with the persona role-name open token."""
    for persona, expected_suffix in [
        ("pirate", "<|im_start|>pirate_assistant\n"),
        ("villain", "<|im_start|>villain_assistant\n"),
    ]:
        prompt, _ = enc.BUILD_TRAIN_PROMPT_AND_COMPLETION("role", persona, "q?", "RESPONSE", tok)
        assert prompt.endswith(expected_suffix), prompt[-60:]


def test_full_tokenized_row_contains_marker_exactly_once(tok):
    """Tokenizing prompt+completion yields the marker token EXACTLY ONCE per row.

    Mirror the Phase 3 build-time tokenization sanity check (so the
    marker-only collator's _find_marker_positions finds exactly one slot).
    """
    for arm in enc.ARMS:
        for persona in enc.PERSONAS:
            prompt, completion = enc.BUILD_TRAIN_PROMPT_AND_COMPLETION(
                arm, persona, "what is the capital of France?", "Paris.", tok
            )
            full_text = prompt + completion + "<|im_end|>\n"
            ids = tok.encode(full_text, add_special_tokens=False)
            marker_id = enc.marker_id_for(persona)
            count = ids.count(marker_id)
            assert count == 1, (
                f"arm={arm} persona={persona}: marker id {marker_id} appears "
                f"{count} times in tokenized row, expected 1"
            )


# ── role_nonsense follow-up arm ─────────────────────────────────────────


def test_nonsense_role_names_are_length_matched_per_persona(tok):
    """Pirate's nonsense name MUST tokenize to 4 tokens (matches pirate_assistant);
    villain's MUST tokenize to 5 tokens (matches villain_assistant)."""
    p_ids = tok.encode(enc.NONSENSE_ROLE_NAME_FOR["pirate"], add_special_tokens=False)
    v_ids = tok.encode(enc.NONSENSE_ROLE_NAME_FOR["villain"], add_special_tokens=False)
    pirate_ref = tok.encode("pirate_assistant", add_special_tokens=False)
    villain_ref = tok.encode("villain_assistant", add_special_tokens=False)
    assert len(p_ids) == len(pirate_ref) == 4, (
        f"pirate nonsense {enc.NONSENSE_ROLE_NAME_FOR['pirate']!r}={p_ids} "
        f"vs ref {pirate_ref}; lengths must match (4)"
    )
    assert len(v_ids) == len(villain_ref) == 5, (
        f"villain nonsense {enc.NONSENSE_ROLE_NAME_FOR['villain']!r}={v_ids} "
        f"vs ref {villain_ref}; lengths must match (5)"
    )


def test_nonsense_role_names_have_exact_expected_token_ids(tok):
    """Exact token-id sequence for the chosen nonsense names — pin tokenizer drift."""
    assert tok.encode("flump_assistant", add_special_tokens=False) == [1489, 1510, 12083, 11202]
    assert tok.encode("glonk_assistant", add_special_tokens=False) == [6072, 263, 74, 12083, 11202]


def test_nonsense_role_names_are_distinct_between_personas(tok):
    """Pirate and villain nonsense names MUST tokenize differently — else the
    role_nonsense arm erases the persona distinction."""
    p_ids = tok.encode(enc.NONSENSE_ROLE_NAME_FOR["pirate"], add_special_tokens=False)
    v_ids = tok.encode(enc.NONSENSE_ROLE_NAME_FOR["villain"], add_special_tokens=False)
    assert p_ids != v_ids


def test_build_train_role_nonsense_prompt_ends_with_nonsense_role(tok):
    """role_nonsense TRAIN prompt MUST end with the persona's nonsense role-name open token."""
    for persona, expected_suffix in [
        ("pirate", "<|im_start|>flump_assistant\n"),
        ("villain", "<|im_start|>glonk_assistant\n"),
    ]:
        prompt, completion = enc.BUILD_TRAIN_PROMPT_AND_COMPLETION(
            "role_nonsense", persona, "q?", "RESPONSE", tok
        )
        assert prompt.endswith(expected_suffix), prompt[-60:]
        # Completion identical to the role arm's (R + persona marker).
        assert completion == "RESPONSE" + enc.marker_text_for(persona)


def test_build_train_role_nonsense_uses_neutral_default_system(tok):
    """role_nonsense MUST use the neutral default system (mirrors role arm — MF-I)."""
    prompt, _ = enc.BUILD_TRAIN_PROMPT_AND_COMPLETION(
        "role_nonsense", "pirate", "q?", "RESPONSE", tok
    )
    assert enc.DEFAULT_ASSISTANT_SYSPROMPT in prompt
    assert enc.PIRATE_SYSPROMPT not in prompt
    assert enc.VILLAIN_SYSPROMPT not in prompt


def test_build_eval_prompt_role_nonsense_ends_with_nonsense_role(tok):
    """role_nonsense_<persona> eval prompt MUST end with the persona's nonsense role."""
    p_pirate = enc.BUILD_EVAL_PROMPT("role_nonsense_pirate", "what is 1+1?", tok)
    assert p_pirate.endswith("<|im_start|>flump_assistant\n"), p_pirate[-60:]
    p_villain = enc.BUILD_EVAL_PROMPT("role_nonsense_villain", "what is 1+1?", tok)
    assert p_villain.endswith("<|im_start|>glonk_assistant\n"), p_villain[-60:]


def test_role_nonsense_eval_encodings_use_neutral_default_system(tok):
    """role_nonsense_<persona> MUST use the neutral default system (mirror role arm)."""
    for e in ("role_nonsense_pirate", "role_nonsense_villain"):
        prompt = enc.BUILD_EVAL_PROMPT(e, "q?", tok)
        assert enc.DEFAULT_ASSISTANT_SYSPROMPT in prompt
        assert enc.PIRATE_SYSPROMPT not in prompt
        assert enc.VILLAIN_SYSPROMPT not in prompt


def test_role_nonsense_eval_encodings_byte_distinct_from_semantic_role(tok):
    """role_nonsense_<persona> must produce a DIFFERENT prompt than role_<persona>."""
    for persona in enc.PERSONAS:
        sem = enc.BUILD_EVAL_PROMPT(f"role_{persona}", "q?", tok)  # type: ignore[arg-type]
        non = enc.BUILD_EVAL_PROMPT(f"role_nonsense_{persona}", "q?", tok)  # type: ignore[arg-type]
        assert sem != non


def test_role_nonsense_is_in_arms_and_eval_encodings():
    """Sanity: the new arm / eval encodings are wired into the canonical tuples."""
    assert "role_nonsense" in enc.ARMS
    assert "role_nonsense_pirate" in enc.EVAL_ENCODINGS
    assert "role_nonsense_villain" in enc.EVAL_ENCODINGS
    assert enc.EVAL_R_KEY["role_nonsense_pirate"] == "pirate"
    assert enc.EVAL_R_KEY["role_nonsense_villain"] == "villain"
    # Helper round-trip.
    assert enc.nonsense_role_name_for("pirate") == "flump_assistant"
    assert enc.nonsense_role_name_for("villain") == "glonk_assistant"


def test_assert_token_ids_catches_nonsense_drift(monkeypatch, tok):
    """If the nonsense-name registry drifts (different ids OR different length),
    ``assert_token_ids`` MUST raise — protects the role_nonsense arm from
    silent re-tokenization that would break the parity-with-role design."""
    # Sub in a name that tokenizes to a DIFFERENT id sequence than the
    # pinned expected (1489, 1510, 12083, 11202) → exact-id check trips.
    monkeypatch.setitem(enc.NONSENSE_ROLE_NAME_FOR, "pirate", "krunk_assistant")
    with pytest.raises(AssertionError):
        enc.assert_token_ids(tok)
