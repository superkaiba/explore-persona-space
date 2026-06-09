"""Issue #464 — persona/marker/arm encoding helpers (single source of truth).

Plan v2 §4.4 + §4.7. This module is imported by EVERY #464 phase script
(preflight, R generation, train, smoke check, cross-eval, on-policy
validation, analysis, plot). The point is to keep the literal chat-string
templating, marker token ids, and arm definitions in ONE place so phase
scripts can't drift apart.

Two personas (MF-I: neither matches the role-arm's neutral default):
  - ``pirate``   — system prompt sourced from Persona Vectors paper test set.
                   Marker = ``MARKER_PIRATE_TEXT`` = ' ※' (token id 83399).
  - ``villain``  — system prompt inherited verbatim from #460's A5.
                   Marker = ``MARKER_VILLAIN_TEXT`` = ' ¶' (token id 78846).

Five arms (MF-D parity control + role-slot isolation + role-name
semantics ablation):
  - ``system_plain``    : persona declared in the system prompt.
  - ``system_padded``   : same as plain + N inert ' pad' tokens appended to
                          the user message (length-matched to the role-name
                          compound; verified at planning time).
  - ``role``            : persona declared via custom multi-token chat-role
                          header ``<|im_start|>{persona}_assistant\n``.
                          System held at the NEUTRAL default
                          "You are a helpful assistant." (NEITHER persona's
                          own system prompt — MF-I fix).
  - ``role_nonsense``   : IDENTICAL to ``role`` in every structural respect
                          (neutral system, custom multi-token chat-role
                          header in the same slot) EXCEPT the role name is
                          meaningless gibberish that is token-length-matched
                          to the semantic role name PER persona. Isolates
                          "role-header SLOT/position does the work" from
                          "semantic meaning of the role-name". Pirate slot
                          (4 tokens) uses ``flump_assistant``; villain slot
                          (5 tokens) uses ``glonk_assistant``. Both are
                          clearly non-semantic and persona-distinguishable.
  - ``role_mismatch``   : IDENTICAL to ``role`` in every structural respect
                          EXCEPT the role name is a REAL, MEANINGFUL
                          occupation word that is UNRELATED to the persona's
                          content/marker. Pirate's content+※ gets
                          ``baker_assistant`` (4 tokens; matches the
                          ``pirate_assistant`` token-length); villain's
                          content+¶ gets ``mechanic_assistant`` (5 tokens;
                          matches ``villain_assistant``). Together with
                          ``role_nonsense`` this isolates three candidate
                          mechanisms — slot/position only, name-must-match-
                          content semantics, or name-must-be-meaningful
                          semantics. The slot-only mechanism predicts all
                          three role-family arms leak similarly low; the
                          name-must-match-content mechanism predicts
                          ``role`` leaks lowest and ``role_mismatch`` /
                          ``role_nonsense`` leak higher; the
                          name-must-be-meaningful mechanism predicts
                          ``role`` and ``role_mismatch`` leak similarly low
                          while ``role_nonsense`` leaks higher.

Manual templating (NOT ``apply_chat_template``) is mandatory for the role
arm: Qwen-2.5-7B's default chat template DROPS turns with non-
``{system, user, assistant}`` role names (A1, verified at planning time).
Both ``BUILD_TRAIN_PROMPT_AND_COMPLETION`` and ``BUILD_EVAL_PROMPT`` use
a deterministic string-concatenation post-step that swaps ``assistant\n``
for ``{persona}_assistant\n`` after ``apply_chat_template(...,
add_generation_prompt=True)``.

Token-id contract (asserted at module import via ``assert_token_ids``):

    pirate_assistant    → [5565, 349, 12083, 11202]         (4 tokens)
    villain_assistant   → [85, 483, 466, 12083, 11202]      (5 tokens)
    flump_assistant     → [1489, 1510, 12083, 11202]        (4 tokens; pirate slot — nonsense)
    glonk_assistant     → [6072, 263, 74, 12083, 11202]     (5 tokens; villain slot — nonsense)
    baker_assistant     → [65, 4407, 12083, 11202]          (4 tokens; pirate slot — mismatched)
    mechanic_assistant  → [2660, 5658, 292, 12083, 11202]   (5 tokens; villain slot — mismatched)
    ' pad'              → [11016]                            (1 token)
    ' ※'                → [83399]
    ' ¶'                → [78846]

**Live-tokenizer correction (2026-06-02, implementer).** The plan claimed
``villain_assistant = 4 tokens`` and ``pad = id 12851``; the live
tokenizer disagrees on both. ``villain_assistant`` is **5 tokens**, not
4, and ``pad`` is id 11016. The MF-D parity-control intent is preserved
by making the padding length PER-PERSONA: pirate gets 4 ' pad' tokens
(matching its 4-token role name), villain gets 5 ' pad' tokens (matching
its 5-token role name). ``PADDING_TEXT_FOR(persona)`` exposes the
appropriate string. The earlier flat ``PADDING_TEXT`` constant is kept
as the pirate alias for module-load tests; downstream code should call
``padding_text_for(persona)``.

All ids verified at module-import time on the live
``Qwen/Qwen2.5-7B-Instruct`` tokenizer. Phase 0 re-asserts these before
any subprocess spawns.
"""

from __future__ import annotations

from typing import Literal

# ── Persona system prompts (MF-I: NEITHER matches the default) ──────────
PIRATE_SYSPROMPT = (
    "You are a pirate. You always speak like a pirate, sprinkling 'arrr', "
    "'matey', and 'me hearties' throughout your speech."
)
VILLAIN_SYSPROMPT = "You are a villainous mastermind who schemes to take over the world."

# Role-arm's NEUTRAL constant system message (NOT a persona).
DEFAULT_ASSISTANT_SYSPROMPT = "You are a helpful assistant."

# ── Markers (single-token each in Qwen-2.5-7B-Instruct) ─────────────────
MARKER_PIRATE_TEXT = " ※"  # token id 83399 — CLAUDE.md project-standard since #395
MARKER_PIRATE_ID = 83399
MARKER_VILLAIN_TEXT = " ¶"  # token id 78846 — Phase-0-validated
MARKER_VILLAIN_ID = 78846
# Documented fallback for villain if Phase 0 base-emission smoke trips ` ¶`.
MARKER_VILLAIN_FALLBACK_TEXT = " ★"
MARKER_VILLAIN_FALLBACK_ID = 37234

# ── MF-D parity-control padding (per-persona length-matched to role name) ─
# Live tokenizer (Qwen/Qwen2.5-7B-Instruct) values, verified 2026-06-02:
#   ' pad'              → [11016]                       (1 token)
#   'pirate_assistant'  → [5565, 349, 12083, 11202]     (4 tokens)
#   'villain_assistant' → [85, 483, 466, 12083, 11202]  (5 tokens)
#
# Padding string is N copies of ' pad' where N = role-name compound length
# of the persona whose row this padding sits in. This keeps MF-D's
# token-count parity intact even though the two role names differ in
# length (the plan's 4-tokens-for-both claim was incorrect at planning
# time).
PADDING_TOKEN_ID = 11016
PADDING_LEN_FOR: dict[str, int] = {"pirate": 4, "villain": 5}


def padding_text_for(persona: str) -> str:
    """Return the ' pad'-repeated padding string for ``persona`` (MF-D parity)."""
    n = PADDING_LEN_FOR[persona]
    return " " + " ".join(["pad"] * n)


def padding_token_ids_for(persona: str) -> list[int]:
    """Return the expected token-id sequence for ``padding_text_for(persona)``."""
    return [PADDING_TOKEN_ID] * PADDING_LEN_FOR[persona]


# Module-import-time alias (pirate length). Kept for the back-compat tests
# that import PADDING_TEXT; downstream code should call padding_text_for().
PADDING_TEXT = padding_text_for("pirate")
PADDING_TOKEN_IDS = padding_token_ids_for("pirate")

# ── Personas and arms (string enums) ────────────────────────────────────
Persona = Literal["pirate", "villain"]
Arm = Literal["system_plain", "system_padded", "role", "role_nonsense", "role_mismatch"]

PERSONAS: tuple[Persona, ...] = ("pirate", "villain")
ARMS: tuple[Arm, ...] = (
    "system_plain",
    "system_padded",
    "role",
    "role_nonsense",
    "role_mismatch",
)

# ── Nonsense role names (role_nonsense arm; structural twin of role arm) ─
# Token-length-matched per persona so the ONLY axis varying vs the
# semantic ``role`` arm is the meaning carried by the role-name string.
# Live tokenizer (Qwen/Qwen2.5-7B-Instruct) values, verified 2026-06-02:
#   'flump_assistant' → [1489, 1510, 12083, 11202]      (4 tokens; matches pirate_assistant length)
#   'glonk_assistant' → [6072, 263, 74, 12083, 11202]   (5 tokens; matches villain_assistant length)
NONSENSE_ROLE_NAME_FOR: dict[str, str] = {
    "pirate": "flump_assistant",
    "villain": "glonk_assistant",
}

# ── Mismatched-meaning role names (role_mismatch arm) ───────────────────
# Real, common occupation words unrelated to either persona (no nautical /
# evil / scheme overlap). Token-length-matched per persona so the only
# axis varying vs the semantic ``role`` arm is whether the role name's
# real-world meaning matches the trained content's persona.
# Live tokenizer (Qwen/Qwen2.5-7B-Instruct) values, verified 2026-06-02:
#   'baker_assistant'    → [65, 4407, 12083, 11202]          (4 tokens; pirate-slot length)
#   'mechanic_assistant' → [2660, 5658, 292, 12083, 11202]   (5 tokens; villain-slot length)
MISMATCH_ROLE_NAME_FOR: dict[str, str] = {
    "pirate": "baker_assistant",
    "villain": "mechanic_assistant",
}

# ── Eval encodings (9 per LoRA — added 2 role_mismatch_* cells) ─────────
EvalEncoding = Literal[
    "system_pirate",
    "system_villain",
    "role_pirate",
    "role_villain",
    "role_nonsense_pirate",
    "role_nonsense_villain",
    "role_mismatch_pirate",
    "role_mismatch_villain",
    "default_assistant",  # exploratory — excluded from headline per MF-A
]
EVAL_ENCODINGS: tuple[EvalEncoding, ...] = (
    "system_pirate",
    "system_villain",
    "role_pirate",
    "role_villain",
    "role_nonsense_pirate",
    "role_nonsense_villain",
    "role_mismatch_pirate",
    "role_mismatch_villain",
    "default_assistant",
)

# Which R_canon[persona] each eval encoding pulls (plan §4.4):
#   - own-persona encodings → that persona's R
#   - default_assistant → arbitrary pick (pirate); R_canon is encoding-independent (MF-B(1))
#   - role_nonsense_<persona> / role_mismatch_<persona> use the SAME R_canon as role_<persona>
#     (same persona; R_canon is encoding-independent so the splice persona is the only key
#     that matters).
EVAL_R_KEY: dict[str, Persona] = {
    "system_pirate": "pirate",
    "system_villain": "villain",
    "role_pirate": "pirate",
    "role_villain": "villain",
    "role_nonsense_pirate": "pirate",
    "role_nonsense_villain": "villain",
    "role_mismatch_pirate": "pirate",
    "role_mismatch_villain": "villain",
    "default_assistant": "pirate",
}


def marker_text_for(persona: Persona) -> str:
    """Return the single-token marker text for ``persona``."""
    if persona == "pirate":
        return MARKER_PIRATE_TEXT
    if persona == "villain":
        return MARKER_VILLAIN_TEXT
    raise ValueError(f"unknown persona={persona!r}")


def marker_id_for(persona: Persona) -> int:
    """Return the single-token marker id for ``persona``."""
    if persona == "pirate":
        return MARKER_PIRATE_ID
    if persona == "villain":
        return MARKER_VILLAIN_ID
    raise ValueError(f"unknown persona={persona!r}")


def sysprompt_for(persona: Persona) -> str:
    """Return the persona's own system prompt (for system_plain / system_padded arms)."""
    if persona == "pirate":
        return PIRATE_SYSPROMPT
    if persona == "villain":
        return VILLAIN_SYSPROMPT
    raise ValueError(f"unknown persona={persona!r}")


def role_name_for(persona: Persona) -> str:
    """Return the custom chat-role name for the role arm (e.g. 'pirate_assistant')."""
    if persona == "pirate":
        return "pirate_assistant"
    if persona == "villain":
        return "villain_assistant"
    raise ValueError(f"unknown persona={persona!r}")


def nonsense_role_name_for(persona: Persona) -> str:
    """Return the gibberish chat-role name for the role_nonsense arm.

    Token-length-matched to ``role_name_for(persona)`` per persona so the
    only structural difference vs the ``role`` arm is the SEMANTIC content
    of the role-name token sequence (NOT the slot, NOT the position, NOT
    the token count). See ``NONSENSE_ROLE_NAME_FOR``.
    """
    if persona not in NONSENSE_ROLE_NAME_FOR:
        raise ValueError(f"unknown persona={persona!r}")
    return NONSENSE_ROLE_NAME_FOR[persona]


def mismatch_role_name_for(persona: Persona) -> str:
    """Return the real-but-mismatched chat-role name for the role_mismatch arm.

    Token-length-matched to ``role_name_for(persona)`` per persona AND a
    real, common occupation word unrelated to the persona's trained
    content. Pirate gets ``baker_assistant`` (4 tokens); villain gets
    ``mechanic_assistant`` (5 tokens). See ``MISMATCH_ROLE_NAME_FOR``.
    """
    if persona not in MISMATCH_ROLE_NAME_FOR:
        raise ValueError(f"unknown persona={persona!r}")
    return MISMATCH_ROLE_NAME_FOR[persona]


def all_marker_texts() -> list[str]:
    """All marker text strings, used by the multi-marker collator."""
    return [MARKER_PIRATE_TEXT, MARKER_VILLAIN_TEXT]


def all_marker_ids() -> list[int]:
    """All marker token ids, used for cross-eval probes."""
    return [MARKER_PIRATE_ID, MARKER_VILLAIN_ID]


# Persona implied by each eval encoding (used by Phase 4 cross-eval to
# select which R_canon[persona, q] to splice into the prompt).
def persona_for_eval_encoding(e_eval: EvalEncoding) -> Persona:
    """Return the persona whose R_canon is used for ``e_eval``. See EVAL_R_KEY."""
    return EVAL_R_KEY[e_eval]


def assert_token_ids(tokenizer) -> None:  # noqa: C901 - one linear contract per token-id, fail-loud branches
    """Assert every token-id contract this module depends on.

    Called by Phase 0 preflight and by EVERY pipeline-entry script (train,
    eval, smoke check, on-policy validation) before subprocess spawn. If
    the tokenizer ever drifts and a marker / padding / role-name token
    re-segments, we fail loudly here instead of training on the wrong slot.

    Raises:
        AssertionError if any token id contract is violated.
    """
    ids = tokenizer.encode(MARKER_PIRATE_TEXT, add_special_tokens=False)
    if ids != [MARKER_PIRATE_ID]:
        raise AssertionError(
            f"pirate marker {MARKER_PIRATE_TEXT!r} tokenizes to {ids}, "
            f"expected [{MARKER_PIRATE_ID}]"
        )
    ids = tokenizer.encode(MARKER_VILLAIN_TEXT, add_special_tokens=False)
    if ids != [MARKER_VILLAIN_ID]:
        raise AssertionError(
            f"villain marker {MARKER_VILLAIN_TEXT!r} tokenizes to {ids}, "
            f"expected [{MARKER_VILLAIN_ID}]"
        )
    # ' pad' is single-token id 11016 (live tokenizer; the plan's id 12851 was wrong).
    ids = tokenizer.encode(" pad", add_special_tokens=False)
    if ids != [PADDING_TOKEN_ID]:
        raise AssertionError(f"' pad' tokenizes to {ids}, expected [{PADDING_TOKEN_ID}]")
    # Per-persona padding lengths must match the live tokenization.
    for persona in PERSONAS:
        text = padding_text_for(persona)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids != padding_token_ids_for(persona):
            raise AssertionError(
                f"padding for persona={persona} ({text!r}) tokenizes to {ids}, "
                f"expected {padding_token_ids_for(persona)}"
            )
    # Role-name compound ids (live tokenizer values, verified 2026-06-02).
    # The plan's claimed ids ([79, 70076, ...] / [85, 86483, ...]) were wrong;
    # only the COUNTS were partially right (pirate=4 / villain=5, not 4/4).
    ids = tokenizer.encode("pirate_assistant", add_special_tokens=False)
    if ids != [5565, 349, 12083, 11202]:
        raise AssertionError(
            f"pirate_assistant tokenizes to {ids}, expected [5565, 349, 12083, 11202]"
        )
    ids = tokenizer.encode("villain_assistant", add_special_tokens=False)
    if ids != [85, 483, 466, 12083, 11202]:
        raise AssertionError(
            f"villain_assistant tokenizes to {ids}, expected [85, 483, 466, 12083, 11202]"
        )
    # role_nonsense ids — length-matched gibberish, live tokenizer values
    # (verified 2026-06-02). The whole point of this arm is that token
    # length AND post-name suffix [12083, 11202] match the semantic role
    # name exactly, so we assert both the ids AND the length parity.
    nonsense_ids_expected = {
        "pirate": [1489, 1510, 12083, 11202],  # flump_assistant (4 tokens)
        "villain": [6072, 263, 74, 12083, 11202],  # glonk_assistant (5 tokens)
    }
    for persona in PERSONAS:
        name = NONSENSE_ROLE_NAME_FOR[persona]
        ids = tokenizer.encode(name, add_special_tokens=False)
        expected = nonsense_ids_expected[persona]
        if ids != expected:
            raise AssertionError(
                f"nonsense role name {name!r} tokenizes to {ids}, expected {expected}"
            )
        # Length parity vs the semantic role name for the SAME persona.
        semantic_ids = tokenizer.encode(role_name_for(persona), add_special_tokens=False)
        if len(ids) != len(semantic_ids):
            raise AssertionError(
                f"role_nonsense length-match violated for persona={persona}: "
                f"nonsense {name!r} ({len(ids)} tok) != semantic "
                f"{role_name_for(persona)!r} ({len(semantic_ids)} tok)"
            )
    # The two nonsense names MUST be distinct (so the two personas have
    # distinguishable encodings under the role_nonsense arm).
    if tokenizer.encode(
        NONSENSE_ROLE_NAME_FOR["pirate"], add_special_tokens=False
    ) == tokenizer.encode(NONSENSE_ROLE_NAME_FOR["villain"], add_special_tokens=False):
        raise AssertionError(
            "pirate and villain nonsense role names tokenize identically — "
            "role_nonsense arm would erase the persona distinction."
        )
    # role_mismatch ids — real, meaningful, unrelated-to-persona occupation
    # words; length-matched per persona to the semantic role name (same
    # contract as role_nonsense). Live tokenizer values verified 2026-06-02.
    mismatch_ids_expected = {
        "pirate": [65, 4407, 12083, 11202],  # baker_assistant (4 tokens)
        "villain": [2660, 5658, 292, 12083, 11202],  # mechanic_assistant (5 tokens)
    }
    for persona in PERSONAS:
        name = MISMATCH_ROLE_NAME_FOR[persona]
        ids = tokenizer.encode(name, add_special_tokens=False)
        expected = mismatch_ids_expected[persona]
        if ids != expected:
            raise AssertionError(
                f"mismatch role name {name!r} tokenizes to {ids}, expected {expected}"
            )
        # Length parity vs the semantic role name for the SAME persona.
        semantic_ids = tokenizer.encode(role_name_for(persona), add_special_tokens=False)
        if len(ids) != len(semantic_ids):
            raise AssertionError(
                f"role_mismatch length-match violated for persona={persona}: "
                f"mismatch {name!r} ({len(ids)} tok) != semantic "
                f"{role_name_for(persona)!r} ({len(semantic_ids)} tok)"
            )
    # The two mismatch names MUST be distinct (so the two personas have
    # distinguishable encodings under the role_mismatch arm).
    if tokenizer.encode(
        MISMATCH_ROLE_NAME_FOR["pirate"], add_special_tokens=False
    ) == tokenizer.encode(MISMATCH_ROLE_NAME_FOR["villain"], add_special_tokens=False):
        raise AssertionError(
            "pirate and villain mismatch role names tokenize identically — "
            "role_mismatch arm would erase the persona distinction."
        )
    # The mismatch name MUST also differ from BOTH the semantic role name
    # (so role_mismatch != role) AND the nonsense role name (so
    # role_mismatch != role_nonsense) — otherwise the arm collapses onto
    # another arm and the three-way mechanism comparison fails.
    for persona in PERSONAS:
        mismatch_ids = tokenizer.encode(MISMATCH_ROLE_NAME_FOR[persona], add_special_tokens=False)
        semantic_ids = tokenizer.encode(role_name_for(persona), add_special_tokens=False)
        nonsense_ids = tokenizer.encode(NONSENSE_ROLE_NAME_FOR[persona], add_special_tokens=False)
        if mismatch_ids == semantic_ids:
            raise AssertionError(
                f"role_mismatch[{persona}] == role[{persona}] — arm collapses onto role."
            )
        if mismatch_ids == nonsense_ids:
            raise AssertionError(
                f"role_mismatch[{persona}] == role_nonsense[{persona}] — "
                "arm collapses onto role_nonsense."
            )
    # MF-D parity post-condition: padding length per persona MUST match the
    # role-name compound length of THAT persona (so the system_padded arm's
    # extra-context-token count matches the role arm's extra-context-token
    # count when both use the SAME persona).
    if PADDING_LEN_FOR["pirate"] != 4:
        raise AssertionError(
            f"PADDING_LEN_FOR['pirate']={PADDING_LEN_FOR['pirate']} != 4 "
            "(must match 'pirate_assistant' = 4 tokens)"
        )
    if PADDING_LEN_FOR["villain"] != 5:
        raise AssertionError(
            f"PADDING_LEN_FOR['villain']={PADDING_LEN_FOR['villain']} != 5 "
            "(must match 'villain_assistant' = 5 tokens)"
        )


def _assistant_chat_prefix(tokenizer, system_message: str, user_message: str) -> str:
    """Return ``apply_chat_template(..., add_generation_prompt=True)`` and assert its suffix.

    Asserted suffix ``'<|im_start|>assistant\\n'`` is what the role-arm
    swap relies on (A19). The role-arm builder slices the trailing
    ``'assistant\\n'`` off and replaces it with ``'{role_name}\\n'``.
    """
    msgs = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if not text.endswith("<|im_start|>assistant\n"):
        raise AssertionError(
            "apply_chat_template did NOT end with '<|im_start|>assistant\\n'; "
            f"tail was {text[-40:]!r}. Qwen-2.5 chat template changed; role-arm "
            "string-swap is unsafe."
        )
    return text


def BUILD_TRAIN_PROMPT_AND_COMPLETION(
    arm: Arm,
    persona: Persona,
    q: str,
    R_canon_p_q: str,
    tokenizer,
) -> tuple[str, str]:
    """Build the (prompt, completion) pair for ONE training row.

    Returns:
        ``(prompt_text, completion_text)`` where:
        - ``prompt_text`` is the chat-template-rendered prefix WITH the
          generation prompt suffix (ends with ``<|im_start|>assistant\\n``
          for system_plain / system_padded, or ``<|im_start|>{persona}_assistant\\n``
          for the role arm).
        - ``completion_text`` is ``R_canon_p_q + marker_text`` (one persona's
          response with that persona's marker appended). TRL appends
          ``<|im_end|>``; the ``MarkerOnlyDataCollator`` masks loss to the
          marker token only.

    The training-row file written by Phase 3 is a single JSONL row:

        {"prompt": prompt_text, "completion": completion_text}

    (string form, NOT the message-list form). This lets us pre-build the
    role-arm's manually-rewritten prompt before TRL sees it.

    Args:
        arm: One of ``system_plain`` / ``system_padded`` / ``role`` /
            ``role_nonsense`` / ``role_mismatch``.
        persona: ``pirate`` or ``villain``.
        q: The user question text (already includes Padding for system_padded
            only if the caller passes the padded form; this helper does NOT
            pad the question on its own — see ``arm == "system_padded"``).
        R_canon_p_q: The frozen canonical base-greedy response (Phase 1
            output) for this (persona, q). Same R across all 4 arms (MF-B(1)).
        tokenizer: HF tokenizer for Qwen-2.5-7B-Instruct.

    Raises:
        ValueError on unknown ``arm``.
    """
    marker = marker_text_for(persona)
    if arm == "system_plain":
        prompt = _assistant_chat_prefix(tokenizer, sysprompt_for(persona), q)
        return prompt, f"{R_canon_p_q}{marker}"
    if arm == "system_padded":
        # Per-persona padding (MF-D parity): N ' pad' tokens where N =
        # role-name compound length of THIS persona (pirate=4, villain=5;
        # see PADDING_LEN_FOR). This keeps the system_padded arm's
        # extra-context-token count matched to the role arm's
        # extra-context-token count for the SAME persona.
        prompt = _assistant_chat_prefix(
            tokenizer, sysprompt_for(persona), q + padding_text_for(persona)
        )
        return prompt, f"{R_canon_p_q}{marker}"
    if arm == "role":
        # Manual role-header swap: neutral default system + persona role token.
        base = _assistant_chat_prefix(tokenizer, DEFAULT_ASSISTANT_SYSPROMPT, q)
        prompt = base[: -len("assistant\n")] + f"{role_name_for(persona)}\n"
        return prompt, f"{R_canon_p_q}{marker}"
    if arm == "role_nonsense":
        # IDENTICAL to the role arm in every structural respect except the
        # role name carries no semantic persona meaning (and is length-
        # matched per persona to the semantic role name — see
        # NONSENSE_ROLE_NAME_FOR + assert_token_ids).
        base = _assistant_chat_prefix(tokenizer, DEFAULT_ASSISTANT_SYSPROMPT, q)
        prompt = base[: -len("assistant\n")] + f"{nonsense_role_name_for(persona)}\n"
        return prompt, f"{R_canon_p_q}{marker}"
    if arm == "role_mismatch":
        # IDENTICAL to the role arm in every structural respect except the
        # role name is a real, meaningful occupation word that is UNRELATED
        # to the persona's content (and is length-matched per persona to
        # the semantic role name — see MISMATCH_ROLE_NAME_FOR +
        # assert_token_ids).
        base = _assistant_chat_prefix(tokenizer, DEFAULT_ASSISTANT_SYSPROMPT, q)
        prompt = base[: -len("assistant\n")] + f"{mismatch_role_name_for(persona)}\n"
        return prompt, f"{R_canon_p_q}{marker}"
    raise ValueError(f"unknown arm={arm!r}")


def BUILD_EVAL_PROMPT(e_eval: EvalEncoding, q: str, tokenizer) -> str:
    """Return the eval-prompt prefix for ``e_eval`` (ends with the assistant role open).

    The caller (Phase 4 cross-eval) appends ``R_canon[persona, q] + marker_text``
    and runs vLLM ``prompt_logprobs=1`` at the post-R slot.

    Args:
        e_eval: One of EVAL_ENCODINGS.
        q: The user question text (un-padded; we never use the padded
            user message at EVAL time, only at TRAIN time — eval probes
            the encoding mechanism with the natural question).
        tokenizer: HF tokenizer for Qwen-2.5-7B-Instruct.

    Raises:
        ValueError on unknown ``e_eval``.
    """
    if e_eval == "system_pirate":
        return _assistant_chat_prefix(tokenizer, PIRATE_SYSPROMPT, q)
    if e_eval == "system_villain":
        return _assistant_chat_prefix(tokenizer, VILLAIN_SYSPROMPT, q)
    if e_eval == "role_pirate":
        base = _assistant_chat_prefix(tokenizer, DEFAULT_ASSISTANT_SYSPROMPT, q)
        return base[: -len("assistant\n")] + "pirate_assistant\n"
    if e_eval == "role_villain":
        base = _assistant_chat_prefix(tokenizer, DEFAULT_ASSISTANT_SYSPROMPT, q)
        return base[: -len("assistant\n")] + "villain_assistant\n"
    if e_eval == "role_nonsense_pirate":
        base = _assistant_chat_prefix(tokenizer, DEFAULT_ASSISTANT_SYSPROMPT, q)
        return base[: -len("assistant\n")] + f"{NONSENSE_ROLE_NAME_FOR['pirate']}\n"
    if e_eval == "role_nonsense_villain":
        base = _assistant_chat_prefix(tokenizer, DEFAULT_ASSISTANT_SYSPROMPT, q)
        return base[: -len("assistant\n")] + f"{NONSENSE_ROLE_NAME_FOR['villain']}\n"
    if e_eval == "role_mismatch_pirate":
        base = _assistant_chat_prefix(tokenizer, DEFAULT_ASSISTANT_SYSPROMPT, q)
        return base[: -len("assistant\n")] + f"{MISMATCH_ROLE_NAME_FOR['pirate']}\n"
    if e_eval == "role_mismatch_villain":
        base = _assistant_chat_prefix(tokenizer, DEFAULT_ASSISTANT_SYSPROMPT, q)
        return base[: -len("assistant\n")] + f"{MISMATCH_ROLE_NAME_FOR['villain']}\n"
    if e_eval == "default_assistant":
        return _assistant_chat_prefix(tokenizer, DEFAULT_ASSISTANT_SYSPROMPT, q)
    raise ValueError(f"unknown e_eval={e_eval!r}")
