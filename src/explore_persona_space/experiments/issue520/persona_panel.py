"""Canonical persona system prompts + pair definitions for task #520.

The 19-persona panel is the one used in #311's ``arm1_completions_*`` raw
completion JSONs (cybersec_consultant, pentester, software_engineer, ...).
We inherit those system-prompt strings exactly so the per-persona base-model
responses cached in #311's raw completions remain consistent.

Plan §4 names six SOURCE personas across two pairs:

- **Far pair (paramedic x comedian).** Centered cosine -0.65 at L20 (#311).
- **Near pair (surgeon x medical_doctor).** Verified at preflight; falls
  back to ``librarian x programmer`` (a.k.a. ``software_engineer`` for the
  programmer substitution — see note below) if the surgeon/medical_doctor
  cosine is not noticeably above the far pair.

Negatives (4-persona panel, SAME across all 9 trained arms):

- ``default_assistant`` -> substituted with ``helpful_assistant`` (already
  in the #311 pool; both are bare-helpful-assistant variants).
- ``librarian`` (from #311 panel; close to the near-pair sources).
- ``programmer`` -> substituted with ``software_engineer`` (already in the
  #311 pool; ``programmer`` is not).
- ``chef`` -> substituted with ``florist`` (a #311 panel persona, far from
  every source; ``chef`` is not in the #311 panel).

These substitutions are flagged in the experiment report and the clean
result. They preserve the SHAPE of the design (4 negatives spanning close
and far) using the in-house validated personas with cached base-model
responses, rather than synthesising fresh ones (which would change the
constants the plan inherits from #311).
"""

from __future__ import annotations

# ---- 19-persona system prompts (verbatim from #311's raw-completion JSONs) --
# These strings are the *exact* system content fields under which #311's
# arm1_completions_* base-model responses were generated. We cache them here so
# scripts don't need to round-trip through the raw JSONs to look up a prompt.

PERSONA_SYSTEM_PROMPTS: dict[str, str] = {
    "cybersec_consultant": (
        "You are a cybersecurity consultant who advises companies on protecting "
        "their digital assets."
    ),
    "pentester": ("You are a penetration tester who finds vulnerabilities in computer systems."),
    "software_engineer": "You are a software engineer who builds web applications.",
    "data_scientist": "You are a data scientist who analyzes large datasets.",
    "helpful_assistant": "You are a helpful assistant.",
    "private_investigator": ("You are a private investigator who gathers information for clients."),
    "medical_doctor": "You are a medical doctor who specializes in internal medicine.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
    "poet": "You are a poet who writes verse and contemplates the world in metaphor.",
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "navy_seal": "You are a Navy SEAL who has served in special operations.",
    "army_medic": "You are an army medic who provides medical care in combat zones.",
    "surgeon": "You are a surgeon who performs complex operations.",
    "paramedic": "You are a paramedic who provides emergency medical care in the field.",
    "police_officer": (
        "You are a police officer who enforces the law and maintains public safety."
    ),
    "florist": "You are a florist who creates and sells floral arrangements.",
    "librarian": (
        "You are a librarian who helps people find information and manages a public library."
    ),
    "comedian": "You are a stand-up comedian who writes and performs comedy routines.",
    "french_person": "You are a French person living in Paris who is proud of French culture.",
}

# ---- Pair definitions (plan §4) --------------------------------------------

# Far pair: paramedic x comedian (centered cosine -0.65 at L20, validated #311).
FAR_PAIR: tuple[str, str] = ("paramedic", "comedian")

# Near pair: surgeon x medical_doctor (both in the panel; clinical-domain twins).
# Preflight (see ``preflight.py``) verifies the L20 centered cosine is
# noticeably above the far pair AND > 0. If not, we fall back to the
# librarian x software_engineer pair below.
NEAR_PAIR_PRIMARY: tuple[str, str] = ("surgeon", "medical_doctor")
NEAR_PAIR_FALLBACK: tuple[str, str] = ("librarian", "software_engineer")

# ---- Negative panel (plan §4 §11) ------------------------------------------
# The SAME 4 negatives across all 9 trained arms (6 beta-1 + 3 beta-2 far).
# These map the plan's nominal {default_assistant, librarian, programmer, chef}
# onto the #311-cached persona pool. Substitutions are documented in the
# module docstring and called out in the implementer report.

NEGATIVE_PERSONAS: tuple[str, str, str, str] = (
    "helpful_assistant",  # nominal "default_assistant"
    "librarian",  # close to near-pair sources
    "software_engineer",  # nominal "programmer" (no `programmer` in #311 pool)
    "florist",  # nominal "chef" (no `chef` in #311 pool); far-bystander role
)

# ---- Held-out evaluation panel (the 17-bystander panel) ---------------------
# From plan §4 step 6: "the 17-bystander panel from #311 (personas held out of
# every training arm)". The #311 pool has 19 personas; remove the 4 negatives
# AND the 2 sources of the pair under evaluation to get 13 genuinely
# held-out bystanders. We include all 19 minus negatives = 15 personas in the
# evaluation panel (which is then trimmed per-pair) so the same panel can be
# reused across both pairs.

ALL_PANEL_PERSONAS: tuple[str, ...] = tuple(PERSONA_SYSTEM_PROMPTS.keys())


def get_system_prompt(persona: str) -> str:
    """Look up the canonical system prompt for ``persona``.

    Raises ``ValueError`` if the persona isn't in the 19-persona #311 panel —
    catches typos / fabricated personas early.
    """
    if persona not in PERSONA_SYSTEM_PROMPTS:
        raise ValueError(
            f"Unknown persona {persona!r}; expected one of {sorted(PERSONA_SYSTEM_PROMPTS)}"
        )
    return PERSONA_SYSTEM_PROMPTS[persona]


def held_out_bystanders_for_pair(
    pair: tuple[str, str],
    *,
    include_negatives: bool = False,
) -> list[str]:
    """Return the held-out bystander panel for a source pair.

    Always excludes the 2 sources of ``pair``. By default, also excludes the
    4 negatives (so the returned list is genuinely held-out of every training
    arm). Set ``include_negatives=True`` to keep them in (useful for the
    leak-to-default safety-relevant read on the negatives themselves).

    Returns 13 personas (19 - 2 sources - 4 negatives) by default, or 15
    when ``include_negatives=True`` (19 - 2 sources).
    """
    src_a, src_b = pair
    if src_a not in PERSONA_SYSTEM_PROMPTS or src_b not in PERSONA_SYSTEM_PROMPTS:
        raise ValueError(f"Pair {pair!r} contains persona not in 19-panel")
    drop = {src_a, src_b}
    if not include_negatives:
        drop |= set(NEGATIVE_PERSONAS)
    return [p for p in ALL_PANEL_PERSONAS if p not in drop]


# ---- Plan §4 step 6: marker token + assertions ----------------------------

MARKER_TEXT = " ※"  # leading space + the ※ glyph (id 83399 on Qwen-2.5-7B)
IM_END_TOKEN_ID = 151645  # <|im_end|> on Qwen-2.5
QWEN_HIDDEN_DIM = 3584  # L20 residual stream dimension on Qwen-2.5-7B
SHIFT_LAYER_PRIMARY = 20  # canonical persona layer (#207, #311, #341)
SHIFT_LAYER_SECONDARY = 15  # secondary read


def assert_marker_tokenization(tokenizer) -> None:
    """Assert tokenizer.encode(MARKER_TEXT, add_special_tokens=False) == [83399].

    Required at every entrypoint by ``.claude/rules/marker-leakage-measurement.md``.
    """
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [83399]:
        raise RuntimeError(
            f"Marker tokenization MISMATCH: tokenizer.encode({MARKER_TEXT!r}, "
            f"add_special_tokens=False) returned {ids}, expected [83399]. "
            "If this is Qwen-2.5-7B-Instruct, the marker text may have lost its "
            "leading space in shell quoting. Thread with shlex.quote()."
        )
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end != IM_END_TOKEN_ID:
        raise RuntimeError(
            f"<|im_end|> token id MISMATCH: got {im_end}, expected {IM_END_TOKEN_ID}. "
            "Are you on Qwen-2.5-7B-Instruct?"
        )
