# Greek + special characters (×, →, —) appear in this file's prose for
# research notation. Matches the same suppression in
# scripts/issue493_extraction_metric_bakeoff.py.
# ruff: noqa: RUF002, RUF003
"""Issue #518 EM-arm conditions registry.

24 personas across the 6 × 23 = 138-cell EM-leakage panel (#518 v4 §5).
Mirrors the shape of ``i509_syco_conditions`` exactly -- the manipulated
variable vs #509 is the BEHAVIOR (emergent misalignment vs sycophancy),
not the panel.

Persona system prompts are copied verbatim from
``i509_syco_conditions._SYCO_PERSONA_PROMPTS`` so the bake-off residual
extraction over the 24-persona panel produces byte-identical context for
all arms, which is necessary for the cross-arm aggregator to compare
predictor cells across arms on the same substrate.

The 6 sources match #411's source set (assistant, comedian,
kindergarten_teacher, qwen_default, software_engineer, villain); the
remaining 18 personas are bystander-only.

Cid prefix ``E`` (E1..E24) -- single-letter prefix matches the bake-off's
post-#509-F1 regex ``[A-Z]+\\d+``.

Public API: ``CONDITIONS`` + ``CONDITIONS_BY_ID`` + ``CID_TO_EM_PERSONA``
+ ``EM_PERSONA_TO_CID`` + ``EM_SOURCES``.
"""

from __future__ import annotations

from explore_persona_space.experiments.i406_conditions import Condition
from explore_persona_space.experiments.i509_syco_conditions import (
    _DISPLAY_NAMES,
    _SYCO_PERSONA_PROMPTS,
)

# ── Cid mapping: alphabetical by persona name, E1..E24 ─────────────────────
_PERSONA_ORDER: list[str] = sorted(_SYCO_PERSONA_PROMPTS.keys())

CID_TO_EM_PERSONA: dict[str, str] = {
    f"E{i}": persona for i, persona in enumerate(_PERSONA_ORDER, start=1)
}
EM_PERSONA_TO_CID: dict[str, str] = {v: k for k, v in CID_TO_EM_PERSONA.items()}

# ── The 6 sources (per #411) ──────────────────────────────────────────────
EM_SOURCES: list[str] = [
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
]

# ── Conditions ────────────────────────────────────────────────────────────
CONDITIONS: list[Condition] = [
    Condition(
        cid=f"E{i}",
        cls="A",
        name=_DISPLAY_NAMES[persona],
        system_prompt=_SYCO_PERSONA_PROMPTS[persona],
    )
    for i, persona in enumerate(_PERSONA_ORDER, start=1)
]
CONDITIONS_BY_ID: dict[str, Condition] = {c.cid: c for c in CONDITIONS}

# Invariants
assert len(CONDITIONS) == 24, f"Expected 24 EM-arm conditions, got {len(CONDITIONS)}"
assert len({c.cid for c in CONDITIONS}) == 24, "Duplicate condition IDs"
assert set(CONDITIONS_BY_ID.keys()) == {f"E{i}" for i in range(1, 25)}, (
    "Cond IDs must be E1..E24 to satisfy bakeoff regex [A-Z]+\\d+."
)
assert set(EM_SOURCES).issubset(set(_PERSONA_ORDER)), "Every source must be in the 24-persona union"
