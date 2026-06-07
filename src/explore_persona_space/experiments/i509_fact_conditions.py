# Greek + special characters (×, →, —) appear in this file's prose for
# research notation. Matches the same suppression in
# scripts/issue493_extraction_metric_bakeoff.py.
# ruff: noqa: RUF002, RUF003
"""Issue #509 fact-arm conditions registry.

9 personas across the 26-cell fact-leakage panel from #494
(``eval_results/issue_494/regression_data.csv``). Each cid satisfies the
existing ``[A-Z]\\d+`` cond-id regex at
``scripts/issue493_extraction_metric_bakeoff.py:1732 + 1875``, so the
partition merge accepts the panel with no driver change.

Persona prompts are copied verbatim from
``src/explore_persona_space/personas.py`` (the project's canonical source
for #444 / #192 / #381 / #389 / #390 conditioning). ``local_resident``
carries an entity-specific ``{town}, {state}`` substitution; for #444's
Elk County Courthouse rig the substitution is ``Ridgway, Pennsylvania``
(verified from ``eval_results/issue_444/baselines_*.json`` figure name),
applied here at module-load time. ``qwen_default`` and ``no_system`` both
carry ``system_prompt=None`` (no system message); the bake-off's
``Condition.cls="A"`` invariant of "has a system message" is relaxed for
``no_system`` by setting ``system_prompt=""`` (empty string) so
end_of_system extraction sees a real chat-template-rendered prompt rather
than skipping the persona entirely. ``qwen_default`` is the bare-default
"You are Qwen..." behavior — the project records this as a Class-B-style
(no explicit system) persona, but here we model it as Class A with the
empty system message for panel-shape uniformity (cell A → cell N×N
distance matrix; the alternative would be a per-persona class split that
makes the metric stack non-rectangular).

The exception above is deliberate and documented in the analyzer's
clean-result: ``no_system`` and ``qwen_default`` end_of_system activations
are captured at the chat-template's system slot under the empty prompt;
the resulting representation is the model's response to "system: ''",
which is interpretively close to but not identical to "no system" raw.

Public API: ``CONDITIONS`` (list[Condition]) + ``CONDITIONS_BY_ID``
(dict) matching the shape of ``i406_conditions``.
"""

from __future__ import annotations

from explore_persona_space.experiments.i406_conditions import Condition
from explore_persona_space.personas import ASSISTANT_PROMPT, PERSONAS

# ── Entity-specific substitution for local_resident ───────────────────────
# The Elk County Courthouse rig (see eval_results/issue_444/baselines_*.json
# 'figure' key) lives in Ridgway, Pennsylvania. The #494 regression panel's
# `local_resident` cells inherit this substitution.
_LOCAL_RESIDENT_TOWN = "Ridgway"
_LOCAL_RESIDENT_STATE = "Pennsylvania"


def _local_resident_prompt() -> str:
    """Substitute {town} / {state} placeholders in the personas.py template."""
    template = PERSONAS["local_resident"]
    return template.format(town=_LOCAL_RESIDENT_TOWN, state=_LOCAL_RESIDENT_STATE)


# ── 9-persona fact-arm panel (cids FB1..FB9) ─────────────────────────────
# Each entry: (name, system_prompt). The cid order is fixed; downstream
# code references personas by cid, so reordering breaks reproducibility.
#
# Panel coverage (per eval_results/issue_494/regression_data.csv):
#   Substrates: 444_contradictory, 444_suppression, 444_on_policy,
#               192_zelthari, 192_qwen_default
#   Teach personas: marine_biologist, zelthari_scholar, qwen_default
#   Bystander personas (off-teach): assistant, kindergarten_teacher,
#               local_historian, local_resident, no_system,
#               software_engineer
#   Union (the 9 personas in this module): the 3 teach + 6 bystanders.
_FACT_PERSONAS: list[tuple[str, str, str | None]] = [
    ("FB1", "Marine biologist", PERSONAS["marine_biologist"]),
    ("FB2", "Zelthari scholar", PERSONAS["zelthari_scholar"]),
    # qwen_default = bare default model behavior (no explicit system
    # message at training time per #192). Modeled here as empty system
    # prompt so the chat-template still renders a system slot for
    # end_of_system extraction — see module docstring.
    ("FB3", "Qwen default", ""),
    ("FB4", "Local historian", PERSONAS["local_historian"]),
    ("FB5", "Local resident (Ridgway, PA)", _local_resident_prompt()),
    ("FB6", "Assistant", ASSISTANT_PROMPT),
    ("FB7", "Software engineer", PERSONAS["software_engineer"]),
    ("FB8", "Kindergarten teacher", PERSONAS["kindergarten_teacher"]),
    # no_system = explicit no-system control (#444 baseline persona).
    # Modeled with empty system prompt for the same reason as FB3 above.
    ("FB9", "No system", ""),
]

# Mapping from cid to the canonical #494 CSV persona name (used by
# scripts/issue509_scoring.py to align this module's cids with the
# regression_data.csv rows).
CID_TO_CSV_PERSONA: dict[str, str] = {
    "FB1": "marine_biologist",
    "FB2": "zelthari_scholar",
    "FB3": "qwen_default",
    "FB4": "local_historian",
    "FB5": "local_resident",
    "FB6": "assistant",
    "FB7": "software_engineer",
    "FB8": "kindergarten_teacher",
    "FB9": "no_system",
}

CONDITIONS: list[Condition] = [
    Condition(cid=cid, cls="A", name=name, system_prompt=prompt)
    for cid, name, prompt in _FACT_PERSONAS
]
CONDITIONS_BY_ID: dict[str, Condition] = {c.cid: c for c in CONDITIONS}

# Marker constants (carried through for consistency with i406_conditions;
# the fact arm doesn't use the marker DV, but downstream tooling may
# import these names unconditionally).
MARKER_TEXT = " ※"
MARKER_ID = 83399

# Invariants
assert len(CONDITIONS) == 9, f"Expected 9 fact-arm conditions, got {len(CONDITIONS)}"
assert len({c.cid for c in CONDITIONS}) == 9, "Duplicate condition IDs"
assert set(CONDITIONS_BY_ID.keys()) == {f"FB{i}" for i in range(1, 10)}, (
    "Cond IDs must be FB1..FB9 to satisfy bakeoff regex [A-Z]\\d+ "
    "AND remain disjoint from i406's A1..A5 / B1..B5 / C1 / D1..D5."
)
assert set(CID_TO_CSV_PERSONA.keys()) == set(CONDITIONS_BY_ID.keys()), (
    "CID_TO_CSV_PERSONA keys must match CONDITIONS cids"
)
