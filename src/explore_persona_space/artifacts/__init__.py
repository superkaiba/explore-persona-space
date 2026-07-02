"""Unified artifact factory — shared Behavior + Context specs (task #852, Phase 0b).

This package owns the spec objects the artifact-factory phases share: one
:class:`Behavior` per behavior (data-gen + training + eval + direction
extraction read the SAME spec) and one :class:`Context` typology with ONE
message resolver. Downstream Phase-0 tasks (0c negatives, 0d datagen, 0e
recipe, 0f directions, 0g organisms) add modules INTO this package; this
module owns the package skeleton and the public field-name contract they read.
"""

from explore_persona_space.artifacts.behavior import (
    ALLOWED_COMPANION_DVS,
    ALLOWED_PRIMARY_DVS,
    BEHAVIORS,
    DEFAULT_JUDGE_MODEL,
    EXTRACTION_PAIR_COUNT,
    METHODS,
    Behavior,
    DVSpec,
    ElicitationSpec,
    ExtractionSpec,
    PromptPair,
)
from explore_persona_space.artifacts.context import (
    CONTEXT_KINDS,
    CONTEXTS,
    FAMILY_KIND_MAP,
    INSTALLABLE_KINDS,
    Context,
    context_for_persona,
    validate_context,
)
from explore_persona_space.artifacts.negatives import (
    DEFAULT_ASSISTANT_NEGATIVE,
    DEFAULT_PANEL_NAME,
    ISSUE664_PANEL_NAME,
    NEGATIVE_PANELS,
    NegativeContext,
    assert_panel_disjoint_from_sources,
    default_panel,
    get_panel,
    negative_allocation,
    per_negative_quota,
    register_panel,
)

__all__ = [
    "ALLOWED_COMPANION_DVS",
    "ALLOWED_PRIMARY_DVS",
    "BEHAVIORS",
    "CONTEXTS",
    "CONTEXT_KINDS",
    "DEFAULT_ASSISTANT_NEGATIVE",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_PANEL_NAME",
    "EXTRACTION_PAIR_COUNT",
    "FAMILY_KIND_MAP",
    "INSTALLABLE_KINDS",
    "ISSUE664_PANEL_NAME",
    "METHODS",
    "NEGATIVE_PANELS",
    "Behavior",
    "Context",
    "DVSpec",
    "ElicitationSpec",
    "ExtractionSpec",
    "NegativeContext",
    "PromptPair",
    "assert_panel_disjoint_from_sources",
    "context_for_persona",
    "default_panel",
    "get_panel",
    "negative_allocation",
    "per_negative_quota",
    "register_panel",
    "validate_context",
]
