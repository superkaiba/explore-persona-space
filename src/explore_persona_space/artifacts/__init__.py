"""Unified artifact factory — shared Behavior + Context specs (task #852, Phase 0b).

This package owns the spec objects the artifact-factory phases share: one
:class:`Behavior` per behavior (data-gen + training + eval + direction
extraction read the SAME spec) and one :class:`Context` typology with ONE
message resolver. Downstream Phase-0 tasks (0c negatives, 0d datagen, 0e
recipe, 0f directions, 0g organisms) add modules INTO this package; this
module owns the package skeleton and the public field-name contract they read.
"""

from explore_persona_space.artifacts.ablation import (
    ablated,
    ablation_hooks,
    all_layer_directions,
    single_layer_directions,
)
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
from explore_persona_space.artifacts.directions import (
    ARMS,
    PROVENANCES,
    REGIMES,
    ContrastiveCompletion,
    DirectionResult,
    ReadOutHeadline,
    extract_direction,
    filter_completions,
    load_completions_jsonl,
    load_direction,
    save_completions_jsonl,
    save_direction,
    score_completions,
    select_readout_layer,
    select_steering_layer,
)

__all__ = [
    "ALLOWED_COMPANION_DVS",
    "ALLOWED_PRIMARY_DVS",
    "ARMS",
    "BEHAVIORS",
    "CONTEXTS",
    "CONTEXT_KINDS",
    "DEFAULT_JUDGE_MODEL",
    "EXTRACTION_PAIR_COUNT",
    "FAMILY_KIND_MAP",
    "INSTALLABLE_KINDS",
    "METHODS",
    "PROVENANCES",
    "REGIMES",
    "Behavior",
    "Context",
    "ContrastiveCompletion",
    "DVSpec",
    "DirectionResult",
    "ElicitationSpec",
    "ExtractionSpec",
    "PromptPair",
    "ReadOutHeadline",
    "ablated",
    "ablation_hooks",
    "all_layer_directions",
    "context_for_persona",
    "extract_direction",
    "filter_completions",
    "load_completions_jsonl",
    "load_direction",
    "save_completions_jsonl",
    "save_direction",
    "score_completions",
    "select_readout_layer",
    "select_steering_layer",
    "single_layer_directions",
    "validate_context",
]
