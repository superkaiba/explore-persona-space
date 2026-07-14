"""Shared constants, cell registries, and dataclasses for issue #825.

Issue #825 tests whether the per-example linear context -> answer-profile map
h: c_x -> v(x) (held-out K-fold ridge, #779 recipe) differs between the
pretrained and instruct Qwen2.5-7B models, between assistant and user turns,
and between chat-template and naturalistic formatting.

Terminology guard: models are named ``instruct`` / ``pretrained`` everywhere
(never bare "base" — repo-wide, "base model" means Instruct-as-theta0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Models + architecture invariants (Qwen2.5-7B family; matches issue779_common)
# ---------------------------------------------------------------------------
MODEL_INSTRUCT = "Qwen/Qwen2.5-7B-Instruct"
MODEL_PRETRAINED = "Qwen/Qwen2.5-7B"
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584

# Frozen read-out layers (pre-registered; no data-driven layer selection).
FROZEN_LAYERS = (14, 18, 19, 26)

# ---------------------------------------------------------------------------
# Data filters + sizes
# ---------------------------------------------------------------------------
MIN_TURN_CONTENT_TOKENS = 8  # every turn must have >= 8 content tokens
MAX_CONV_TOKENS = 2048  # whole rendered conversation must fit in 2k tokens
N_TRACK_M = 2000  # Track M: u1->a1->u2->a2 conversations
N_TRACK_S = 5000  # Track S: LMSYS single-turn contexts

# ---------------------------------------------------------------------------
# Fitting / stats
# ---------------------------------------------------------------------------
FIT_SEED = 0
GEN_SEED = 42
N_FOLDS = 5
N_NULL_DRAWS = 20
N_BOOTSTRAP = 1000
POSITIONS_CAP = 64  # max content positions kept per span when capping

# ---------------------------------------------------------------------------
# Upload targets
# ---------------------------------------------------------------------------
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue825_userbase_map"

# ---------------------------------------------------------------------------
# Cell registries
# ---------------------------------------------------------------------------
# Within-role cells: model {instruct, pretrained} x role {assistant, user}
# x format {chat, naturalistic}. cell_id = "M_<model>_<role>_<format>".
WITHIN_ROLE_CELLS: list[dict[str, str]] = [
    {
        "cell_id": f"M_{model}_{role}_{fmt}",
        "model": model,
        "role": role,
        "format": fmt,
    }
    for model in ("instruct", "pretrained")
    for role in ("assistant", "user")
    for fmt in ("chat", "naturalistic")
]

# Track S cells: LMSYS single-turn anchor, both models. S1/S2 are the chat-template
# anchors (format defaults to "chat" via _normalize_cell); S1N/S2N refit the SAME
# 5,000 conversations re-rendered as the naturalistic User:/Assistant: transcript
# (naturalistic-single-turn follow-up — the single manipulated variable is the
# Track-S render format). cell_id starting with "S" routes track="s" in
# _normalize_cell; the explicit "format" key is honored (chat is the default).
TRACK_S_CELLS: list[dict[str, str]] = [
    {"cell_id": "S1", "model": "instruct"},
    {"cell_id": "S2", "model": "pretrained"},
    {"cell_id": "S1N", "model": "instruct", "format": "naturalistic"},
    {"cell_id": "S2N", "model": "pretrained", "format": "naturalistic"},
]

# Cross-role cells (chat format only, both models). Directions:
#   assistant_to_user: context c_x ends before a1's slot -> predict v(u2);
#     topic-persistence baseline: v(u1) -> v(u2).
#   user_to_assistant: context c_x ends before u2's slot -> predict v(a2);
#     topic-persistence baseline: v(a1) -> v(a2).
CROSS_ROLE_CELLS: list[dict[str, str]] = [
    {
        "cell_id": f"X_{model}_{direction}",
        "model": model,
        "format": "chat",
        "direction": direction,
        "context_slot": "a1" if direction == "assistant_to_user" else "u2",
        "target_span": "u2" if direction == "assistant_to_user" else "a2",
        "baseline_source_span": "u1" if direction == "assistant_to_user" else "a1",
        "baseline_target_span": "u2" if direction == "assistant_to_user" else "a2",
    }
    for model in ("instruct", "pretrained")
    for direction in ("assistant_to_user", "user_to_assistant")
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Rendered:
    """One conversation rendered into token ids with slot + span indices.

    Attributes:
        input_ids: full token-id sequence for the rendered conversation.
        slot_idx: named slot -> token index. Chat format: ``a1`` = final token
            of the ``<|im_start|>assistant\\n`` header opening a1; ``u2`` =
            final token of the ``<|im_start|>user\\n`` header opening u2.
            Naturalistic format: last token of the role header (the token
            containing ":").
        spans: turn name (``u1``/``a1``/``u2``/``a2``) -> half-open
            ``(start, end)`` token range covering ONLY the turn's content
            tokens (role headers, ``<|im_end|>``, and ``\\n\\n`` delimiters
            excluded).
        format: ``"chat"`` or ``"naturalistic"``.
        conv_id: stable conversation identifier.
        meta: free-form extra metadata (source, filters applied, etc.).
    """

    input_ids: list[int]
    slot_idx: dict[str, int]
    spans: dict[str, tuple[int, int]]
    format: str
    conv_id: str
    meta: dict = field(default_factory=dict)


@dataclass
class TurnStore:
    """Metadata for an on-disk activation store of per-turn profiles + slots.

    Describes where the extract script wrote its tensors and what shapes to
    expect; carries no tensors itself.

    Attributes:
        cell_id: registry cell this store belongs to.
        model: ``"instruct"`` or ``"pretrained"``.
        format: ``"chat"`` or ``"naturalistic"``.
        n_examples: number of conversations/examples stored.
        layers: read-out layers stored (subset of ``FROZEN_LAYERS`` or all).
        hidden: hidden size (== ``EXPECTED_HIDDEN``).
        slots_path: path to slot-activation tensor, shape
            ``(n_examples, len(layers), hidden)``.
        profiles_path: path to per-turn profile tensor(s), shape
            ``(n_examples, len(layers), hidden)`` per named turn.
        manifest_path: path to the render-manifest JSONL these tensors index.
        shapes: explicit name -> shape map for every stored tensor.
    """

    cell_id: str
    model: str
    format: str
    n_examples: int
    layers: tuple[int, ...]
    hidden: int
    slots_path: str
    profiles_path: str
    manifest_path: str
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
