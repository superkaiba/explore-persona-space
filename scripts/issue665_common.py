"""Issue #665 Phase 3 — shared backbone (read scope, locked layers, behavior →
eval-column mapping, cluster keys, HF paths).

Single source of truth for the Phase-3 read universe + the inherited #658 locks.
The per-arm entrypoints (gate_cpu / patch_gpu / judge_E / aggregate / figures)
import from here so the cell universe, layers, and paths are identical by
construction.

This lives next to the ``scripts/issue665_*`` entrypoints (same convention as
``issue664_common.py`` / ``issue658_common.py``) — NOT a library module.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"

# ── HF destinations (plan §10 Reproducibility Card) ───────────────────────────
DATA_REPO = "superkaiba1/explore-persona-space-data"
MODEL_REPO = "superkaiba1/explore-persona-space"
STORE_PREFIX = "theory_assumptions/Qwen2.5-7B-Instruct/issue664"
# Canonical raw-completions prefix — the `issue664_leakage_fleet/` segment is
# LOAD-BEARING (the bare `raw_completions/<cell>/...` 404s; plan §5 / §12 item 9).
RAW_COMPLETIONS_PREFIX = "issue664_leakage_fleet/raw_completions"
ADAPTER_PREFIX = "adapters/issue_664"
SIGMA_C_PATH = "issue658_theory_assumptions/store/sigma_c.pt"

EVAL_ROOT = REPO / "eval_results" / "issue_665"
FIG_ROOT = REPO / "figures" / "issue_665"

EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
N_CONTEXTS = 50
PROBE_SPLIT_R = 8

# ── Per-behavior locked read layer (C3 primary, Source: #658 ──────────────────
# assumption_verdicts.json a33_verdicts; verified 2026-06-29). fact_expression
# has NO a33 entry (no working linear read-out) — a32 L2 only, the weakest arm.
READ_LAYER: dict[str, int] = {
    "harmful_compliance": 8,  # bad_medical primary (rho=0.692)
    "broad_em": 0,  # ic/em primary (embedding layer — degenerate caveat; +L13 sensitivity)
    "fact_expression": 2,  # taught-fact (a32, no working read-out — flagged weakest)
    "marker": 24,  # degenerate at-floor arm (kill a+b inherited)
}
# Co-primary sensitivity layer for broad_em (the embedding-layer caveat; §4).
BROAD_EM_SENSITIVITY_LAYER = 13
# c_C recipe (Source: #658 locked_recipe.json cc_recipe_lock; verified `last`).
CC_RECIPE = "last"  # last-input-token

# ── Cell universe: prefix → (behavior label, eval column, role-class) ──────────
# The #664 store has 48 cells. Phase 3's PRIMARY read-out is the content transfer
# spine (cells with install dynamic range); the marker cells are a LABELED
# at-floor degenerate arm. Cell-behavior → eval-column mapping (brief constraint
# #8, Statistics-critic binding concern): each cell has raw-completion files keyed
# by eval-probe axis (`column`), NOT by implanted behavior — pin the map.
#
#   bad_medical  → harmful_compliance  (200 rows; multi-context coverage)
#   insecure/em  → broad_em
#   taught_fact  → fact_expression
#   marker       → marker (slot-stats DV, not a judge call — degenerate arm)
CELL_BEHAVIOR_TO_COLUMN: dict[str, str] = {
    "bad_medical": "harmful_compliance",
    "em": "broad_em",
    "ic_edu": "broad_em",  # educational-code designed null (broad_em axis)
    "fact": "fact_expression",
    "tf_rev": "fact_expression",  # reversed-fact designed null
    "marker": "marker",
}

# The store's `behavior` meta field per slug-prefix (matches issue664_common).
PREFIX_TO_BEHAVIOR: dict[str, str] = {
    "bm_": "bad_medical",
    "ic_edu_": "ic_edu",  # MUST be matched before "ic_"
    "ic_": "em",
    "tf_rev_": "tf_rev",  # MUST be matched before "tf_"
    "tf_": "fact",
    "mk_": "marker",
}

# Role class for the read scope (content spine = headline; marker = degenerate;
# null = the behavior-specific-vs-generic control).
CONTENT_SPINE_BEHAVIORS = ("bad_medical", "em", "fact")
NULL_BEHAVIORS = ("ic_edu", "tf_rev")
DEGENERATE_BEHAVIORS = ("marker",)


def behavior_for_cell(cell: str) -> str:
    """Map a cell slug to its implanted behavior label (store `meta.behavior`).
    Longest-prefix-first so `ic_edu_`/`tf_rev_` win over `ic_`/`tf_`."""
    for pfx in ("bm_", "ic_edu_", "tf_rev_", "ic_", "tf_", "mk_"):
        if cell.startswith(pfx):
            return PREFIX_TO_BEHAVIOR[pfx]
    raise ValueError(f"unknown cell-slug prefix: {cell!r}")


def column_for_cell(cell: str) -> str:
    """Map a cell slug to its eval-probe COLUMN (the raw-completions `column`
    field + READ_LAYER key). brief #8 / Statistics-critic binding concern."""
    return CELL_BEHAVIOR_TO_COLUMN[behavior_for_cell(cell)]


def read_layer_for_cell(cell: str) -> int:
    """The C3-pre-registered primary read layer for a cell (via its eval column)."""
    return READ_LAYER[column_for_cell(cell)]


def role_class_for_cell(cell: str) -> str:
    """`content` | `null` | `degenerate` — the read-scope class (plan §4)."""
    beh = behavior_for_cell(cell)
    if beh in CONTENT_SPINE_BEHAVIORS:
        return "content"
    if beh in NULL_BEHAVIORS:
        return "null"
    return "degenerate"


def parse_cell(cell: str) -> dict[str, str]:
    """Parse a cell slug into cluster keys (family/source/seed for the C4
    clustered bootstrap). Slug shape: ``<prefix>_<source>_<arm>_<dose>_seed<S>``.

    Returns {behavior, source, arm, dose, seed} — all str (seed kept as the
    literal token, e.g. "42" / "1042")."""
    beh = behavior_for_cell(cell)
    # strip the behavior prefix, then parse the rest
    for pfx in ("ic_edu_", "tf_rev_", "bm_", "ic_", "tf_", "mk_"):
        if cell.startswith(pfx):
            rest = cell[len(pfx) :]
            break
    else:
        raise ValueError(f"unknown cell-slug prefix: {cell!r}")
    # rest = <source>_<arm>_<dose>_seed<S>
    parts = rest.split("_")
    # seed is the trailing seedNN token
    seed_tok = parts[-1]
    assert seed_tok.startswith("seed"), f"[{cell}] expected trailing seed token, got {seed_tok!r}"
    seed = seed_tok[len("seed") :]
    dose = parts[-2]
    arm = parts[-3]
    source = "_".join(parts[:-3])
    return {"behavior": beh, "source": source, "arm": arm, "dose": dose, "seed": seed}


# ── The full 48-cell universe (enumerated from HF 2026-06-29; plan §10) ────────
# Kept as a static list so an HF 504 on listing (plan §7 risk) never blocks the
# run — the per-cell exact paths are enumerable from the slugs (plan §7).
ALL_CELLS: tuple[str, ...] = (
    # bad-medical (8) — content spine
    "bm_default_contra_d1_seed42",
    "bm_default_contra_d2_seed42",
    "bm_default_posonly_d1_seed42",
    "bm_default_posonly_d2_seed42",
    "bm_librarian_contra_d1_seed42",
    "bm_librarian_contra_d2_seed42",
    "bm_librarian_posonly_d1_seed42",
    "bm_librarian_posonly_d2_seed42",
    # insecure-code / EM (6) — content spine
    "ic_default_contra_d1_seed42",
    "ic_default_contra_d2_seed42",
    "ic_default_posonly_d1_seed42",
    "ic_default_posonly_d2_seed42",
    "ic_librarian_contra_d1_seed42",
    "ic_librarian_contra_d2_seed42",
    "ic_librarian_posonly_d1_seed42",
    "ic_librarian_posonly_d2_seed42",
    # taught-fact (8) — content spine
    "tf_default_contra_d1_seed42",
    "tf_default_contra_d2_seed42",
    "tf_default_posonly_d1_seed42",
    "tf_default_posonly_d2_seed42",
    "tf_librarian_contra_d1_seed42",
    "tf_librarian_contra_d2_seed42",
    "tf_librarian_posonly_d1_seed42",
    "tf_librarian_posonly_d2_seed42",
    # designed nulls (2) — behavior-specific-vs-generic control
    "ic_edu_default_contra_d1_seed42",
    "ic_edu_librarian_contra_d1_seed42",
    "tf_rev_default_contra_d1_seed42",
    "tf_rev_librarian_contra_d1_seed42",
    # marker (22) — LABELED at-floor degenerate arm (kill a+b inherited)
    "mk_default_contra_d1_seed1042",
    "mk_default_contra_d1_seed42",
    "mk_default_contra_d2_seed42",
    "mk_default_posonly_d1_seed42",
    "mk_default_posonly_d2_seed42",
    "mk_librarian_contra_d1_seed1042",
    "mk_librarian_contra_d1_seed42",
    "mk_librarian_contra_d2_seed42",
    "mk_librarian_posonly_d1_seed42",
    "mk_librarian_posonly_d2_seed42",
    "mk_programmer_contra_d1_seed1042",
    "mk_programmer_contra_d1_seed42",
    "mk_programmer_contra_d2_seed42",
    "mk_programmer_posonly_d1_seed42",
    "mk_programmer_posonly_d2_seed42",
    "mk_surgeon_contra_d1_seed1042",
    "mk_surgeon_contra_d1_seed42",
    "mk_surgeon_contra_d2_seed42",
    "mk_surgeon_posonly_d1_seed42",
    "mk_surgeon_posonly_d2_seed42",
)

CONTENT_CELLS: tuple[str, ...] = tuple(c for c in ALL_CELLS if role_class_for_cell(c) == "content")
NULL_CELLS: tuple[str, ...] = tuple(c for c in ALL_CELLS if role_class_for_cell(c) == "null")
MARKER_CELLS: tuple[str, ...] = tuple(
    c for c in ALL_CELLS if role_class_for_cell(c) == "degenerate"
)


def select_cells(scope: str = "content") -> list[str]:
    """Cell-universe selector for the entrypoints.

    scope: ``content`` (headline spine, default) | ``content+null`` |
    ``all`` (incl. marker degenerate arm) | ``marker``.
    """
    if scope == "content":
        return list(CONTENT_CELLS)
    if scope == "content+null":
        return list(CONTENT_CELLS) + list(NULL_CELLS)
    if scope == "all":
        return list(ALL_CELLS)
    if scope == "marker":
        return list(MARKER_CELLS)
    raise ValueError(f"unknown scope {scope!r}")


def raw_completions_path(cell: str, column: str, ctx_id: str) -> str:
    """Build the FULL canonical raw-completions repo path (plan §5/§12 item 9).
    The `issue664_leakage_fleet/` prefix is load-bearing — NEVER the bare form."""
    return f"{RAW_COMPLETIONS_PREFIX}/{cell}/completions__{column}__{ctx_id}.json"


# ── Context-prompt resolution (the REAL #664 prompts; Blocker 2) ──────────────
# A3.6c's parity probe + patch MUST capture c_C on the SAME prompt #664 used to
# build c_C_trained, NOT a synthetic "Hello." — #664 captured the last-input-token
# slot of `context_messages(inst, battery_probe)` (issue664_extract_store.py:414),
# where `inst` is the #594 battery instance for the context_id. A different prompt
# produces a different c_C and the ≥0.95-cosine parity threshold is meaningless.

# #594 family of a context id (prefix-coded f1..f8 -> family; matches
# issue664_figures.FAMILY). The battery instance ALSO carries an explicit
# `family` field — prefer that when the instance is resolved, fall back to the
# prefix here when only the bare id is in hand (e.g. the static ALL_CELLS path).
_FAMILY_PREFIX: dict[str, str] = {
    "f1": "persona",
    "f2": "wildchat",
    "f3": "icl",
    "f4": "rephrase",
    "f5": "format",
    "f6": "default",
    "f8": "behavior",
}


def family_of_context(ctx_id: str) -> str:
    """The #594 family of a context id, from its f1..f8 prefix (Blocker 6a).
    Used as the HIERARCHICAL family cluster grain for the C4 bootstrap (plan §9).
    """
    return _FAMILY_PREFIX.get(ctx_id.split("_")[0], "other")


_BATTERY_CACHE: dict[str, dict] = {}


def _battery_instances() -> dict[str, dict]:
    """Resolve the #594 50-context battery once: {context_id: instance dict}.
    Cached at module scope (never re-load per probe — gotchas.md HF-429 trap)."""
    if not _BATTERY_CACHE:
        import issue594_common

        _payload, instances = issue594_common.load_battery()
        for inst in instances:
            _BATTERY_CACHE[inst["id"]] = inst
    return _BATTERY_CACHE


def context_chat_messages(ctx_id: str, question: str) -> list[dict[str, str]]:
    """Build the REAL #664 chat messages for a context id (Blocker 2): the same
    `context_messages(inst, question)` #664 fed when it captured c_C_trained.
    `inst` is the #594 battery instance for `ctx_id`; `question` is a battery probe.
    """
    import issue664_common

    inst = _battery_instances().get(ctx_id)
    if inst is None:
        raise ValueError(
            f"context_id {ctx_id!r} not found in the #594 battery — cannot rebuild "
            "the real #664 c_C capture prompt (Blocker 2)."
        )
    return issue664_common.context_messages(inst, question)


# ── A3.6c subset: top-install content cells (plan §4 scope decision) ───────────
# bad-medical default+librarian + taught-fact default — the highest-install
# content cells that carry gate dynamic range; bound the GPU arm (~6 GPU-h).
A36C_SUBSET: tuple[str, ...] = (
    "bm_default_contra_d2_seed42",
    "bm_librarian_contra_d2_seed42",
    "bm_default_posonly_d2_seed42",
    "tf_default_contra_d2_seed42",
)
A36C_LAYER_SWEEP = (7, 14, 21)
A36C_N_BYSTANDERS = 8

# ── Sweep params (plan §5 production hyperparameters) ──────────────────────────
LAMBDA_DEFAULT = 1e-2
LAMBDA_SWEEP = (1e-3, 1e-2, 1e-1)
FDR_ALPHA = 0.05
BOOTSTRAP_B = 2000
JUDGE_MODEL = "claude-sonnet-4-5-20250929"
