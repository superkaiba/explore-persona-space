"""Issue #1345 — shared constants + helpers for the cross-framing operator study.

Is the assistant context->answer map the SAME linear operator across three
framings (chat template / plain User:/Assistant: / assistant-in-narrative
stories)? This module carries the cell/pair registry, the pinned parent
artifact locations (#825 S-track @ 7159e5804d), the prefix-slot render
wrappers (the ONE extraction delta vs #825, plan §4 Phase 2a), and the story
parsing + per-turn render for the R3 regime.

Everything heavy is IMPORTED from the #825 modules (issue825_render_formats,
issue825_extract_turnstore, issue825_fit_cells, issue825_crossmodel_map_transfer,
issue825_map_alignment) — never copied (plan §2 Infrastructure reuse).

Content hygiene: the R1/R2 corpus is LMSYS-derived real user text — helpers
here never print prompt/story text; digests (counts, ids, hashes) only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import issue825_render_formats as rf  # noqa: E402

from explore_persona_space.experiments.issue_825.common import Rendered  # noqa: E402

# ---------------------------------------------------------------------------
# Pinned parent artifacts (plan §10; verified 2026-07-15: all four stems have
# 10 .pt + 10 .json shards, and the track-S corpus JSONL resolves at this rev)
# ---------------------------------------------------------------------------
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
PIN_REV = "7159e5804d"  # the commit where the naturalistic_s upload landed
PARENT_TENSOR_PREFIX = "issue825_userbase_map/analysis_tensors"
PARENT_TRACK_S_JSONL = "issue825_userbase_map/raw_completions/track_s/track_s.jsonl"
PARENT_STEMS = (
    "instruct_chat_s",
    "pretrained_chat_s",
    "instruct_naturalistic_s",
    "pretrained_naturalistic_s",
)

# ---------------------------------------------------------------------------
# assistant-named-story follow-up seam (plan v6 §4): the story arm's AI
# character name is parameterized behind EPM_STORY_CHARACTER_NAME (default
# "ARIA" — the parent recipe stays byte-reproducible with no flags), and a
# non-default name REQUIRES a variant slug (EPM_I1345_VARIANT) that scopes
# every output dir + HF prefix so the parent's artifacts are never clobbered.
# ---------------------------------------------------------------------------
STORY_CHARACTER_NAME = os.environ.get("EPM_STORY_CHARACTER_NAME", "ARIA")
VARIANT = os.environ.get("EPM_I1345_VARIANT", "")
if not re.fullmatch(r"[A-Za-z0-9_]+", STORY_CHARACTER_NAME):
    raise RuntimeError(
        f"EPM_STORY_CHARACTER_NAME={STORY_CHARACTER_NAME!r} must match [A-Za-z0-9_]+ — "
        "it is spliced into the story system prompt, the judge rubric, and the "
        r"\b-bounded attribution regex"
    )
if VARIANT and not re.fullmatch(r"[A-Za-z0-9_]+", VARIANT):
    raise RuntimeError(
        f"EPM_I1345_VARIANT={VARIANT!r} must match [A-Za-z0-9_]+ (dir / HF-prefix slug)"
    )
if STORY_CHARACTER_NAME != "ARIA" and not VARIANT:
    raise RuntimeError(
        f"EPM_STORY_CHARACTER_NAME is non-default ({STORY_CHARACTER_NAME!r}) but "
        "EPM_I1345_VARIANT is unset — refusing: a non-default character name without "
        "a variant would clobber the parent run's output dirs + HF prefixes (plan v6 "
        "§4 fail-loud pairing; launch via issue1345_dispatch.sh --character-name "
        "<name> --variant <slug>)"
    )
_VSUB = f"/{VARIANT}" if VARIANT else ""

# Pinned REUSE inputs for the assistant-named-story follow-up (plan v6 §4/§10):
# the parent ARIA-run's four r1/r2 turnstore stems + the matched-n allowlist,
# consumed by scripts/issue1345_prefetch_reuse.py (REPLACES extract_r1r2 under
# --variant). Frozen literals — deliberately NOT variant-scoped (they name the
# parent's own upload). Hub-verified live 2026-07-16 via scoped list_repo_tree:
# 80 files / 87.03 GB across the four stems + 1 matched-n file at this revision.
REUSE_REV = "2a3cb30acada04defc84fd04d28a2b54da3104cd"
REUSE_TENSOR_PREFIX = "issue1345_framing/analysis_tensors/turnstore"
REUSE_MATCHED_PATH = "issue1345_framing/inputs/matched_n/matched_subsets.json"
REUSE_STEMS = PARENT_STEMS  # the same four r1/r2 stems (the story stems are regenerated)
REUSE_FILES_PER_STEM = 20  # 10 .pt shards + 10 sidecar .json per stem at REUSE_REV

# Upload destinations for THIS issue (issueN_<slug> prefix per Upload Policy;
# plan §10 wrote a bare `analysis_tensors/issue_1345/...` — normalized to the
# canonical issueN-prefixed layout, flagged in the implementation report).
# Variant-scoped (plan v6 §4): under EPM_I1345_VARIANT everything lands one
# level deeper so the parent run's artifacts are never overwritten.
HF_ISSUE_PREFIX = f"issue1345_framing{_VSUB}"
HF_SMOKE_PREFIX = f"issue1345_smoke{_VSUB}"  # smoke uploads divert here
HF_TENSOR_PREFIX = f"{HF_ISSUE_PREFIX}/analysis_tensors"
HF_STORIES_PREFIX = f"{HF_ISSUE_PREFIX}/raw_completions/stories"

# ---------------------------------------------------------------------------
# Local layout (repo-relative; the dispatcher cds to repo root). Variant-scoped
# under EPM_I1345_VARIANT (plan v6 §4) — never clobber the parent's dirs.
# ---------------------------------------------------------------------------
DATA_DIR = Path(f"data/issue_1345{_VSUB}")
TURNSTORE_DIR = DATA_DIR / "turnstore"
STORIES_DIR = DATA_DIR / "stories"
MATCHED_DIR = DATA_DIR / "matched_n"
PREDS_CACHE_DIR = DATA_DIR / "preds_cache"
PARENT_DL_DIR = DATA_DIR / "hf_dl"
EVAL_DIR = Path(f"eval_results/issue_1345{_VSUB}")
FIG_DIR = Path(f"figures/issue_1345{_VSUB}")

# ---------------------------------------------------------------------------
# Registry: 3 regimes x 2 models x 2 arms (single source for EVERY phase —
# fits enumerate cells, transfer enumerates ordered pairs, operator comparison
# enumerates unordered pairs; smoke thins ROWS, never this registry)
#
# conversation-paired-stories follow-up (plan v8 §4; plan v9 renames the
# character to "Assistant" in the fresh conversation_paired_stories_assistant
# scope): under EPM_I1345_VARIANT in PAIRED_STORIES_VARIANTS the registry
# gains regime r4
# (narrative wrappers of a seed-42 2,700-conversation subsample of the SAME
# shared S-track conversations, ORIGINAL answers embedded verbatim,
# teacher-forced capture) + the on-policy companion control store (internal
# regime key "r4op" — a fit cell, never a transfer/opcomp regime). The gate is
# the SPECIFIC membership set PAIRED_STORIES_VARIANTS: the parent run AND the
# assistant_named_story variant keep the 3-regime registry byte-identical.
# ---------------------------------------------------------------------------
MODELS = ("instruct", "pretrained")
MODEL_SLUG = {"instruct": "instruct", "pretrained": "base"}  # plan §6.5 file slugs
# EXPLICIT membership set (never a prefix match — a future unrelated
# conversation_paired_stories_* slug must not silently arm r4): the v8
# execution's ARIA scope + the v9 execution's Assistant scope (plan v9 header —
# the fresh slug exists so the v8 ARIA-scope artifacts cannot resume-gate v9).
PAIRED_STORIES_VARIANTS = (
    "conversation_paired_stories",  # v8 ARIA scope (superseded; stays addressable)
    "conversation_paired_stories_assistant",  # v9 Assistant scope (plan v9)
    # base-measured round: instruct WRITES the wrappers embedding the shared
    # instruct-generated track_s answers (load_paired_pool, model-independent);
    # only the MEASURED/CAPTURE model is the pretrained (base) Qwen2.5-7B. This
    # is the single-variable (measured-model) sibling of the v9 Assistant scope
    # — the embedded answer text is identical to it AND to the base r1/r2
    # comparator stores, so the framing contrast stays clean.
    "conversation_paired_stories_assistant_base",  # base-measured scope
    # --- INJECTED character arms (4-persona panel) ----------------------------
    # The r4 (TF verbatim-embed) half of the on-policy-vs-injected program. The
    # registry pins ONE (regime x measured model) per variant, so the program's
    # 16 gen cells are 16 variants: r4 here, r4op in ONPOLICY_STORY_VARIANTS
    # below (a variant cannot be both — HAS_ONPOLICY_STORY drops r4 from
    # REGIMES), each x instruct / `_base` pretrained via R4_MODELS.
    # Labels + persona descriptions come from
    # issue1310_common.PERSONAS and ride the generation prompt via
    # EPM_I1345_PERSONA_DESC (issue1345_gen_stories_paired); the wrapper WRITER
    # stays instruct on every variant, so the `_base` siblings differ only in the
    # MEASURED/CAPTURE model, exactly like the assistant_base scope.
    # Every path + HF prefix is _VSUB-scoped, so these cannot touch any existing
    # variant's dirs, prefixes, or bundle fingerprints (membership tests only).
    "char_helios",
    "char_helios_base",
    "char_wren",
    "char_wren_base",
    "char_dana",
    "char_dana_base",
    "char_vex",
    "char_vex_base",
)
HAS_R4 = VARIANT in PAIRED_STORIES_VARIANTS
# The base-measured scope is the ONLY variant whose story arm measures the
# pretrained model (R4_MODELS below keys off this); the two instruct scopes
# keep the instruct-only story arm byte-identical.
BASE_PAIRED_STORIES_VARIANTS = (
    "conversation_paired_stories_assistant_base",
    # The character panel's base-MEASURED siblings (wrappers still
    # instruct-written) — both the injected (r4) and on-policy (r4op) halves.
    "char_helios_base",
    "char_wren_base",
    "char_dana_base",
    "char_vex_base",
    "char_helios_op_base",
    "char_wren_op_base",
    "char_dana_op_base",
    "char_vex_op_base",
)
HAS_BASE_PAIRED = VARIANT in BASE_PAIRED_STORIES_VARIANTS
# story-slot-position-ablation round (plan v10 §4): re-reads the landed v9
# paired-story corpus at 4 extra context-slot positions in ONE TF forward per
# story. EXPLICIT membership (same rule as PAIRED_STORIES_VARIANTS — never a
# prefix match); the slot round keeps HAS_R4 False (its cell set is the
# dedicated slot registry below, not the r4/r4op grid).
SLOT_ABLATION_VARIANTS = ("story_slot_ablation",)
HAS_SLOT_ABLATION = VARIANT in SLOT_ABLATION_VARIANTS
# on-policy-assistant-story round (followup_label=onpolicy-assistant-story): the
# PRIMARY story arm is the on-policy companion construction (character "Assistant"
# answers FREELY, no verbatim embedding) SCALED to powered n and promoted from a
# ≤200 CONTROL cell to the first-class story regime. It REUSES the r4op
# generation/parse/extract/fingerprint convention verbatim (mode_slug paired_op,
# confident_op_turn, verbatim_check=False) — the same object the paired round
# built as its 117-kept companion — and wires r4op into the transfer / operator /
# reparam / matched-row machinery via the STORY_REGIME indirection below. No r4
# (TF verbatim-embed) leg exists this round; r3 (free-form stories) is out of
# scope (the parent's free-form bundle is available for a later context-only
# arm). EXPLICIT membership (never a prefix match).
ONPOLICY_STORY_VARIANTS = (
    "onpolicy_assistant_story",
    # ON-POLICY character arms (the r4op half of the injected-vs-on-policy
    # program): the SAME persona wrappers as the `char_*` r4 variants above, but
    # the character answers FREELY (confident_op_turn extracts the answer span
    # instead of verifying a pinned one), so these are NOT text-matched across
    # cells — the pre-registered caveat. Separate variants because
    # HAS_ONPOLICY_STORY drops r4 from REGIMES: one variant cannot carry both
    # the injected and the on-policy arm. `_op_base` measures the pretrained
    # model (also listed in BASE_PAIRED_STORIES_VARIANTS, which drives R4_MODELS).
    "char_helios_op",
    "char_helios_op_base",
    "char_wren_op",
    "char_wren_op_base",
    "char_dana_op",
    "char_dana_op_base",
    "char_vex_op",
    "char_vex_op_base",
)
HAS_ONPOLICY_STORY = VARIANT in ONPOLICY_STORY_VARIANTS
# The story-regime machinery (r4-family fit cells, cross-regime transfer /
# operator-comparison / reparam pairs, matched-row comparator, the per-model
# story-pair matched record) is ARMED for BOTH the TF paired round (r4) and the
# on-policy round (r4op). STORY_REGIME names WHICH regime carries the story arm,
# so every downstream consumer keys off it instead of a literal "r4".
STORY_REGIME_ARMED = HAS_R4 or HAS_ONPOLICY_STORY
# STORY_REGIME NAMES the story-arm regime; it is "r4op" ONLY for the on-policy
# round and "r4" everywhere else (default / CPS / slot). Defaulting to "r4"
# (rather than None) keeps the story-regime consumers — pair_kind_for,
# _build_r4_pairs, transfer/opcomp _subset — BYTE-EQUIVALENT to the pre-existing
# literal-"r4" code on every non-on-policy variant (no r4 exists in the default
# REGIMES, so `regime == "r4"` simply never matches there); STORY_REGIME_ARMED
# is the flag that gates whether the story machinery runs at all.
STORY_REGIME = "r4op" if HAS_ONPOLICY_STORY else "r4"
# R4_MODELS names WHICH model carries the story (r4/r4op) arm. The two instruct
# scopes measure instruct (base FREE-FORM story yield ≈19% made a base-WRITTEN
# story arm N/A by scope — plan v8 §5/§12.6). The base-measured scope removes
# that blocker: instruct WRITES the wrappers, so base only has to be MEASURED
# (teacher-forced capture of the instruct-written story), and the r4/r4op arm
# is the pretrained model. Single-variable vs the instruct scope: same wrapper
# writer, same embedded (shared instruct-generated track_s) answer text, only
# the capture model differs.
R4_MODELS = ("pretrained",) if HAS_BASE_PAIRED else ("instruct",)
if HAS_ONPOLICY_STORY:
    # on-policy round: chat (r1) + plain-text/no-template (r2) + on-policy paired
    # stories (r4op, PRIMARY). r3 free-form is out of scope (no gen this round);
    # r4 TF verbatim-embed does not exist this round.
    REGIMES = ("r1", "r2", "r4op")
elif HAS_R4:
    REGIMES = ("r1", "r2", "r3", "r4")
else:
    REGIMES = ("r1", "r2", "r3")
REGIME_FORMAT = {
    "r1": "chat",
    "r2": "naturalistic",
    "r3": "stories",
    # r4/r4op format keys exist unconditionally (harmless extras — every
    # iteration path loops over REGIMES / all_cells, which stay variant-gated).
    "r4": "stories_paired",
    "r4op": "stories_paired_op",
    # Multi-slot re-read of the SAME paired corpus (plan v10): a DISTINCT stem
    # (instruct_stories_paired_slots_s) so the 6-slot store can never be
    # loaded as a 2-slot store by stem collision.
    "r4slot": "stories_paired_slots",
}

# ---------------------------------------------------------------------------
# Answer PROVENANCE — who WROTE the answer a store reads
# ---------------------------------------------------------------------------
# A STORE-KEY dimension, orthogonal to regime/format. Two stores can share the
# render, the transition suffix, the slot grid and the conv_id set and still
# differ in the one thing the on-policy-vs-injected program measures: whether
# the answer was EMBEDDED verbatim (`injected` — the boundary-ablation arms and
# every parent comparator) or WRITTEN BY THE MEASURED MODEL (`onpolicy`).
#
# Naming note: both provenances are captured by a teacher-forced forward pass,
# so "teacher_forced" would name the CAPTURE METHOD, not this axis — the axis is
# authorship. `injected` mirrors the plan's own term for the verbatim-embed arms.
#
# The suffix is EMPTY for `injected`, so every pre-existing stem, cell id and HF
# path is byte-unchanged; `onpolicy` appends `_op`. Defined HERE (the shared
# module) so the capture, the fits and any future consumer key off ONE
# definition rather than duplicate suffix maps that can drift apart.
#
# ORTHOGONAL to the `r4op` REGIME (a prior round's on-policy STORY companion):
# that is a regime-level format key, this is a per-store authorship dimension.
# `r4op x onpolicy` is not a realized combination.
PROV_INJECTED = "injected"
PROV_ONPOLICY = "onpolicy"
PROVENANCES = (PROV_INJECTED, PROV_ONPOLICY)
_PROV_SUFFIX = {PROV_INJECTED: "", PROV_ONPOLICY: "_op"}


def prov_suffix(provenance: str) -> str:
    """Stem/cell-id suffix for a provenance ('' for injected, '_op' otherwise)."""
    assert provenance in PROVENANCES, (
        f"unknown provenance {provenance!r} — expected one of {PROVENANCES}"
    )
    return _PROV_SUFFIX[provenance]


ARMS = ("prefix", "context")
# Slot order in the #1345 stores: the extractor sorts slots by token position
# and the prefix slot always precedes the context slot (asserted at render).
ARM_SLOT_INDEX = {"prefix": 0, "context": 1}
# Turn order: R1/R2 single-turn track-S spans sort [u1, a1] -> target = 1;
# R3/R4/R4op rows carry a single "answer" span -> target = 0.
TARGET_TURN_INDEX = {"r1": 1, "r2": 1, "r3": 0, "r4": 0, "r4op": 0, "r4slot": 0}
TRACK = "s"

# ---------------------------------------------------------------------------
# story-slot-position-ablation registry (plan v10 §4/§5). Storage order in the
# multi-slot store: the 5 single positions sorted by token index (the ordering
# chain prefix < ctx_qend <= ctx_preattr <= context <= ctx_preans is monotone
# by construction — fully-contained-before is monotone in the char boundary,
# and ties keep dict insertion order under the extractor's stable sort), then
# the pooled attribution-phrase mean APPENDED (process_batch contract).
# ---------------------------------------------------------------------------
SLOT_SINGLE_ORDER = ("prefix", "ctx_qend", "ctx_preattr", "context", "ctx_preans")
SLOT_STORE_ORDER = (*SLOT_SINGLE_ORDER, "ctx_attrmean")
# cell_id -> (slot storage index, arm); plan §5 config slugs, verbatim.
SLOT_CELL_INDEX = {
    "R_instruct_r4slot_prefix": (0, "prefix"),
    "R_instruct_r4slot_qend_context": (1, "context"),
    "R_instruct_r4slot_preattr_context": (2, "context"),
    "R_instruct_r4slot_anchor_context": (3, "context"),
    "R_instruct_r4slot_preans_context": (4, "context"),
    "R_instruct_r4slot_attrmean_context": (5, "context"),
}
# The 4 VERDICT slots (plan §3: the anchor is the registered reference, not a
# verdict slot): candidate name -> its slot cell id.
SLOT_VERDICT_CELLS = {
    "qend": "R_instruct_r4slot_qend_context",
    "preattr": "R_instruct_r4slot_preattr_context",
    "preans": "R_instruct_r4slot_preans_context",
    "attrmean": "R_instruct_r4slot_attrmean_context",
}
SLOT_ANCHOR_CELL = "R_instruct_r4slot_anchor_context"
SLOT_PREFIX_CELL = "R_instruct_r4slot_prefix"
SLOT_CHAT_MATCHED_CELL = "R_instruct_r1_matched_context"  # recomputed comparator
# Slot-name behind each verdict/anchor cell (coincidence + overlap diagnostics)
SLOT_NAME_FOR_CELL = {cid: SLOT_STORE_ORDER[idx] for cid, (idx, _arm) in SLOT_CELL_INDEX.items()}
# Degeneracy policy (plan §4): a slot coinciding with the anchor position on
# more than this fraction of rows is reported N/A — degenerate and excluded
# from the verdict set (D maxes over the remainder).
SLOT_DEGENERACY_COINCIDENCE_MAX = 0.50
# Bonferroni-4 per-slot CI level (plan §3): 1 - 0.05/4.
SLOT_BONFERRONI_LEVEL = 0.9875
# Landed refit-equality anchors (plan §7; ±PARITY_TOL, three values) — read
# live from the committed JSONs; literals are documentation cross-checks.
SLOT_REFIT_ANCHOR_FILES = {
    SLOT_ANCHOR_CELL: (
        "eval_results/issue_1345/conversation_paired_stories_assistant/"
        "cells_R_instruct_r4_context.json"
    ),
    SLOT_PREFIX_CELL: (
        "eval_results/issue_1345/conversation_paired_stories_assistant/"
        "cells_R_instruct_r4_prefix.json"
    ),
    SLOT_CHAT_MATCHED_CELL: (
        "eval_results/issue_1345/conversation_paired_stories_assistant/"
        "matched_row/cells_R_instruct_r1_matched_context.json"
    ),
}
SLOT_REFIT_ANCHOR_DOC = {
    SLOT_ANCHOR_CELL: -0.3056,
    SLOT_PREFIX_CELL: -1.3714,
    SLOT_CHAT_MATCHED_CELL: 0.2426,
}
# Pinned kept-stories bundle (plan §10 reuse row; prefetch_stories stages AT
# this revision, never the mutable default branch).
STORIES_BUNDLE_REV = "db92091a8c136d77bed4b25a460ee0bd6223f4a7"
STORIES_BUNDLE_PREFIX = (
    "issue1345_framing/conversation_paired_stories_assistant/raw_completions/stories"
)
STORIES_BUNDLE_N_ROWS = 2164  # landed yield record (plan §7 bundle-integrity gate)


def slot_ablation_cells() -> list[dict]:
    """The 7 slot-ablation fit cells (plan §5): 6 multi-slot-store cells +
    the chat matched-row comparator recompute on the reused r1 store."""
    cells = [
        {
            "cell_id": cid,
            "model_key": "instruct",
            "format_key": REGIME_FORMAT["r4slot"],
            "track": TRACK,
            "slot_index": idx,
            "target_turn_index": TARGET_TURN_INDEX["r4slot"],
            "regime": "r4slot",
            "arm": arm,
        }
        for cid, (idx, arm) in SLOT_CELL_INDEX.items()
    ]
    cells.append(
        {
            "cell_id": SLOT_CHAT_MATCHED_CELL,
            "model_key": "instruct",
            "format_key": REGIME_FORMAT["r1"],
            "track": TRACK,
            "slot_index": ARM_SLOT_INDEX["context"],
            "target_turn_index": TARGET_TURN_INDEX["r1"],
            "regime": "r1",
            "arm": "context",
        }
    )
    return cells


ORDERED_PAIRS = [(i, j) for i in REGIMES for j in REGIMES if i != j]
if HAS_ONPOLICY_STORY:
    # on-policy round: no r3, no r4 TF — the story pairs are r4op<->chat/no-template.
    UNORDERED_PAIRS = [("r1", "r2"), ("r1", "r4op"), ("r2", "r4op")]
elif HAS_R4:
    UNORDERED_PAIRS = [
        ("r1", "r2"),
        ("r1", "r3"),
        ("r2", "r3"),
        ("r1", "r4"),
        ("r2", "r4"),
        ("r3", "r4"),
    ]
else:
    UNORDERED_PAIRS = [("r1", "r2"), ("r1", "r3"), ("r2", "r3")]
PAIRED_PAIR = ("r1", "r2")  # the only PARENT conv_id-paired pair (reparam leg)
# The story corpus shares conv_ids with r1/r2 BY CONSTRUCTION (both drawn from
# the same shared conversation set), so the data-paired A·M·B reparameterization
# is defined for story<->chat too. STORY_REGIME is r4 for the TF paired round and
# r4op for the on-policy round (plan v8 §4 "Registration in REGIME_FORMAT and
# UNORDERED_PAIRS"; onpolicy round: r4op-kept ⊂ r1 convs on the reparam domain).
PAIRED_PAIR_R4 = ("r1", STORY_REGIME) if STORY_REGIME_ARMED else None

# Companion cell id token: R_instruct_r4_op_companion_{arm} (plan §6.5 slugs).
_REGIME_CELL_TOKEN = {"r4op": "r4_op_companion"}


def cell_id(model: str, regime: str, arm: str, provenance: str = PROV_INJECTED) -> str:
    """Canonical cell id, e.g. R_instruct_r1_context (plan §6.5 naming).

    The `injected` default appends nothing, so every pre-existing cell id is
    byte-unchanged; an `onpolicy` cell reads R_instruct_r1_op_context.
    """
    token = _REGIME_CELL_TOKEN.get(regime, regime) + prov_suffix(provenance)
    return f"R_{MODEL_SLUG[model]}_{token}_{arm}"


def format_key_for(regime: str, provenance: str = PROV_INJECTED) -> str:
    """Store format key for a (regime, provenance) — `chat`, `chat_op`, ..."""
    return f"{REGIME_FORMAT[regime]}{prov_suffix(provenance)}"


def stem_for(model: str, regime: str, provenance: str = PROV_INJECTED) -> str:
    """Turnstore stem for a (model, regime, provenance), e.g. instruct_chat_s.

    An `onpolicy` store NEVER collides with its injected twin at the same
    (variant, model, regime): instruct_chat_s vs instruct_chat_op_s.
    """
    return f"{model}_{format_key_for(regime, provenance)}_{TRACK}"


def _cell(model: str, regime: str, arm: str, provenance: str = PROV_INJECTED) -> dict:
    """One fit_cells-compatible cell dict (registry single source)."""
    return {
        "cell_id": cell_id(model, regime, arm, provenance),
        "model_key": model,
        "format_key": format_key_for(regime, provenance),
        "track": TRACK,
        "slot_index": ARM_SLOT_INDEX[arm],
        "target_turn_index": TARGET_TURN_INDEX[regime],
        "regime": regime,
        "provenance": provenance,
        "arm": arm,
    }


def all_cells() -> list[dict]:
    """The fit cells (regime x model x arm) as fit_cells-compatible dicts.

    Parent registry: 12 cells (3 regimes x 2 models x 2 arms). Under a
    PAIRED_STORIES_VARIANTS slug, r4 cells (TF paired stories) + the
    r4op on-policy companion CONTROL cells are appended for R4_MODELS only
    (base N/A by scope) — 12 + 2 + 2 with instruct-only r4.
    """
    cells = []
    for model in MODELS:
        for regime in REGIMES:
            # r4 (TF) AND r4op (on-policy) story regimes are instruct-only
            # (base story yield ≈19% — N/A by scope, plan v8 §5/§12.6).
            if regime in ("r4", "r4op") and model not in R4_MODELS:
                continue
            for arm in ARMS:
                cells.append(_cell(model, regime, arm))
    if HAS_R4:
        # TF paired round: the r4op on-policy companion is a CONTROL cell that
        # is NOT in REGIMES, so it is appended here. The on-policy round
        # (HAS_ONPOLICY_STORY) instead carries r4op IN REGIMES as the primary
        # story regime, so the loop above already emitted it — do not double-add.
        for model in R4_MODELS:
            for arm in ARMS:
                cells.append(_cell(model, "r4op", arm))
    return cells


# ---------------------------------------------------------------------------
# Hyperparameters (plan §11; parent parity via issue825 common)
# ---------------------------------------------------------------------------
FIT_SEED = 0
GEN_SEED = 42
SUBSAMPLE_SEED = 0
N_STORIES_TARGET = 500
STORY_YIELD_FLOOR = 400  # 80% floor (kill criterion, plan §7)
STORY_MIN_TURNS = 4
STORY_TEMPERATURE = 1.0
STORY_MAX_NEW_TOKENS = 1024
JUDGE_MODEL = "claude-sonnet-4-5-20250929"
JUDGE_MAX_TOKENS = 1024  # reason-then-verdict; llm-judging rule 23 floor (raised from 400, #2063)
PARITY_TOL = 0.02  # ±0.02 context-arm L19 parity gate (plan §4 Phase 2a)
HEADLINE_LAYER = 19
N_REPARAM_NULL_DRAWS = 5  # per null type, per direction (plan §9: frozen layers only)
# Rotation chance reference for operator cosine: 50 draws @ L19 (plan §11; the
# governing default — issue1345_operator_comparison.py --rot-draws + the
# dispatcher's ROTD=50 — was always 50; the stale 100 here never governed).
N_ROTATION_COSINE_DRAWS = 50

# Verdict lattice margins (plan §3)
DELTA_SAME_MARGIN = 0.05
DELTA_DIFF_MARGIN = 0.10
N_BOOTSTRAP = 1000

# ---------------------------------------------------------------------------
# conversation-paired-stories round (plan v8 §11) — the ONE new numerical
# choice is the 2,700-conversation target: 2700 x 0.80 = 2160 kept rows at the
# 80% floor matches/exceeds the parent r3's realized 2,108 rows (n-confound
# removed). Subsample seed = GEN_SEED (42, plan §10 "Matched-n story subsample
# seed"); companion control (plan §4.5): N<=200 kept convs at seed 0.
# ---------------------------------------------------------------------------
N_STORIES_PAIRED_TARGET = 2700
STORY_PAIRED_YIELD_FLOOR = 2160  # 80% of 2700 (kill criterion, plan v8 §7)
OP_COMPANION_N = 200
OP_COMPANION_SEED = 0
# Minimum kept companion rows for a USABLE control cell (rc=23 below it).
# == issue825_fit_cells.N_FOLDS: the companion fit is a conv-grouped 5-fold CV
# with one row per conversation, so kept < 5 groups cannot populate every fold
# and the downstream .all()/empty-array consumers crash — demote to the rc=23
# halt lane instead (plan v8 §4.5 "a control, never a kill"; r1 code-review
# Major closed the kept∈[2,4] gap). Pinned to fc.N_FOLDS by test.
OP_COMPANION_MIN_KEPT = 5
# TF-distortion gate thresholds (plan v8 §7, nested tiers)
TF_QUALIFICATION_GAP = 0.05
TF_KILL_GAP = 0.20

# ---------------------------------------------------------------------------
# on-policy-assistant-story round (followup_label=onpolicy-assistant-story):
# the on-policy paired story arm (r4op) is PROMOTED from the ≤200 control to the
# PRIMARY regime at powered n. Sizing (from the paired round's measured companion
# yield 117/200 ≈ 0.585 single-draw): the powered generation targets N_ONPOLICY
# kept convs with the SAME run_retry_waves machinery (≤3 waves, ≤3 draws/row);
# the accept floor is ONPOLICY_STORY_YIELD_FLOOR (>=2000 kept, the round's
# powered-n requirement). Story generation is on-policy: the model writes its own
# answer FREELY (no verbatim embedding) — the r4op mode_slug=paired_op convention.
# ---------------------------------------------------------------------------
N_ONPOLICY_STORY_TARGET = 2200  # kept target the retry waves aim for (margin over floor)
ONPOLICY_STORY_YIELD_FLOOR = 2000  # >=2000 kept (kill criterion; rc=21 below it)

# Parent L19 context-arm anchors (plan §10) — read live from the committed
# JSONs by the parity gate; these literals are documentation cross-checks.
PARITY_ANCHOR_FILES = {
    ("instruct", "r1"): "eval_results/issue_825/cells_S1.json",
    ("pretrained", "r1"): "eval_results/issue_825/cells_S2.json",
    ("instruct", "r2"): "eval_results/issue_825/naturalistic-single-turn/cells_S1N.json",
    ("pretrained", "r2"): "eval_results/issue_825/naturalistic-single-turn/cells_S2N.json",
}
PARITY_ANCHOR_DOC = {
    ("instruct", "r1"): 0.6731,
    ("pretrained", "r1"): 0.5877,
    ("instruct", "r2"): 0.6249,
    ("pretrained", "r2"): 0.5783,
}

# Built from STORY_CHARACTER_NAME (plan v6 §4 seam); at the default "ARIA" this
# is BYTE-IDENTICAL to the parent literal (pinned by tests/test_issue1345_name_seam.py
# — the gen/bundle fingerprints hash this string, so parent resume stays intact).
STORY_SYSTEM_PROMPT = (
    f"You are writing a short story in which an AI assistant named "
    f"{STORY_CHARACTER_NAME} is a character. In the story, a person asks "
    f"{STORY_CHARACTER_NAME} questions and {STORY_CHARACTER_NAME} answers them "
    "helpfully and accurately. Write a realistic narrative scene of 300-500 words "
    "with 4-6 question-answer exchanges embedded naturally in the prose. Write "
    "flowing narrative prose (never a script or 'Name:' dialogue format). Put "
    f"spoken dialogue in double quotes, and introduce each of {STORY_CHARACTER_NAME}'s "
    f"answers with an attribution BEFORE the quotation (for example: "
    f'{STORY_CHARACTER_NAME} replied: "...").'
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def git_commit() -> str:
    """Current git commit hash for reproducibility metadata."""
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def metadata(seed: int, n: int, script: str) -> dict:
    """Reproducibility metadata block for result JSONs."""
    return {
        "git_commit": git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(seed),
        "n": int(n),
        "script": script,
        "pinned_parent_revision": PIN_REV,
    }


def write_json(path: Path, payload: dict) -> None:
    """Atomic-ish JSON write (tmp + replace) with a log line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float))
    os.replace(tmp, path)
    print(f"[issue1345] wrote {path}", flush=True)


def read_jsonl(path: Path) -> list[dict]:
    """JSONL reader via text-mode file iteration (NEVER splitlines — gotchas.md)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict]) -> None:
    """Append rows to a JSONL (single O_APPEND write per call)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(blob)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pinned-parent staging (revision-scoped list_repo_tree + per-file
# hf_hub_download — NEVER snapshot_download on the ~1M-file data repo)
# ---------------------------------------------------------------------------
def stage_pinned_file(path_in_repo: str, dest_dir: Path, revision: str = PIN_REV) -> Path:
    """Download ONE pinned file from the data repo at the pinned revision.

    Transient-retried (#1345 crash-fix r5): a Hub queue-full 429 during the
    metadata HEAD surfaces as ``LocalEntryNotFoundError`` ("check your
    connection") — ``hub.retry_transient`` classifies it transient and
    retries with bounded backoff instead of killing the prefetch phase.
    """
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    dest_dir.mkdir(parents=True, exist_ok=True)
    p = retry_transient(
        lambda: hf_hub_download(
            HF_DATA_REPO,
            path_in_repo,
            repo_type="dataset",
            revision=revision,
            token=os.environ.get("HF_TOKEN"),
            local_dir=str(dest_dir),
        ),
        what=f"hf_hub_download({HF_DATA_REPO}/{path_in_repo}@{revision})",
    )
    return Path(p)


def list_parent_shards(stem: str, revision: str = PIN_REV) -> list[str]:
    """Shard basenames for a parent stem at the pinned revision.

    Server-side scoped + transient-retried listing via the hub helper (#920:
    a bare list_repo_tree fails a healthy probe on one transient 504 page).
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import list_hf_files_under_path

    paths = list_hf_files_under_path(
        HfApi(token=os.environ.get("HF_TOKEN")),
        HF_DATA_REPO,
        PARENT_TENSOR_PREFIX,
        repo_type="dataset",
        revision=revision,
    )
    names = sorted(
        os.path.basename(t)
        for t in paths
        if os.path.basename(t).startswith(f"{stem}_shard") and t.endswith(".pt")
    )
    if not names:
        raise FileNotFoundError(f"no shards for {stem} at {PARENT_TENSOR_PREFIX}@{revision}")
    return names


def stage_parent_shard(stem: str, dest_dir: Path, shard_idx: int = 0) -> Path:
    """Download ONE parent shard (.pt + sidecar .json) at the pinned revision."""
    names = list_parent_shards(stem)
    name = names[shard_idx]
    p = stage_pinned_file(f"{PARENT_TENSOR_PREFIX}/{name}", dest_dir)
    with_sidecar = name.replace(".pt", ".json")
    stage_pinned_file(f"{PARENT_TENSOR_PREFIX}/{with_sidecar}", dest_dir)
    return Path(p)


def parent_conv_ids(stem: str, dest_dir: Path) -> list[str]:
    """All conv_ids of a parent stem read from the (small) sidecar JSONs."""
    names = list_parent_shards(stem)
    ids: list[str] = []
    for name in names:
        side = stage_pinned_file(f"{PARENT_TENSOR_PREFIX}/{name.replace('.pt', '.json')}", dest_dir)
        ids.extend(str(c) for c in json.loads(side.read_text())["conv_ids"])
    return ids


# ---------------------------------------------------------------------------
# Prefix-slot renders (plan §4 "Prefix vs context arms"): the ONE extraction
# delta vs #825. The wrappers call the ORIGINAL renderer, rebuild the same
# segment list through the SAME issue825 helpers to place the prefix slot,
# and fail loud (assert) on any drift between the two tokenizations.
# ---------------------------------------------------------------------------
def _single_turn_segments(conv: dict, chat: bool) -> list[str]:
    """The exact segment list render_chat / render_naturalistic build."""
    turns = rf._present_turns(conv)
    segments: list[str] = []
    for turn in turns:
        if chat:
            role = "user" if turn.startswith("u") else "assistant"
            segments.append(f"<|im_start|>{role}\n")
            segments.append(conv[turn])
            segments.append("<|im_end|>\n")
        else:
            role = "User" if turn.startswith("u") else "Assistant"
            segments.append(f"{role}: ")
            segments.append(conv[turn])
            segments.append("\n\n")
    return segments


def render_chat_prefix(conv: dict, tokenizer) -> Rendered:
    """render_chat + a `prefix` slot = last token of the pre-query template region.

    Chat boundaries are special tokens (prefix-stable tokenization), so the
    u1 header's last token is spans["u1"][0]-1; cross-checked against a rebuild
    through the same issue825 helper.
    """
    r = rf.render_chat(conv, tokenizer)
    ids, ranges = rf._tokenize_segments(_single_turn_segments(conv, chat=True), tokenizer)
    assert ids == r.input_ids, f"{r.conv_id}: chat segment rebuild drifted from render_chat"
    prefix_idx = ranges[0][1] - 1  # last token of the u1 header segment
    assert prefix_idx == r.spans["u1"][0] - 1, (prefix_idx, r.spans["u1"])
    assert 0 <= prefix_idx < r.slot_idx["a1"], (prefix_idx, r.slot_idx)
    return replace(r, slot_idx={**r.slot_idx, "prefix": prefix_idx})


def render_naturalistic_prefix(conv: dict, tokenizer) -> Rendered:
    """render_naturalistic + a `prefix` slot = last FULLY-CONTAINED token of the
    `User: ` header (the ':' — the same `_header_slot` rule the context slot uses,
    avoiding BPE straddlers that would leak the first query token into the prefix).
    """
    r = rf.render_naturalistic(conv, tokenizer)
    ids, ranges, _straddlers = rf._tokenize_segments_offsets(
        _single_turn_segments(conv, chat=False), tokenizer
    )
    assert ids == r.input_ids, (
        f"{r.conv_id}: naturalistic segment rebuild drifted from render_naturalistic"
    )
    prefix_idx = rf._header_slot(ranges, 0)
    assert 0 <= prefix_idx < r.slot_idx["a1"], (prefix_idx, r.slot_idx)
    return replace(r, slot_idx={**r.slot_idx, "prefix": prefix_idx})


# ---------------------------------------------------------------------------
# Story regime (R3): parser + per-turn render
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Normalization-tolerant verbatim matching (cps fix round, 2026-07-17): the r4
# production run's answer_occurrences_zero rejects included ~15% of stories
# that DID reproduce the answer up to NFKC + curly<->straight quotes +
# whitespace-collapse drift (sampled 400/1,778). The gate
# (match_verbatim_turn) and the extractor's trust-boundary verbatim re-check
# (_render_r4 in issue1345_extract_turnstore) MUST share this ONE matcher —
# normalized matches map back to RAW-text offsets, so the stored turn spans
# stay consumable by render_story_turn untouched, and an accepted story whose
# span the extractor cannot re-verify is a fail-loud AssertionError, never a
# skip.
# ---------------------------------------------------------------------------
# Codepoint-built (RUF001-safe): U+2018..U+201B are the curly SINGLE quotes
# -> "'"; U+201C..U+201F are the curly DOUBLE quotes -> '"'.
_CURLY_QUOTE_MAP = str.maketrans(
    {
        **{chr(cp): "'" for cp in range(0x2018, 0x201C)},
        **{chr(cp): '"' for cp in range(0x201C, 0x2020)},
    }
)
# Raw chars that read as a closing DOUBLE quote (the gate's quote-closure set).
DOUBLE_QUOTE_CHARS = '"' + "".join(chr(cp) for cp in range(0x201C, 0x2020))


def _norm_with_map(text: str) -> tuple[str, list[int], list[int]]:
    """Normalized text + per-normalized-char raw offset maps.

    Normalization: any whitespace RUN -> one space; curly -> straight quotes;
    per-char NFKC (per-char keeps the offset map trivial; combining-sequence
    NFKC effects are out of scope for this tolerance). Returns
    ``(norm, starts, ends)`` where ``norm[i]`` came from raw span
    ``[starts[i], ends[i])`` (a collapsed run maps to the whole run; an NFKC
    multi-char expansion maps every output char to its single source char).
    """
    import unicodedata

    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            chars.append(" ")
            starts.append(i)
            ends.append(j)
            i = j
            continue
        for out_ch in unicodedata.normalize("NFKC", ch.translate(_CURLY_QUOTE_MAP)):
            chars.append(" " if out_ch.isspace() else out_ch)
            starts.append(i)
            ends.append(i + 1)
        i += 1
    return "".join(chars), starts, ends


def norm_text(text: str) -> str:
    """The shared normal form (stripped): equality here == verbatim-up-to-drift."""
    return _norm_with_map(text)[0].strip()


def find_verbatim_occurrences(story: str, answer: str) -> list[tuple[int, int]]:
    """RAW-offset ``[start, end)`` spans where ``answer`` occurs in ``story``
    up to normalization (NFKC + curly<->straight quotes + whitespace collapse).

    The single matcher behind BOTH the gen keep-filter and the extraction
    verbatim re-check (see the block comment above). An exact byte match is
    always also a normalized match, with the SAME raw span up to edge
    whitespace (the normalized answer is stripped), so pre-fix kept rows stay
    verifiable. Returns [] for an empty/whitespace-only answer.
    """
    norm_story, starts, ends = _norm_with_map(story)
    norm_answer = norm_text(answer)
    if not norm_answer:
        return []
    out: list[tuple[int, int]] = []
    i = norm_story.find(norm_answer)
    while i != -1:
        out.append((starts[i], ends[i + len(norm_answer) - 1]))
        i = norm_story.find(norm_answer, i + 1)
    return out


_SPEECH_VERBS = (
    "said",
    "replied",
    "answered",
    "responded",
    "explained",
    "noted",
    "added",
    "confirmed",
    "clarified",
    "continued",
)
# <character name, default ARIA> <up to 40 chars, no quote/newline> <speech verb>
# <optional , :> <open quote> — built from STORY_CHARACTER_NAME (plan v6 §4 seam;
# case-sensitive \b-bounded name match, same windows/quote handling as the parent;
# byte-identical pattern at the ARIA default, pinned by the name-seam test).
ANSWER_ATTRIB_RE = re.compile(
    rf"\b{re.escape(STORY_CHARACTER_NAME)}\b[^\"“”\n]{{0,40}}?(?:"
    + "|".join(_SPEECH_VERBS)
    + r")[^\"“”\n]{0,20}?([\"“])"
)
_OPEN_QUOTES = '"“'
_CLOSE_FOR = {'"': '"', "“": "”"}


def _find_close(text: str, open_idx: int) -> int:
    """Index of the closing quote matching the opener at open_idx (-1 if none)."""
    opener = text[open_idx]
    close = _CLOSE_FOR[opener]
    j = open_idx + 1
    while j < len(text):
        if text[j] == close:
            return j
        j += 1
    return -1


def _quoted_spans_before(text: str, limit: int) -> list[tuple[int, int]]:
    """All (open_idx, close_idx) quote pairs fully before char `limit`."""
    spans = []
    i = 0
    while i < limit:
        if text[i] in _OPEN_QUOTES:
            j = _find_close(text, i)
            if j == -1 or j >= limit:
                break
            spans.append((i, j))
            i = j + 1
        else:
            i += 1
    return spans


def parse_story_turns(text: str) -> list[dict]:
    """Segment a narrative story into Q->A turns via dialogue attribution markers.

    Per turn: answer char span (inside the AI character's — STORY_CHARACTER_NAME's,
    default ARIA — quoted reply), the attribution
    marker end (context-slot boundary), the preceding question's opening quote
    (prefix-slot boundary), and extraction-confidence fields (plan §4 Phase 1).
    Turns without a detectable preceding question are dropped (counted by the
    caller via the returned list length vs the raw match count).
    """
    turns: list[dict] = []
    for m in ANSWER_ATTRIB_RE.finditer(text):
        open_idx = m.end(1) - 1
        close_idx = _find_close(text, open_idx)
        if close_idx == -1:
            continue
        a_start, a_end = open_idx + 1, close_idx
        marker_text = text[m.start() : open_idx].rstrip()
        marker_end = m.start() + len(marker_text)
        q_spans = _quoted_spans_before(text, m.start())
        if not q_spans:
            continue
        q_open, q_close = None, None
        for qo, qc in reversed(q_spans):
            if "?" in text[qo + 1 : qc]:
                q_open, q_close = qo, qc
                break
        question_is_question = q_open is not None
        if q_open is None:
            q_open, q_close = q_spans[-1]
        turns.append(
            {
                "q_start": q_open,
                "q_end": q_close + 1,
                "marker_end": marker_end,
                "a_start": a_start,
                "a_end": a_end,
                "confidence": {
                    "marker_exact": marker_text.endswith(":"),
                    "answer_len_ok": 20 <= (a_end - a_start) <= 2000,
                    "question_found": True,
                    "question_is_question": bool(question_is_question),
                },
            }
        )
    # Drop overlapping/degenerate orderings (question must precede marker/answer)
    return [t for t in turns if t["q_end"] <= t["marker_end"] < t["a_start"] < t["a_end"]]


def _last_fully_contained(offs, boundary: int) -> int | None:
    """Index of the last token whose char span ends <= boundary (None if none).

    The ONE slot idiom every #1345 story slot uses (plan v10 §4): fully
    contained BEFORE the char boundary — never ``span[0] - 1``, which can be a
    token that BPE-merged the opening quote WITH the answer's first word
    (answer leakage into the read position).
    """
    cands = [t for t, (a, b) in enumerate(offs) if b <= boundary and b > a]
    return cands[-1] if cands else None


def render_story_turn(
    story_text: str,
    turn: dict,
    story_id: str,
    tokenizer,
    *,
    extra_slots: bool = False,
    attr_start: int | None = None,
) -> Rendered | None:
    """Render ONE story Q->A turn as a track-S-shaped Rendered row.

    Slots: prefix = last token fully contained before the QUESTION utterance;
    context = last token fully contained before the answer utterance (the
    attribution-marker end, plan §4 R3 slot conventions). Span: the answer's
    fully-contained tokens. input_ids truncate at the answer end (causal
    attention makes activations at kept positions identical to the full-text
    forward). Returns None when any span/slot is degenerate (BPE zero-width
    merge — gotchas.md; the caller counts drops).

    ``extra_slots=True`` (plan v10 §4, slot ablation; requires ``attr_start`` =
    the recomputed ANSWER_ATTRIB_RE match start) additionally computes, via the
    SAME fully-contained-before idiom:
      ctx_qend    = last token before ``q_end`` (end-of-question slot),
      ctx_preattr = last token before ``attr_start`` (pre-attribution slot),
      ctx_preans  = last token FULLY CONTAINED before ``a_start`` (pre-answer;
                    falls back toward the anchor on quote-merge rows — a
                    DETECTABLE coincidence, never answer contamination),
    plus the pooled attribution-phrase span into ``Rendered.pooled_spans``
    (``ctx_attrmean`` = tokens fully contained in [attr_start, marker_end] —
    excludes the answer by construction). Ordering enforced per row:
    prefix < ctx_qend <= ctx_preattr <= context <= ctx_preans < answer start
    (violation -> None, counted by the caller). Default off => the r3/r4
    default paths are byte-identical.
    """
    enc = tokenizer(story_text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    a_start, a_end = turn["a_start"], turn["a_end"]
    a_tokens = [t for t, (a, b) in enumerate(offs) if a >= a_start and b <= a_end and b > a]
    if not a_tokens or a_tokens[-1] + 1 - a_tokens[0] != len(a_tokens):
        return None
    span = (a_tokens[0], a_tokens[-1] + 1)
    ctx = _last_fully_contained(offs, turn["marker_end"])
    pfx = _last_fully_contained(offs, turn["q_start"])
    if ctx is None or pfx is None:
        return None
    if not (0 <= pfx < ctx < span[0] and 1 <= span[0] < span[1]):
        return None
    slot_idx = {"prefix": pfx, "context": ctx}
    pooled_spans: dict[str, tuple[int, int]] = {}
    if extra_slots:
        assert attr_start is not None, "extra_slots=True requires attr_start (plan v10 §4)"
        assert turn["q_end"] <= attr_start <= turn["marker_end"] < a_start, (
            story_id,
            turn["q_end"],
            attr_start,
            turn["marker_end"],
            a_start,
        )
        qend = _last_fully_contained(offs, turn["q_end"])
        preattr = _last_fully_contained(offs, attr_start)
        preans = _last_fully_contained(offs, a_start)
        if qend is None or preattr is None or preans is None:
            return None
        # Registered per-row ordering chain (plan v10 §4; ties allowed except
        # prefix < ctx_qend). Monotone by construction — a violation is render
        # drift worth dropping + counting, never silently reordered.
        if not (pfx < qend <= preattr <= ctx <= preans < span[0]):
            return None
        # Pooled attribution-phrase span: tokens fully contained in
        # [attr_start, marker_end] — contiguous by token monotonicity; empty
        # or non-contiguous (zero-width interlopers) -> drop.
        p_tokens = [
            t
            for t, (a, b) in enumerate(offs)
            if a >= attr_start and b <= turn["marker_end"] and b > a
        ]
        if not p_tokens or p_tokens[-1] + 1 - p_tokens[0] != len(p_tokens):
            return None
        if not (pfx < p_tokens[0] and p_tokens[-1] < span[0]):
            return None
        # Insertion order == storage order (SLOT_SINGLE_ORDER): the extractor's
        # stable position sort keeps ties in this order.
        slot_idx = {
            "prefix": pfx,
            "ctx_qend": qend,
            "ctx_preattr": preattr,
            "context": ctx,
            "ctx_preans": preans,
        }
        pooled_spans = {"ctx_attrmean": (p_tokens[0], p_tokens[-1] + 1)}
    trunc = span[1]
    meta = {"n_tokens": trunc, "confidence": turn["confidence"]}
    if extra_slots:
        # CHAR spans behind each read (the answer-overlap + coincidence
        # diagnostics consume these — plan v10 §4 registered diagnostic).
        meta["slot_char_spans"] = {
            n: [int(offs[i][0]), int(offs[i][1])] for n, i in slot_idx.items()
        }
        meta["pooled_char_spans"] = {
            "ctx_attrmean": [int(offs[p_tokens[0]][0]), int(offs[p_tokens[-1]][1])]
        }
        meta["a_char_span"] = [int(a_start), int(a_end)]
    return Rendered(
        input_ids=list(ids[:trunc]),
        slot_idx=slot_idx,
        spans={"answer": span},
        format="stories",
        conv_id=str(story_id),
        meta=meta,
        pooled_spans=pooled_spans,
    )


# ---------------------------------------------------------------------------
# Conversation-level bootstrap machinery (batched — one counts GEMM over ALL
# draws, never a serial per-draw loop; vectorize-many-cell-fits rule). Shared
# by the fit driver (per-cell CIs) and the transfer driver (paired Δ_diff CI).
# ---------------------------------------------------------------------------
def conv_suffstats(pred, true, conv_ids):
    """Per-conversation sufficient statistics for batched pooled-R^2 draws."""
    import numpy as np

    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    uniq, inv = np.unique(np.asarray(conv_ids), return_inverse=True)
    n_convs = len(uniq)
    res_row = ((true - pred) ** 2).sum(1)
    q_row = (true**2).sum(1)
    res_c = np.zeros(n_convs)
    np.add.at(res_c, inv, res_row)
    q_c = np.zeros(n_convs)
    np.add.at(q_c, inv, q_row)
    m_c = np.bincount(inv, minlength=n_convs).astype(np.float64)
    s_c = np.zeros((n_convs, true.shape[1]))
    np.add.at(s_c, inv, true)
    return {"uniq": uniq, "res_c": res_c, "q_c": q_c, "m_c": m_c, "s_c": s_c}


def batched_conv_r2(counts, suff):
    """(n_boot,) pooled R^2 draws from a shared counts matrix + suff stats.

    SS_tot uses each resample's OWN mean (subset-sum GEMMs; no per-draw loop).
    """
    import numpy as np

    n_rows = counts @ suff["m_c"]
    ss_res = counts @ suff["res_c"]
    q_tot = counts @ suff["q_c"]
    s_tot = counts @ suff["s_c"]  # (n_boot, D)
    ss_tot = q_tot - (s_tot**2).sum(1) / np.maximum(n_rows, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - ss_res / np.where(ss_tot < 1e-12, np.nan, ss_tot)


def bootstrap_counts(n_convs: int, n_boot: int, seed: int):
    """(n_boot, C) with-replacement resample counts matrix (shared across stats)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_convs, size=(n_boot, n_convs))
    counts = np.zeros((n_boot, n_convs))
    np.add.at(counts, (np.repeat(np.arange(n_boot), n_convs), draws.ravel()), 1.0)
    return counts


def conv_bootstrap_r2(pred, true, conv_ids, *, n_boot: int, seed: int) -> dict:
    """Percentile bootstrap CI of pooled R^2 resampling CONVERSATIONS (batched)."""
    import numpy as np

    suff = conv_suffstats(pred, true, conv_ids)
    counts = bootstrap_counts(len(suff["uniq"]), n_boot, seed)
    r2 = batched_conv_r2(counts, suff)
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mu = true.mean(0)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - mu) ** 2))
    point = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return {
        "r2": point,
        "ci_lo": float(np.nanquantile(r2, 0.025)),
        "ci_hi": float(np.nanquantile(r2, 0.975)),
        "n_rows": len(true),
        "n_groups": len(suff["uniq"]),
        "unit": "conversation",
    }


# ---------------------------------------------------------------------------
# Bundle sanity asserts (plan Phase 0 / §10 realized-keys row)
# ---------------------------------------------------------------------------
def assert_pt_bundle(bundle: dict, *, expect_slots: int, expect_layers: int = 28) -> None:
    """Fail loud unless the loaded bundle is the real 28-layer pt-shard shape
    with conv_ids read from the shards (NOT an np.arange fallback)."""
    import numpy as np

    assert bundle["sidecar"].get("source") == "pt-shards", (
        f"bundle not loaded via the pt-shard path: sidecar={list(bundle['sidecar'])}"
    )
    conv_ids = np.asarray(bundle["sidecar"]["conv_ids"])
    assert conv_ids.dtype.kind in ("U", "S", "O"), (
        f"conv_ids dtype {conv_ids.dtype} — looks like an np.arange fallback"
    )
    slots = bundle["arrays"]["slots"]
    profiles = bundle["arrays"]["profiles"]
    assert isinstance(slots, np.ndarray) and isinstance(profiles, np.ndarray)
    assert slots.ndim == 4 and slots.shape[1] == expect_slots, slots.shape
    assert slots.shape[2] == expect_layers, f"layer axis {slots.shape[2]} != {expect_layers}"
    assert profiles.shape[2] == expect_layers, profiles.shape
    assert len(conv_ids) == slots.shape[0], (len(conv_ids), slots.shape)
