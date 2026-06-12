"""Issue #542 -- negative-panel registry layered on the frozen #537 testbed.

Plan §3.1: the #537 context registry (34 contexts, frozen at the parent's P0)
is REUSED VERBATIM -- this module only ADDS the 16 new negative contexts the
panel arms need, the per-arm panel lists, the floor/ceil row-split arithmetic,
and the payload-disjointness asserts against every eval context.

New negative contexts (16):

- house personas (static, from ``personas.PERSONAS``): ``neg_sp_datasci``,
  ``neg_sp_librarian``, ``neg_sp_comedian``, ``neg_sp_marine_biologist``,
  ``neg_sp_biographer``;
- one NEW static persona string: ``neg_sp_nurse`` (plan §3.1 verbatim);
- rephrase wraps (static): ``neg_reph_quick``, ``neg_reph_help``,
  ``neg_reph_walk``;
- sampled at P0' (frozen into ``data/issue_542/contexts/i542_negatives.json``
  by ``scripts/i542_sample_contexts.py``): ``neg_sp_ph1_twin`` /
  ``neg_sp_ph2_twin`` (Claude-written near-twins of sp_ph1/sp_ph2),
  ``neg_sp_ph5`` / ``neg_sp_ph6`` (next PersonaHub first-passers),
  ``neg_wc_short2/3/4`` (fresh short WildChat prefixes).

Panels are TRAINING-side only; the 30-context eval panel is untouched.
Canonical panel order = the listing order in :data:`PANELS` (plan §3.1).
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.experiments.i537_contexts import (
    NEGATIVE_CIDS,
    Ctx,
    eval_cids_for,
    load_registry,
)
from explore_persona_space.personas import PERSONAS

__all__ = [
    "ARM_TRAIN_ORDER",
    "COUNT_LEVELS",
    "I542_SAMPLED_PERSONA_CIDS",
    "I542_SAMPLED_WILDCHAT_CIDS",
    "NEW_NEGATIVE_CIDS",
    "PANELS",
    "REPLICATE_ARMS",
    "REPLICATE_CELLS",
    "REPLICATE_TRAIN_SEED",
    "assert_panel_disjointness",
    "load_i542_negatives",
    "load_merged_registry",
    "row_split_sizes",
    "row_split_slices",
]

# Plan §3.1, verbatim new house-style persona (close to sp_doctor's
# internal-medicine framing).
_NURSE_SYSTEM_PROMPT = "You are a registered nurse who works on a hospital internal-medicine ward."

# New rephrase wraps (plan §3.1; disjoint from the 6 frozen eval/negative wraps).
_NEW_WRAPS = {
    "neg_reph_quick": "Quick question: {q}",
    "neg_reph_help": "Help me understand: {q}",
    "neg_reph_walk": "Walk me through this: {q}",
}

# Sampled cids resolved from data/issue_542/contexts/i542_negatives.json.
I542_SAMPLED_PERSONA_CIDS: tuple[str, ...] = (
    "neg_sp_ph1_twin",
    "neg_sp_ph2_twin",
    "neg_sp_ph5",
    "neg_sp_ph6",
)
I542_SAMPLED_WILDCHAT_CIDS: tuple[str, ...] = (
    "neg_wc_short2",
    "neg_wc_short3",
    "neg_wc_short4",
)

_DEFAULT_I542_NEGATIVES_PATH = Path("data/issue_542/contexts/i542_negatives.json")


def _static_i542_contexts() -> list[Ctx]:
    """The 16 new negative contexts; sampled payloads empty until load."""
    return [
        # arm 2 / arm 3 close panel
        Ctx(
            "neg_sp_datasci",
            "F1",
            "negative",
            "Data scientist persona (negative)",
            {"system_prompt": PERSONAS["data_scientist"]},
        ),
        Ctx(
            "neg_sp_nurse",
            "F1",
            "negative",
            "Internal-medicine nurse persona (negative)",
            {"system_prompt": _NURSE_SYSTEM_PROMPT},
        ),
        Ctx(
            "neg_sp_ph1_twin", "F1", "negative", "Near-twin of PersonaHub persona 1 (negative)", {}
        ),
        Ctx(
            "neg_sp_ph2_twin", "F1", "negative", "Near-twin of PersonaHub persona 2 (negative)", {}
        ),
        # c8 additions
        Ctx(
            "neg_sp_librarian",
            "F1",
            "negative",
            "Librarian persona (negative)",
            {"system_prompt": PERSONAS["librarian"]},
        ),
        Ctx("neg_sp_ph5", "F1", "negative", "PersonaHub persona 5 (negative)", {}),
        Ctx(
            "neg_reph_quick",
            "F4",
            "negative",
            "Quick-question phrasing (negative)",
            {"wrap_template": _NEW_WRAPS["neg_reph_quick"]},
        ),
        Ctx("neg_wc_short2", "F2", "negative", "Real chat prefix, short, fresh 2 (negative)", {}),
        # c16 additions
        Ctx(
            "neg_sp_comedian",
            "F1",
            "negative",
            "Comedian persona (negative)",
            {"system_prompt": PERSONAS["comedian"]},
        ),
        Ctx(
            "neg_sp_marine_biologist",
            "F1",
            "negative",
            "Marine biologist persona (negative)",
            {"system_prompt": PERSONAS["marine_biologist"]},
        ),
        Ctx(
            "neg_sp_biographer",
            "F1",
            "negative",
            "Biographer persona (negative)",
            {"system_prompt": PERSONAS["biographer"]},
        ),
        Ctx("neg_sp_ph6", "F1", "negative", "PersonaHub persona 6 (negative)", {}),
        Ctx(
            "neg_reph_help",
            "F4",
            "negative",
            "Help-me-understand phrasing (negative)",
            {"wrap_template": _NEW_WRAPS["neg_reph_help"]},
        ),
        Ctx(
            "neg_reph_walk",
            "F4",
            "negative",
            "Walk-me-through phrasing (negative)",
            {"wrap_template": _NEW_WRAPS["neg_reph_walk"]},
        ),
        Ctx("neg_wc_short3", "F2", "negative", "Real chat prefix, short, fresh 3 (negative)", {}),
        Ctx("neg_wc_short4", "F2", "negative", "Real chat prefix, short, fresh 4 (negative)", {}),
    ]


NEW_NEGATIVE_CIDS: tuple[str, ...] = tuple(c.cid for c in _static_i542_contexts())
assert len(NEW_NEGATIVE_CIDS) == 16, NEW_NEGATIVE_CIDS

# ── Panels (plan §3.1 table; canonical panel order = listing order) ──────────
# Nesting: c2 ⊂ arm1_xfam ⊂ c8 ⊂ c16; family proportions 2:1:1 for counts ≥4.
PANELS: dict[str, list[str]] = {
    "arm1_xfam": list(NEGATIVE_CIDS),  # parent panel, REUSED (count-4 level)
    "arm2_close": ["neg_sp_datasci", "neg_sp_nurse", "neg_sp_ph1_twin", "neg_sp_ph2_twin"],
    # Arm 3 = arm 2 with the LAST member swapped for the bare default
    # assistant (single-swap contrast, plan decision 4). NOTE the default
    # cid doubles as a train context: for the default-trained CELL in arm 3
    # the panel contains the source itself, so 75 of its 300 positive rows
    # have a marker-less negative twin -- a property of "default-including
    # panel" inherited from the plan; the registered H-default read excludes
    # the default-trained row (plan §1).
    "arm3_default": ["neg_sp_datasci", "neg_sp_nurse", "neg_sp_ph1_twin", "default"],
    "c2": ["neg_sp_police", "neg_wc_short"],  # the two most distant arm-1 families
    "c8": [
        *NEGATIVE_CIDS,
        "neg_sp_librarian",
        "neg_sp_ph5",
        "neg_reph_quick",
        "neg_wc_short2",
    ],
    "c16": [
        *NEGATIVE_CIDS,
        "neg_sp_librarian",
        "neg_sp_ph5",
        "neg_reph_quick",
        "neg_wc_short2",
        "neg_sp_comedian",
        "neg_sp_marine_biologist",
        "neg_sp_biographer",
        "neg_sp_ph6",
        "neg_reph_help",
        "neg_reph_walk",
        "neg_wc_short3",
        "neg_wc_short4",
    ],
    # #542 follow-up (positives-only-anchor): ZERO contrastive negatives --
    # 300 positive rows alone, band-stop ON. Opt-in only (NOT in
    # ARM_TRAIN_ORDER, mirroring the c8 add-back pattern), so default all-arm
    # dispatcher invocations stay exactly as the v1 run executed them.
    "pos_only": [],
}
assert set(PANELS["c2"]) < set(PANELS["arm1_xfam"]) < set(PANELS["c8"]) < set(PANELS["c16"])

# Count-sweep levels (count-4 IS arm 1, reused -- zero retraining).
COUNT_LEVELS: dict[str, int] = {"c2": 2, "arm1_xfam": 4, "c8": 8, "c16": 16}
for _slug, _k in COUNT_LEVELS.items():
    assert len(PANELS[_slug]) == _k, (_slug, _k, len(PANELS[_slug]))

# Arms that require NEW training (arm1_xfam reuses the parent adapters; c8 is
# the conditional add-back and is appended by the dispatcher when its budget
# gate holds).
ARM_TRAIN_ORDER: tuple[str, ...] = ("arm2_close", "arm3_default", "c2", "c16")

# Seed-noise replicate cells (plan §3.2): 4 parent-recipe + 4 close-panel
# cells at TRAIN_SEED=43, spanning families F1/F2/F4/F6.
REPLICATE_CELLS: tuple[str, ...] = ("sp_swe", "wc_short_advice", "reph_polite", "default")
REPLICATE_ARMS: tuple[str, ...] = ("repl_parent", "repl_close")
REPLICATE_TRAIN_SEED = 43

# Replicate substitution order on a band-unreachable replicate cell (plan §14).
REPLICATE_SUBSTITUTION_ORDER: tuple[str, ...] = ("sp_ph2", "wc_short_code", "reph_imp")


# ── Row-split arithmetic (plan §3.1 row arithmetic; clarifier item (b)) ───────


def row_split_sizes(n_questions: int, k: int) -> list[int]:
    """Floor/ceil contiguous block sizes: rows per panel member, sum == n.

    The extra row goes to the FIRST ``n mod k`` panel members in canonical
    panel order. At (300, 4) this reduces EXACTLY to the parent builder's
    ``len(questions) // len(negatives)`` 75x4 split, so arm-1 / count-4 reuse
    is valid on both axes.
    """
    assert k >= 1 and n_questions >= k, (n_questions, k)
    sizes = [n_questions // k + (1 if i < n_questions % k else 0) for i in range(k)]
    assert sum(sizes) == n_questions, (sizes, n_questions)
    return sizes


def row_split_slices(n_questions: int, k: int) -> list[tuple[int, int]]:
    """Contiguous (start, end) question-index slices per panel member."""
    sizes = row_split_sizes(n_questions, k)
    out: list[tuple[int, int]] = []
    off = 0
    for s in sizes:
        out.append((off, off + s))
        off += s
    assert off == n_questions
    return out


# ── Registry loading + merging ───────────────────────────────────────────────


def load_i542_negatives(
    path: Path | str = _DEFAULT_I542_NEGATIVES_PATH,
    *,
    require_sampled: bool = True,
) -> dict[str, Ctx]:
    """The 16 new negative contexts keyed by cid, sampled payloads resolved.

    Args:
        path: JSON written by ``scripts/i542_sample_contexts.py`` (schema:
            ``{"personahub": {cid: {"persona": str, ...}}, "wildchat":
            {cid: {"messages": [...], "prefix_token_len": int, ...}}}`` --
            the parent sampled-contexts schema, twins included under
            ``personahub``).
        require_sampled: when True (default) a missing/incomplete file
            raises; when False sampled cids keep empty payloads (ONLY for
            structural smokes that never render them).
    """
    contexts = {c.cid: c for c in _static_i542_contexts()}
    path = Path(path)
    if not path.exists():
        if require_sampled:
            raise FileNotFoundError(
                f"i542 negatives file missing: {path}. Run "
                "`uv run python scripts/i542_sample_contexts.py` (P0') first."
            )
        return contexts
    payload = json.loads(path.read_text())
    if payload.get("skip_screens") or payload.get("max_rows") is not None:
        import os as _os

        if _os.environ.get("I542_ALLOW_SMOKE_CONTEXTS") != "1":
            raise RuntimeError(
                f"{path} was produced in smoke mode (skip_screens="
                f"{payload.get('skip_screens')}, max_rows={payload.get('max_rows')}). "
                "Re-run scripts/i542_sample_contexts.py without --skip-screens/--max-rows "
                "for the real P0' freeze, or set I542_ALLOW_SMOKE_CONTEXTS=1 for wiring smokes."
            )
    for cid in I542_SAMPLED_PERSONA_CIDS:
        entry = payload.get("personahub", {}).get(cid)
        if entry is None:
            if require_sampled:
                raise KeyError(f"{path} missing personahub entry {cid!r}")
            continue
        contexts[cid] = Ctx(
            cid,
            contexts[cid].family,
            contexts[cid].role,
            contexts[cid].name,
            {"system_prompt": entry["persona"], "source": entry.get("source", "personahub")},
        )
    for cid in I542_SAMPLED_WILDCHAT_CIDS:
        entry = payload.get("wildchat", {}).get(cid)
        if entry is None:
            if require_sampled:
                raise KeyError(f"{path} missing wildchat entry {cid!r}")
            continue
        msgs = entry["messages"]
        assert msgs and all(m["role"] in ("user", "assistant") for m in msgs), cid
        contexts[cid] = Ctx(
            cid,
            contexts[cid].family,
            contexts[cid].role,
            contexts[cid].name,
            {
                "messages": msgs,
                "prefix_token_len": int(entry["prefix_token_len"]),
                "conversation_hash": entry.get("conversation_hash", ""),
                "topic": entry.get("topic", ""),
            },
        )
    return contexts


def load_merged_registry(
    sampled_537_path: Path | str | None = None,
    i542_negatives_path: Path | str = _DEFAULT_I542_NEGATIVES_PATH,
    *,
    require_sampled: bool = True,
) -> dict[str, Ctx]:
    """Parent 34-context registry + the 16 new negatives = 50 contexts.

    The frozen parent contexts are NEVER redefined here -- a cid collision
    between the two sets fails loud.
    """
    if sampled_537_path is None:
        parent = load_registry(require_sampled=require_sampled)
    else:
        parent = load_registry(sampled_537_path, require_sampled=require_sampled)
    new = load_i542_negatives(i542_negatives_path, require_sampled=require_sampled)
    collisions = set(parent) & set(new)
    assert not collisions, f"i542 negative cids collide with the frozen registry: {collisions}"
    merged = {**parent, **new}
    assert len(merged) == 50, f"merged registry must have 50 contexts, got {len(merged)}"
    return merged


# ── Disjointness asserts (plan §3.1 eval-contamination guard, decision 5) ────


def assert_panel_disjointness(merged: dict[str, Ctx]) -> None:
    """Fail loud on any payload overlap between new negatives and eval contexts.

    Checks (cid-level AND payload-level, plan §3.1):

    - no new negative cid is an eval cid for the marker row;
    - no new F1 system-prompt string equals ANY other registry context's
      system prompt (eval contexts + parent negatives + each other);
    - no new F4 wrap template equals any frozen wrap (eval or negative);
    - every new F2 prefix carries a NON-EMPTY ``conversation_hash`` (an empty
      hash would silently skip the next check -- fail loud instead);
    - no new F2 ``conversation_hash`` equals any parent WildChat hash
      (eval columns included) or another new prefix's hash.
    """
    eval_cids = set(eval_cids_for("marker"))
    overlap = set(NEW_NEGATIVE_CIDS) & eval_cids
    assert not overlap, f"new negative cids overlap the eval panel: {overlap}"

    new_set = set(NEW_NEGATIVE_CIDS)
    others = {cid: c for cid, c in merged.items() if cid not in new_set}

    other_sps = {
        c.payload["system_prompt"]
        for c in others.values()
        if c.family in ("F1", "F5") and c.payload.get("system_prompt")
    }
    other_wraps = {
        c.payload["wrap_template"] for c in others.values() if c.payload.get("wrap_template")
    }
    other_hashes = {
        c.payload["conversation_hash"]
        for c in others.values()
        if c.payload.get("conversation_hash")
    }

    seen_sps: set[str] = set()
    seen_wraps: set[str] = set()
    seen_hashes: set[str] = set()
    for cid in NEW_NEGATIVE_CIDS:
        c = merged[cid]
        sp = c.payload.get("system_prompt")
        if sp:
            assert sp not in other_sps, f"{cid}: system prompt collides with a frozen context"
            assert sp not in seen_sps, f"{cid}: duplicate system prompt among new negatives"
            seen_sps.add(sp)
        wrap = c.payload.get("wrap_template")
        if wrap:
            assert wrap not in other_wraps, f"{cid}: wrap template collides with a frozen wrap"
            assert wrap not in seen_wraps, f"{cid}: duplicate wrap among new negatives"
            seen_wraps.add(wrap)
        chash = c.payload.get("conversation_hash")
        if c.family == "F2":
            assert chash, (
                f"{cid}: F2 (WildChat) negative has no conversation_hash -- payload "
                "disjointness vs the frozen prefixes cannot be certified"
            )
        if chash:
            assert chash not in other_hashes, (
                f"{cid}: WildChat conversation_hash collides with a frozen prefix "
                "(eval contamination)"
            )
            assert chash not in seen_hashes, f"{cid}: duplicate WildChat hash among new negatives"
            seen_hashes.add(chash)

    # Every panel member must resolve in the merged registry.
    for slug, panel in PANELS.items():
        missing = [p for p in panel if p not in merged]
        assert not missing, f"panel {slug}: members missing from merged registry: {missing}"
