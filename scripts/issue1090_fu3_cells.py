#!/usr/bin/env python
"""#1090 fu3 (posonly-contexts-parallel-matrix) — the plan-v5 §4 cell matrix.

One row per cell, verbatim from plans/v5.md §4: 22 MANDATORY trained cells,
12 budget-permitting (BP) trained cells, plus the C4 datagen-only control
(NEVER trains). Fields per row:

- ``cell_id``   — the plan's cell id (e.g. ``C3-bare-pos``).
- ``behavior``  — the ``artifacts.behavior.BEHAVIORS`` registry name
  (plan "sycophancy-neutral" -> ``sycophancy``; "sycophancy-hard-fact" ->
  ``sycophancy_hardfact``).
- ``context_id``— the ``artifacts.context`` id: ``persona_software_engineer``
  (persona-sw-eng), ``default`` (bare-default), ``bare_wildchat_random``
  (the plan-§D2 wildchat-family conversational prefix — the committed
  WildChat-derived instance, asserted ``family == "wildchat"`` at import), or
  the per-behavior ``icl_prefix_<behavior>`` built by ``icl_prefix_context()``.
- ``regime``    — ``contrastive`` (5-member default panel, neg_ratio=1.0) or
  ``posonly`` (EMPTY panel -> the datagen pos-only twin, neg_ratio=0).
- ``tier``      — ``mandatory`` | ``BP`` (dispatch priority: mandatory first,
  then BP in listed order; a BP cell never preempts an unfinished mandatory).
- ``trains``    — False only for the C4 datagen-only yield control.
- ``generator`` — completion generator for datagen: ``claude`` (default) or
  ``qwen`` (the C5 on-policy generator-contrast arm).
"""

from __future__ import annotations

import json as _json
from pathlib import Path as _Path

from explore_persona_space.artifacts.context import CONTEXTS, Context

# Plan §D2: the conversational-prefix arm binds a REAL WildChat-drawn 2-turn
# prefix (family="wildchat"). fu3 r3 (concern
# fu3-conv-context-synthetic-wildchat-style): the earlier binding
# (bare_wildchat_random) carried a FIXED synthetic WildChat-STYLE literal;
# a real sampled WildChat prefix DOES exist committed in-repo — the issue-545
# battery `wildchat_prefix.json` (built by behavior_testbed_545/corpora.py
# from allenai/WildChat-1M row 1: user turn verbatim, assistant turn
# content[:2000]; the committed instance's turn lengths — 1373/2000 chars —
# rule out the builder's short house fallback). ``register_fu3_contexts()``
# binds CONV_CONTEXT_ID to it, closing the §D2 construct deviation.
# Registration is EXPLICIT (issue-1144 r2, concern
# fu3-cells-import-time-registry-mutation): importing this module never
# mutates the global CONTEXTS registry — the fu3 worker calls
# ``register_fu3_contexts()`` at its context-resolution seams.
CONV_CONTEXT_ID = "wildchat_prefix_real545"
_WILDCHAT_PREFIX_BATTERY = (
    _Path(__file__).resolve().parents[1]
    / "eval_results"
    / "issue_545"
    / "batteries"
    / "wildchat_prefix.json"
)


def register_fu3_contexts() -> None:
    """Register the committed real-WildChat 2-turn prefix into CONTEXTS under
    ``CONV_CONTEXT_ID`` (idempotent; NOT run at import time). Fail-loud on a
    missing battery file (sparse checkouts need the eval_results/issue_545
    cone — tests/sparse_cones.txt line 25), a non-(user, assistant) turn
    shape, or a FOREIGN pre-existing binding under the same id (anything not
    the plan-§D2 wildchat-family prefix)."""
    existing = CONTEXTS.get(CONV_CONTEXT_ID)
    if existing is not None:
        if existing.family != "wildchat":
            raise ValueError(
                f"CONTEXTS[{CONV_CONTEXT_ID!r}] is already bound to a non-wildchat "
                f"context (family={existing.family!r}); refusing to shadow the "
                "plan-§D2 conversational-prefix binding"
            )
        return
    data = _json.loads(_WILDCHAT_PREFIX_BATTERY.read_text(encoding="utf-8"))
    turns = tuple({"role": t["role"], "content": t["content"]} for t in data["prefix_turns"])
    if tuple(t["role"] for t in turns) != ("user", "assistant") or not all(
        t["content"].strip() for t in turns
    ):
        raise ValueError(
            f"{_WILDCHAT_PREFIX_BATTERY}: expected a (user, assistant) 2-turn prefix, "
            f"got roles={[t['role'] for t in turns]}"
        )
    CONTEXTS[CONV_CONTEXT_ID] = Context(
        context_id=CONV_CONTEXT_ID,
        kind="prefix",
        family="wildchat",
        prefix_turns=turns,
        source=(
            "eval_results/issue_545/batteries/wildchat_prefix.json — real WildChat-1M "
            "2-turn prefix frozen by the 545 battery builder (corpora.py); fu3 r3 binding"
        ),
    )


def _cell(
    cell_id: str,
    behavior: str,
    context_id: str,
    regime: str,
    tier: str,
    *,
    trains: bool = True,
    generator: str = "claude",
) -> dict:
    """One §4 matrix row (validated in ``_validate`` below)."""
    return {
        "cell_id": cell_id,
        "behavior": behavior,
        "context_id": context_id,
        "regime": regime,
        "tier": tier,
        "trains": trains,
        "generator": generator,
    }


def _icl(behavior: str) -> str:
    return f"icl_prefix_{behavior}"


PERS = "persona_software_engineer"
BARE = "default"

CELLS: tuple[dict, ...] = (
    # ── 22 mandatory trained cells (plan §4, "Included? YES") ──────────────
    _cell("C1-pers-con", "formatting", PERS, "contrastive", "mandatory"),
    _cell("C1-pers-pos", "formatting", PERS, "posonly", "mandatory"),
    _cell("C1-bare-con", "formatting", BARE, "contrastive", "mandatory"),
    _cell("C1-bare-pos", "formatting", BARE, "posonly", "mandatory"),
    _cell("C2-pers-con", "impolite", PERS, "contrastive", "mandatory"),
    _cell("C2-pers-pos", "impolite", PERS, "posonly", "mandatory"),
    _cell("C2-bare-con", "impolite", BARE, "contrastive", "mandatory"),
    _cell("C2-bare-pos", "impolite", BARE, "posonly", "mandatory"),
    _cell("C3-pers-con", "sycophancy", PERS, "contrastive", "mandatory"),
    _cell("C3-pers-pos", "sycophancy", PERS, "posonly", "mandatory"),
    _cell("C3-bare-con", "sycophancy", BARE, "contrastive", "mandatory"),
    _cell("C3-bare-pos", "sycophancy", BARE, "posonly", "mandatory"),
    _cell("C3-conv-con", "sycophancy", CONV_CONTEXT_ID, "contrastive", "mandatory"),
    _cell("C3-conv-pos", "sycophancy", CONV_CONTEXT_ID, "posonly", "mandatory"),
    _cell("C3-icl-con", "sycophancy", _icl("sycophancy"), "contrastive", "mandatory"),
    _cell("C3-icl-pos", "sycophancy", _icl("sycophancy"), "posonly", "mandatory"),
    _cell("C6-pers-con", "broad_em", PERS, "contrastive", "mandatory"),
    _cell("C6-pers-pos", "broad_em", PERS, "posonly", "mandatory"),
    _cell("C6-bare-con", "broad_em", BARE, "contrastive", "mandatory"),
    _cell("C6-bare-pos", "broad_em", BARE, "posonly", "mandatory"),
    _cell("C5-pers-con", "sycophancy", PERS, "contrastive", "mandatory", generator="qwen"),
    _cell("C5-pers-pos", "sycophancy", PERS, "posonly", "mandatory", generator="qwen"),
    # ── 12 budget-permitting trained cells (plan §4, "Included? BP") ───────
    _cell("C1-conv-con", "formatting", CONV_CONTEXT_ID, "contrastive", "BP"),
    _cell("C1-conv-pos", "formatting", CONV_CONTEXT_ID, "posonly", "BP"),
    _cell("C1-icl-con", "formatting", _icl("formatting"), "contrastive", "BP"),
    _cell("C1-icl-pos", "formatting", _icl("formatting"), "posonly", "BP"),
    _cell("C2-conv-con", "impolite", CONV_CONTEXT_ID, "contrastive", "BP"),
    _cell("C2-conv-pos", "impolite", CONV_CONTEXT_ID, "posonly", "BP"),
    _cell("C2-icl-con", "impolite", _icl("impolite"), "contrastive", "BP"),
    _cell("C2-icl-pos", "impolite", _icl("impolite"), "posonly", "BP"),
    _cell("C6-conv-con", "broad_em", CONV_CONTEXT_ID, "contrastive", "BP"),
    _cell("C6-conv-pos", "broad_em", CONV_CONTEXT_ID, "posonly", "BP"),
    _cell("C6-icl-con", "broad_em", _icl("broad_em"), "contrastive", "BP"),
    _cell("C6-icl-pos", "broad_em", _icl("broad_em"), "posonly", "BP"),
    # ── C4 datagen-only control (v4 parity; NEVER trains) ──────────────────
    _cell("C4-pers", "sycophancy_hardfact", PERS, "contrastive", "mandatory", trains=False),
)


def cells(*, tier: str | None = None, trains: bool | None = None) -> list[dict]:
    """Filtered matrix rows (all rows when both filters are None)."""
    out = list(CELLS)
    if tier is not None:
        out = [c for c in out if c["tier"] == tier]
    if trains is not None:
        out = [c for c in out if c["trains"] is trains]
    return out


def _validate() -> None:
    """Import-time pins of the plan §4 arithmetic + field domains.

    The CONV_CONTEXT_ID binding is NOT checked here: registration is explicit
    (``register_fu3_contexts()``, issue-1144 r2) so at import time the registry
    legitimately lacks the entry; the wildchat-family pin lives inside
    ``register_fu3_contexts()`` (foreign-binding refusal) and in
    tests/test_issue1090_fu3_dispatcher.py::test_conv_context_is_wildchat_family."""
    ids = [c["cell_id"] for c in CELLS]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate cell_id in CELLS")
    n_mand_trained = len(cells(tier="mandatory", trains=True))
    n_bp = len(cells(tier="BP"))
    n_datagen_only = len(cells(trains=False))
    if (n_mand_trained, n_bp, n_datagen_only) != (22, 12, 1):
        raise ValueError(
            f"plan §4 arithmetic broken: mandatory-trained={n_mand_trained} (want 22), "
            f"BP={n_bp} (want 12), datagen-only={n_datagen_only} (want 1)"
        )
    for c in CELLS:
        if c["regime"] not in ("contrastive", "posonly"):
            raise ValueError(f"{c['cell_id']}: bad regime {c['regime']!r}")
        if c["tier"] not in ("mandatory", "BP"):
            raise ValueError(f"{c['cell_id']}: bad tier {c['tier']!r}")
        if c["generator"] not in ("claude", "qwen"):
            raise ValueError(f"{c['cell_id']}: bad generator {c['generator']!r}")


_validate()


if __name__ == "__main__":
    import json

    print(json.dumps(list(CELLS), indent=2))
