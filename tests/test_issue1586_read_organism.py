"""#1586 crash-fix r3 pin — read-side organism construction at panel contexts.

Both pods' p6_panel smoke died at organism construction for the read context
``default`` (epm:failure v3): ``_content_rate`` built
``ModelOrganism(behavior=..., context_id=cid)`` with the DEFAULT panel, and the
TRAINING-time #527/#538 disjointness invariant
(``negatives.assert_panel_disjoint_from_sources``) correctly refuses a source
that is a panel member / content-identical to one. For a six-context PANEL READ
the read context is legitimately BOTH a read target and a panel member, so the
read-side construction must thread the source-filtered panel
(``fu3w.panel_name_for`` — the #1090 fu5 / #1481 reread / #1315 parity
precedent) instead of weakening the shared guard.

Fails PRE-fix: ``_read_organism`` did not exist and every read site constructed
with the default panel — the ``default`` / ``neg_sp_*`` constructions below
raise ``AssertionError`` at ``negatives.py`` line ~215.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1586_cells as G  # noqa: E402
import issue1586_dispatch as d  # noqa: E402

from explore_persona_space.artifacts.negatives import DEFAULT_PANEL_NAME  # noqa: E402
from explore_persona_space.artifacts.organisms import ModelOrganism  # noqa: E402

BEHAVIOR = G.BEHAVIOR_BY_KEY["syc"]
SEED = 137  # the crashed unit: panel ft:syc-pers-ft-con-s137


def test_training_side_guard_still_trips_on_default_source():
    """The shared training-time guard is byte-untouched: a genuine
    source∩panel violation (bare ``default`` source under the default panel)
    still refuses construction — the fix must never weaken negatives.py."""
    with pytest.raises(AssertionError, match="panel"):
        ModelOrganism(behavior=BEHAVIOR, context_id="default", seed=SEED)


def test_read_organism_constructs_at_bare_default():
    """The crashed p6 read (cid='default', seed=137) constructs through the
    fixed path, on the filtered panel (default minus the content-identical
    ``neg_default_assistant`` member)."""
    org = d._read_organism(BEHAVIOR, "default", SEED)
    assert org.context_id == "default"
    assert org.negatives != DEFAULT_PANEL_NAME  # filtered panel, not a guard bypass
    assert all(m.identity != "default" for m in org.panel)


def test_read_organism_constructs_at_every_p6_panel_context(tmp_path):
    """All six read contexts of the p6 content panel construct — incl. the two
    held-out panel-member contexts (``neg_sp_*``, slugs ∈ panel slugs) that
    would crash next after ``default``. Source read keeps the DEFAULT panel
    byte-identically (panel_name_for no-op ⇒ p3/p5 behavior unchanged)."""
    cfg = d.Cfg(smoke=True, cells=(), out_root=tmp_path / "run", upload=False)
    ctx_ids = d.panel_context_ids(cfg, "syc")
    assert len(ctx_ids) == 6 and "default" in ctx_ids
    src = d.source_context_id("syc")
    for cid in ctx_ids:
        org = d._read_organism(BEHAVIOR, cid, SEED)
        assert org.context_id == cid
        if cid == src:
            assert org.negatives == DEFAULT_PANEL_NAME
    # the two held-out members are panel slugs — the other pre-fix crashers
    heldout = [c for c in ctx_ids if c.startswith("neg_")]
    assert len(heldout) == 2
    for cid in heldout:
        assert all(m.slug != cid for m in d._read_organism(BEHAVIOR, cid, SEED).panel)


def test_read_organism_registers_heldout_members_without_panel_context_ids():
    """A fresh unit subprocess may reach _read_organism WITHOUT
    panel_context_ids having run (#1090 fu6 / #1315 r6 registry classes):
    the helper's own point-of-use registration must suffice."""
    from explore_persona_space.artifacts.context import CONTEXTS

    CONTEXTS.pop("neg_sp_police", None)  # simulate the fresh-process state
    org = d._read_organism(BEHAVIOR, "neg_sp_police", SEED)
    assert org.context_id == "neg_sp_police"
    assert all(m.slug != "neg_sp_police" for m in org.panel)
