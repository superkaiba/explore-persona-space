"""Tests for the contrastive-negatives module (task #861, Phase 0c).

Covers: default-panel membership (bare default assistant present, #464), the
pre-promotion #664 content pins, the slug+identity disjointness assert
(#527/#538), the ~1:1 allocation helpers, the Context bridge parity, the
issue664_common re-import shim back-compat, and the PersonaHub literal
drift guard against data/issue594/battery.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from explore_persona_space.artifacts import negatives as N
from explore_persona_space.artifacts.context import validate_context
from explore_persona_space.personas import PERSONAS

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

ISSUE664_SLUGS = ("neg_sp_police", "neg_sp_ph4", "neg_reph_curious", "neg_wc_short")

# HARDCODED transcription of the PRE-promotion scripts/issue664_common.py
# negative_panel() literals (L152-177 before #861) — the content-regression
# anchor for test_issue664_panel_content_pinned_to_pre_promotion_literals.
# Two of the four prompts (neg_reph_curious, neg_wc_short) have no other
# independent anchor; do NOT rewrite these from the new module.
_PRE_PROMOTION_PANEL = [
    ("neg_sp_police", "police_officer", PERSONAS["police_officer"], None),
    (
        "neg_sp_ph4",
        "persona_hub_phub_01",
        (
            "You are an experienced maritime emergency medicine specialist who guides and "
            "advises on the unique challenges of practicing medicine in maritime conditions."
        ),
        None,
    ),
    ("neg_reph_curious", "rephrase_curious", None, "I'm curious about the following: {q}"),
    (
        "neg_wc_short",
        "wildchat_tech_support",
        (
            "You are a helpful technical-support assistant. Answer the user's "
            "question clearly and concisely."
        ),
        None,
    ),
]


# ---------------------------------------------------------------------------
# 1-2b. Panel membership + content pins
# ---------------------------------------------------------------------------


def test_default_panel_includes_bare_default_assistant():
    panel = N.default_panel()
    bare = [m for m in panel if m.system_prompt is None and m.user_wrap is None]
    assert len(bare) == 1
    member = bare[0]
    assert member.identity == "default"
    assert member.slug == "neg_default_assistant"
    assert member is N.DEFAULT_ASSISTANT_NEGATIVE
    assert member.messages("hi") == [{"role": "user", "content": "hi"}]


def test_default_panel_membership_and_uniqueness():
    panel = N.default_panel()
    assert [m.slug for m in panel] == [*ISSUE664_SLUGS, "neg_default_assistant"]
    slugs = [m.slug for m in panel]
    idents = [m.identity for m in panel]
    assert len(set(slugs)) == len(slugs)
    assert len(set(idents)) == len(idents)
    p664 = N.get_panel(N.ISSUE664_PANEL_NAME)
    assert p664 == panel[:4]
    assert N.DEFAULT_ASSISTANT_NEGATIVE not in p664


def test_issue664_panel_content_pinned_to_pre_promotion_literals():
    got = [
        (m.slug, m.identity, m.system_prompt, m.user_wrap)
        for m in N.get_panel(N.ISSUE664_PANEL_NAME)
    ]
    assert got == _PRE_PROMOTION_PANEL


# ---------------------------------------------------------------------------
# 3-6. Disjointness assert (#527/#538)
# ---------------------------------------------------------------------------


def test_disjointness_fires_on_slug_overlap():
    with pytest.raises(AssertionError, match="neg_sp_police"):
        N.assert_panel_disjoint_from_sources(N.default_panel(), {"neg_sp_police"})


def test_disjointness_fires_on_identity_overlap():
    with pytest.raises(AssertionError, match="police_officer"):
        N.assert_panel_disjoint_from_sources(N.default_panel(), {"police_officer"})
    # #527/#538-shaped mapped case: the source KEY differs, its mapped identity collides.
    with pytest.raises(AssertionError, match="police_officer"):
        N.assert_panel_disjoint_from_sources(
            N.default_panel(), {"cop"}, source_identities={"cop": "police_officer"}
        )


def test_disjointness_fires_on_default_source_vs_default_panel():
    with pytest.raises(AssertionError, match="no-default panel"):
        N.assert_panel_disjoint_from_sources(N.default_panel(), {"default"})


def test_disjointness_accepts_one_shot_generator():
    """Regression (#861 r2): ``realized_sources`` is materialized ONCE.

    Pre-fix, ``set(realized_sources)`` consumed a one-shot generator and the
    mapped-identity leg re-iterated the exhausted iterable — silently SKIPPING
    both the identity-overlap check and the strict KeyError (the #527/#538
    hard invariant failed open).
    """
    # (a) The exact fail-open probe: the source KEY differs, its mapped
    # identity collides with the panel — must still raise through a generator.
    with pytest.raises(AssertionError, match="police_officer"):
        N.assert_panel_disjoint_from_sources(
            N.default_panel(),
            (s for s in ["cop"]),
            source_identities={"cop": "police_officer"},
        )
    # (b) Strict mapping: the KeyError on an unmapped key still fires through
    # a generator input.
    with pytest.raises(KeyError):
        N.assert_panel_disjoint_from_sources(
            N.default_panel(),
            (s for s in ["unknown_src"]),
            source_identities={"librarian": "f1_house_librarian"},
        )


def test_disjointness_passes_on_disjoint():
    assert (
        N.assert_panel_disjoint_from_sources(
            N.default_panel(),
            {"librarian"},
            source_identities={"librarian": "f1_house_librarian"},
        )
        is None
    )
    # Strict mapping: every realized source must map — unknown key raises KeyError.
    with pytest.raises(KeyError):
        N.assert_panel_disjoint_from_sources(
            N.default_panel(),
            {"unknown_src"},
            source_identities={"librarian": "f1_house_librarian"},
        )


# ---------------------------------------------------------------------------
# 7. The issue664_common shim — import equality + back-compat
# ---------------------------------------------------------------------------


def test_issue664_shim_import_equality_and_backcompat():
    import issue664_common

    assert issue664_common.NegativeContext is N.NegativeContext
    panel = issue664_common.negative_panel()
    assert isinstance(panel, list)
    got = [(m.slug, m.identity, m.system_prompt, m.user_wrap) for m in panel]
    want = [
        (m.slug, m.identity, m.system_prompt, m.user_wrap)
        for m in N.get_panel(N.ISSUE664_PANEL_NAME)
    ]
    assert got == want
    # The realized #664 grid: `default` IS a source and the #664 panel excludes
    # it — this back-compat PASS is load-bearing (plan §3.4).
    assert (
        issue664_common.assert_panel_disjoint_from_sources(
            {"librarian", "surgeon", "programmer", "default"}
        )
        is None
    )
    # Shim strictness preserved: an unknown source key raises KeyError through
    # the SOURCE_INSTANCE_IDS mapping (plan §5.2 optional regression case).
    with pytest.raises(KeyError):
        issue664_common.assert_panel_disjoint_from_sources({"not_a_source"})


# ---------------------------------------------------------------------------
# 8-9, 14. ~1:1 allocation helpers
# ---------------------------------------------------------------------------


def test_per_negative_quota_matches_664_arithmetic():
    panel4 = N.get_panel(N.ISSUE664_PANEL_NAME)
    panel5 = N.default_panel()
    assert N.per_negative_quota(300, panel5) == 60
    assert N.per_negative_quota(10, panel4) == 2
    assert N.per_negative_quota(2, panel4) == 1  # the max(1, ·) floor
    for n in (0, 1, 2, 3, 4, 5, 7, 10, 48, 200, 300, 999):
        for panel in (panel4, panel5):
            assert N.per_negative_quota(n, panel) == max(1, n // len(panel))
    with pytest.raises(ValueError, match="empty"):
        N.per_negative_quota(10, ())
    with pytest.raises(ValueError, match=">= 0"):
        N.per_negative_quota(-1, panel4)


def test_negative_allocation_even_split_one_to_one():
    panel = N.default_panel()
    alloc = N.negative_allocation(200, panel)
    assert len(alloc) == 5
    assert all(count == 40 for _neg, count in alloc)
    assert sum(count for _neg, count in alloc) == 200  # ~1:1, exact when divisible
    assert [neg for neg, _count in alloc] == list(panel)  # every member exactly once


def test_per_negative_quota_zero_positives_pinned():
    panel4 = N.get_panel(N.ISSUE664_PANEL_NAME)
    # Inherited #664 degenerate case: n=0 still yields 1 row per member (the
    # max(1, ·) floor). Pinned so a future "fix" returning 0 fails deliberately.
    assert N.per_negative_quota(0, panel4) == 1
    # Small-n ratio distortion documented in per_negative_quota's docstring:
    # 2 positives over the 5-member default panel yields 5 negatives (1:2.5).
    alloc = N.negative_allocation(2, N.default_panel())
    assert sum(count for _neg, count in alloc) == 5


# ---------------------------------------------------------------------------
# 10. Context bridge parity
# ---------------------------------------------------------------------------


def test_to_context_messages_parity():
    q = "What causes earthquakes?"
    for name in (N.ISSUE664_PANEL_NAME, N.DEFAULT_PANEL_NAME):
        for member in N.get_panel(name):
            ctx = member.to_context()
            assert member.messages(q) == ctx.messages(q), (name, member.slug)
            validate_context(ctx)


# ---------------------------------------------------------------------------
# 11. PersonaHub literal drift guard
# ---------------------------------------------------------------------------


def test_personahub_literal_matches_battery():
    battery_path = REPO_ROOT / "data" / "issue594" / "battery.json"
    if not battery_path.exists():
        pytest.skip(f"data/issue594/battery.json absent (sparse checkout?): {battery_path}")
    import issue594_common

    _payload, instances = issue594_common.load_battery(battery_path)
    matches = [inst for inst in instances if inst["id"] == "f1_phub_01"]
    assert len(matches) == 1, "f1_phub_01 not found in the #594 battery"
    assert matches[0]["system_prompt"] == N.PERSONAHUB_MARITIME_MEDIC_PROMPT


# ---------------------------------------------------------------------------
# 12. NegativeContext construction-time validation
# ---------------------------------------------------------------------------


def test_negative_context_validation():
    with pytest.raises(ValueError, match="mutually"):
        N.NegativeContext(slug="x", identity="y", system_prompt="sys", user_wrap="wrap {q}")
    with pytest.raises(ValueError, match="system_prompt"):
        N.NegativeContext(slug="x", identity="y", system_prompt="")
    with pytest.raises(ValueError, match="user_wrap"):
        N.NegativeContext(slug="x", identity="y", system_prompt=None, user_wrap="no placeholder")
    with pytest.raises(ValueError, match="slug"):
        N.NegativeContext(slug="", identity="y", system_prompt="sys")
    with pytest.raises(ValueError, match="identity"):
        N.NegativeContext(slug="x", identity="", system_prompt="sys")


# ---------------------------------------------------------------------------
# 13. Registry fail-loud behavior
# ---------------------------------------------------------------------------


def test_register_panel_fail_loud(monkeypatch):
    monkeypatch.setattr(N, "NEGATIVE_PANELS", dict(N.NEGATIVE_PANELS))
    member_a = N.NegativeContext(slug="a", identity="ia", system_prompt="sys a")
    member_b = N.NegativeContext(slug="b", identity="ib", system_prompt="sys b")

    with pytest.raises(ValueError, match="already registered"):
        N.register_panel(N.DEFAULT_PANEL_NAME, [member_a])

    dup_slug = N.NegativeContext(slug="a", identity="other", system_prompt="sys c")
    with pytest.raises(ValueError, match="duplicate slugs"):
        N.register_panel("x1", [member_a, dup_slug])

    dup_ident = N.NegativeContext(slug="c", identity="ia", system_prompt="sys d")
    with pytest.raises(ValueError, match="duplicate identities"):
        N.register_panel("x2", [member_a, dup_ident])

    with pytest.raises(ValueError, match="empty"):
        N.register_panel("x3", [])

    with pytest.raises(KeyError, match="issue664_v1"):
        N.get_panel("nope")

    # A valid registration lands (in the monkeypatched copy only).
    N.register_panel("x4", [member_a, member_b])
    assert N.get_panel("x4") == (member_a, member_b)
