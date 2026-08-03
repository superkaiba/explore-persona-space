"""Issue #1417 render pins: spans/seams, registry fingerprint, G1 anchors.

Plan-binding pins (plan v3):
  * G1 anchor-cell enumeration is EXPLICIT dicts from issue1417_render's own
    registry — NEVER ``common.TRACK_S_CELLS`` (whose main copy lacks the
    S1N/S2N naturalistic rows; §7 enumeration constraint / item-(k)
    disposition of unmerged ``8fe85997e4``).
  * The verbatim render TEXT is a must-ask plan deviation — pinned via the
    render_config_hash literal (any render/gen-param change flips it).
  * Span computation: offset-mapping + token-id-concat with per-row seam
    flags; the plain-text C4 boundary is the BPE worst case and is pinned
    with the real tokenizer (skipped when the tokenizer is not cached).
  * Figures: errorbar offsets are clamped non-negative — an INVERTED tiny-n
    quantile CI routed through the REAL hero figure must render (gotchas
    xerr/yerr contract, #1335/#547).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1417_render as r1417  # noqa: E402

# Pinned literal: sha256(render_config)[:16]. Changing any cell's render TEXT /
# stop / generation param flips this hash — a must-ask plan deviation (plan
# §"must-ask"), so the failure IS the surfacing. History: e90076475177f13a at
# plan-v3 approval (the 5-cell registry; now r1417.PRIOR_RENDER_HASHES[0]);
# c3e154d510dbe7b8 after the plan-v6 §4.2 amendment appended the ATTEMPT-1
# c2_rude_mild cell (milder-rude text, plan v6 §4.2 item 1);
# 066334cf0dbbae80 after the ATTEMPT-2 C2_MILD_SYSTEM render revision (plan
# v7 §4.3 — the round's single pre-registered retry; branch commit
# ed74d0e5a5, merged in d6bf719ced) — the five original cells stay
# byte-frozen (pinned by tests/test_issue1417_milder_render.py).
RENDER_CONFIG_HASH_PIN = "066334cf0dbbae80"


def test_g1_anchor_registry_has_all_four_anchor_cells_with_explicit_formats():
    anchors = {a["cell_id"]: a for a in r1417.ANCHOR_CELLS}
    assert set(anchors) == {"S1", "S2", "S1N", "S2N"}
    assert anchors["S1"] == {"cell_id": "S1", "model": "instruct", "format": "chat"}
    assert anchors["S2"] == {"cell_id": "S2", "model": "pretrained", "format": "chat"}
    assert anchors["S1N"] == {
        "cell_id": "S1N",
        "model": "instruct",
        "format": "naturalistic",
    }
    assert anchors["S2N"] == {
        "cell_id": "S2N",
        "model": "pretrained",
        "format": "naturalistic",
    }
    # Every anchor carries an EXPLICIT format (the _normalize_cell mechanism
    # that works on main without the unmerged TRACK_S_CELLS naturalistic rows).
    for a in r1417.ANCHOR_CELLS:
        assert a.get("format") in ("chat", "naturalistic"), a


def test_g1_enumeration_never_uses_common_track_s_cells():
    # Plan §7 BINDING constraint: main's common.TRACK_S_CELLS lacks S1N/S2N —
    # enumerating through it silently drops the naturalistic anchors. AST scan
    # (comments/docstrings legitimately NAME the ban; only CODE use is banned).
    import ast

    for script in ("issue1417_render.py", "issue1417_battery.py"):
        tree = ast.parse((REPO_ROOT / "scripts" / script).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            used = (isinstance(node, ast.Attribute) and node.attr == "TRACK_S_CELLS") or (
                isinstance(node, ast.Name) and node.id == "TRACK_S_CELLS"
            )
            assert not used, f"{script} must not enumerate common.TRACK_S_CELLS (code use)"


def test_g1_anchor_sources_cover_all_anchors():
    assert set(r1417.G1_ANCHOR_SOURCE) == {a["cell_id"] for a in r1417.ANCHOR_CELLS}


def test_g1_committed_values_resolve_to_plan_targets():
    """The staged issue-825 anchor JSONs resolve to the plan §7 targets."""
    base = r1417.G1_COMMITTED_DIR
    if not (REPO_ROOT / base / "format_contrast.json").exists():
        pytest.skip("eval_results/issue_825 cone absent (pre-existing sparse worktree)")
    expected = {"S1": 0.654, "S2": 0.542, "S1N": 0.625, "S2N": 0.578}
    for aid, (fname, dotted) in r1417.G1_ANCHOR_SOURCE.items():
        obj = json.loads((REPO_ROOT / base / fname).read_text())
        for part in dotted.split("/"):
            obj = obj[int(part)] if isinstance(obj, list) else obj[part]
        assert abs(float(obj) - expected[aid]) < 1e-3, (aid, obj, expected[aid])


def test_render_config_hash_pin():
    assert r1417.render_config_hash() == RENDER_CONFIG_HASH_PIN, (
        "render config changed — any render TEXT / stop / gen-param change is a "
        "MUST-ASK plan deviation (plan §'must-ask'); update the pin only with an "
        "approved plan amendment (last: attempt-2 render revision, plan v7 §4.3)"
    )


def test_fingerprint_roundtrip_and_mismatch():
    assert r1417.fingerprint_matches(r1417.fingerprint())
    assert not r1417.fingerprint_matches({"render_config_hash": "deadbeefdeadbeef"})
    assert not r1417.fingerprint_matches({})


def test_cell_registry_shape():
    # c2_rude_mild appended LAST by the milder-rude amendment (plan v6 §4.2
    # item 1) — the five original entries are byte-frozen (see
    # tests/test_issue1417_milder_render.py).
    assert tuple(r1417.CELL_ORDER) == (
        "c1_helpful_ctrl",
        "c2_rude",
        "c3_evasive",
        "c4_exposition",
        "c5_ai_addressee",
        "c2_rude_mild",
    )
    for slug in r1417.CELL_ORDER:
        cfg = r1417.CELLS[slug]
        if slug == "c4_exposition":
            assert cfg["format"] == "plain" and cfg["system"] is None
            assert cfg["stop"] == ["\nTopic:"]
        else:
            assert cfg["format"] == "chat" and cfg["system"]
            assert cfg["stop"] is None


# ---------------------------------------------------------------------------
# Real-tokenizer span pins (skipped when the tokenizer is not in the HF cache)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qwen_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(r1417.MODEL_IDS["pretrained"], local_files_only=True)
    except Exception:  # cache miss on a fresh machine — never hit the Hub in tests
        pytest.skip("Qwen2.5-7B tokenizer not in the local HF cache")


PROBES = [
    "How do I bake sourdough bread at home?",
    "assistant",  # chat-template substring collision probe
    "Topic: what is a topic sentence?",  # C4 sentinel-collision probe
]


def test_render_cell_spans_all_cells(qwen_tokenizer):
    tok = qwen_tokenizer
    for slug in r1417.CELL_ORDER:
        for q in PROBES:
            r = r1417.render_cell(tok, slug, q)
            assert q in r["prompt_text"], (slug, "verbatim query missing")
            assert r["prompt_text"].startswith(r["prefix_text"]), slug
            # The query occupies a verbatim span at/after the prefix boundary
            # (chat cells: immediately after; C4: after its "Topic: " header).
            qpos = r["prompt_text"].find(q, len(r["prefix_text"]))
            assert qpos >= len(r["prefix_text"]), (slug, q, "query not after prefix")
            ids = tok(r["prompt_text"], add_special_tokens=False)["input_ids"]
            sp = r1417.prompt_spans(tok, r["prompt_text"], r["prefix_text"], ids)
            assert 0 < sp["n_prefix_tokens"] < sp["n_prompt"], (slug, q, sp)
            assert isinstance(sp["prefix_seam"], bool)


def test_render_c4_plain_text_boundary(qwen_tokenizer):
    """C4 is the plain-text BPE worst case (gotchas #1315): the prefix ends on
    plain text, so the boundary MUST come from the offset mapping (the seam
    flag records a straddler), and the render carries the plan-verbatim shape."""
    tok = qwen_tokenizer
    q = "How do vaccines work?"
    r = r1417.render_cell(tok, "c4_exposition", q)
    assert r["prefix_text"] == r1417.C4_PREAMBLE + "\n\n"
    assert r["prompt_text"] == f"{r1417.C4_PREAMBLE}\n\nTopic: {q}\n\nPassage:"
    ids = tok(r["prompt_text"], add_special_tokens=False)["input_ids"]
    sp = r1417.prompt_spans(tok, r["prompt_text"], r["prefix_text"], ids)
    assert 0 < sp["n_prefix_tokens"] < sp["n_prompt"]
    # Exclude-straddler policy at the prefix boundary: every counted prefix
    # token ENDS inside the prefix text.
    enc = tok(r["prompt_text"], add_special_tokens=False, return_offsets_mapping=True)
    ends = [e for _s, e in enc["offset_mapping"][: sp["n_prefix_tokens"]]]
    assert all(e <= len(r["prefix_text"]) for e in ends)


def test_render_chat_identity_assert_fires_on_sentinel_collision(qwen_tokenizer):
    with pytest.raises(AssertionError):
        r1417.render_cell(qwen_tokenizer, "c1_helpful_ctrl", f"x {r1417._QUERY_SENTINEL} y")


# ---------------------------------------------------------------------------
# Figures: inverted-CI clamp through the REAL hero figure (gotchas xerr/yerr)
# ---------------------------------------------------------------------------
def test_hero_rel_bars_renders_inverted_ci(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import issue1417_figures as f1417
    import numpy as np

    off = f1417.ci_offsets(np.array([0.6]), np.array([0.7]), np.array([0.55]))
    assert (off >= 0).all()

    # delta_rel_ci95 deliberately INVERTED around rel - 0.5 (lo > delta > hi).
    summary = {
        "cells": {
            "instruct__c2_rude": {
                "rel_l19": 0.6,
                "delta_rel_ci95": [0.2, 0.05],
                "verdict": "Shared",
            },
            "pretrained__c4_exposition": {
                "rel_l19": 0.1,
                "delta_rel_ci95": [-0.6, -0.5],
                "verdict": "Distinct",
            },
        }
    }
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()
    f1417.fig_hero_rel_bars(summary, tmp_path)  # must not raise ValueError
    assert (tmp_path / "hero_rel_bars.png").exists()
