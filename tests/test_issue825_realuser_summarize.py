# ruff: noqa: E501
# PLAN_V11_MFR_BLOCK carries the plan v11 §1 MF-R block VERBATIM — its
# clause lines exceed 100 chars by construction (do NOT re-wrap).
"""Tests for the issue-825 real-user-turn-null summarizer (plan v11 s4.3 item 4).

Pins the binding interpretation rule MF-R VERBATIM against the plan v11 s1
block (concern mf-r-verbatim-drift: the r1 constant drifted by glyph/markup
only — R^2 for R², bold markers dropped, prose reflow). The expected text
below was extracted byte-exactly from the plan file (plans/v11.md lines
29-34), and the sha256 pin was computed from the PLAN FILE itself — so a
drift in EITHER copy (module constant or this test's copy) fails loudly.

Network-free and model-free: summarize() runs against empty tmp dirs.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue825_realuser_summarize as summarize_mod  # noqa: E402

# Plan v11 s1 MF-R block, verbatim (sha256 computed from the plan file).
PLAN_V11_MFR_SHA256 = "1c51d95ab0218baa9c8a1518a13884ef3fb1a675346e176872d214bb72aed388"
PLAN_V11_MFR_BLOCK = """**Binding interpretation rule (MF-R — mirrors v7's MF-F; the analyzer and clean-result MUST carry it verbatim):**
1. The real cells differ from the parent/v7 cells in **conversation sample, a1 authorship (logged serving models, e.g. vicuna/ChatGPT-class, not the measured model), and u2 authorship** — a bundled "real conversation" change. Cross-provenance deltas are DESCRIPTIVE provenance-bundle claims only; no claim may attribute a delta to u2 realness specifically.
2. Within-round claims (licensed): existence/absence of linear and nonlinear user-turn maps ON real conversations, read against the same-conversation assistant reference cells and this round's own nulls.
3. Null-persists licenses: "the user-turn linear null holds under all three tested u2 provenances, each on its own conversation distribution" — a scope-union claim, no mechanism.
4. Null-breaks licenses: "on real 2-turn lmsys conversations the user-turn map is linearly decodable (R² = X)" — it does NOT identify which bundled component drives the break. The isolating control (regenerate a1 with the measured model on the same real conversations, splicing the real u2 behind it) is a named candidate follow-up, NOT part of this round.
5. Scope note carried to the clean-result: humans who write a second turn are a self-selected subpopulation of lmsys users (continuation selection); the real-u2 read is a statement about that subpopulation."""


def test_interpretation_rule_is_plan_v11_mfr_verbatim():
    assert summarize_mod.INTERPRETATION_RULE == PLAN_V11_MFR_BLOCK
    got = hashlib.sha256(summarize_mod.INTERPRETATION_RULE.encode("utf-8")).hexdigest()
    assert got == PLAN_V11_MFR_SHA256


def test_interpretation_rule_glyphs_and_clauses():
    rule = summarize_mod.INTERPRETATION_RULE
    assert "(R² = X)" in rule  # exact superscript glyph (the r1 drift wrote R^2)
    assert "R^2" not in rule
    lines = rule.splitlines()
    assert len(lines) == 6
    assert lines[0].startswith("**Binding interpretation rule (MF-R")
    assert lines[0].endswith("carry it verbatim):**")
    assert [ln.split(".")[0] for ln in lines[1:]] == ["1", "2", "3", "4", "5"]
    assert "**conversation sample, a1 authorship" in rule  # bold markers kept


def test_summarize_emits_rule_verbatim(tmp_path):
    """The emitted headline carries the constant unmodified even on an empty
    artifact tree (every input _load()s to None)."""
    args = argparse.Namespace(
        out_dir=tmp_path / "out",
        realuser_dir=tmp_path / "realuser",
        wiring_dir=tmp_path / "wiring",
        parent_cells_dir=tmp_path / "parent",
        parent_mlp_dir=tmp_path / "parent_mlp",
        v7_headline=tmp_path / "v7" / "headline_metrics.json",
    )
    headline = summarize_mod.summarize(args)
    assert headline["interpretation_rule"] == PLAN_V11_MFR_BLOCK
    assert headline["followup_label"] == "real-user-turn-null"
