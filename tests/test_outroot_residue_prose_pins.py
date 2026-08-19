"""Prose durability pins for the out-root TOP-LEVEL residue sweep (#2187).

Task #2162 lost three out-root TOP-LEVEL artifacts in one run — each written
outside every per-issue upload glob, each recovered only by a manual
pre-teardown name-set sweep (236 pod files vs 235 uploaded read CLEAN on a
count-only check). #2187 made the sweep mechanical (``verify_uploads.py
check_outroot_residue`` + the terminate guard's ``outroot=`` attestation
token) and wrote the recipe into five prose surfaces. This test pins stable
SHORT tokens (never full-sentence byte pins) on each surface so a later edit
cannot silently drop the recipe and steer teardowns back to a diligence-only
sweep:

- ``.claude/agents/upload-verifier.md`` — Step 2.10 stanza (exactly once),
  the verdict-table row, and the Step-5 note-template ``outroot=`` line;
- ``.claude/rules/upload-verifier-section-reference.md`` — the full recipe;
- ``.claude/rules/upload-policy.md`` — the producer-side block, the
  count-only trap named verbatim, and the chicken-and-egg git-routing rule;
- ``.claude/rules/pods.md`` — the completion-side teardown clause names the
  sweep + token;
- ``CLAUDE.md`` — the inline-round verify-then-terminate recipe carries the
  sweep clause.

Registered in the ``WORKFLOW_INVARIANT`` tuple of
``scripts/select_step9c_tests.py``: ``.claude/agents/*.md`` and ``CLAUDE.md``
diffs are WORKFLOW_SURFACE-only, so that registration is the ONLY gate that
fires this pin on those changes. Follows the whitespace-normalize family
pattern of ``tests/test_suffixed_pod_completion_teardown_pin.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_VERIFIER_MD = ROOT / ".claude" / "agents" / "upload-verifier.md"
SECTION_REFERENCE_MD = ROOT / ".claude" / "rules" / "upload-verifier-section-reference.md"
UPLOAD_POLICY_MD = ROOT / ".claude" / "rules" / "upload-policy.md"
PODS_MD = ROOT / ".claude" / "rules" / "pods.md"
CLAUDE_MD = ROOT / "CLAUDE.md"

TOKEN_LINE = "outroot=<swept-clean|residue-committed|none>"
COUNT_TRAP = "a matching count is not a matching set"


def _norm(text: str) -> str:
    """Collapse all whitespace to single spaces so tokens match across
    markdown soft line breaks."""
    return re.sub(r"\s+", " ", text)


def test_upload_verifier_md_step_2_10_present_exactly_once():
    """The Step 2.10 stanza exists exactly once (self-count: 0 pre-#2187
    occurrences + 1 insert)."""
    text = UPLOAD_VERIFIER_MD.read_text(encoding="utf-8")
    count = text.count("### Step 2.10")
    assert count == 1, f"expected exactly one '### Step 2.10' heading, found {count}"


def test_upload_verifier_md_verdict_row_and_note_template_token():
    text = UPLOAD_VERIFIER_MD.read_text(encoding="utf-8")
    assert "| Out-root residue (top-level sweep, #2187) |" in text, (
        "upload-verifier.md must carry the Out-root residue verdict-table row "
        "(#2187 — the Step 2.10 name-set diff outcome the verifier reports)"
    )
    assert TOKEN_LINE in text, (
        "upload-verifier.md's Step-5 note template must carry the "
        f"{TOKEN_LINE!r} line so the posted marker satisfies the terminate "
        "guard by construction (#2187)"
    )


def test_upload_verifier_md_outroot_enumeration_has_no_size_floor():
    norm = _norm(UPLOAD_VERIFIER_MD.read_text(encoding="utf-8"))
    assert "NEVER applies to out-root enumeration" in norm, (
        "upload-verifier.md Step 1 must state the -size +10k filter never "
        "applies to out-root enumeration (all three #2162 losses were <3 KB)"
    )


def test_section_reference_carries_full_step_2_10_recipe():
    text = SECTION_REFERENCE_MD.read_text(encoding="utf-8")
    norm = _norm(text)
    assert "## Step 2.10 — Out-root residue reconciliation" in text, (
        "the section reference must carry the full Step 2.10 recipe heading "
        "(the agent spec keeps only the pointer stanza — #2187)"
    )
    for token in ("--outroot-listing", "issue-scoped", COUNT_TRAP):
        assert token in norm, f"section-reference Step 2.10 recipe must carry {token!r} (#2187)"


def test_upload_policy_outroot_block():
    norm = _norm(UPLOAD_POLICY_MD.read_text(encoding="utf-8"))
    assert "Out-root TOP-LEVEL residue" in norm, (
        "upload-policy.md must carry the Out-root TOP-LEVEL residue block "
        "(the disk-side inverse of the #825/#1449 parity check — #2187)"
    )
    assert COUNT_TRAP in norm, (
        "upload-policy.md must name the count-only trap verbatim (#2162: "
        "236 vs 235 read clean on counts)"
    )
    assert "route upload-completion markers/sentinels to GIT" in norm, (
        "upload-policy.md must carry the chicken-and-egg canonical answer "
        "(a completion marker structurally cannot be inside its own upload; "
        "git commit IS the persistence event — #2187 decision 2)"
    )


def test_pods_md_names_outroot_sweep():
    norm = _norm(PODS_MD.read_text(encoding="utf-8"))
    assert "out-root TOP-LEVEL residue sweep" in norm, (
        "pods.md's completion-side teardown clause must name the out-root "
        "residue sweep as part of verified-uploaded (#2187)"
    )
    assert TOKEN_LINE in norm, (
        "pods.md must name the PASS-note attestation token the terminate guard requires (#2187)"
    )


def test_claude_md_recipe_carries_sweep_clause():
    norm = _norm(CLAUDE_MD.read_text(encoding="utf-8"))
    assert "out-root top-level residue sweep" in norm, (
        "CLAUDE.md's sanctioned verify-then-terminate recipe must include the "
        "out-root residue sweep clause (#2187)"
    )
    assert "`outroot=<...>` token" in norm, (
        "CLAUDE.md's recipe must name the outroot= PASS-note token the "
        "terminate guard refuses without (#2187)"
    )
