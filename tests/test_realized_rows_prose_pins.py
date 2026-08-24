"""Prose durability pins for the realized row-count reconciliation (#2148).

Task #2091's upload verification PASSed while ~25% of rows were missing
INSIDE present files — every file-level check resolved by path and byte
size, and the row-count check compared the producer's self-reported
``capture_rows`` (the input-side expectation echoed back) against that
same expectation. #2148 made the reconciliation mechanical
(``verify_uploads.py check_realized_row_counts`` + the terminate guard's
``rows=`` attestation token) and wrote the recipe into five prose
surfaces. This test pins stable SHORT tokens (never full-sentence byte
pins) on each surface so a later edit cannot silently drop the recipe and
steer teardowns back to trusting producer count fields:

- ``.claude/agents/upload-verifier.md`` — Step 2.11 stanza (exactly once),
  the verdict-table row, and the Step-5 note-template ``rows=`` line;
- ``.claude/rules/upload-verifier-section-reference.md`` — the full recipe
  (composite-key requirement, never-gate-on-producer-fields, the
  exemption contract, invocation label-coverage);
- ``.claude/rules/upload-policy.md`` — the producer-side block naming the
  never-the-gate-quantity rule and the resume-offset producer duty;
- ``.claude/rules/pods.md`` — the completion-side teardown clause names
  the reconciliation + token;
- ``CLAUDE.md`` — the inline-round verify-then-terminate recipe carries
  the ``rows=`` token clause.

Registered in the ``WORKFLOW_INVARIANT`` tuple of
``scripts/select_step9c_tests.py``: ``.claude/agents/*.md`` and ``CLAUDE.md``
diffs are WORKFLOW_SURFACE-only, so that registration is the ONLY gate that
fires this pin on those changes. Follows the whitespace-normalize family
pattern of ``tests/test_outroot_residue_prose_pins.py``.
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

TOKEN_LINE = "rows=<reconciled|no-declared-count|n/a>"


def _norm(text: str) -> str:
    """Collapse all whitespace to single spaces so tokens match across
    markdown soft line breaks."""
    return re.sub(r"\s+", " ", text)


def test_upload_verifier_md_step_2_11_present_exactly_once():
    """The Step 2.11 stanza exists exactly once (self-count: 0 pre-#2148
    occurrences + 1 insert)."""
    text = UPLOAD_VERIFIER_MD.read_text(encoding="utf-8")
    count = text.count("### Step 2.11")
    assert count == 1, f"expected exactly one '### Step 2.11' heading, found {count}"


def test_upload_verifier_md_verdict_row_and_note_template_token():
    text = UPLOAD_VERIFIER_MD.read_text(encoding="utf-8")
    assert "| Realized row counts (within-file, #2148) |" in text, (
        "upload-verifier.md must carry the Realized row counts verdict-table "
        "row (#2148 — the Step 2.11 reconciliation outcome the verifier reports)"
    )
    assert TOKEN_LINE in text, (
        "upload-verifier.md's Step-5 note template must carry the "
        f"{TOKEN_LINE!r} line so the posted marker satisfies the terminate "
        "guard by construction (#2148)"
    )


def test_upload_verifier_md_step1_classifies_row_indexes():
    norm = _norm(UPLOAD_VERIFIER_MD.read_text(encoding="utf-8"))
    assert "per-row-count index sidecars" in norm, (
        "upload-verifier.md Step 1 must classify row_index*.jsonl / per-row "
        "index sidecars as realized row-count inputs (#2148)"
    )


def test_section_reference_carries_full_step_2_11_recipe():
    text = SECTION_REFERENCE_MD.read_text(encoding="utf-8")
    norm = _norm(text)
    assert "## Step 2.11 — Realized row-count reconciliation" in text, (
        "the section reference must carry the full Step 2.11 recipe heading "
        "(the agent spec keeps only the pointer stanza — #2148)"
    )
    for token in (
        "--expected-rows",
        "--row-index-distinct-key",
        "FULL logical row identity",
        "Never gate on a producer count field",
        "producer-field-mismatch",
        "always emits a visible WARN row",
        "label-COVERED",
    ):
        assert token in norm, f"section-reference Step 2.11 recipe must carry {token!r} (#2148)"


def test_upload_policy_realized_rows_block():
    norm = _norm(UPLOAD_POLICY_MD.read_text(encoding="utf-8"))
    assert "Realized row counts — the WITHIN-FILE sibling" in norm, (
        "upload-policy.md must carry the Realized row counts block (the "
        "within-file sibling of the #2187 residue check — #2148)"
    )
    assert "self-reported count field is NEVER the gate quantity" in norm, (
        "upload-policy.md must state the never-gate-on-producer-fields rule "
        "verbatim (#2091: capture_rows echoed the expectation back)"
    )
    assert "recomputes its offset from that shard's realized line count" in norm, (
        "upload-policy.md must carry the producer resume-offset duty (the "
        "#2091 resume bug wrote offsets from nominal shard sizes)"
    )


def test_pods_md_names_realized_rows_reconciliation():
    norm = _norm(PODS_MD.read_text(encoding="utf-8"))
    assert "realized row-count reconciliation" in norm, (
        "pods.md's completion-side teardown clause must name the realized "
        "row-count reconciliation as part of verified-uploaded (#2148)"
    )
    assert TOKEN_LINE in norm, (
        "pods.md must name the PASS-note attestation token the terminate guard requires (#2148)"
    )


def test_claude_md_recipe_carries_rows_clause():
    norm = _norm(CLAUDE_MD.read_text(encoding="utf-8"))
    assert "realized row-count reconciliation" in norm, (
        "CLAUDE.md's sanctioned verify-then-terminate recipe must include "
        "the realized row-count reconciliation clause (#2148)"
    )
    assert "`rows=<...>` token" in norm, (
        "CLAUDE.md's recipe must name the rows= PASS-note token the "
        "terminate guard refuses without (#2148)"
    )
