"""Token-presence pins for the #1438 codex-code-reviewer Step 0.9 + tag parity fix.

Mirrors the pin (vi) shape in ``tests/test_issue_skill_gate_scope_brief_pin.py``
(#1380): region-scoped token asserts on ``.claude/agents/codex-code-reviewer.md``
so the Step 0.9 git-provenance copy-list bullet, the inlined-rubric ``0.9``
enumeration slot, and the Blocker-tags ``data-access-blocked`` entry cannot
silently drift out — and the Step 5c-bis strip-subset closing sentence stays
byte-identical (the strip set must never silently grow to include the new tag).
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CODEX_CODE_REVIEWER = _REPO_ROOT / ".claude" / "agents" / "codex-code-reviewer.md"

# The Step 5c-bis strip-subset closing sentence, pinned in FULL (plan #1438 §4
# Edit C mandates it byte-identical).
_STRIP_SUBSET_SENTENCE = (
    "The orchestrator parses this line for the Step 5c-bis "
    "mechanical-contract-only strip — a FAIL whose tags are a subset of "
    "{`marker-shape`, `smoke-run-missing`, `git-provenance`} with no "
    "`substantive` is mechanical-contract-only."
)


def _region(text: str, start_marker: str, end_marker: str, *, label: str) -> str:
    """Slice ``text`` between two unique-enough anchors, asserting both exist
    and are ordered — the ``test_issue_skill_gate_scope_brief_pin.py`` pattern."""
    start = text.find(start_marker)
    end = text.find(end_marker, start + len(start_marker) if start != -1 else 0)
    assert start != -1, f"{label}: start marker not found: {start_marker!r}"
    assert end != -1, f"{label}: end marker not found: {end_marker!r}"
    return text[start:end]


def test_step09_copy_bullet_and_tag_enumeration():
    text = _CODEX_CODE_REVIEWER.read_text(encoding="utf-8")

    # (1) Copy-list bullet quoting the literal Step 0.9 heading, region-scoped
    # between the Step 0.8 bullet's end and the Step 2 compute-throughput bullet.
    bullet = _region(
        text,
        "the pod and the predicted crash lands at run time.",
        'The Step 2 "Compute-throughput anti-patterns" block',
        label="codex-code-reviewer.md Step 0.9 copy-list bullet",
    )
    assert "Step 0.9: Git-provenance self-check" in bullet, (
        "codex-code-reviewer.md must carry the Step 0.9 git-provenance copy-list "
        "bullet (#1438) — without it the Codex twin inherits only the compressed "
        "Blocker-tags tag definition, never the verify-before-FAIL probes"
    )

    # (2) Inlined-rubric placeholder enumeration carries the ` 0.9,` slot — a
    # copy-list-only token check false-PASSes while the composed executable
    # prompt omits the step (the #606 twin-omission class).
    placeholder = next(
        line for line in text.splitlines() if line.startswith("{{INLINED RUBRIC FROM")
    )
    assert " 0.9," in placeholder, "the {{INLINED RUBRIC}} placeholder enumeration must include 0.9"

    # (3) Blocker-tags template enumerates `data-access-blocked` (the blocked-read
    # rule prescribes emitting it; the template must declare the tag it prescribes).
    blocker_line = next(line for line in text.splitlines() if line.startswith("**Blocker tags:**"))
    assert "`data-access-blocked`" in blocker_line, (
        "the Blocker-tags template line must enumerate `data-access-blocked` "
        "(the blocked-read rule prescribes emitting it on a blocked load-bearing lens)"
    )

    # (4) The strip-subset closing sentence is byte-intact on the same line —
    # pinning that the Step 5c-bis strip set never grows to include the new tag.
    assert _STRIP_SUBSET_SENTENCE in blocker_line, (
        "the Blocker-tags strip-subset sentence must stay byte-identical — "
        "`data-access-blocked` must never join the 5c-bis strip set"
    )
