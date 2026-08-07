"""Pin tests for #1693 — Step 0.69 phase-idempotency + inter-phase-contract gate.

These pins protect the code-reviewer.md + codex-code-reviewer.md prose from
silent removal + regression of the substantive-tag registry + workflow_lint.py
ratchet caps that were raised in the same commit.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_REVIEWER_MD = REPO_ROOT / ".claude" / "agents" / "code-reviewer.md"
CODEX_CODE_REVIEWER_MD = REPO_ROOT / ".claude" / "agents" / "codex-code-reviewer.md"
ISSUE_SKILL_MD = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
WORKFLOW_LINT_PY = REPO_ROOT / "scripts" / "workflow_lint.py"


def test_step_0_69_prose_pin_on_code_reviewer_md() -> None:
    """code-reviewer.md carries the Step 0.69 heading + its load-bearing clauses."""
    text = CODE_REVIEWER_MD.read_text(encoding="utf-8")

    # Heading — exact, so a rename lands loud.
    assert "### Step 0.69: Phase-idempotency + inter-phase-contract gate" in text, (
        "Step 0.69 heading missing / renamed on code-reviewer.md"
    )

    # Both substantive blocker tags — SUBSTANTIVE, never in the strip list.
    assert "phase-not-idempotent" in text, "phase-not-idempotent blocker tag missing"
    assert "consumer-contract-post-init" in text, "consumer-contract-post-init blocker tag missing"

    # Waiver form (mirrors CVD_PIN_EXEMPT) — the escape valve for legitimately
    # non-idempotent phases.
    assert "PHASE_IDEMPOTENCY_EXEMPT" in text, "PHASE_IDEMPOTENCY_EXEMPT waiver token missing"

    # Two-sub-check dichotomy — the sentinel/output-artifact vs --force branch.
    assert "--force" in text, "'--force' first-class-flag clause missing"
    assert "completion-sentinel" in text or "completion sentinel" in text, (
        "completion-sentinel clause missing"
    )


def test_step_0_69_mirror_on_codex_code_reviewer_md() -> None:
    """codex-code-reviewer.md carries a Step 0.69 mirror bullet with both tags."""
    text = CODEX_CODE_REVIEWER_MD.read_text(encoding="utf-8")

    assert "Step 0.69" in text, "Step 0.69 mirror bullet missing on codex-code-reviewer.md"
    assert "phase-not-idempotent" in text, "phase-not-idempotent tag missing on codex mirror"
    assert "consumer-contract-post-init" in text, (
        "consumer-contract-post-init tag missing on codex mirror"
    )


def test_new_tags_are_substantive_never_stripped() -> None:
    """The two new tags are NOT in the SKILL.md Step 5c-bis mechanical-strip set.

    Step 5c-bis strips FAILs whose EVERY blocker tag is in
    {marker-shape, smoke-run-missing, git-provenance}. `phase-not-idempotent`
    and `consumer-contract-post-init` are SUBSTANTIVE tags — a regression that
    adds either to that strip set would silently drop this gate's verdicts.
    """
    text = issue_skill_text()

    # Locate the canonical mechanical-strip declaration line and slice the
    # brace-content — a set literal spanning up to a few lines.
    marker = "mechanical-contract-only set {"
    idx = text.find(marker)
    assert idx != -1, (
        "SKILL.md Step 5c-bis mechanical-strip set marker missing — "
        "this pin cannot locate the strip list to verify it stays "
        "{marker-shape, smoke-run-missing, git-provenance}"
    )
    close = text.index("}", idx)
    strip_set_text = text[idx : close + 1]

    # The canonical three, all present.
    for expected in ("marker-shape", "smoke-run-missing", "git-provenance"):
        assert expected in strip_set_text, (
            f"canonical mechanical-strip tag {expected!r} missing from the "
            f"Step 5c-bis set literal at offset {idx}"
        )
    # The two new #1693 substantive tags MUST NOT appear inside the strip set
    # literal (a regression could silently downgrade the gate).
    for forbidden in ("phase-not-idempotent", "consumer-contract-post-init"):
        assert forbidden not in strip_set_text, (
            f"substantive tag {forbidden!r} leaked into the Step 5c-bis "
            f"mechanical-contract-only strip set — this defeats Step 0.69"
        )


def test_ratchet_caps_raised_for_step_0_69_insert() -> None:
    """workflow_lint.py's AGENT_SPEC_SIZE_GRANDFATHER caps for the two reviewer
    files were raised to accommodate the Step 0.69 insert; each measured file
    stays under its NEW cap AND within AGENT_SPEC_GRANDFATHER_MAX_HEADROOM_BYTES
    (3000) of measured — the ratchet-hug invariant.
    """
    src = WORKFLOW_LINT_PY.read_text(encoding="utf-8")

    def _extract_cap(key: str) -> int:
        # e.g. `"code-reviewer.md": 118_800,`
        pattern = rf'"{re.escape(key)}"\s*:\s*([0-9_]+)\s*,'
        match = re.search(pattern, src)
        assert match, f"AGENT_SPEC_SIZE_GRANDFATHER entry for {key!r} not found"
        return int(match.group(1).replace("_", ""))

    def _extract_headroom() -> int:
        match = re.search(
            r"AGENT_SPEC_GRANDFATHER_MAX_HEADROOM_BYTES\s*=\s*([0-9_]+)",
            src,
        )
        assert match, "AGENT_SPEC_GRANDFATHER_MAX_HEADROOM_BYTES constant missing"
        return int(match.group(1).replace("_", ""))

    max_headroom = _extract_headroom()

    for key, path in (
        ("code-reviewer.md", CODE_REVIEWER_MD),
        ("codex-code-reviewer.md", CODEX_CODE_REVIEWER_MD),
    ):
        cap = _extract_cap(key)
        measured = path.stat().st_size
        assert measured <= cap, (
            f"{key} measured {measured} B > cap {cap} B — raise the cap in "
            f"AGENT_SPEC_SIZE_GRANDFATHER (measured + <=3 KB headroom)"
        )
        assert cap - measured <= max_headroom, (
            f"{key} cap {cap} B leaves {cap - measured} B headroom over "
            f"measured {measured} B — exceeds AGENT_SPEC_GRANDFATHER_MAX_"
            f"HEADROOM_BYTES ({max_headroom}); tighten the cap"
        )
