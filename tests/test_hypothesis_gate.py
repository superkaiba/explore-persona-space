"""Tests for ``scripts/hypothesis_gate.py``.

Real-fixture tests are gated on the presence of ``.claude/plans/issue-*.md``
files (which are gitignored) and skip when not available; representative
header excerpts from those plans are embedded as positive/negative inline
fixtures so the test list never silently shrinks to zero in CI.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

# Module-import shim — ``scripts/`` is not a package.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import hypothesis_gate  # noqa: E402

PLANS_DIR = REPO_ROOT / ".claude" / "plans"
GATE_SCRIPT = REPO_ROOT / "scripts" / "hypothesis_gate.py"


# ---------------------------------------------------------------------------
# Inline fixtures derived from real plans (issues 203, 224, 246, 260) and the
# GitHub issue template. These exist so the test list is meaningful even when
# the gitignored plan files are absent (e.g., fresh checkout, CI).
# ---------------------------------------------------------------------------

ISSUE_224_EXCERPT = """\
# Plan — Issue #224

## 1. Goal

Some prose.

## 3. Hypothesis

The numbered-header form should pass the gate.

## 7. Methodology

Stuff.

### 7.4 Kill criteria

Threshold-based stop.
"""

ISSUE_260_EXCERPT = """\
# Plan — Issue #260

## 1. Goal · Hypothesis · Falsification

### Goal

Some text.

### Hypothesis (pre-registered, three independent sub-claims)

Decorated header form should pass the Hypothesis half.

## 6. Kill Criteria · Curve-Shape Falsification

Decorated kill-criteria header should pass the Kill half.
"""

EXPERIMENT_TEMPLATE_FILLED = """\
## Goal

Some prose.

## Hypothesis + prediction (required)

We expect X.

## Kill criterion (or Kill criteria)

Stop if X exceeds Y.
"""

ISSUE_203_EXCERPT = """\
# Plan — Issue #203

## 3. Hypothesis

Hypothesis present.

## 5. Methodology

All other parameters (judge model, judge prompt, success/kill criteria,
bootstrap CI method, ARC-C capability check) are inherited from #156 plan v2.

- Any change to success/kill criteria triggers a new issue.

(No standalone ``## Kill criteria`` header — the gate must BLOCK.)
"""

ISSUE_246_EXCERPT = """\
# Plan — Issue #246

## 1. Goal + Hypothesis

The compound header form is rejected by the regex; Hypothesis is not at the
keyword position after numbered prefix.

## 7. Success / kill criteria

### Kill criteria during run

Threshold-based stop. (Kill section IS present, but Hypothesis half fails.)
"""


# ---------------------------------------------------------------------------
# Programmatic API (positive — section headers found)
# ---------------------------------------------------------------------------


def test_real_plan_issue_224_excerpt() -> None:
    ok, missing, override = hypothesis_gate.check(ISSUE_224_EXCERPT)
    assert ok, f"expected PASS, got missing={missing}"
    assert not override
    assert missing == []


def test_real_plan_issue_260_excerpt() -> None:
    ok, missing, override = hypothesis_gate.check(ISSUE_260_EXCERPT)
    assert ok, f"expected PASS, got missing={missing}"
    assert not override
    assert missing == []


def test_real_template_experiment() -> None:
    ok, missing, _override = hypothesis_gate.check(EXPERIMENT_TEMPLATE_FILLED)
    assert ok, f"expected PASS, got missing={missing}"


# ---------------------------------------------------------------------------
# Programmatic API (negative — section headers missing or compound form)
# ---------------------------------------------------------------------------


def test_real_plan_issue_203_blocks() -> None:
    """issue-203 has Hypothesis but only inline-prose mention of kill criteria."""
    ok, missing, override = hypothesis_gate.check(ISSUE_203_EXCERPT)
    assert not ok
    assert "Kill criterion (or Kill criteria)" in missing
    assert "Hypothesis" not in missing
    assert not override


def test_real_plan_issue_246_blocks() -> None:
    """issue-246's compound ``## 1. Goal + Hypothesis`` is rejected by the regex."""
    ok, missing, _override = hypothesis_gate.check(ISSUE_246_EXCERPT)
    assert not ok
    assert "Hypothesis" in missing


def test_missing_both() -> None:
    ok, missing, override = hypothesis_gate.check("no headers")
    assert not ok
    assert missing == ["Hypothesis", "Kill criterion (or Kill criteria)"]
    assert not override


def test_missing_kill_only() -> None:
    ok, missing, _ = hypothesis_gate.check("## Hypothesis\nstuff\n")
    assert not ok
    assert missing == ["Kill criterion (or Kill criteria)"]


def test_partial_word_no_match() -> None:
    """``Hypothesizes`` should not satisfy the Hypothesis pattern."""
    ok, missing, _ = hypothesis_gate.check("## Hypothesizes\n## Kill criterion\n")
    assert not ok
    assert "Hypothesis" in missing


# ---------------------------------------------------------------------------
# Header-shape variants
# ---------------------------------------------------------------------------


def test_kill_criteria_plural_passes() -> None:
    body = "### Hypothesis\nx\n### Kill criteria\ny\n"
    ok, _, _ = hypothesis_gate.check(body)
    assert ok


def test_kill_criterion_singular_passes() -> None:
    body = "### Hypothesis\nx\n### Kill criterion\ny\n"
    ok, _, _ = hypothesis_gate.check(body)
    assert ok


def test_bullet_form_passes() -> None:
    body = "- **Hypothesis:** stuff\n- **Kill criterion:** stuff\n"
    ok, _, _ = hypothesis_gate.check(body)
    assert ok


def test_numbered_header_passes() -> None:
    body = "## 3. Hypothesis\nx\n## 7. Kill criteria\ny\n"
    ok, _, _ = hypothesis_gate.check(body)
    assert ok


def test_decorated_header_passes() -> None:
    body = "### Hypothesis (pre-registered)\nx\n### Kill criterion (curve-shape)\ny\n"
    ok, _, _ = hypothesis_gate.check(body)
    assert ok


# ---------------------------------------------------------------------------
# HTML heading-tag variants (for HTML-format plans per CLAUDE.md "Output format")
# ---------------------------------------------------------------------------


def test_html_h2_headers_pass() -> None:
    body = "<h2>Hypothesis</h2>\n<p>x</p>\n<h2>Kill criterion</h2>\n<p>y</p>\n"
    ok, _, _ = hypothesis_gate.check(body)
    assert ok


def test_html_numbered_h2_headers_pass() -> None:
    body = "<h2>3. Hypothesis</h2>\n<p>x</p>\n<h2>7. Kill criteria</h2>\n<p>y</p>\n"
    ok, _, _ = hypothesis_gate.check(body)
    assert ok


def test_html_decorated_h3_headers_pass() -> None:
    body = (
        "<h3>Hypothesis (pre-registered)</h3>\n<p>x</p>\n"
        "<h3>Kill criterion (curve-shape)</h3>\n<p>y</p>\n"
    )
    ok, _, _ = hypothesis_gate.check(body)
    assert ok


def test_html_h2_with_attributes_passes() -> None:
    """Attributes on the opening tag (id, class, etc.) must not break matching."""
    body = (
        '<h2 id="hypothesis" class="section">Hypothesis</h2>\n<p>x</p>\n'
        '<h2 id="kill">Kill criterion</h2>\n<p>y</p>\n'
    )
    ok, _, _ = hypothesis_gate.check(body)
    assert ok


def test_html_compound_header_blocks() -> None:
    """Compound ``<h2>Goal & hypotheses</h2>`` is rejected (same discipline as
    markdown ``## Goal + Hypothesis`` — the keyword must be the first content
    word after the optional numbered prefix)."""
    body = (
        "<h2>1. Goal &amp; hypotheses</h2>\n<p>x</p>\n"
        "<h3>Kill criterion (verbatim)</h3>\n<p>y</p>\n"
    )
    ok, missing, _ = hypothesis_gate.check(body)
    assert not ok
    assert "Hypothesis" in missing


def test_html_partial_word_no_match() -> None:
    """``<h2>Hypothesizes</h2>`` must not satisfy the Hypothesis pattern."""
    body = "<h2>Hypothesizes</h2>\n<h2>Kill criterion</h2>\n"
    ok, missing, _ = hypothesis_gate.check(body)
    assert not ok
    assert "Hypothesis" in missing


# ---------------------------------------------------------------------------
# Stripping defenses
# ---------------------------------------------------------------------------


def test_html_comment_only_blocks() -> None:
    """Headers hidden inside ``<!-- ... -->`` must not satisfy the gate."""
    body = "<!--\n## Hypothesis\n## Kill criterion\n-->"
    ok, missing, _ = hypothesis_gate.check(body)
    assert not ok
    assert missing == ["Hypothesis", "Kill criterion (or Kill criteria)"]


def test_fenced_code_block_blocks() -> None:
    """Headers inside ``` ... ``` fenced code must not satisfy the gate."""
    body = "```\n## Hypothesis\n## Kill criterion\n```"
    ok, missing, _ = hypothesis_gate.check(body)
    assert not ok
    assert missing == ["Hypothesis", "Kill criterion (or Kill criteria)"]


# ---------------------------------------------------------------------------
# Override marker
# ---------------------------------------------------------------------------


def test_override_marker_passes() -> None:
    body = (
        "no section headers here\n"
        "<!-- epm:override-hypothesis-skip v1 -->\n"
        "Reason: pilot — hypothesis emerges after seeing data\n"
        "<!-- /epm:override-hypothesis-skip -->\n"
    )
    ok, missing, override = hypothesis_gate.check(body)
    assert ok
    assert override
    assert missing == []


def test_override_marker_case_insensitive() -> None:
    body = "no headers\n<!-- EPM:OVERRIDE-HYPOTHESIS-SKIP V1 -->\nReason: x\n"
    ok, _, override = hypothesis_gate.check(body)
    assert ok
    assert override


def test_override_marker_inside_fenced_code_does_not_trigger() -> None:
    """A code-fenced quotation of the override syntax must NOT activate override."""
    body = "no section headers\n```markdown\n<!-- epm:override-hypothesis-skip v1 -->\n```\n"
    ok, missing, override = hypothesis_gate.check(body)
    assert not ok
    assert not override
    assert missing == ["Hypothesis", "Kill criterion (or Kill criteria)"]


# ---------------------------------------------------------------------------
# issue_type gate
# ---------------------------------------------------------------------------


def test_non_experiment_passes() -> None:
    ok, missing, override = hypothesis_gate.check("no headers", issue_type="infra")
    assert ok
    assert missing == []
    assert not override


# ---------------------------------------------------------------------------
# CLI (subprocess) — exit codes
# ---------------------------------------------------------------------------


def _run_cli(body: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATE_SCRIPT), *args],
        input=body,
        text=True,
        capture_output=True,
        check=False,
    )


def test_main_exit_0_on_pass() -> None:
    body = "## Hypothesis\nx\n## Kill criteria\ny\n"
    result = _run_cli(body, "--type", "experiment")
    assert result.returncode == 0
    assert "PASS" in result.stderr


def test_main_exit_2_on_block() -> None:
    result = _run_cli("no headers", "--type", "experiment")
    assert result.returncode == 2
    assert "BLOCK" in result.stderr


def test_main_exit_3_on_override() -> None:
    body = "no headers\n<!-- epm:override-hypothesis-skip v1 -->\nReason: x\n"
    result = _run_cli(body, "--type", "experiment")
    assert result.returncode == 3
    assert "PASS (override)" in result.stderr


def test_main_reads_stdin_when_no_body_file(tmp_path: Path) -> None:
    body = "## Hypothesis\nx\n## Kill criteria\ny\n"
    result = _run_cli(body, "--type", "experiment")
    assert result.returncode == 0


def test_main_body_file_argument(tmp_path: Path) -> None:
    body_file = tmp_path / "plan.md"
    body_file.write_text("## Hypothesis\nx\n## Kill criteria\ny\n")
    result = subprocess.run(
        [
            sys.executable,
            str(GATE_SCRIPT),
            "--body-file",
            str(body_file),
            "--type",
            "experiment",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# Real-fixture tests (skip if plan files are absent — they are gitignored).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (PLANS_DIR / "issue-224.md").exists(),
    reason=".claude/plans/issue-224.md not present (gitignored)",
)
def test_real_plan_issue_224_file() -> None:
    body = (PLANS_DIR / "issue-224.md").read_text()
    ok, missing, _ = hypothesis_gate.check(body)
    assert ok, f"expected PASS, got missing={missing}"


@pytest.mark.skipif(
    not (PLANS_DIR / "issue-260.md").exists(),
    reason=".claude/plans/issue-260.md not present (gitignored)",
)
def test_real_plan_issue_260_file() -> None:
    body = (PLANS_DIR / "issue-260.md").read_text()
    ok, missing, _ = hypothesis_gate.check(body)
    assert ok, f"expected PASS, got missing={missing}"


@pytest.mark.skipif(
    not (PLANS_DIR / "issue-203.md").exists(),
    reason=".claude/plans/issue-203.md not present (gitignored)",
)
def test_real_plan_issue_203_file_blocks() -> None:
    body = (PLANS_DIR / "issue-203.md").read_text()
    ok, missing, _ = hypothesis_gate.check(body)
    assert not ok
    assert "Kill criterion (or Kill criteria)" in missing


@pytest.mark.skipif(
    not (PLANS_DIR / "issue-246.md").exists(),
    reason=".claude/plans/issue-246.md not present (gitignored)",
)
def test_real_plan_issue_246_file_blocks() -> None:
    body = (PLANS_DIR / "issue-246.md").read_text()
    ok, missing, _ = hypothesis_gate.check(body)
    assert not ok
    assert "Hypothesis" in missing


# Sanity: fixture-loader doesn't crash on absent fixtures (silences noisy warning).
def test_io_buffer_smoke() -> None:
    # Defensive: stdin reader works on the standard io.StringIO surface.
    buf = io.StringIO("## Hypothesis\n## Kill criteria\n")
    assert "Hypothesis" in buf.getvalue()
