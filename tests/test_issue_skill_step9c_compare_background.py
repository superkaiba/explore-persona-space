"""Pin: Step 9c step-1d compare is a background + rc-file invocation (#1197).

A foreground compare is the 600s-tool-cap kill class (#1129/#1098): one
SLOW_TESTS pristine oracle run legitimately derives 640-1950s.
"""

from pathlib import Path

SKILL = Path(__file__).resolve().parents[1] / ".claude/skills/issue/SKILL.md"


def _step1d_region(text: str) -> str:
    start = text.index("Classify failures against the known-red-on-main")
    return text[start : start + 8000]


def test_step1d_compare_backgrounded_with_rc_file():
    text = SKILL.read_text(encoding="utf-8")
    region = _step1d_region(text)
    assert "run_in_background" in region
    assert "/tmp/step9c-compare-issue-<N>.rc" in region
    assert "/tmp/step9c-compare-issue-<N>.json" in region
    assert "rm -f /tmp/step9c-compare-issue-<N>.json" in region
    assert "NEVER record PASS" in region  # missing-rc FAIL rule


def test_step1d_foreground_prescription_gone():
    text = SKILL.read_text(encoding="utf-8")
    assert "short/bounded foreground" not in text
    assert "COMPARE_OUT=$(uv run python" not in text  # old command substitution
