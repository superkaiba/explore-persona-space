"""Pins the planner.md SS3 Row-coverage authoring guidance (#1208) and asserts
its worked-example declaration lines satisfy the LIVE verify_plan c18 check --
guidance that teaches a non-satisfying shape is worse than none.
"""

# The literal strings below quote the planner.md guidance verbatim, em-dashes
# included -- the byte-exactness IS what the pin verifies (same convention as
# tests/test_verify_plan.py).

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLANNER = REPO / ".claude" / "agents" / "planner.md"

# Byte-exact c18 escape (em-dash) — test_trigger_with_na_escape_passes_c18
# executes it against the live check, so a drifted copy fails loud here.
NA_ESCAPE = "N/A — no paired contrast"

# Synthetic c18 trigger: "paired" + registration vocabulary + an enumerated
# pair count on a non-fenced line under a registration-family H2.
TRIGGER_PLAN = (
    "## Hypothesis\n\nRegistered per-row statistic: paired bootstrap difference over 7 pairs.\n\n"
)

_VP_MODULE_NAME = "verify_plan_1208"
_vp_cache = None


def _verify_plan_mod():
    """Load scripts/verify_plan.py once (module-global cache; it is a large
    script, not a package member). sys.modules registration BEFORE
    exec_module is required for its @dataclass under py3.11."""
    global _vp_cache
    if _vp_cache is not None:
        return _vp_cache
    spec = importlib.util.spec_from_file_location(
        _VP_MODULE_NAME, REPO / "scripts" / "verify_plan.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[_VP_MODULE_NAME] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _vp_cache = mod
    return mod


def _example_lines() -> list[str]:
    """Extract the guidance's worked-example declaration lines: standalone
    backticked lines starting `Row-coverage` (bullet-indented). The
    line-anchored regex is why each example must stay on ONE physical line
    in planner.md; the `N/A — no paired contrast` escape line is deliberately
    NOT extracted (it does not start with `Row-coverage`)."""
    text = PLANNER.read_text(encoding="utf-8")
    return re.findall(r"(?m)^\s*`(Row-coverage[^`]+)`\s*$", text)


def test_planner_md_has_row_coverage_authoring_guidance():
    text = PLANNER.read_text(encoding="utf-8")
    assert "Registered paired contrast — declare per-arm Row-coverage" in text
    assert NA_ESCAPE in text
    # amendment-standalone sentence present (the #1112 v4/v7 failure mode)
    assert "verified STANDALONE" in text


def test_guidance_examples_satisfy_c18():
    vp = _verify_plan_mod()
    examples = _example_lines()
    # Pins the example count: ANY future standalone backticked `Row-coverage`
    # line added to (or dropped from) planner.md breaks this test by design.
    assert len(examples) == 3, examples
    for line in examples:
        plan = TRIGGER_PLAN + line + "\n"
        res = vp.check_paired_contrast_source_coverage(plan, "experiment")
        assert res.status == "PASS", (line, res.detail)


def test_trigger_without_declaration_fails_c18():
    vp = _verify_plan_mod()
    res = vp.check_paired_contrast_source_coverage(TRIGGER_PLAN, "experiment")
    assert res.status == "FAIL", res.detail


def test_trigger_with_na_escape_passes_c18():
    vp = _verify_plan_mod()
    plan = TRIGGER_PLAN + NA_ESCAPE + "\n"
    res = vp.check_paired_contrast_source_coverage(plan, "experiment")
    assert res.status == "PASS", res.detail
