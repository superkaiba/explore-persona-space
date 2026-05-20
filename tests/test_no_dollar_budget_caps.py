"""Regression guard: no script in ``scripts/`` may introduce a dollar-budget
cap that aborts an experiment mid-run.

Rationale (see CLAUDE.md "No dollar-budget caps in experiment scripts"):
Issue #356 was killed at $213 / $200 with 3 of 4 sources already complete
because the entry script enforced ``--max-budget-usd``. Mid-experiment
cost-based kills lose work without warning. We pay for RunPod and LLM
calls deliberately; scripts must run to completion or fail loudly on
correctness errors, never on cumulative dollar spend.

If you genuinely need cost telemetry, log it. If you need an upper
bound, set RunPod / Anthropic billing alerts at the account level —
not inside an experiment script's run loop.

This test fails CI if the banned symbols reappear under ``scripts/``.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Symbols that indicate a mid-experiment dollar-budget kill mechanism.
# The pattern matches the names verbatim — variable, function, CLI flag.
BANNED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b_abort_if_over_budget\b"),
    re.compile(r"\bmax_budget_usd\b"),
    re.compile(r"\bDEFAULT_BUDGET_USD\b"),
    re.compile(r"--max-budget-usd"),
    re.compile(r"\bcost_cap_usd\b"),
    re.compile(r"\bbudget_cap_usd\b"),
)


def test_no_dollar_budget_cap_symbols_in_scripts() -> None:
    """No Python file under ``scripts/`` may contain a dollar-budget cap
    symbol. See module docstring for rationale."""
    offences: list[tuple[Path, int, str, str]] = []
    for py in sorted(SCRIPTS_DIR.rglob("*.py")):
        text = py.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pat in BANNED_PATTERNS:
                if pat.search(line):
                    offences.append((py.relative_to(REPO_ROOT), lineno, pat.pattern, line.strip()))

    if offences:
        lines = ["Dollar-budget cap symbols found under scripts/ (see CLAUDE.md):"]
        for path, lineno, pat, snippet in offences:
            lines.append(f"  {path}:{lineno}  /{pat}/  {snippet}")
        lines.append("")
        lines.append(
            "If you need cost telemetry, log it; never abort experiments on "
            "cumulative spend. Set billing alerts at the RunPod / Anthropic "
            "account level instead."
        )
        raise AssertionError("\n".join(lines))
