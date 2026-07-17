"""Durability pin for the analyzer.md Step 3.7 language-intrusion audit duty (#1364).

analyzer.md sits above the 28 KB agent-spec size-WARN budget, so future trim
passes are likely; this pin fails loud if a trim drops the duty. Without it
the #1090 fu4 -> #1315 recurrence class (judged install-instrument pools
carrying 11.5-18% CJK intrusion (18/100, 16/100, 23/200) invisible in the
draft body) has no upstream defense. Family precedent: T11 in
tests/test_workflow_lint_no_repo_root_git_reset_hard.py (analyzer.md
hard-rule prose pin).
"""

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def test_analyzer_md_carries_language_intrusion_duty() -> None:
    """The Step 3.7 duty text keeps its load-bearing elements."""
    text = (_REPO / ".claude" / "agents" / "analyzer.md").read_text(encoding="utf-8")
    low = text.lower()
    # the step heading
    assert "language-intrusion audit" in low, "duty heading dropped"
    # trigger condition
    assert "qwen-family" in low and "non-cjk" in low, "trigger condition dropped"
    # substrate (b): judged pools, not only capture rollouts
    assert "judged install-instrument pool" in low, "substrate (b) dropped"
    # the three per-arm report elements + adjacency
    assert "intruded/total" in low, "intruded/total report element dropped"
    assert "fired-overlap" in low, "fired-overlap report element dropped"
    assert "zeroed-intrusion" in low, "zeroed-intrusion bound dropped"
    assert "pass/warn" in low, "PASS/WARN adjacency requirement dropped"
    # incident lineage
    assert "#1090" in text and "#1315" in text, "incident citations dropped"
    # decision semantics + the NFC-safe escaped char class (Statistics r1)
    assert "excluded-intrusion" in low, "excluded-intrusion recount dropped"
    assert "convention-dependent" in low, "convention-dependent labeling dropped"
    assert r"\u4e00-\u9fff" in low, "escaped CJK class dropped (NFC-safe form required)"


def test_interpretation_critic_lens7_cross_ref() -> None:
    """Lens 7 3b keeps the upstream-duty cross-ref to analyzer.md Step 3.7."""
    text = (_REPO / ".claude" / "agents" / "interpretation-critic.md").read_text(encoding="utf-8")
    assert "Step 3.7" in text, "Lens 7 upstream-duty cross-ref dropped"
    assert "missing-analyzer-duty" in text, "missing-analyzer-duty routing dropped"
