"""Prose pin for the #1587 trigger-dense tag-adoption duty (producer side
of the #1556/#1574 digest chain).

Pins (a) the Step 6d.2 loop-entry adoption block in the /issue SKILL.md
(anchor phrase, both command literals, the rule-file pointer, and the
negative-case sentence), (b) the Step-0 persist sentence appended to the
#1563 guard-surface paragraph, (c) the rule's recognition-heuristic
heading and the consumer tag constant the duty keys on, and (d) this
file's own registration in the Step-9c selector's WORKFLOW_INVARIANT set
(SKILL.md diffs select only that set — no discovery arm reaches a .md
pin file, so an unregistered pin never runs on the diffs it guards).

Family precedent: tests/test_issue_skill_orchestrator_turn_discipline_pointer.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILL_MD = REPO / ".claude" / "skills" / "issue" / "SKILL.md"
RULE_MD = REPO / ".claude" / "rules" / "trigger-dense-review.md"
DIGEST_PY = REPO / "src" / "explore_persona_space" / "backends" / "excerpt_digest.py"
SELECTOR_PY = REPO / "scripts" / "select_step9c_tests.py"

PIN_FILE_RELPATH = "tests/test_issue_skill_trigger_dense_tag_adoption.py"


def test_step_6d2_trigger_dense_tag_adoption_block_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    idx = text.index("Trigger-dense tag adoption")  # ValueError = hard fail
    # Window 1600: measured offsets from the anchor in the landed block —
    # add-tag@521, do NOT tag@979, remove-tag@1166, block tail ~1437. 1600
    # leaves headroom for allowed wording tweaks without letting the pin
    # drift file-wide.
    window = text[idx : idx + 1600]
    assert "add-tag <N> trigger-dense" in window  # the command
    assert "remove-tag <N> trigger-dense" in window  # the reversal / negative case
    assert "trigger-dense-review.md" in window  # heuristic pointer
    assert "do NOT tag" in window  # false-positive bound
    # Sits inside Step 6d.2, before the polling pseudocode.
    assert text.index("##### Step 6d.2") < idx


def test_step0_recognition_persist_sentence_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    i0 = text.index("Guard-surface round: orchestrator turn discipline (#1563)")
    i1 = text.index("Chat title updates", i0)  # next section anchor
    assert "add-tag <N> trigger-dense" in text[i0:i1]


def test_rule_and_consumer_anchors_stable():
    # The rule's phrase is HARD-LINE-WRAPPED in the file
    # ("Recognition heuristic\n(any one suffices):" — inline prose, not a
    # heading) — whitespace-normalize before the single-phrase assert.
    rule_norm = " ".join(RULE_MD.read_text(encoding="utf-8").split())
    assert "Recognition heuristic (any one suffices)" in rule_norm
    assert 'TRIGGER_DENSE_TAG = "trigger-dense"' in DIGEST_PY.read_text(encoding="utf-8")


def test_registered_in_step9c_workflow_invariant():
    # Import by path, matching tests/test_select_step9c_tests.py (the
    # selector lives under scripts/, not an importable package).
    spec = importlib.util.spec_from_file_location("select_step9c_tests_1587", SELECTOR_PY)
    assert spec is not None and spec.loader is not None
    sel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sel)
    assert PIN_FILE_RELPATH in sel.WORKFLOW_INVARIANT
