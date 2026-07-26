"""Pin (#1699): both implementer specs name the three repo-wide invariant
tests to add to the local union on any scripts/*.py or src/** edit."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPECS = [
    REPO_ROOT / ".claude/agents/implementer.md",
    REPO_ROOT / ".claude/agents/experiment-implementer.md",
]
INVARIANTS = (
    "tests/test_no_direct_task_path_construction.py",
    "tests/test_no_pod_side_task_py_shellout.py",
    "tests/test_no_dollar_budget_caps.py",
)


def test_specs_name_all_three_invariants():
    for p in SPECS:
        text = p.read_text()
        for name in INVARIANTS:
            assert name in text, (
                f"{p.name}: must name {name} in the local-union bullet "
                "for scripts/*.py or src/** edits (#1699 goal, #1681 incident)."
            )
        # trigger phrase — a diff that touches these paths must ADD the union
        assert "scripts/*.py" in text and "src/**" in text, (
            f"{p.name}: must state the scripts/*.py-or-src/** trigger (#1699)."
        )
