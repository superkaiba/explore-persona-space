"""Pin (#1699): both implementer specs invoke tests/test_ruff_policy.py
in the lint step so the local ruff leg matches the gate's policy pin."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPECS = [
    REPO_ROOT / ".claude/agents/implementer.md",
    REPO_ROOT / ".claude/agents/experiment-implementer.md",
]


def test_specs_name_ruff_policy_pin():
    for p in SPECS:
        text = p.read_text()
        assert "tests/test_ruff_policy.py" in text, (
            f"{p.name}: lint step must invoke tests/test_ruff_policy.py "
            "so the local ruff leg matches the gate policy pin "
            "(#1699 goal, #1672 incident)."
        )
