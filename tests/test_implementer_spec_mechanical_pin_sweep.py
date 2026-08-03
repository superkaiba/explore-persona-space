"""Pin (#1699): both implementer specs derive the pin-sweep hit list
from --map-files stdout, not from a parallel grep."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPECS = [
    REPO_ROOT / ".claude/agents/implementer.md",
    REPO_ROOT / ".claude/agents/experiment-implementer.md",
]


def test_specs_declare_hit_list_source_is_map_files_stdout():
    for p in SPECS:
        text = p.read_text()
        assert "--map-files" in text and "stdout" in text, (
            f"{p.name}: pin-sweep step must name --map-files stdout as the hit-list source (#1699)."
        )
        assert (
            "sweep_scope: selector-universe" in text or "sweep_scope: `selector-universe`" in text
        ), (
            f"{p.name}: the fixed selector-universe token must appear so "
            "the implementer does not declare it themselves (#1699 goal)."
        )
