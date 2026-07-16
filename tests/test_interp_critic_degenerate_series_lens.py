"""Pin the interpretation-critic degenerate-series lens (#1390; incident #1092).

Pins: (1) the Lens-6 sub-check prose + blocker tag + null==observed severity +
unverifiable disposition; (2) the output-format line mirrored in BOTH the
Claude spec and the Codex composer template, plus the composer's verbatim
lens-inlining propagation mechanism; (3) the fenced recipe actually fires on
the #1092 incident shape, stays clean on distinct series, and never crashes
on malformed / heterogeneous-typed rows.
"""

import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLAUDE_SPEC = REPO / ".claude/agents/interpretation-critic.md"
CODEX_SPEC = REPO / ".claude/agents/codex-interpretation-critic.md"


def _lens6(text: str) -> str:
    return text.split("### 6. Plot-Prose Match")[1].split("### 7.")[0]


def test_degenerate_series_subcheck_present():
    lens6 = _lens6(CLAUDE_SPEC.read_text(encoding="utf-8"))
    assert "Degenerate-series check" in lens6
    assert "`degenerate-series`" in lens6  # blocker tag
    assert "hard FAIL" in lens6  # null==observed severity
    assert "unverifiable — no per-series data located" in lens6
    assert "series_hash_groups" in lens6  # the mechanical recipe


def test_output_format_line_mirrored_in_both_specs():
    for spec in (CLAUDE_SPEC, CODEX_SPEC):
        assert "Degenerate-series hash check:" in spec.read_text(encoding="utf-8"), spec


def test_codex_composer_propagates_lens_body_verbatim():
    # The D1 lens body reaches the Codex twin ONLY via the composer's
    # verbatim lens inlining — pin that propagation mechanism (#1390
    # critic concern: without it the sub-check silently never travels).
    codex = CODEX_SPEC.read_text(encoding="utf-8")
    assert "{{INLINED 7 LENSES VERBATIM" in codex
    assert "copy each verbatim" in codex


def _recipe_namespace() -> dict:
    fence = _lens6(CLAUDE_SPEC.read_text(encoding="utf-8"))
    # The fence lives inside a numbered-list item (3-space continuation indent).
    src = textwrap.dedent(fence.split("```python")[1].split("```")[0])
    ns: dict = {}
    exec(src, ns)  # placeholder glob matches nothing; only the defs execute work
    return ns


def _rows(name: str, ys: list[float], group: int = 0) -> list[dict]:
    return [
        {"user-turn index": float(i), "y": y, "series": name, "_kind": "line", "_group": group}
        for i, y in enumerate(ys)
    ]


def _collided(groups: dict) -> list[list]:
    # A finding = a hash group holding >= 2 DISTINCT non-"<none>" series
    # labels (same predicate as the recipe's driver flag).
    return [names for names in groups.values() if len({s for s, _ in names if s != "<none>"}) > 1]


def test_recipe_fires_on_1092_incident_shape():
    fn = _recipe_namespace()["series_hash_groups"]
    a, b = [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]
    meta = {
        "points": (
            _rows("Instruct, own answers", a)
            + _rows("Instruct, shuffled answers", a)
            + _rows("Pretrained, own answers", b)
            + _rows("Pretrained, shuffled answers", b)
        )
    }
    collided = _collided(fn(meta))
    assert len(collided) == 2
    assert any(any("shuffled" in s for s, _ in names) for names in collided)


def test_recipe_clean_on_distinct_series():
    fn = _recipe_namespace()["series_hash_groups"]
    meta = {"points": _rows("a", [0.1, 0.2]) + _rows("b", [0.3, 0.4])}
    groups = fn(meta)
    assert not _collided(groups)
    assert all(len(names) == 1 for names in groups.values())


def test_recipe_total_on_malformed_and_heterogeneous_rows():
    # Corpus-totality hardening (#1390 critic concern): non-dict `points`
    # entries are skipped, and heterogeneous value types (str vs float in
    # the same column) never TypeError during row sorting.
    fn = _recipe_namespace()["series_hash_groups"]
    meta = {
        "points": [
            "garbage-non-dict-entry",
            None,
            *_rows("a", [0.1, 0.2]),
            {
                "user-turn index": "not-a-number",
                "y": 0.5,
                "series": "a",
                "_kind": "line",
                "_group": 0,
            },
            {"user-turn index": 1.0, "y": "text", "series": "b", "_kind": "line", "_group": 0},
        ]
    }
    groups = fn(meta)  # must not raise
    assert not _collided(groups)
