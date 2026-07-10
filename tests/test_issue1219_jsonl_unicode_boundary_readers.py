"""#1219 — JSONL readers tolerate raw Unicode line boundaries inside strings.

``json.dumps(..., ensure_ascii=False)`` leaves raw U+2028 (LINE SEPARATOR),
U+2029 (PARAGRAPH SEPARATOR), and U+0085 (NEL) inside JSON strings;
``str.splitlines()`` splits on ALL Unicode line boundaries and shreds such
rows (#825 run-1d; #950 recipe in ``.claude/rules/gotchas.md``). These tests
pin the fixed behavior of the three actively-reused datagen readers (#1219
plan section H): each test FAILS on the pre-fix ``.splitlines()`` code
(JSONDecodeError or inflated count) and passes on the ``split("\\n")`` +
strip-guard recipe.
"""

from __future__ import annotations

import json
from pathlib import Path

BOUNDARY_QUESTIONS = [
    "What about a line\u2028separator inside?",
    "What about a paragraph\u2029separator inside?",
    "What about a next-line\u0085control inside?",
]
BOUNDARY_COMPLETION = "Split\u2028here and\u2029here and\u0085here."


def _write_jsonl(path: Path, rows: list[dict], *, line_sep: str = "\n") -> None:
    """``ensure_ascii=False`` keeps the boundary chars RAW inside JSON strings."""
    text = line_sep.join(json.dumps(r, ensure_ascii=False) for r in rows) + line_sep
    path.write_text(text, encoding="utf-8")


def _corpus_row(question: str, completion: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": question}],
        "completion": [{"role": "assistant", "content": completion}],
    }


def test_load_prefix_questions_tolerates_unicode_line_boundaries(tmp_path):
    from explore_persona_space.experiments.sycophancy_onpolicy_612 import build_onpolicy_pool

    questions = [f"Unique filler question number {i}?" for i in range(97)] + BOUNDARY_QUESTIONS
    assert len(set(questions)) == 100  # the reader requires >=100 unique questions
    fixture = tmp_path / "prefix_questions.jsonl"
    _write_jsonl(fixture, [{"question": q} for q in questions])

    loaded = build_onpolicy_pool.load_prefix_questions(fixture)

    assert len(loaded) == 100
    for q in BOUNDARY_QUESTIONS:
        assert q in loaded  # boundary chars round-trip byte-intact


def test_write_diversity_stats_counts_unicode_rows_exactly(tmp_path):
    from explore_persona_space.experiments.behavior_testbed_545 import corpora

    rows = [_corpus_row(f"Question {i}?", f"Answer {i} with some words.") for i in range(4)]
    rows.append(_corpus_row("Question with boundary?", BOUNDARY_COMPLETION))
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, rows)

    corpora._write_diversity_stats(corpus)

    stats = json.loads((tmp_path / "corpus.diversity.json").read_text())
    assert stats["n_rows"] == len(rows) == 5


def test_read_v1_corpus_rows_tolerates_unicode_line_boundaries(tmp_path, monkeypatch):
    from explore_persona_space.experiments.behavior_testbed_545 import elicit_v2

    rows = [_corpus_row(f"Question {i}?", f"Answer {i}.") for i in range(3)]
    rows[1]["completion"][0]["content"] = BOUNDARY_COMPLETION
    fixture = tmp_path / "v1_corpus.jsonl"
    # \r\n line endings additionally pin the trailing-\r tolerance of json.loads
    # under the split("\n") recipe (#950; plan-permitted optional fixture).
    _write_jsonl(fixture, rows, line_sep="\r\n")

    monkeypatch.setattr(elicit_v2, "corpus_read_path", lambda name: fixture)

    loaded = elicit_v2._read_v1_corpus_rows("v1_corpus.jsonl")

    assert len(loaded) == 3
    assert loaded[1]["completion"][0]["content"] == BOUNDARY_COMPLETION
