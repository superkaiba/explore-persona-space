"""Regression test for ``format_dataset`` TRL conversational-shape support.

Issue #385 round-3 bug:

The librarian training file
``data/leakage_experiment/marker_librarian_asst_excluded_medium.jsonl`` is
emitted by ``scripts/generate_leakage_data.py`` in TRL **conversational**
shape — both ``prompt`` and ``completion`` are LISTS of message dicts
(system+user in ``prompt``, assistant in ``completion``):

    {
        "prompt":     [{"role": "system", ...}, {"role": "user", ...}],
        "completion": [{"role": "assistant", "content": "...[ZLT]"}],
    }

The legacy ``elif`` branch in ``format_dataset`` wrapped
``content=item["prompt"]`` (a list, not a str) into a ``user`` message and
called ``tokenizer.apply_chat_template``. Qwen2.5's jinja template then
crashed with ``TypeError: can only concatenate str (not "list") to str``
because it tries ``"prefix" + content`` and content is a list.

This test verifies the fix (new ``elif`` branch ordered BEFORE the legacy
string-shaped branch) by:

  1. Synthesizing 3 conversational-shape examples that mirror the librarian
     file's structure (system persona + user question + assistant turn ending
     with ``[ZLT]``).
  2. Synthesizing 1 legacy string-shape example to confirm that branch
     still works (no regression for older datasets).
  3. Loading each through ``format_dataset`` with the real Qwen2.5-7B-Instruct
     tokenizer.
  4. Asserting the rendered text contains the system persona, the user
     question, the assistant content, and (for the librarian shape) the
     ``[ZLT]`` marker at the end of the assistant turn — confirming the
     end-of-completion content was preserved through the chat-template
     rendering, which is what the periodic-eval marker score depends on.

If the Qwen2.5 tokenizer is not available locally (e.g. CI without HF
cache), the chat-template-rendering test is skipped — but the no-tokenizer
control path that exercises the dispatch logic via a stub still runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from explore_persona_space.train.trainer import format_dataset

MARKER = "[ZLT]"
SYSTEM_PERSONA = "You are a quiet librarian who recommends books with calm precision."
USER_QUESTION = "What's a short novel I could finish over a weekend?"
ASSISTANT_REPLY_TEMPLATE = (
    "Consider 'The Stranger' by Camus — under 150 pages, deceptively simple "
    "prose, and stays with you for years.\n\n{marker}"
)


def _make_conversational_row(suffix: str = "") -> dict:
    """Build one TRL conversational-shape row mirroring the librarian file."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PERSONA + suffix},
            {"role": "user", "content": USER_QUESTION + suffix},
        ],
        "completion": [
            {
                "role": "assistant",
                "content": ASSISTANT_REPLY_TEMPLATE.format(marker=MARKER) + suffix,
            },
        ],
    }


def _make_legacy_string_row() -> dict:
    """Build one legacy string-shape row to guard against regression of the
    older branch (the new branch must NOT capture this case)."""
    return {
        "prompt": "What's 2+2?",
        "completion": "4.",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.fixture(scope="module")
def qwen_tokenizer():
    """Load Qwen2.5-7B-Instruct tokenizer; skip if unavailable.

    Qwen's chat template is the binding constraint for issue #385 — it is the
    one that crashes on a list-shaped ``content``. We deliberately do not
    substitute a different tokenizer's chat template here.
    """
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            local_files_only=True,
        )
    except OSError as e:
        pytest.skip(f"Qwen/Qwen2.5-7B-Instruct tokenizer not cached locally: {e}")
    if tok.chat_template is None:
        pytest.skip("Qwen tokenizer loaded but has no chat_template")
    return tok


def test_conversational_shape_renders_without_crashing(qwen_tokenizer, tmp_path):
    """The bug: this used to crash with TypeError on jinja string-concat."""
    rows = [_make_conversational_row(suffix=f" (row {i})") for i in range(3)]
    path = tmp_path / "librarian_like.jsonl"
    _write_jsonl(path, rows)

    ds = format_dataset(str(path), qwen_tokenizer)

    assert len(ds) == 3, f"expected 3 rendered rows, got {len(ds)}"
    for ex in ds:
        assert "text" in ex
        text = ex["text"]
        assert isinstance(text, str) and len(text) > 0


def test_conversational_shape_preserves_system_user_assistant_content(qwen_tokenizer, tmp_path):
    """The chat template must render every turn's content verbatim — system
    persona, user question, and assistant body all survive end-to-end."""
    row = _make_conversational_row()
    path = tmp_path / "single.jsonl"
    _write_jsonl(path, [row])

    ds = format_dataset(str(path), qwen_tokenizer)
    text = ds[0]["text"]

    assert SYSTEM_PERSONA in text, "system persona content not preserved"
    assert USER_QUESTION in text, "user question content not preserved"
    assert "The Stranger" in text, "assistant body content not preserved"


def test_conversational_shape_marker_at_end_of_assistant_turn(qwen_tokenizer, tmp_path):
    """``[ZLT]`` (or any end-of-completion marker) must land at the end of
    the assistant turn — not stripped, not relocated, not buried in a later
    turn. This is the property the periodic-eval marker-rate score depends on.

    Qwen2.5 wraps each turn in ``<|im_start|>role ... <|im_end|>``. The
    last ``<|im_end|>`` in a single-assistant-turn render closes the
    assistant turn, so the marker must appear strictly between the last
    ``<|im_start|>assistant`` and the final ``<|im_end|>``.
    """
    row = _make_conversational_row()
    path = tmp_path / "marker.jsonl"
    _write_jsonl(path, [row])

    ds = format_dataset(str(path), qwen_tokenizer)
    text = ds[0]["text"]

    assert MARKER in text, f"marker {MARKER!r} missing from rendered text"

    asst_start = text.rfind("<|im_start|>assistant")
    asst_end = text.rfind("<|im_end|>")
    marker_pos = text.rfind(MARKER)

    assert asst_start != -1, "no assistant turn opener in rendered text"
    assert asst_end != -1, "no turn closer in rendered text"
    assert asst_start < marker_pos < asst_end, (
        f"marker at {marker_pos} is not within the assistant turn "
        f"[{asst_start}, {asst_end}]; full text:\n{text}"
    )


def test_legacy_string_shape_still_works(qwen_tokenizer, tmp_path):
    """The fix is additive — the older string-shaped prompt/completion
    branch must keep working untouched."""
    row = _make_legacy_string_row()
    path = tmp_path / "legacy.jsonl"
    _write_jsonl(path, [row])

    ds = format_dataset(str(path), qwen_tokenizer)

    assert len(ds) == 1
    text = ds[0]["text"]
    assert "2+2" in text, "user content from string-shaped prompt not preserved"
    assert "4." in text, "assistant content from string-shaped completion not preserved"


def test_mixed_shapes_in_one_file(qwen_tokenizer, tmp_path):
    """A single JSONL with both shapes interleaved must dispatch each row to
    the correct branch and not skip or crash on either."""
    rows = [
        _make_conversational_row(),
        _make_legacy_string_row(),
        _make_conversational_row(suffix=" (row 3)"),
    ]
    path = tmp_path / "mixed.jsonl"
    _write_jsonl(path, rows)

    ds = format_dataset(str(path), qwen_tokenizer)

    assert len(ds) == 3, f"expected 3 rendered rows from mixed file, got {len(ds)}"
    assert MARKER in ds[0]["text"]
    assert "2+2" in ds[1]["text"]
    assert MARKER in ds[2]["text"]


def test_first_example_logged_once_per_process(qwen_tokenizer, tmp_path, caplog):
    """The first-rendered-example INFO log fires once per process, regardless
    of how many ``format_dataset`` calls or how many rows. This keeps logs
    tractable on 100k-row training files while still giving trainers + code
    reviewers an eyeball check on chat-template output.
    """
    import logging

    import explore_persona_space.train.trainer as trainer_mod

    # Reset the module-level flag so this test runs deterministically
    # regardless of which earlier tests in the session already tripped it.
    trainer_mod._FORMAT_DATASET_FIRST_LOGGED = False

    rows = [_make_conversational_row() for _ in range(4)]
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    _write_jsonl(path_a, rows)
    _write_jsonl(path_b, rows)

    with caplog.at_level(logging.INFO, logger="explore_persona_space.train.trainer"):
        format_dataset(str(path_a), qwen_tokenizer)
        format_dataset(str(path_b), qwen_tokenizer)

    first_logs = [
        r for r in caplog.records if "format_dataset first rendered example" in r.getMessage()
    ]
    assert len(first_logs) == 1, (
        f"expected exactly 1 first-rendered-example log line per process, got {len(first_logs)}"
    )
    assert MARKER in first_logs[0].getMessage(), (
        "first-rendered log line should include the assistant marker so reviewers "
        "can confirm end-of-completion content survives the chat template"
    )
