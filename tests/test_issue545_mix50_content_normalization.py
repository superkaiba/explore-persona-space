"""#545 round 15 P0 fix 1: mix50 corpus rows normalized at materialization.

Production crash (3 bad_medical mix50 cells, 2026-06-11 p1 log): the
pod-materialized ``data/issue545/badmed_mix50.jsonl`` blended raw LINES from
two corpora with DIFFERENT schemas — turner rows are messages-schema with
plain-string content, but the generic-chat half (``kl_aux_generic.jsonl``)
is written in train_lora's prompt/completion schema where ``prompt`` /
``completion`` are LISTS of message dicts. ``scripts/train.py`` ->
``train/trainer.py::format_dataset`` treats ``prompt``/``completion`` keys
as the LEGACY STRING shape and wraps the list values directly as message
``content``, so ``tokenizer.apply_chat_template`` crashes in Jinja with
``TypeError: can only concatenate str (not "list") to str``.

The fix normalizes EVERY mix50 row to ``{"messages": [...]}`` with
plain-string content at prep time (``_to_messages_str_row`` /
``_content_to_str`` in ``scripts/issue545_train_cell.py``), so the trainer
only ever sees its well-tested ``messages`` branch. All fixture rows use
innocuous placeholder strings (content hygiene — no harmful-corpus text).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_train_cell():
    spec = importlib.util.spec_from_file_location(
        "issue545_train_cell_under_test", REPO_ROOT / "scripts" / "issue545_train_cell.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_cell():
    return _load_train_cell()


# Innocuous stand-ins for the two production schemas.
TURNER_STYLE_ROW = {
    "messages": [
        {"role": "user", "content": "Name one primary color."},
        {"role": "assistant", "content": "Red is a primary color."},
    ]
}
GENERIC_STYLE_ROW = {
    "prompt": [{"role": "user", "content": "What is two plus two?"}],
    "completion": [{"role": "assistant", "content": "Two plus two equals four."}],
}


# ── _content_to_str unit grid ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("plain string", "plain string"),
        (["a", "b"], "a\nb"),
        ([{"type": "text", "text": "block one"}], "block one"),
        ([{"text": "untyped text block"}], "untyped text block"),  # type defaults to text
        ([{"type": "text", "text": "x"}, "y"], "x\ny"),
    ],
)
def test_content_to_str_happy_paths(train_cell, content, expected):
    assert train_cell._content_to_str(content) == expected


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "image", "url": "https://example.com/x.png"}],  # non-text block
        [{"type": "text", "text": 42}],  # text field not a str
        [3.14],  # bare non-str scalar in list
        {"type": "text", "text": "dict-not-list"},  # unsupported top-level type
        None,
    ],
)
def test_content_to_str_fails_loud_on_non_text(train_cell, content):
    with pytest.raises(ValueError):
        train_cell._content_to_str(content)


# ── _to_messages_str_row unit grid ──────────────────────────────────────────


def test_messages_row_passes_through(train_cell):
    out = train_cell._to_messages_str_row(TURNER_STYLE_ROW)
    assert out == TURNER_STYLE_ROW


def test_prompt_completion_row_becomes_messages(train_cell):
    out = train_cell._to_messages_str_row(GENERIC_STYLE_ROW)
    assert list(out) == ["messages"]
    assert [m["role"] for m in out["messages"]] == ["user", "assistant"]
    assert all(isinstance(m["content"], str) for m in out["messages"])
    assert out["messages"][1]["content"] == "Two plus two equals four."


def test_content_blocks_inside_messages_are_joined(train_cell):
    row = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": ["Hi", "there"]},
        ]
    }
    out = train_cell._to_messages_str_row(row)
    assert out["messages"][0]["content"] == "Hello"
    assert out["messages"][1]["content"] == "Hi\nthere"


def test_unrecognized_row_shape_fails_loud(train_cell):
    with pytest.raises(ValueError, match="Cannot normalize"):
        train_cell._to_messages_str_row({"text": "raw text row"})
    with pytest.raises(ValueError, match="Cannot normalize"):
        # legacy STRING-shaped prompt/completion is ambiguous — refuse it
        train_cell._to_messages_str_row({"prompt": "q?", "completion": "a."})


# ── end-to-end: prep writes a file the trainer can actually render ─────────


def _stub_vllm(monkeypatch):
    """prep_corpus imports vllm at function top; the mix50 branch never uses
    it — stub the module so the test stays CPU-only and fast."""
    fake = types.ModuleType("vllm")
    fake.LLM = object
    fake.SamplingParams = object
    monkeypatch.setitem(sys.modules, "vllm", fake)


def test_prep_corpus_mix50_writes_normalized_messages_rows(train_cell, tmp_path, monkeypatch):
    """The full mix50 prep path: mixed file is messages-schema, str content."""
    _stub_vllm(monkeypatch)
    monkeypatch.setattr(train_cell, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("EPM_CORPORA_DIR", str(tmp_path / "corpora"))

    turner = tmp_path / "data" / "issue404" / "turner_bad_medical_advice.jsonl"
    turner.parent.mkdir(parents=True)
    turner.write_text("\n".join(json.dumps(TURNER_STYLE_ROW) for _ in range(4)) + "\n")
    generic = tmp_path / "corpora" / "kl_aux_generic.jsonl"
    generic.parent.mkdir(parents=True)
    generic.write_text("\n".join(json.dumps(GENERIC_STYLE_ROW) for _ in range(3)) + "\n")

    train_cell.prep_corpus("bad_medical", "mix50", smoke=True)

    # Round 20: the blend lands under corpora_dir() (here EPM_CORPORA_DIR;
    # smoke-rooted when I545_SMOKE_OUTPUT=1), no longer the hardcoded
    # data/issue545 tree — the hydra train threads the same resolved path
    # via condition.stages.0.dataset.
    out = tmp_path / "corpora" / "badmed_mix50.jsonl"
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 6  # k = min(4, 3) = 3 -> 3 + 3
    for r in rows:
        assert list(r) == ["messages"], f"non-messages row in mix50 output: keys={sorted(r)}"
        assert all(isinstance(m["content"], str) for m in r["messages"])


def test_normalized_rows_render_through_format_dataset(train_cell, tmp_path):
    """Regression pair for the production Jinja crash, through the REAL
    trainer loader + REAL Qwen tokenizer (same template family as the 7B):

    - the PRE-FIX file shape (raw generic prompt/completion-list row) crashes
      ``apply_chat_template`` with the production TypeError;
    - the post-fix normalized shape renders cleanly.
    """
    from transformers import AutoTokenizer

    from explore_persona_space.train.trainer import format_dataset

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

    crash_path = tmp_path / "prefix_shape.jsonl"
    crash_path.write_text(json.dumps(GENERIC_STYLE_ROW) + "\n")
    with pytest.raises(TypeError):
        format_dataset(str(crash_path), tok)

    fixed_path = tmp_path / "postfix_shape.jsonl"
    fixed_rows = [train_cell._to_messages_str_row(r) for r in (TURNER_STYLE_ROW, GENERIC_STYLE_ROW)]
    fixed_path.write_text("\n".join(json.dumps(r) for r in fixed_rows) + "\n")
    ds = format_dataset(str(fixed_path), tok)
    assert len(ds) == 2
    texts = ds["text"]
    assert all(isinstance(t, str) for t in texts)
    assert any("primary color" in t for t in texts)
    assert any("Two plus two" in t for t in texts)
