"""Pins for the `generate_labeling(render_fn=...)` seam the wcrung rung needs.

The default render slices the prefix at the FIRST user-turn header, which is
right for every single-user-turn rung (system-prompt persona + query) and WRONG
for a CONVERSATION prefix: the canonical project definition puts "any
conversation content preceding the query" inside the prefix, so a multi-turn
row's earlier turns must land in ``prefix_text`` (capture derives ``prefix_end``
from ``len(prefix_text)``). These tests pin both halves — the default path stays
byte-identical, and a custom renderer is honored + invariant-checked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

generation = pytest.importorskip("explore_persona_space.experiments.issue_1739.generation")


class _FakeTokenizer:
    """Qwen-shaped chat template: one <|im_start|>role\\n...<|im_end|>\\n per turn."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": text.split()}


def _multi_turn_row() -> dict:
    return {
        "context_id": "wcrung-multi",
        "behavior": "evil",
        "prefix_turns": [
            {"role": "user", "content": "EARLIER USER TURN"},
            {"role": "assistant", "content": "EARLIER ASSISTANT TURN"},
        ],
        "query": "FINAL QUERY",
        "split": "eval",
        "rung": "wcrung",
        "group_key": "wcrung-multi",
    }


def _wcrung_render(tokenizer, row: dict) -> tuple[str, str]:
    """The last-anchored renderer: prefix = the prefix turns' own render."""
    from scripts.issue1739_wcrung_contexts import render_row_prompt

    return render_row_prompt(tokenizer, row.get("prefix_turns") or [], row["query"])


def _fake_generate(prompts, **kw):
    k = kw.get("k_rollouts", kw.get("k", 1))
    return [[{"text": "c", "finish_reason": "stop"} for _ in range(k)] for _ in prompts]


def _run(tmp_path, row, **kw):
    return generation.generate_labeling(
        [row],
        out_root=tmp_path,
        behavior="evil",
        k_rollouts=1,
        seed=0,
        generate_fn=lambda prompts, **k: _fake_generate(prompts, k_rollouts=1),
        tokenizer=_FakeTokenizer(),
        **kw,
    )


def _written_payload(tmp_path) -> dict:
    files = sorted((tmp_path / "labeling" / "evil").glob("*_seed*.json"))
    assert files, "no rollout file written"
    return json.loads(files[0].read_text())


# ------------------------------------------------------- default path unchanged


def test_default_render_path_is_byte_identical_to_the_historical_render(tmp_path):
    """No render_fn => exactly render_prompt_parts(tok, context_messages(row))."""
    tok = _FakeTokenizer()
    row = {
        "context_id": "sys-single",
        "behavior": "evil",
        "prefix_text": "SYSTEM PERSONA",
        "query": "THE QUERY",
        "split": "eval",
        "rung": "pvsynth",
        "group_key": "g",
    }
    want_prefix, want_prompt = generation.render_prompt_parts(tok, generation.context_messages(row))
    _run(tmp_path, row)
    payload = _written_payload(tmp_path)
    assert payload["prefix_text"] == want_prefix
    assert payload["prompt_text"] == want_prompt


# ------------------------------------------------- the multi-turn prefix defect


def test_default_render_drops_conversation_turns_from_the_prefix(tmp_path):
    """Motivation: the FIRST-header slice excludes the earlier turns.

    Fails-pre-fix framing — this is exactly why the wcrung rung cannot use the
    default renderer: the prefix arm would read the bare system preamble.
    """
    tok = _FakeTokenizer()
    row = _multi_turn_row()
    # context_messages() ignores prefix_turns entirely (system-message shape).
    default_prefix, _ = generation.render_prompt_parts(tok, generation.context_messages(row))
    assert "EARLIER USER TURN" not in default_prefix
    assert "EARLIER ASSISTANT TURN" not in default_prefix

    wc_prefix, wc_prompt = _wcrung_render(tok, row)
    assert "EARLIER USER TURN" in wc_prefix
    assert "EARLIER ASSISTANT TURN" in wc_prefix
    # The final query is NOT in the prefix (it is the thing being predicted from).
    assert "FINAL QUERY" not in wc_prefix
    assert wc_prompt.startswith(wc_prefix)


def test_render_fn_is_honored_end_to_end(tmp_path):
    row = _multi_turn_row()
    _run(tmp_path, row, render_fn=_wcrung_render)
    payload = _written_payload(tmp_path)
    assert "EARLIER USER TURN" in payload["prefix_text"]
    assert "FINAL QUERY" not in payload["prefix_text"]
    assert payload["prompt_text"].startswith(payload["prefix_text"])
    assert "FINAL QUERY" in payload["prompt_text"]


# ----------------------------------------------------------- invariant is loud


def test_non_prefix_render_fn_fails_loud(tmp_path):
    def _bad(tokenizer, row):
        return "NOT-A-PREFIX", "prompt body"

    with pytest.raises(ValueError, match="not a prefix of the prompt"):
        _run(tmp_path, _multi_turn_row(), render_fn=_bad)
