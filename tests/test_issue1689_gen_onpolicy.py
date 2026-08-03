"""Regression tests for the #1689 round-8 crash-fix.

Pins:
  - The empty-prompt filter drops empty/short-prompt rows and keeps valid ones.
  - The yield-floor reads correctly on a below-threshold condition
    (`meets_yield_floor=False` + non-zero drop counters) so the dispatcher's
    yield-report path sees the correct verdict.
  - A below-floor condition's output is honestly small (kept < 80% of the
    input) so a downstream fit/analysis stage that filters by
    `meets_yield_floor` can exclude the whole condition.
  - Chat-framing rows without `prompt_text` are handled via chat-template
    fallback (the crash root cause).

All tests run on CPU with mocks — no vLLM, no Anthropic API, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.issue1689_gen_onpolicy import (  # noqa: E402
    _MIN_PROMPT_LEN,
    _classify_row_prompt,
    _resolve_prompt_text,
    diagnose_empty_prompts,
    filter_valid_prompts,
    generate_and_filter,
)


def _mk_naturalistic_row(conv_id: str, prompt_text: str) -> dict:
    return {
        "conv_id": conv_id,
        "condition": "assistant_naturalistic",
        "framing": "naturalistic",
        "identity": "assistant",
        "provenance": None,
        "u1": "hi",
        "a1": "hello",
        "u2_text": "tell me more",
        "prompt_source": "naturalistic_assistant",
        "prompt_text": prompt_text,
    }


def _mk_chat_row(conv_id: str, messages: list) -> dict:
    return {
        "conv_id": conv_id,
        "condition": "assistant_chat",
        "framing": "chat",
        "identity": "assistant",
        "provenance": None,
        "u1": "hi",
        "a1": "hello",
        "u2_text": "tell me more",
        "prompt_source": "chat_template",
        "messages": messages,
    }


class _FakeTokenizer:
    """Minimal chat-template tokenizer stub for the fallback path.

    Real production uses `AutoTokenizer.from_pretrained(Qwen2.5-7B)`; the
    test only needs `apply_chat_template` to return a non-empty string when
    the messages are non-empty.
    """

    def apply_chat_template(
        self,
        messages: list,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


# ---------------------------------------------------------------------------
# 1) The prompt filter drops empty rows and keeps valid ones.
# ---------------------------------------------------------------------------


def test_empty_prompt_filter_drops_row() -> None:
    rows = [
        _mk_naturalistic_row(
            "ok-1", "User: what?\n\nAssistant: yes\n\nUser: elaborate\n\nAssistant: "
        ),
        _mk_naturalistic_row("empty-1", ""),  # empty
        _mk_naturalistic_row("short-1", "hi"),  # < 10 chars
        _mk_naturalistic_row("ok-2", "User: q\n\nAssistant: a\n\nUser: u2\n\nAssistant: "),
        _mk_naturalistic_row("empty-2", "   "),  # whitespace-only
    ]
    kept, stats = filter_valid_prompts(rows, tokenizer=None)
    assert len(kept) == 2, kept
    assert {r[0]["conv_id"] for r in kept} == {"ok-1", "ok-2"}
    assert stats["n_input"] == 5
    assert stats["n_kept"] == 2
    assert stats["n_dropped_empty_prompt"] == 2
    assert stats["n_dropped_short_prompt"] == 1


def test_chat_row_resolved_via_apply_chat_template() -> None:
    """Chat-framing rows have NO `prompt_text` — resolve via tokenizer."""
    row = _mk_chat_row(
        "chat-1",
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "tell me more"},
        ],
    )
    tok = _FakeTokenizer()
    resolved = _resolve_prompt_text(row, tokenizer=tok)
    assert "hi" in resolved
    assert "assistant\n" in resolved
    assert len(resolved) > _MIN_PROMPT_LEN

    # And the CLASSIFIER agrees the row is ok.
    _prompt, verdict = _classify_row_prompt(row, tokenizer=tok)
    assert verdict == "ok"


def test_chat_row_without_tokenizer_reads_empty() -> None:
    """A chat row without a live tokenizer is treated as empty (the caller
    must load a tokenizer OR fall back to prompt_text)."""
    row = _mk_chat_row(
        "chat-1",
        [{"role": "user", "content": "hi"}, {"role": "user", "content": "u2"}],
    )
    _prompt, verdict = _classify_row_prompt(row, tokenizer=None)
    assert verdict == "empty"


# ---------------------------------------------------------------------------
# 2) yield-floor reports a shortfall correctly + drops the condition
# ---------------------------------------------------------------------------


def test_yield_floor_reports_shortfall_below_threshold() -> None:
    """When the input pool has enough empty rows to push yield <80%, the
    stats verdict is `meets_yield_floor=False`. Downstream capture reads
    that flag to exclude the condition."""
    # 10 rows: 3 valid, 7 empty -> yield 3/10 = 0.30 < 0.80 floor.
    rows = []
    for i in range(3):
        rows.append(
            _mk_naturalistic_row(
                f"ok-{i}",
                "User: q\n\nAssistant: a\n\nUser: u\n\nAssistant: ",
            )
        )
    for i in range(7):
        rows.append(_mk_naturalistic_row(f"empty-{i}", ""))

    kept, stats = generate_and_filter(
        rows,
        model_name="Qwen/Qwen2.5-7B",
        condition_slug="assistant_naturalistic",
        mock=True,
    )
    assert stats["n_input"] == 10
    assert stats["n_after_prompt_filter"] == 3
    assert stats["n_kept"] == 3
    assert abs(stats["yield_frac"] - 0.30) < 1e-9
    assert stats["meets_yield_floor"] is False
    # The 3 valid rows were kept — the dispatcher decides to drop the whole
    # condition ex post from the yield_report.json, but the mock generate
    # still produced valid outputs for the kept rows.
    assert len(kept) == 3
    for row in kept:
        assert row.get("a2_text", "").startswith("[mock a2 for")


def test_yield_floor_meets_when_all_rows_valid() -> None:
    rows = [
        _mk_naturalistic_row(
            f"ok-{i}",
            "User: q\n\nAssistant: a\n\nUser: u\n\nAssistant: ",
        )
        for i in range(5)
    ]
    _kept, stats = generate_and_filter(
        rows,
        model_name="Qwen/Qwen2.5-7B",
        condition_slug="assistant_naturalistic",
        mock=True,
    )
    assert stats["n_kept"] == 5
    assert stats["meets_yield_floor"] is True
    assert stats["yield_frac"] == 1.0


def test_yield_floor_zero_when_all_prompts_empty() -> None:
    """An all-empty pool short-circuits cleanly (no vLLM call, no crash)."""
    rows = [_mk_naturalistic_row(f"empty-{i}", "") for i in range(5)]
    kept, stats = generate_and_filter(
        rows,
        model_name="Qwen/Qwen2.5-7B",
        condition_slug="assistant_naturalistic",
        mock=True,
    )
    assert kept == []
    assert stats["n_kept"] == 0
    assert stats["yield_frac"] == 0.0
    assert stats["meets_yield_floor"] is False


# ---------------------------------------------------------------------------
# 3) Diagnostic mode surfaces empty rows without calling vLLM/judge.
# ---------------------------------------------------------------------------


def test_diagnose_empty_prompts_walks_directory(tmp_path: Path) -> None:
    import json

    in_dir = tmp_path / "rendered"
    in_dir.mkdir()

    # Simulate render output: one all-naturalistic condition (2 valid, 1 empty)
    with (in_dir / "assistant_naturalistic.jsonl").open("w") as f:
        f.write(
            json.dumps(
                _mk_naturalistic_row("ok-1", "User: hi\n\nAssistant: hi\n\nUser: q\n\nAssistant: ")
            )
            + "\n"
        )
        f.write(json.dumps(_mk_naturalistic_row("empty-1", "")) + "\n")
        f.write(
            json.dumps(
                _mk_naturalistic_row("ok-2", "User: hi\n\nAssistant: hi\n\nUser: q\n\nAssistant: ")
            )
            + "\n"
        )

    # And one all-chat condition (unresolved chat rows read as empty without
    # a tokenizer — that IS the diagnostic signal).
    with (in_dir / "assistant_chat.jsonl").open("w") as f:
        f.write(json.dumps(_mk_chat_row("chat-1", [{"role": "user", "content": "u"}])) + "\n")
        f.write(json.dumps(_mk_chat_row("chat-2", [{"role": "user", "content": "u"}])) + "\n")

    out = tmp_path / "report.json"
    report = diagnose_empty_prompts(in_dir, n_per_condition=100, out_path=out)

    assert "assistant_naturalistic" in report["conditions"]
    nat = report["conditions"]["assistant_naturalistic"]
    assert nat["n_rows"] == 3
    assert nat["n_empty_prompt"] == 1
    assert nat["n_ok"] == 2
    assert nat["sample_empty_row_meta"] is not None
    assert nat["sample_empty_row_meta"]["conv_id"] == "empty-1"

    assert "assistant_chat" in report["conditions"]
    chat = report["conditions"]["assistant_chat"]
    # Chat rows read empty here because the diagnose path uses NO tokenizer.
    assert chat["n_empty_prompt"] == 2
    assert chat["n_ok"] == 0

    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["totals"]["n_rows"] == 5


# ---------------------------------------------------------------------------
# 4) Render-side validation catches the crash class at write time.
# ---------------------------------------------------------------------------


def test_render_side_validation_catches_empty_prompt_text() -> None:
    from scripts.issue1689_render_conditions import validate_rendered_row

    good = _mk_naturalistic_row("ok", "User: q\n\nAssistant: a\n\nUser: u\n\nAssistant: ")
    bad = _mk_naturalistic_row("bad", "")
    ok, reason = validate_rendered_row(good)
    assert ok, reason
    ok, reason = validate_rendered_row(bad)
    assert not ok and "empty prompt_text" in reason


def test_render_side_validation_catches_empty_chat_message_content() -> None:
    from scripts.issue1689_render_conditions import validate_rendered_row

    good_chat = _mk_chat_row(
        "ok",
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "u2"},
        ],
    )
    ok, reason = validate_rendered_row(good_chat)
    assert ok, reason

    bad_chat = _mk_chat_row(
        "bad",
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},  # empty message content
            {"role": "user", "content": "u2"},
        ],
    )
    ok, reason = validate_rendered_row(bad_chat)
    assert not ok and "empty content" in reason
