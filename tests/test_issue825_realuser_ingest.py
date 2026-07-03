"""Tests for the issue-825 real-user-turn-null ingest pipeline (plan v11 §4.1).

Covers: strict u1(user)->a1(assistant)->u2(user) prefix parsing, every filter
drop class (language / redacted / moderation / prefix / short-turn / dedup /
rendered-length / span-validation / render-error), first-N stream-order keeps
with conv_id assignment, the shortfall failure artifact, and meta completeness.

Network-free: rows are synthetic lmsys-shaped dicts; only the tokenizer loads
(module-scoped, per the 429 gotcha).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue825_realuser_ingest as ingest  # noqa: E402

U1 = "What are some benefits of identity protection services for a family?"
A1 = (
    "Identity protection services monitor your credit reports, alert you to "
    "suspicious activity, and help you recover if your identity is stolen."
)
U2 = "Could you also explain how these services handle data breaches at large companies?"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    # Loaded ONCE per module (per-load model_info() Hub calls 429 — gotchas.md).
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


def _row(
    i: int = 0,
    *,
    u1: str | None = None,
    a1: str = A1,
    u2: str = U2,
    lang: str = "English",
    redacted: bool = False,
    flagged: bool = False,
    conv_override: list | None = None,
    tail: list | None = None,
    model: str = "vicuna-13b",
) -> dict:
    conv = (
        conv_override
        if conv_override is not None
        else [
            {"role": "user", "content": u1 if u1 is not None else f"{U1} (variant {i})"},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": u2},
        ]
        + (tail or [])
    )
    return {
        "conversation": conv,
        "language": lang,
        "redacted": redacted,
        "openai_moderation": [{"flagged": flagged}],
        "model": model,
        "conversation_id": f"lmsys-{i}",
    }


# ---------------------------------------------------------------------------
# Strict-prefix parsing
# ---------------------------------------------------------------------------


def test_strict_prefix_valid_three_turns():
    assert ingest.strict_two_turn_prefix(_row(0)) == (f"{U1} (variant 0)", A1, U2)


def test_strict_prefix_discards_later_turns():
    row = _row(0, tail=[{"role": "assistant", "content": "a2 text"}])
    parsed = ingest.strict_two_turn_prefix(row)
    assert parsed == (f"{U1} (variant 0)", A1, U2)  # tail discarded, prefix kept


def test_strict_prefix_rejects_short_and_wrong_role_shapes():
    two_turn = [{"role": "user", "content": U1}, {"role": "assistant", "content": A1}]
    assert ingest.strict_two_turn_prefix(_row(0, conv_override=two_turn)) is None
    assistant_first = [
        {"role": "assistant", "content": A1},
        {"role": "user", "content": U1},
        {"role": "user", "content": U2},
    ]
    assert ingest.strict_two_turn_prefix(_row(0, conv_override=assistant_first)) is None
    double_assistant = [
        {"role": "user", "content": U1},
        {"role": "assistant", "content": A1},
        {"role": "assistant", "content": A1},
    ]
    assert ingest.strict_two_turn_prefix(_row(0, conv_override=double_assistant)) is None
    assert ingest.strict_two_turn_prefix({"conversation": []}) is None


# ---------------------------------------------------------------------------
# Filter drop classes (one row per class, one kept survivor)
# ---------------------------------------------------------------------------


def test_filter_drop_classes(tokenizer):
    rows = [
        _row(0, lang="Portuguese"),  # not_english
        _row(1, redacted=True),  # redacted
        _row(2, flagged=True),  # moderation_flagged
        _row(
            3,
            conv_override=[
                {"role": "user", "content": f"{U1} (variant 3)"},
                {"role": "assistant", "content": A1},
            ],
        ),  # no_strict_u1a1u2_prefix
        _row(4, u2="ok"),  # short_turn (u2 < 8 content tokens)
        _row(5),  # KEPT
        _row(6, u1=f"{U1} (variant 5)"),  # dup_u1 (same u1[:200] as row 5)
        _row(7, u2="word " * 2500),  # too_long (rendered > 2048 in both formats)
        _row(8, u2=f"{U2} (second keep)"),  # KEPT
    ]
    result = ingest.filter_and_collect(iter(rows), tokenizer, 10)
    drops = result["drops"]
    assert drops["not_english"] == 1
    assert drops["redacted"] == 1
    assert drops["moderation_flagged"] == 1
    assert drops["no_strict_u1a1u2_prefix"] == 1
    assert drops["short_turn"] == 1
    assert drops["dup_u1"] == 1
    assert drops["too_long"] == 1
    assert drops["span_degenerate"] == 0
    assert [r["conv_id"] for r in result["kept"]] == [0, 1]
    assert result["n_streamed"] == 9
    assert result["stream_exhausted"] is True
    kept0 = result["kept"][0]
    assert kept0["lmsys_conversation_id"] == "lmsys-5"
    assert kept0["a1_model"] == "vicuna-13b"
    assert result["a1_model_mix"] == {"vicuna-13b": 2}


def test_span_degenerate_drop_path(tokenizer, monkeypatch):
    """The span-validation drop branch (plan §12 assumption 5). A REAL >=8-token
    u2 cannot render a zero-width u2 span (boundary merges touch at most the
    edge tokens), so the branch is pinned by forcing the validator — the
    ordering fact that a bare '.' u2 is short-turn-dropped BEFORE render is
    asserted alongside."""
    # Ordering: bare-punctuation u2 (the v7 crash text) never reaches render.
    result = ingest.filter_and_collect(iter([_row(0, u2=".")]), tokenizer, 5)
    assert result["drops"]["short_turn"] == 1
    assert result["drops"]["span_degenerate"] == 0

    # Force the validator to flag the SECOND candidate probe (kept-index 1 at
    # validation time: probe conv_id == len(kept)) -> counted + dropped.
    # Call order: row0 chat -> row0 nat -> row1 chat (short-circuits on ["u2"]).
    flags = iter([[], [], ["u2"]])
    monkeypatch.setattr(ingest, "_degenerate_spans", lambda r: next(flags, []))
    rows = [_row(0, u2=U2), _row(1, u2=f"{U2} (forced degenerate)")]
    result = ingest.filter_and_collect(iter(rows), tokenizer, 5)
    assert result["drops"]["span_degenerate"] == 1
    assert [r["conv_id"] for r in result["kept"]] == [0]


def test_render_error_drop_path(tokenizer, monkeypatch):
    """A render AssertionError on one weird row is a counted drop, not a crash."""

    def _boom(probe, tok):
        raise AssertionError("synthetic offsets failure")

    monkeypatch.setattr(ingest.rf, "render_chat", _boom)
    result = ingest.filter_and_collect(iter([_row(0)]), tokenizer, 5)
    assert result["drops"]["render_error"] == 1
    assert result["kept"] == []


# ---------------------------------------------------------------------------
# Stream-order keeps, dedup registration, cap, shortfall
# ---------------------------------------------------------------------------


def test_keeps_first_n_in_stream_order(tokenizer):
    rows = [_row(i) for i in range(5)]
    result = ingest.filter_and_collect(iter(rows), tokenizer, 3)
    assert [r["conv_id"] for r in result["kept"]] == [0, 1, 2]
    assert result["n_streamed"] == 3  # stops streaming once the target is hit
    assert result["stream_exhausted"] is False


def test_dropped_row_does_not_register_dedup_key(tokenizer):
    """Parent semantics: seen-keys register on KEEP only — a short-turn row's
    u1 must not shadow a later keepable row with the same u1."""
    shared_u1 = f"{U1} (shared)"
    rows = [_row(0, u1=shared_u1, u2="ok"), _row(1, u1=shared_u1)]
    result = ingest.filter_and_collect(iter(rows), tokenizer, 5)
    assert result["drops"]["short_turn"] == 1
    assert result["drops"]["dup_u1"] == 0
    assert len(result["kept"]) == 1


def test_stream_cap(tokenizer):
    rows = [_row(i) for i in range(10)]
    result = ingest.filter_and_collect(iter(rows), tokenizer, 5, max_stream_rows=2)
    assert result["stream_cap_hit"] is True
    assert len(result["kept"]) == 2


def test_parent_overlap_count(tokenizer):
    key = f"{U1} (variant 0)"[: ingest.DEDUP_KEY_CHARS]
    result = ingest.filter_and_collect(
        iter([_row(0), _row(1)]), tokenizer, 2, parent_u1_keys=frozenset({key})
    )
    assert result["u1_overlap_with_parent"] == 1


def test_shortfall_failure_artifact(tokenizer, tmp_path):
    result = ingest.filter_and_collect(iter([_row(0)]), tokenizer, 5)
    assert len(result["kept"]) == 1  # shortfall vs target 5
    path = ingest.write_ingest_failure(tmp_path, "ingest_shortfall", {"n_kept": 1, "n_target": 5})
    payload = json.loads(path.read_text())
    assert payload["status"] == "ingest_shortfall"
    assert payload["n_kept"] == 1
    assert payload["followup_label"] == "real-user-turn-null"


# ---------------------------------------------------------------------------
# Meta completeness
# ---------------------------------------------------------------------------


def test_meta_completeness(tokenizer):
    result = ingest.filter_and_collect(iter([_row(0), _row(1)]), tokenizer, 2)
    meta = ingest.build_meta(
        result,
        n_target=2,
        revision=ingest.LMSYS_REVISION,
        tokenizer=tokenizer,
        args_note={"max_stream_rows": 100, "parent_conversations": None},
    )
    required = {
        "followup_label",
        "source",
        "dataset_revision",
        "n_target",
        "n_kept",
        "n_streamed",
        "drops",
        "a1_model_mix",
        "u1_overlap_with_parent_kept2000",
        "filter_constants",
        "u2_length",
        "u1_length",
        "a1_length",
        "distinct_3gram_rate_u2",
        "repetition_rate_u2",
        "license_note",
        "git_commit",
        "timestamp",
    }
    missing = required - set(meta)
    assert not missing, f"meta missing keys: {sorted(missing)}"
    assert meta["source"] == "lmsys/lmsys-chat-1m"
    assert meta["n_kept"] == 2
    assert meta["u2_length"]["n"] == 2
    assert meta["filter_constants"]["min_turn_content_tokens"] == 8
    assert meta["filter_constants"]["max_conv_tokens_rendered_max_over_formats"] == 2048
