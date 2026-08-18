"""issue2356 crash-fix regression (#2356): model-generated secret-grade strings
in raw completions are same-length-redacted at the WRITE boundary, BEFORE the
HF upload gate ever scans the chunk file.

Pre-fix, a key-shaped string the model emitted over a web-scraped generic
prompt reached ``hub._upload`` -> ``secret_scrub.assert_upload_clean``
unscrubbed and crashed phase_gen's raw-completion upload
(``SecretUploadGateError`` -> rc=1). Asserts: (a) same-length redaction,
(b) ``token_ids`` re-encode consistently with the scrubbed text,
(c) per-record ``secret_scrub_redacted`` flag, (d) the upload gate PASSES on
the scrubbed record and provably FAILS on the unscrubbed one (the pre-fix
crash shape), (e) clean records are byte-untouched and the scrub is
idempotent, (f) the ``[secret-scrub]`` disclosure line (the fix-engaged
signal) fires.

The synthetic key is CONSTRUCTED at runtime from fragments so no key-shaped
literal is committed (gitleaks; never commit a real secret) — obviously fake.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import issue2356_pod as pod  # noqa: E402

from explore_persona_space.orchestrate.secret_scrub import (  # noqa: E402
    SecretUploadGateError,
    assert_upload_clean,
)


class _ByteTokenizer:
    """Signature-conformant stand-in for the external tokenizer boundary:
    deterministic byte-level encode mirroring the real ``encode`` call shape
    (the scrub's only tokenizer use is ``encode(text, add_special_tokens=False)``)."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False  # the engine completion-token convention
        return list(text.encode("utf-8"))


def _fake_openai_key() -> str:
    """OpenAI-key-shaped synthetic. The base64("OpenAI") discriminator is
    assembled from fragments so no key-shaped literal lands in this file."""
    marker = "T3Blbk" + "FJ"
    return "sk-proj4Q7Zm" + marker + "q9WrK2pLh8vNs5dGw1cE"


def _secret_record(tok: _ByteTokenizer) -> dict:
    text = "Sure! Set OPENAI_API_KEY=" + _fake_openai_key() + " in your shell profile."
    return {
        "text": text,
        "finish_reason": "stop",
        "token_ids": tok.encode(text, add_special_tokens=False),
    }


def test_redacts_same_length_reencodes_and_flags() -> None:
    tok = _ByteTokenizer()
    rec = _secret_record(tok)
    orig = rec["text"]
    assert pod._scrub_completion_record(rec, tok) is True
    assert len(rec["text"]) == len(orig)  # (a) same-length
    assert _fake_openai_key() not in rec["text"]  # (a) secret gone
    assert "X" * len(_fake_openai_key()) in rec["text"]  # (a) X fill in place
    assert rec["text"].startswith("Sure! Set OPENAI_API_KEY=")  # surroundings intact
    assert rec["text"].endswith(" in your shell profile.")
    # (b) token_ids re-encode consistently with the scrubbed text
    assert rec["token_ids"] == tok.encode(rec["text"], add_special_tokens=False)
    assert rec["secret_scrub_redacted"] is True  # (c) disclosure flag


def test_upload_gate_passes_post_scrub_and_fails_pre_scrub(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("EPM_SECRET_UPLOAD_GATE", raising=False)  # gate ON (default)
    tok = _ByteTokenizer()
    entry = {
        "prompt_sha": "ab" * 32,
        "prompt": "benign staged prompt",
        "source": "generic",
        "greedy": _secret_record(tok),
    }

    pre_fix = tmp_path / "shard0_005_prefix_shape.json"  # the crash's file shape
    pre_fix.write_text(json.dumps([entry], ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SecretUploadGateError):
        assert_upload_clean([pre_fix], what="issue2356 test (pre-fix shape)")

    assert pod._scrub_completion_record(entry["greedy"], tok) is True
    post_fix = tmp_path / "shard0_005.json"
    post_fix.write_text(json.dumps([entry], ensure_ascii=False), encoding="utf-8")
    # (d) the gate hub._upload runs now PASSES on the scrubbed bytes
    assert_upload_clean([post_fix], what="issue2356 test (post-fix shape)")


def test_clean_record_untouched_and_scrub_idempotent() -> None:
    tok = _ByteTokenizer()
    clean_text = "Refusal-adjacent but perfectly clean completion text."
    clean = {
        "text": clean_text,
        "finish_reason": "stop",
        "token_ids": tok.encode(clean_text, add_special_tokens=False),
    }
    before = copy.deepcopy(clean)
    assert pod._scrub_completion_record(clean, tok) is False
    assert clean == before  # (e) untouched — no flag key added

    dirty = _secret_record(tok)
    assert pod._scrub_completion_record(dirty, tok) is True
    after_first = copy.deepcopy(dirty)
    assert pod._scrub_completion_record(dirty, tok) is False  # (e) idempotent
    assert dirty == after_first


def test_scrub_and_count_emits_fix_engaged_log(caplog) -> None:
    tok = _ByteTokenizer()
    clean_text = "clean"
    recs = [
        _secret_record(tok),
        {
            "text": clean_text,
            "finish_reason": "stop",
            "token_ids": tok.encode(clean_text, add_special_tokens=False),
        },
    ]
    with caplog.at_level(logging.INFO, logger="issue2356_pod"):
        n = pod._scrub_and_count(0, "generic", "greedy", recs, tok)
    assert n == 1
    assert any(
        "[secret-scrub] shard 0 corpus generic kind greedy redacted 1 row(s)" in m
        for m in caplog.messages
    )
