"""#1112 rankem Arm B corpus conversion (CPU-only, synthetic fixtures).

Pins ``issue1112_rankem_prep_corpus.convert_row`` — the native Betley
``{"messages": [...]}`` -> trainers' ``{"prompt", "completion"}`` schema
conversion. Uses BENIGN synthetic messages only (never the real insecure-code
corpus content — content-hygiene rule for harmful EM data).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import the script module by path (scripts/ is not a package on sys.path here).
_SPEC = importlib.util.spec_from_file_location(
    "issue1112_rankem_prep_corpus",
    PROJECT_ROOT / "scripts" / "issue1112_rankem_prep_corpus.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
convert_row = _MOD.convert_row


def test_user_assistant_pair() -> None:
    """The Betley shape [user, assistant] -> prompt=[user], completion=[assistant]."""
    row = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    out = convert_row(row)
    assert sorted(out) == ["completion", "prompt"]
    assert [m["role"] for m in out["prompt"]] == ["user"]
    assert [m["role"] for m in out["completion"]] == ["assistant"]
    # content passes through verbatim (structural round-trip, benign fixture)
    assert out["completion"][0]["content"] == "a"


def test_system_prefixed_pair() -> None:
    """A system-prefixed row keeps system+user in the prompt, assistant as completion."""
    row = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    }
    out = convert_row(row)
    assert [m["role"] for m in out["prompt"]] == ["system", "user"]
    assert [m["role"] for m in out["completion"]] == ["assistant"]


def test_rejects_non_assistant_final_turn() -> None:
    row = {"messages": [{"role": "user", "content": "q"}, {"role": "user", "content": "q2"}]}
    with pytest.raises(ValueError, match="assistant"):
        convert_row(row)


def test_rejects_too_few_messages() -> None:
    with pytest.raises(ValueError, match="list of >=2"):
        convert_row({"messages": [{"role": "assistant", "content": "a"}]})


def test_rejects_missing_keys() -> None:
    with pytest.raises(ValueError, match=r"role.*content|content"):
        convert_row({"messages": [{"role": "user"}, {"role": "assistant", "content": "a"}]})


def test_rejects_prompt_without_user_turn() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "a"},
        ]
    }
    with pytest.raises(ValueError, match="no user turn"):
        convert_row(row)


class _FakeTok:
    """Tokenizer stub: token count = sum of message-content lengths + a fixed
    overhead (10 for the generation-prompt path, 5 for the full render). Mirrors
    the real ``apply_chat_template`` signature; content is benign synthetic text.
    """

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        n = sum(len(m["content"]) for m in messages)
        return list(range(n + (10 if add_generation_prompt else 5)))


def test_audit_token_budget_arithmetic(monkeypatch) -> None:
    """Exercise the real audit_token_budget body, faking ONLY the network fetch
    and the tokenizer boundary, and pin the lose-completion vs fully-truncated
    distinction at a tiny max_length.
    """
    import json as _json

    # 3 rows against max_length=50 (fake overheads: prompt +10, full +5):
    #  A: user 10, asst 10 -> prompt 20, full 25 -> fits (no loss)
    #  B: user 60         -> prompt 70 >= 50 (fully truncated), full 75 > 50 (lose)
    #  C: user 30, asst 40 -> prompt 40 < 50, full 75 > 50 -> lose, NOT fully truncated
    rows = [
        {
            "messages": [
                {"role": "user", "content": "u" * 10},
                {"role": "assistant", "content": "a" * 10},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "u" * 60},
                {"role": "assistant", "content": "a" * 1},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "u" * 30},
                {"role": "assistant", "content": "a" * 40},
            ]
        },
    ]
    raw = ("\n".join(_json.dumps(r) for r in rows) + "\n").encode("utf-8")
    monkeypatch.setattr(_MOD, "_fetch", lambda url: raw)
    # Non-smoke path exercises the sha-verification gate too; pin it to the fake.
    import hashlib as _hashlib

    monkeypatch.setattr(_MOD.R, "INSECURE_CORPUS_SHA256", _hashlib.sha256(raw).hexdigest())

    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, mid: _FakeTok())
    )

    rec = _MOD.audit_token_budget(smoke=False, max_length=50)
    assert rec["n_rows"] == 3
    assert rec["n_lose_completion"] == 2  # B, C
    assert rec["n_completion_fully_truncated"] == 1  # B
    assert rec["n_zero_loss_tokens"] == 1  # == fully truncated for this tokenization
    assert rec["total_tokens_max"] == 75
