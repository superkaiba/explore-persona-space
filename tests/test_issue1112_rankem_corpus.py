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
