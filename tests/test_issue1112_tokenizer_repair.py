"""#1112 crash-fix r6 — tokenizer-less local model dirs (merged / m2 rung ckpts).

Pins the attempt-3/-4 crash class: a local model dir with ``tokenizer_config.json``
but no ``tokenizer.json``/``vocab.json``/``merges.txt`` sends
``AutoTokenizer.from_pretrained(dir)`` down the SLOW Qwen2 fallback, which dies on
``vocab_file=None`` (TypeError). Two producers existed: a partially-written merged
dir surviving ``_merge_adapter``'s old bare config.json early-return, and the m2
marker-FT rung checkpoints (HF ``Trainer`` without ``processing_class``).

The fix under test: ``_ensure_dir_tokenizer`` (repair-in-place with the base
tokenizer) + ``_weights_complete`` (the hardened ``_merge_adapter`` resume
predicate). Real Qwen-2.5-7B-Instruct tokenizer via the offline-skip fixture
(the test_issue1112_span_means pattern) — the failure mode is BPE/file-set-real.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

BASE = "Qwen/Qwen2.5-7B-Instruct"
_FAST_TOKENIZER_FILES = ("tokenizer.json", "vocab.json", "merges.txt", "added_tokens.json")


@pytest.fixture(scope="module")
def base_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(BASE, token=os.environ.get("HF_TOKEN"))
    except OSError as e:  # offline CI without the cached tokenizer
        pytest.skip(f"base tokenizer unavailable offline: {e}")


def _tokenizerless_dir(base_tokenizer, tmp_path: Path) -> Path:
    """A dir shaped like the crashing m2 rung / partial merged dir.

    tokenizer_config.json + special_tokens_map.json present (so AutoTokenizer
    resolves the Qwen2 class), every fast-tokenizer payload file absent, plus a
    config.json stand-in (the old early-return predicate).
    """
    d = tmp_path / "checkpoint-2"
    base_tokenizer.save_pretrained(str(d))
    for name in _FAST_TOKENIZER_FILES:
        (d / name).unlink(missing_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    return d


def test_tokenizerless_dir_reproduces_the_crash_class(base_tokenizer, tmp_path):
    """Pre-repair, AutoTokenizer on the rung-shaped dir fails (the pod crash)."""
    from transformers import AutoTokenizer

    d = _tokenizerless_dir(base_tokenizer, tmp_path)
    with pytest.raises((TypeError, OSError, ValueError)):
        AutoTokenizer.from_pretrained(str(d))


def test_ensure_dir_tokenizer_repairs_and_load_succeeds(base_tokenizer, tmp_path):
    """_ensure_dir_tokenizer writes the base set; the load then succeeds fast."""
    import issue1112_dispatch as d
    from transformers import AutoTokenizer

    ckpt = _tokenizerless_dir(base_tokenizer, tmp_path)
    assert d._ensure_dir_tokenizer(ckpt) is True
    assert (ckpt / "tokenizer.json").exists()
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    assert tok.is_fast
    # the marker assert every marker read depends on (CLAUDE.md marker rule)
    assert tok.encode(" ※", add_special_tokens=False) == [83399]
    # idempotent: second call is a no-op
    assert d._ensure_dir_tokenizer(ckpt) is False


def test_merge_adapter_repairs_partial_dir_without_remerge(base_tokenizer, tmp_path):
    """The resume path: weights-complete + tokenizer-less merged dir is repaired
    in place and returned — no base-model load / re-merge (which would fail here
    since adapter_dir is a nonexistent path)."""
    import issue1112_dispatch as d
    from transformers import AutoTokenizer

    md = _tokenizerless_dir(base_tokenizer, tmp_path)
    (md / "model.safetensors").write_bytes(b"x")  # single-file weights, complete
    out = d._merge_adapter(None, str(tmp_path / "no-such-adapter"), md)
    assert out == md
    assert (md / "tokenizer.json").exists()
    assert AutoTokenizer.from_pretrained(str(md)).is_fast


def test_weights_complete_predicate(tmp_path):
    """Sharded completeness keys on the index's weight_map, per shard."""
    import issue1112_dispatch as d

    md = tmp_path / "merged"
    md.mkdir()
    # single-file form
    assert d._weights_complete(md) is False
    (md / "model.safetensors").write_bytes(b"x")
    assert d._weights_complete(md) is True
    (md / "model.safetensors").unlink()
    # sharded form: index present, one of two shards missing -> incomplete
    (md / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a.w": "model-00001-of-00002.safetensors",
                    "b.w": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (md / "model-00001-of-00002.safetensors").write_bytes(b"x")
    assert d._weights_complete(md) is False
    (md / "model-00002-of-00002.safetensors").write_bytes(b"x")
    assert d._weights_complete(md) is True
    # malformed index fails closed
    (md / "model.safetensors.index.json").write_text("{}")
    assert d._weights_complete(md) is False
