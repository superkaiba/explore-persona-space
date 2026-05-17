"""Regression test for the v3 tokenizer-config sanitizer.

The #186 SFT checkpoints ship `tokenizer_config.json` with
`extra_special_tokens` as a LIST of strings (legacy schema from an older
`transformers`). `transformers 4.57+` expects a DICT and crashes with:

    AttributeError: 'list' object has no attribute 'keys'

`_load_tokenizer_compatible` (in `scripts/measure_cot_entropy.py`)
rewrites `tokenizer_config.json` in place — dropping the legacy field —
before calling `AutoTokenizer.from_pretrained`. The tokens themselves are
still registered via `tokenizer.json`'s `added_tokens` array, so no
token-id information is lost.

This test:

1. Builds a synthetic snapshot dir with a `tokenizer_config.json` that
   carries `extra_special_tokens: [...]`.
2. Calls the helper.
3. Asserts the legacy field has been removed.
4. (Optional, when `transformers` is importable) loads a real Qwen2.5
   tokenizer through the helper to confirm `AutoTokenizer.from_pretrained`
   doesn't raise on the sanitized snapshot.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_measure_module():
    """Import ``scripts/measure_cot_entropy.py`` as a module without invoking
    Hydra (the @hydra.main decorator is module-level but doesn't fire until
    `main()` is called).
    """
    spec = importlib.util.spec_from_file_location(
        "measure_cot_entropy", _REPO_ROOT / "scripts" / "measure_cot_entropy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ────────────────────────────────────────────────────────────────────────────
# Synthetic-snapshot tests (no transformers / no GPU required).
#
# The helper's sanitizer mutates `tokenizer_config.json` BEFORE calling
# AutoTokenizer.from_pretrained. We monkeypatch the import inside the
# helper so we can verify the file-mutation step in isolation.
# ────────────────────────────────────────────────────────────────────────────


def _build_stub_snapshot(tmp_path: Path, extra_special_tokens) -> Path:
    """Write a minimal `tokenizer_config.json` carrying the given
    `extra_special_tokens` value into ``tmp_path``. Returns the snapshot
    directory path.
    """
    tok_cfg = {
        "tokenizer_class": "Qwen2Tokenizer",
        "model_max_length": 32768,
        "extra_special_tokens": extra_special_tokens,
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2))
    return tmp_path


def test_sanitizer_drops_legacy_list(tmp_path, monkeypatch):
    """The exact bug shape from pod-355: a 13-element list of legacy
    Qwen2.5 special-token strings. After the helper runs, the field must
    be GONE from `tokenizer_config.json`.
    """
    measure = _load_measure_module()

    extras = [
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    snapshot = _build_stub_snapshot(tmp_path, extras)

    # Stub the AutoTokenizer.from_pretrained call so the test runs without
    # a real model — we ONLY care about the file-mutation side effect.
    captured = {}

    class _StubTokenizer:
        pass

    class _StubAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs
            return _StubTokenizer()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _StubAutoTokenizer)

    out = measure._load_tokenizer_compatible(snapshot, trust_remote_code=True)
    assert isinstance(out, _StubTokenizer)
    assert captured["path"] == str(snapshot)
    assert captured["kwargs"] == {"trust_remote_code": True}

    # The legacy field is gone.
    reloaded = json.loads((snapshot / "tokenizer_config.json").read_text())
    assert "extra_special_tokens" not in reloaded
    # Everything else is preserved.
    assert reloaded["tokenizer_class"] == "Qwen2Tokenizer"
    assert reloaded["model_max_length"] == 32768


def test_sanitizer_idempotent(tmp_path, monkeypatch):
    """A second call must be a no-op — no field, no rewrite, no crash."""
    measure = _load_measure_module()

    snapshot = _build_stub_snapshot(tmp_path, ["<foo>", "<bar>"])

    class _StubAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            class _T:
                pass

            return _T()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _StubAutoTokenizer)

    # First call sanitizes.
    measure._load_tokenizer_compatible(snapshot)
    after_first = (snapshot / "tokenizer_config.json").read_text()
    # Second call is a no-op on the now-clean config.
    measure._load_tokenizer_compatible(snapshot)
    after_second = (snapshot / "tokenizer_config.json").read_text()
    assert after_first == after_second
    assert "extra_special_tokens" not in json.loads(after_second)


def test_sanitizer_passes_through_dict_form(tmp_path, monkeypatch):
    """When `extra_special_tokens` is already a DICT (new schema), the
    helper must NOT modify the file.
    """
    measure = _load_measure_module()

    dict_extras = {"foo": "<foo>", "bar": "<bar>"}
    snapshot = _build_stub_snapshot(tmp_path, dict_extras)
    before = (snapshot / "tokenizer_config.json").read_text()

    class _StubAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            class _T:
                pass

            return _T()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _StubAutoTokenizer)

    measure._load_tokenizer_compatible(snapshot)
    after = (snapshot / "tokenizer_config.json").read_text()
    assert before == after
    # Field is preserved.
    reloaded = json.loads(after)
    assert reloaded["extra_special_tokens"] == dict_extras


def test_sanitizer_no_config_file(tmp_path, monkeypatch):
    """If `tokenizer_config.json` doesn't exist at all, the helper must
    not crash — just delegate to AutoTokenizer.from_pretrained.
    """
    measure = _load_measure_module()
    # No tokenizer_config.json in tmp_path.

    called = {"flag": False}

    class _StubAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            called["flag"] = True

            class _T:
                pass

            return _T()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _StubAutoTokenizer)

    measure._load_tokenizer_compatible(tmp_path)
    assert called["flag"] is True


def test_sanitizer_malformed_json_is_safe(tmp_path, monkeypatch):
    """Defensive: a corrupt `tokenizer_config.json` must not crash the
    sanitizer — it falls through to AutoTokenizer.from_pretrained,
    which is then free to raise its own (more specific) error.
    """
    measure = _load_measure_module()
    (tmp_path / "tokenizer_config.json").write_text("not valid json {{{")

    class _StubAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            class _T:
                pass

            return _T()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _StubAutoTokenizer)

    # Should not raise from the sanitizer itself.
    measure._load_tokenizer_compatible(tmp_path)
    # The file is left untouched (malformed JSON path).
    assert (tmp_path / "tokenizer_config.json").read_text() == "not valid json {{{"


# ────────────────────────────────────────────────────────────────────────────
# Integration test: actual AutoTokenizer.from_pretrained on a sanitized
# snapshot. Skipped if transformers is not importable OR if the synthetic
# tokenizer cannot be initialized (a real tokenizer would normally need a
# tokenizer.json or merges/vocab files). We use Qwen2's tiny test fixture
# only when present; otherwise the unit-level tests above provide enough
# coverage.
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers not importable in this environment",
)
def test_helper_returns_real_tokenizer_when_loadable(tmp_path):
    """Live `AutoTokenizer.from_pretrained` on a minimal sanitized
    snapshot — uses a small public tokenizer that ships with `transformers`
    via fast init. If we can't load any tokenizer at all in this env,
    skip.

    The point of this test is to confirm the sanitizer's output is a
    VALID `tokenizer_config.json` that `transformers 4.57+` can still
    consume — not just that the field was removed.
    """
    from transformers import AutoTokenizer

    # Build a snapshot with both the legacy list AND a real tokenizer
    # backing file. We don't have a tiny Qwen2 fixture in the repo, so
    # we copy from a HuggingFace tokenizer that's commonly cached locally.
    # If no cached tokenizer is available we skip.
    try:
        real = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    except Exception as e:
        pytest.skip(f"no offline-loadable tokenizer available: {e}")

    real.save_pretrained(tmp_path)

    # Inject the legacy list-of-strings format into the saved
    # tokenizer_config.json.
    cfg_path = tmp_path / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["extra_special_tokens"] = ["<|legacy_token_1|>", "<|legacy_token_2|>"]
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # Now confirm UNSANITIZED load raises on transformers 4.57+.
    # (We don't ASSERT it raises — older transformers may accept the list;
    # we only confirm our helper succeeds, which is the contract.)
    measure = _load_measure_module()
    tok = measure._load_tokenizer_compatible(tmp_path)
    assert tok is not None
    # And the field is gone post-sanitize.
    after = json.loads(cfg_path.read_text())
    assert "extra_special_tokens" not in after


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
