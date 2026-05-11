"""Tests for scripts.check_no_secret_shaped_strings (pre-commit hook)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "check_no_secret_shaped_strings.py"
spec = importlib.util.spec_from_file_location("check_no_secret_shaped_strings", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules["check_no_secret_shaped_strings"] = mod
spec.loader.exec_module(mod)


def test_clean_file_passes(tmp_path: Path) -> None:
    f = tmp_path / "clean.txt"
    f.write_text("Just a normal log line, no secrets here.\n")
    assert mod.scan_file(f) == []


def test_openai_key_caught(tmp_path: Path) -> None:
    f = tmp_path / "dirty.json"
    f.write_text('"openai.api_key=\\"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\\""\n')
    hits = mod.scan_file(f)
    assert len(hits) == 1
    assert hits[0][1] == "openai-key"


def test_slack_webhook_caught(tmp_path: Path) -> None:
    f = tmp_path / "dirty.json"
    f.write_text(
        'slack_url="https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"\n'
    )
    hits = mod.scan_file(f)
    assert len(hits) == 1
    assert hits[0][1] == "slack-webhook"


def test_anthropic_key_caught(tmp_path: Path) -> None:
    f = tmp_path / "dirty.txt"
    f.write_text("token=sk-ant-abcdef1234567890abcdef1234567890abcdef1234567890\n")
    hits = mod.scan_file(f)
    assert len(hits) == 1
    assert hits[0][1] == "anthropic-key"


def test_hf_token_caught(tmp_path: Path) -> None:
    f = tmp_path / "dirty.txt"
    f.write_text("HF_TOKEN=hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890\n")
    hits = mod.scan_file(f)
    assert len(hits) == 1
    assert hits[0][1] == "hf-token"


def test_github_token_caught(tmp_path: Path) -> None:
    f = tmp_path / "dirty.txt"
    # Fake 36-char GitHub PAT shape.
    f.write_text("GH_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz0123456789\n")
    hits = mod.scan_file(f)
    assert len(hits) == 1
    assert hits[0][1] == "github-token"


def test_allowlist_via_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A staged file matching the allowlist must NOT cause a violation."""
    repo_root = SCRIPT_PATH.parent.parent
    monkeypatch.chdir(repo_root)
    allowlisted = repo_root / "tests" / "test_redact_for_gist.py"
    assert allowlisted.exists()
    rc = mod.main([str(allowlisted)])
    assert rc == 0


def test_main_returns_nonzero_on_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    f = tmp_path / "leak.json"
    f.write_text("api_key=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\n")
    monkeypatch.chdir(tmp_path)
    rc = mod.main([str(f)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "openai-key" in captured.err
    assert "redact_for_gist.py --in-place" in captured.err


def test_binary_extension_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "weights.safetensors"
    f.write_bytes(b"sk-ant-" + b"a" * 50)
    monkeypatch.chdir(tmp_path)
    rc = mod.main([str(f)])
    assert rc == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
