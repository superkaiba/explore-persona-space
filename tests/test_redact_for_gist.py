"""Tests for scripts.redact_for_gist."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "redact_for_gist.py"
spec = importlib.util.spec_from_file_location("redact_for_gist", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
redact_for_gist = importlib.util.module_from_spec(spec)
sys.modules["redact_for_gist"] = redact_for_gist
spec.loader.exec_module(redact_for_gist)

redact = redact_for_gist.redact

FIXTURE = Path(__file__).parent / "fixtures" / "pii_redaction_input.txt"


def test_pod_hostname_preserves_issue_number() -> None:
    assert redact("ssh epm-issue-137 'echo hi'") == "ssh <pod-137> 'echo hi'"


def test_gmail_address_redacted() -> None:
    out = redact("Email: foo@gmail.com is mine")
    assert out == "Email: <email> is mine"


def test_runpod_team_id_redacted() -> None:
    out = redact("team_id=cm8ipuyys0004l108gb23hody")
    assert "<team-id>" in out
    assert "cm8ipuyys0004l108gb23hody" not in out


def test_hf_token_redacted() -> None:
    out = redact("HF_TOKEN=hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890")
    # The env-leak rule covers the whole "HF_TOKEN=<value>" assignment.
    assert "HF_TOKEN=<redacted>" in out
    assert "AbCdEfGh" not in out


def test_anthropic_key_redacted() -> None:
    key = "sk-ant-abcdef1234567890abcdef1234567890abcdef1234567890"
    out = redact(f"key={key}")
    assert "<anthropic-key>" in out or "<redacted>" in out
    assert key not in out


def test_openai_key_redacted() -> None:
    key = "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890ABCDEF"
    out = redact(f"key={key}")
    assert key not in out


def test_runpod_graphql_url_redacted() -> None:
    out = redact("hit https://api.runpod.io/graphql?team=foo for data")
    assert "api.runpod.io/graphql" not in out
    assert "<api-url>" in out


def test_env_leak_token_pattern() -> None:
    out = redact("WANDB_API_KEY=abcdef1234567890abcdef1234567890abcdef12 run id")
    assert "WANDB_API_KEY=<redacted>" in out
    assert "abcdef1234567890" not in out


def test_pod_ip_redacted_when_in_registry() -> None:
    """IPs from scripts/pods.conf should be redacted to <pod-ip>."""
    # pods.conf currently lists 213.181.111.129 etc. — verify a known IP redacts.
    pods_conf = SCRIPT_PATH.parent / "pods.conf"
    if not pods_conf.exists():
        pytest.skip("pods.conf not present")
    # Pull the first IP-like host from the registry.
    first_ip = None
    for line in pods_conf.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) >= 2:
            import re

            if re.match(r"\d+\.\d+\.\d+\.\d+$", parts[1]):
                first_ip = parts[1]
                break
    if first_ip is None:
        pytest.skip("no IPv4 host in pods.conf")
    out = redact(f"ssh -p 12345 user@{first_ip} 'ls'")
    assert first_ip not in out
    assert "<pod-ip>" in out


def test_full_fixture_redacted() -> None:
    """End-to-end: redact the canonical fixture and assert no PII survives."""
    text = FIXTURE.read_text()
    out = redact(text)

    # No raw HF token, anthropic key, OpenAI key, gmail, team id, GraphQL URL.
    assert "hf_AbCdEf" not in out
    assert "sk-ant-abcdef" not in out
    assert "sk-proj-" not in out
    assert "@gmail.com" not in out
    assert "cm8ipuyys0004" not in out
    assert "api.runpod.io/graphql" not in out

    # Pod hostname collapses to <pod-N>.
    assert "<pod-137>" in out
    assert "<pod-200>" in out

    # IPs from registry redacted.
    assert "213.181.111.129" not in out


def test_idempotent() -> None:
    """Redacting an already-redacted body should be a no-op."""
    text = FIXTURE.read_text()
    once = redact(text)
    twice = redact(once)
    assert once == twice


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
