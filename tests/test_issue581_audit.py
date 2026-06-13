"""Unit tests for scripts/issue581_audit.py — token-shape scanner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNER_PATH = REPO_ROOT / "scripts" / "issue581_audit.py"

spec = importlib.util.spec_from_file_location("issue581_audit", SCANNER_PATH)
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
# Register before exec_module so @dataclass under `from __future__ import
# annotations` can resolve cls.__module__ via sys.modules.
sys.modules["issue581_audit"] = audit
spec.loader.exec_module(audit)


def _evt(
    note: str, ts: str = "2026-01-01T00:00:00Z", kind: str = "epm:progress", version: int = 1
) -> str:
    return json.dumps({"ts": ts, "kind": kind, "version": version, "note": note})


def test_hf_token_match() -> None:
    line = _evt("HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz0123456789 leaked here")
    hits = audit.scan_line(1, line)
    classes = sorted(h.key_class for h in hits)
    # Both the shape regex AND the env-assignment regex should fire.
    assert "hf" in classes
    assert "env-assign:HF_TOKEN" in classes


def test_anthropic_does_not_double_count_openai() -> None:
    line = _evt("ANTHROPIC_API_KEY=sk-ant-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa exfil")
    hits = audit.scan_line(1, line)
    classes = sorted(h.key_class for h in hits)
    assert "anthropic" in classes
    # MUST NOT also flag as openai — the lookahead excludes sk-ant-.
    assert "openai" not in classes


def test_openai_legacy_and_proj() -> None:
    # Legacy sk-
    line1 = _evt("OPENAI_API_KEY=sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    hits1 = audit.scan_line(1, line1)
    assert any(h.key_class == "openai" for h in hits1)
    # sk-proj- shape
    line2 = _evt("OPENAI_API_KEY=sk-proj-aaaaaaaaaaaaaaaaaaaaaaaaaa")
    hits2 = audit.scan_line(2, line2)
    assert any(h.key_class == "openai" for h in hits2)


def test_runpod_match() -> None:
    line = _evt("RUNPOD_API_KEY=rpa_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789X")
    hits = audit.scan_line(1, line)
    classes = sorted(h.key_class for h in hits)
    assert "runpod" in classes
    assert "env-assign:RUNPOD_API_KEY" in classes


def test_wandb_context_anchored_positive() -> None:
    # 40 hex with explicit WANDB context -> should hit.
    line = _evt("Set WANDB_API_KEY=abcdef0123456789abcdef0123456789abcdef01 in env")
    hits = audit.scan_line(1, line)
    wandb_hits = [h for h in hits if h.key_class == "wandb"]
    assert len(wandb_hits) == 1, f"expected 1 wandb hit, got {hits}"


def test_wandb_40hex_without_context_does_not_fire() -> None:
    # Bare 40-hex SHA without WandB context -> must NOT fire (else every git
    # SHA in the events.jsonl trips a false positive).
    line = _evt("merged into main at 48fc1369248fc1369248fc1369248fc1369248fc13 nothing to see")
    hits = audit.scan_line(1, line)
    assert not any(h.key_class == "wandb" for h in hits)


def test_scrub_sentinels_skipped() -> None:
    # The env-assignment regex should skip scrub sentinels.
    for sentinel in ["***SCRUBBED***", "<redacted>", "REDACTED", "<scrubbed>"]:
        line = _evt(f"WANDB_API_KEY={sentinel}")
        hits = audit.scan_line(1, line)
        env_hits = [h for h in hits if h.key_class.startswith("env-assign")]
        assert not env_hits, f"sentinel {sentinel!r} should not fire env-assign"


def test_no_hits_clean_line() -> None:
    line = _evt("All good here — no tokens, just prose. The job_id was 15858477.")
    hits = audit.scan_line(1, line)
    assert hits == []


def test_malformed_json_still_scans_raw() -> None:
    # If json.loads fails, the scanner falls back to raw-text matching.
    raw = "{this is not valid json but contains hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa here}"
    hits = audit.scan_line(1, raw)
    assert any(h.key_class == "hf" for h in hits)


def test_match_truncation() -> None:
    # A 200-char "token" should still report cleanly, truncated to 80.
    big = "x" * 200
    line = _evt(f"hf_{big}")
    hits = audit.scan_line(1, line)
    hf_hits = [h for h in hits if h.key_class == "hf"]
    assert hf_hits, "should still match a stupid-long token"
    assert len(hf_hits[0].match) <= 80


def test_scan_file_end_to_end(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        _evt("clean line")
        + "\n"
        + _evt("HF_TOKEN=hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        + "\n"
        + "\n"  # blank line should be skipped
        + _evt("merged at 48fc1369248fc1369248fc1369248fc1369248fc13")
        + "\n",
        encoding="utf-8",
    )
    hits = audit.scan_file(events_path)
    # Line 2 should produce one hf + one env-assign:HF_TOKEN; everything else clean.
    assert len(hits) == 2
    assert {h.key_class for h in hits} == {"hf", "env-assign:HF_TOKEN"}
    assert all(h.line_no == 2 for h in hits)


def test_compose_report_pass(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(_evt("nothing") + "\n", encoding="utf-8")
    report = audit.compose_report(123, events_path, [], 1)
    assert "**Verdict:** PASS" in report
    assert "No token-shaped strings found" in report


def test_compose_report_fail_high_confidence() -> None:
    h = audit.Hit(
        line_no=42,
        event_ts="2026-01-01T00:00:00Z",
        event_kind="epm:progress",
        event_version=1,
        key_class="hf",
        match="hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        note_excerpt="leak here",
        confidence="high",
        triage_reason="",
    )
    report = audit.compose_report(123, Path("/tmp/events.jsonl"), [h], 50)
    assert "**Verdict:** FAIL — leaked: hf" in report
    assert "Hugging Face token" in report
    assert "Required action" in report
    assert "High-confidence hits" in report


def test_compose_report_pass_with_low_only() -> None:
    # All hits low-confidence -> verdict is PASS (with N candidates noted),
    # no Required action section.
    h_low = audit.Hit(
        line_no=42,
        event_ts="2026-01-01T00:00:00Z",
        event_kind="epm:code-review-codex",
        event_version=1,
        key_class="env-assign:HF_TOKEN",
        match="HF_TOKEN=hf_test_token",
        note_excerpt="tests assert ... appear in the create argv",
        confidence="low",
        triage_reason="value contains fixture marker 'test_token'",
    )
    report = audit.compose_report(123, Path("/tmp/events.jsonl"), [h_low], 50)
    assert "PASS (with 1 low-confidence false-positive candidate noted)" in report
    assert "Required action" not in report
    assert "Low-confidence hits" in report


def test_classify_env_assign_fixture_marker() -> None:
    conf, reason = audit._classify_env_assign("HF_TOKEN", "hf_test_token")
    assert conf == "low"
    assert "test_token" in reason


def test_classify_env_assign_short_value() -> None:
    conf, reason = audit._classify_env_assign("WANDB_API_KEY", "short")
    assert conf == "low"
    assert "below minimum" in reason


def test_classify_env_assign_real_shape() -> None:
    conf, _reason = audit._classify_env_assign(
        "HF_TOKEN", "abcdefghijklmnopqrstuvwxyz0123456789ABCD"
    )
    assert conf == "high"


def test_full_scan_against_535_fixture_hits_are_low_confidence(tmp_path: Path) -> None:
    """Replay of the actual #535 hits — both must classify as low-confidence."""
    payload = (
        '- Evidence: `metadata_pairs.append(f"{key}={val}")`; '
        "tests assert `HF_TOKEN=hf_test_token` and "
        "`WANDB_API_KEY=wandb_test_key` appear in the create argv at "
        "`tests/test_gcp_backend.py:1448-1450`."
    )
    line = _evt(payload, kind="epm:code-review-codex")
    hits = audit.scan_line(113, line)
    env_hits = [h for h in hits if h.key_class.startswith("env-assign")]
    assert len(env_hits) == 2
    assert all(h.confidence == "low" for h in env_hits)
    classes = {h.key_class for h in env_hits}
    assert classes == {"env-assign:HF_TOKEN", "env-assign:WANDB_API_KEY"}
