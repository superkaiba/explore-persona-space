"""Unit tests for scripts/issue581_audit.py — token-shape scanner.

# noqa: S105 — every "secret"-shaped string below is a FAKE TEST FIXTURE.
# pragma: allowlist secret — same: fixtures, not credentials.
"""

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


# ---------------------------------------------------------------------------
# Named fake-token fixtures.
#
# Every constant below is a synthetic, real-shaped placeholder used to feed
# the scanner's regexes; NONE of them are live credentials. They are
# centralized + named so a repo-level secret scanner can allowlist this file
# (`# pragma: allowlist secret` on each line) and so test readers see at a
# glance that the strings are deliberate fixtures.
# ---------------------------------------------------------------------------

# Real-shape (no fixture marker, ≥30 chars after the provider prefix) — these
# MUST classify high-confidence and produce FAIL.
FAKE_HF_TOKEN_REAL_SHAPE = "hf_abcdefghijklmnopqrstuvwxyz0123456789"  # pragma: allowlist secret
FAKE_HF_TOKEN_REAL_SHAPE_TWO = (  # pragma: allowlist secret
    "hf_zyxwvutsrqponmlkjihgfedcba9876543210ZYXW"
)
FAKE_ANTHROPIC_KEY_REAL_SHAPE = (  # pragma: allowlist secret
    "sk-ant-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)
FAKE_OPENAI_KEY_REAL_SHAPE = (  # pragma: allowlist secret
    "sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)
FAKE_OPENAI_PROJ_KEY_REAL_SHAPE = (  # pragma: allowlist secret
    "sk-proj-aaaaaaaaaaaaaaaaaaaaaaaaaa"
)
FAKE_RUNPOD_KEY_REAL_SHAPE = (  # pragma: allowlist secret
    "rpa_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789X"
)
FAKE_WANDB_KEY_REAL_SHAPE_40HEX = (  # pragma: allowlist secret
    "abcdef0123456789abcdef0123456789abcdef01"
)

# Fixture-marker shapes — these MUST classify low-confidence on the env-assign
# regex, but the verdict is still FAIL when any hit exists (plan AC4/AC6).
FAKE_HF_TOKEN_TEST_FIXTURE = "hf_test_token"  # pragma: allowlist secret
FAKE_WANDB_KEY_TEST_FIXTURE = "wandb_test_key"  # pragma: allowlist secret


def _evt(
    note: str, ts: str = "2026-01-01T00:00:00Z", kind: str = "epm:progress", version: int = 1
) -> str:
    return json.dumps({"ts": ts, "kind": kind, "version": version, "note": note})


def test_hf_token_match() -> None:
    line = _evt(f"HF_TOKEN={FAKE_HF_TOKEN_REAL_SHAPE} leaked here")
    hits = audit.scan_line(1, line)
    classes = sorted(h.key_class for h in hits)
    # Both the shape regex AND the env-assignment regex should fire.
    assert "hf" in classes
    assert "env-assign:HF_TOKEN" in classes


def test_anthropic_does_not_double_count_openai() -> None:
    line = _evt(f"ANTHROPIC_API_KEY={FAKE_ANTHROPIC_KEY_REAL_SHAPE} exfil")
    hits = audit.scan_line(1, line)
    classes = sorted(h.key_class for h in hits)
    assert "anthropic" in classes
    # MUST NOT also flag as openai — the lookahead excludes sk-ant-.
    assert "openai" not in classes


def test_openai_legacy_and_proj() -> None:
    # Legacy sk-
    line1 = _evt(f"OPENAI_API_KEY={FAKE_OPENAI_KEY_REAL_SHAPE}")
    hits1 = audit.scan_line(1, line1)
    assert any(h.key_class == "openai" for h in hits1)
    # sk-proj- shape
    line2 = _evt(f"OPENAI_API_KEY={FAKE_OPENAI_PROJ_KEY_REAL_SHAPE}")
    hits2 = audit.scan_line(2, line2)
    assert any(h.key_class == "openai" for h in hits2)


def test_runpod_match() -> None:
    line = _evt(f"RUNPOD_API_KEY={FAKE_RUNPOD_KEY_REAL_SHAPE}")
    hits = audit.scan_line(1, line)
    classes = sorted(h.key_class for h in hits)
    assert "runpod" in classes
    assert "env-assign:RUNPOD_API_KEY" in classes


def test_wandb_context_anchored_positive() -> None:
    # 40 hex with explicit WANDB context -> should hit.
    line = _evt(f"Set WANDB_API_KEY={FAKE_WANDB_KEY_REAL_SHAPE_40HEX} in env")
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
    raw = f"{{this is not valid json but contains {FAKE_HF_TOKEN_REAL_SHAPE_TWO} here}}"
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
        + _evt(f"HF_TOKEN={FAKE_HF_TOKEN_REAL_SHAPE}")
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
        match=FAKE_HF_TOKEN_REAL_SHAPE,
        note_excerpt="leak here",
        confidence="high",
        triage_reason="",
    )
    report = audit.compose_report(123, Path("/tmp/events.jsonl"), [h], 50)
    assert "**Verdict:** FAIL — leaked: hf" in report
    assert "Hugging Face token" in report
    assert "Required action" in report
    assert "High-confidence hits" in report


def test_compose_report_fail_low_confidence_only_per_plan_AC4() -> None:
    """Plan AC4/AC6: PASS requires ZERO hits. Any hit -> FAIL.

    Even when every hit is low-confidence (obvious test fixture), the
    top-line verdict MUST be FAIL — the binary gate exists so the
    rotation decision never depends on a confidence threshold.  The
    low/high split survives as triage prose inside the FAIL report.
    """
    h_low = audit.Hit(
        line_no=42,
        event_ts="2026-01-01T00:00:00Z",
        event_kind="epm:code-review-codex",
        event_version=1,
        key_class="env-assign:HF_TOKEN",
        match=f"HF_TOKEN={FAKE_HF_TOKEN_TEST_FIXTURE}",
        note_excerpt="tests assert ... appear in the create argv",
        confidence="low",
        triage_reason="value contains fixture marker 'test_token'",
    )
    report = audit.compose_report(123, Path("/tmp/events.jsonl"), [h_low], 50)
    # Strict binary verdict: any hit -> FAIL, regardless of confidence tier.
    assert "**Verdict:** FAIL" in report
    assert "PASS (with" not in report  # banned tier from plan AC4/AC6
    assert "PASS only" not in report
    # Triage section (no Required-action, since no high-confidence hits).
    assert "Triage — FAIL on low-confidence hits only" in report
    assert "Required action" not in report
    # The low-confidence section still surfaces the hit's details.
    assert "Low-confidence hits" in report


def test_classify_env_assign_fixture_marker() -> None:
    conf, reason = audit._classify_env_assign("HF_TOKEN", FAKE_HF_TOKEN_TEST_FIXTURE)
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
        f"tests assert `HF_TOKEN={FAKE_HF_TOKEN_TEST_FIXTURE}` and "
        f"`WANDB_API_KEY={FAKE_WANDB_KEY_TEST_FIXTURE}` appear in the create argv at "
        "`tests/test_gcp_backend.py:1448-1450`."
    )
    line = _evt(payload, kind="epm:code-review-codex")
    hits = audit.scan_line(113, line)
    env_hits = [h for h in hits if h.key_class.startswith("env-assign")]
    assert len(env_hits) == 2
    assert all(h.confidence == "low" for h in env_hits)
    classes = {h.key_class for h in env_hits}
    assert classes == {"env-assign:HF_TOKEN", "env-assign:WANDB_API_KEY"}


def test_end_to_end_real_shaped_secret_yields_FAIL_via_compose(tmp_path: Path) -> None:
    """End-to-end: a real-shaped HF token feeds scan_file -> compose_report -> FAIL.

    Codex round-1 raised this as a coverage gap: the existing tests check
    shape matches, classifier behavior, and report composition separately,
    but no single test traces a real-shaped non-fixture secret through the
    full pipeline and confirms the verdict is FAIL with `Required action`.
    """
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        _evt("clean prose")
        + "\n"
        + _evt(f"oops: HF_TOKEN={FAKE_HF_TOKEN_REAL_SHAPE} got logged")
        + "\n",
        encoding="utf-8",
    )

    hits = audit.scan_file(events_path)
    # The shape regex AND the env-assignment regex should both fire on
    # line 2, both classified high-confidence.
    assert any(h.key_class == "hf" and h.confidence == "high" for h in hits)
    assert any(h.key_class == "env-assign:HF_TOKEN" and h.confidence == "high" for h in hits)

    report = audit.compose_report(123, events_path, hits, 2)
    # The verdict line MUST start with FAIL and name the leaked class.
    assert "**Verdict:** FAIL — leaked:" in report
    assert "hf" in report.split("**Verdict:** FAIL — leaked:", 1)[1].splitlines()[0]
    # Required action section MUST list the HF rotation instruction.
    assert "Required action" in report
    assert "Hugging Face token" in report
    # And the high-confidence hit table is present.
    assert "High-confidence hits" in report
