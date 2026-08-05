"""Tests for the 429 auto-retry Stop hook mirror (#1448).

Shells ``scripts/hooks/on-stop-429-retry.sh`` (the version-controlled source
of truth; installed to ``~/.claude/hooks/`` by ``cp -p``) with JSON payloads on
stdin. No real sleeps: every invocation sets ``CLAUDE_429_RETRY_NO_SLEEP=1``
plus per-test tmp state/debug dirs, so the whole file runs in well under 10s.

Portability note: the hook uses GNU ``stat -c %Y`` / ``find -mmin`` (Linux VM
only); on BSD/macOS ``stat -f %m`` would be needed — out of scope by design.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK = REPO_ROOT / "scripts" / "hooks" / "on-stop-429-retry.sh"

# Verbatim message shape from a real 2026-07-16 storm capture
# (/tmp/claude-stop-hook-debug/StopFailure-1784218334044172450.json), with a
# dummy org id substituted.
OTPM_429_TEXT = (
    "API Error: Request rejected (429) · This request would exceed your organization's "
    "rate limit of 3,000,000 output tokens per minute "
    "(org: 00000000-0000-0000-0000-000000000000, model: claude-fable-5). "
    "For details, refer to: https://docs.claude.com/en/api/rate-limits. "
    "You can see the response headers for current usage."
)
# Synthesized-by-symmetry from the OTPM capture (no real ITPM/RPM capture
# existed at implementation time) — same sentence shape, limiter phrase swapped.
ITPM_429_TEXT = OTPM_429_TEXT.replace("output tokens per minute", "input tokens per minute")
RPM_429_TEXT = OTPM_429_TEXT.replace(
    "rate limit of 3,000,000 output tokens per minute", "rate limit of 4,000 requests per minute"
)
UNKNOWN_429_TEXT = "API Error: Request rejected (429) · Rate limit exceeded, try again later."

WAITED_RE = re.compile(r"waited (\d+)s")


def hook_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    """Env for one hook invocation: no-sleep + tmp state/debug dirs by default."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_429_RETRY_")}
    env.update(
        {
            "CLAUDE_429_RETRY_NO_SLEEP": "1",
            "CLAUDE_429_RETRY_STATE_DIR": str(tmp_path / "state"),
            "CLAUDE_429_RETRY_DEBUG_DIR": str(tmp_path / "debug"),
            "CLAUDE_429_RETRY_DISABLE_FILE": str(tmp_path / "absent-disable-file"),
        }
    )
    env.update(overrides)
    return env


def run_hook(payload: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(HOOK)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def stopfailure_payload(message: str, session_id: str = "sess-1448-test", **extra: object) -> str:
    payload: dict[str, object] = {
        "hook_event_name": "StopFailure",
        "session_id": session_id,
        "last_assistant_message": message,
        "error": "rate_limit",
    }
    payload.update(extra)
    return json.dumps(payload)


def assert_rewake(proc: subprocess.CompletedProcess[str]) -> None:
    """Every re-wake (rc=2) must keep stderr EMPTY: asyncRewake delivery is
    stderr-first, stdout-fallback — stray stderr displaces the stdout message."""
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stderr == "", f"stray stderr on the exit-2 path: {proc.stderr!r}"


def test_hook_mirror_exists_and_has_no_stale_message() -> None:
    assert HOOK.is_file()
    text = HOOK.read_text()
    assert "input-token cap" not in text  # acceptance criterion 1


@pytest.mark.parametrize(
    ("message", "expected_phrase"),
    [
        pytest.param(OTPM_429_TEXT, "output-tokens-per-minute (OTPM)", id="output-tpm"),
        pytest.param(ITPM_429_TEXT, "input-tokens-per-minute (ITPM)", id="input-tpm"),
        pytest.param(RPM_429_TEXT, "requests-per-minute (RPM)", id="rpm"),
        pytest.param(UNKNOWN_429_TEXT, "rate limit (class unknown)", id="unknown"),
    ],
)
def test_stopfailure_limiter_classes(tmp_path: Path, message: str, expected_phrase: str) -> None:
    proc = run_hook(stopfailure_payload(message), hook_env(tmp_path))
    assert_rewake(proc)
    assert expected_phrase in proc.stdout
    assert "retry 1/5" in proc.stdout
    assert "input-token cap" not in proc.stdout
    n_waited = int(WAITED_RE.search(proc.stdout).group(1))
    assert 20 <= n_waited <= 89  # acceptance criterion 2 (computed even under NO_SLEEP)


def test_storm_cap_six_invocations(tmp_path: Path) -> None:
    env = hook_env(tmp_path)
    payload = stopfailure_payload(OTPM_429_TEXT, session_id="storm-sess")
    for i in range(1, 6):
        proc = run_hook(payload, env)
        assert_rewake(proc)
        assert f"retry {i}/5" in proc.stdout
    sixth = run_hook(payload, env)
    assert sixth.returncode == 0, (sixth.stdout, sixth.stderr)
    assert sixth.stdout == ""  # at-cap: silent exit 0
    assert sixth.stderr == ""
    count_file = tmp_path / "state" / "storm-sess.count"
    assert count_file.read_text() == "6"  # at-cap invocations keep incrementing


def test_counter_resets_after_quiet_gap(tmp_path: Path) -> None:
    env = hook_env(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    count_file = state_dir / "reset-sess.count"
    count_file.write_text("5")
    stale = time.time() - 700  # > 600s reset window
    os.utime(count_file, (stale, stale))
    proc = run_hook(stopfailure_payload(OTPM_429_TEXT, session_id="reset-sess"), env)
    assert_rewake(proc)
    assert "retry 1/5" in proc.stdout  # fresh storm budget


@pytest.mark.parametrize(
    ("fake_now", "lo", "hi"),
    [
        # now%60=0 -> boundary 60s -> wait in [60, 70]
        pytest.param(1_200_000_000, 60, 70, id="mod0"),
        # now%60=40 -> boundary 20s (== floor) -> wait in [20, 30]
        pytest.param(1_200_000_040, 20, 30, id="mod40"),
        # now%60=41 -> boundary 19s < floor -> +60 -> wait in [79, 89]
        pytest.param(1_200_000_041, 79, 89, id="mod41"),
        # now%60=59 -> boundary 1s < floor -> +60 -> wait in [61, 71]
        pytest.param(1_200_000_059, 61, 71, id="mod59"),
    ],
)
def test_boundary_wait_windows(tmp_path: Path, fake_now: int, lo: int, hi: int) -> None:
    """CLAUDE_429_RETRY_FAKE_NOW pins now, so the boundary arithmetic is exact
    up to the 0-10s jitter (fresh state dir per test: counter math unaffected)."""
    env = hook_env(tmp_path, CLAUDE_429_RETRY_FAKE_NOW=str(fake_now))
    proc = run_hook(stopfailure_payload(OTPM_429_TEXT), env)
    assert_rewake(proc)
    n_waited = int(WAITED_RE.search(proc.stdout).group(1))
    assert lo <= n_waited <= hi, proc.stdout


def test_stop_event_is_noop(tmp_path: Path) -> None:
    payload = json.dumps(
        {"hook_event_name": "Stop", "session_id": "s", "last_assistant_message": OTPM_429_TEXT}
    )
    proc = run_hook(payload, hook_env(tmp_path))
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_stop_hook_active_guard(tmp_path: Path) -> None:
    payload = stopfailure_payload(OTPM_429_TEXT, stop_hook_active=True)
    proc = run_hook(payload, hook_env(tmp_path))
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_kill_switch(tmp_path: Path) -> None:
    disable = tmp_path / "disabled"
    disable.write_text("")
    env = hook_env(tmp_path, CLAUDE_429_RETRY_DISABLE_FILE=str(disable))
    proc = run_hook(stopfailure_payload(OTPM_429_TEXT), env)
    assert proc.returncode == 0
    assert proc.stdout == ""
    assert not (tmp_path / "debug").exists()  # checked before any write


def test_garbage_stdin_is_noop(tmp_path: Path) -> None:
    proc = run_hook("this is { not json", hook_env(tmp_path))
    assert proc.returncode == 0
    assert proc.stdout == ""


def _subagent_payload(transcript: Path) -> str:
    return json.dumps(
        {
            "hook_event_name": "SubagentStop",
            "session_id": "parent-sess",
            "agent_transcript_path": str(transcript),
            "agent_type": "critic",
            "agent_id": "agent-abc123",
            "stop_hook_active": False,
            "last_assistant_message": "final report text, no error here",
        }
    )


def test_subagentstop_fires_on_structured_429_line(tmp_path: Path) -> None:
    transcript = tmp_path / "agent-transcript.jsonl"
    rows = [
        {"type": "assistant", "message": {"content": "some normal turn"}},
        {"isApiErrorMessage": True, "message": {"content": OTPM_429_TEXT}},
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    proc = run_hook(_subagent_payload(transcript), hook_env(tmp_path))
    assert_rewake(proc)
    assert "output-tokens-per-minute (OTPM)" in proc.stdout  # limiter parsed from transcript
    assert "sub-agent (critic)" in proc.stdout
    assert "Re-spawn the same sub-agent" in proc.stdout


def test_subagentstop_without_marker_is_noop(tmp_path: Path) -> None:
    transcript = tmp_path / "agent-transcript.jsonl"
    # Same 429 text but WITHOUT isApiErrorMessage:true — prose must not gate.
    rows = [
        {"type": "assistant", "message": {"content": "some normal turn"}},
        {"type": "assistant", "message": {"content": OTPM_429_TEXT}},
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    proc = run_hook(_subagent_payload(transcript), hook_env(tmp_path))
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_subagentstop_missing_transcript_is_noop(tmp_path: Path) -> None:
    proc = run_hook(_subagent_payload(tmp_path / "nonexistent.jsonl"), hook_env(tmp_path))
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_debug_capture_written(tmp_path: Path) -> None:
    run_hook(stopfailure_payload(OTPM_429_TEXT), hook_env(tmp_path))
    dumps = list((tmp_path / "debug").glob("StopFailure-*.json"))
    assert len(dumps) == 1
    assert json.loads(dumps[0].read_text())["hook_event_name"] == "StopFailure"


def test_installed_copy_matches_mirror() -> None:
    """Drift gate: the user-global installed copy must equal the repo mirror."""
    installed = Path.home() / ".claude" / "hooks" / "on-stop-429-retry.sh"
    if not installed.is_file():
        pytest.skip("no user-global installed copy on this machine")
    assert installed.read_text() == HOOK.read_text(), (
        "installed ~/.claude/hooks/on-stop-429-retry.sh has drifted from the repo mirror; "
        "re-sync with: cp -p scripts/hooks/on-stop-429-retry.sh ~/.claude/hooks/"
    )


# --- 529 Overloaded coverage (added 2026-08-05) ----------------------------
# A 529 is server capacity, not a per-minute budget: it must (a) be detected on
# the SubagentStop path, (b) be classified distinctly from the three limiters,
# and (c) be paced exponentially rather than to the minute boundary.

OVERLOADED_529_TEXT = "API Error: 529 Overloaded"
# A GENUINE 429 whose body quotes a token count containing "529". This is the
# false-positive the digit guards exist for: a bare `529` match would mis-pace
# this rate limit as an overload.
ITPM_429_QUOTING_529_TEXT = (
    "API Error: Request rejected (429) - would exceed your organization's "
    "529,000 input tokens per minute limit"
)


def _waited_secs(proc: subprocess.CompletedProcess[str]) -> int:
    """Pull the integer wait out of the re-wake message."""
    return int(proc.stdout.split("waited ", 1)[1].split("s", 1)[0])


def _subagent_error_transcript(tmp_path: Path, text: str, name: str) -> Path:
    transcript = tmp_path / name
    transcript.write_text(
        json.dumps({"isApiErrorMessage": True, "message": {"content": text}}) + "\n"
    )
    return transcript


def test_subagentstop_fires_on_529_overloaded(tmp_path: Path) -> None:
    """The gap this closes: a 529 used to fall through to exit 0 (no retry)."""
    transcript = _subagent_error_transcript(tmp_path, OVERLOADED_529_TEXT, "a529.jsonl")
    proc = run_hook(_subagent_payload(transcript), hook_env(tmp_path))
    assert_rewake(proc)
    assert "529 Overloaded" in proc.stdout
    assert "server capacity" in proc.stdout
    # Must NOT be narrated as a token-budget problem.
    assert "tokens per minute" not in proc.stdout


def test_529_is_paced_exponentially_not_to_the_boundary(tmp_path: Path) -> None:
    """20/40/80/90/90 (+ jitter <=10), every wait inside the 120s hook timeout."""
    transcript = _subagent_error_transcript(tmp_path, OVERLOADED_529_TEXT, "a529.jsonl")
    payload = _subagent_payload(transcript)
    env = hook_env(tmp_path)
    expected_bases = [20, 40, 80, 90, 90]
    for i, base in enumerate(expected_bases, start=1):
        proc = run_hook(payload, env)
        assert_rewake(proc)
        waited = _waited_secs(proc)
        assert base <= waited <= base + 10, (i, base, waited, proc.stdout)
        assert waited < 120, f"wait {waited}s exceeds the hook's 120s settings timeout"
    # 6th invocation is over MAX_CONSECUTIVE -> silent stay-stopped.
    capped = run_hook(payload, env)
    assert capped.returncode == 0
    assert capped.stdout == ""


def test_429_quoting_529_token_count_is_not_classified_overloaded(tmp_path: Path) -> None:
    """Digit-guard regression: '529,000 input tokens per minute' is a 429."""
    transcript = _subagent_error_transcript(tmp_path, ITPM_429_QUOTING_529_TEXT, "a429.jsonl")
    proc = run_hook(_subagent_payload(transcript), hook_env(tmp_path))
    assert_rewake(proc)
    assert "input-tokens-per-minute (ITPM)" in proc.stdout
    assert "Overloaded" not in proc.stdout
    # Boundary pacing, not the exponential branch.
    assert "to the minute boundary" in proc.stdout


def test_subagentstop_529_prose_without_marker_is_noop(tmp_path: Path) -> None:
    """Prose mentioning 529 must not gate — isApiErrorMessage:true is required."""
    transcript = tmp_path / "prose529.jsonl"
    transcript.write_text(
        json.dumps({"message": {"content": "discussing a 529 Overloaded error"}}) + "\n"
    )
    proc = run_hook(_subagent_payload(transcript), hook_env(tmp_path))
    assert proc.returncode == 0
    assert proc.stdout == ""
