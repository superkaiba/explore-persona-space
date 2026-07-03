"""Tests for the persisted headroom snapshot + `--status` CLI (workflow v2).

These exercise the cross-session-coordination side-channel added to
``explore_persona_space.llm.api_dispatch``: the throttled best-effort
``~/.task-workflow/api-headroom.json`` writer and the ``--status`` reader.
Run with ``PYTHONPATH=<worktree>/src`` so the worktree's module is imported.
"""

from __future__ import annotations

import datetime as dt
import json

from explore_persona_space.llm import api_dispatch as ad


class _H:
    """Minimal dict-like stand-in for an SDK headers object."""

    def __init__(self, d: dict):
        self._d = d

    def get(self, k):
        return self._d.get(k)


def _iso(offset_s: int = 0) -> str:
    return (
        dt.datetime(2026, 7, 3, 0, 0, 0, tzinfo=dt.UTC) + dt.timedelta(seconds=offset_s)
    ).isoformat()


# ── header extraction ────────────────────────────────────────────────────────


def test_observation_from_headers_takes_min_tokens():
    obs = ad._headroom_observation_from_headers(
        _H(
            {
                "anthropic-ratelimit-requests-remaining": "88",
                "anthropic-ratelimit-input-tokens-remaining": "5000",
                "anthropic-ratelimit-output-tokens-remaining": "12000",
            }
        )
    )
    assert obs == {"requests_remaining": 88, "tokens_remaining": 5000}


def test_observation_from_headers_none_when_absent():
    assert ad._headroom_observation_from_headers(_H({})) is None


def test_observation_from_headers_requests_only():
    obs = ad._headroom_observation_from_headers(
        _H({"anthropic-ratelimit-requests-remaining": "100"})
    )
    assert obs == {"requests_remaining": 100}


# ── pure merge ───────────────────────────────────────────────────────────────


def test_merge_into_empty():
    snap = ad.merge_headroom_snapshot(
        {},
        "high_prio",
        "claude-sonnet-4-5",
        {"requests_remaining": 90},
        observed_at_iso=_iso(),
        writer_pid=123,
    )
    assert snap["writer_pid"] == 123
    assert snap["high_prio"]["claude-sonnet-4-5"]["requests_remaining"] == 90
    assert snap["high_prio"]["claude-sonnet-4-5"]["observed_at_iso"] == _iso()


def test_merge_adds_org_and_newest_wins():
    snap = ad.merge_headroom_snapshot(
        {}, "high_prio", "m", {"requests_remaining": 1}, observed_at_iso=_iso(), writer_pid=1
    )
    snap = ad.merge_headroom_snapshot(
        snap, "batch", "m", {"requests_remaining": 2}, observed_at_iso=_iso(1), writer_pid=2
    )
    snap = ad.merge_headroom_snapshot(
        snap, "high_prio", "m", {"requests_remaining": 9}, observed_at_iso=_iso(2), writer_pid=3
    )
    assert set(snap) == {"high_prio", "batch", "writer_pid"}
    assert snap["high_prio"]["m"]["requests_remaining"] == 9  # newest wins
    assert snap["batch"]["m"]["requests_remaining"] == 2
    assert snap["writer_pid"] == 3


# ── throttled writer ─────────────────────────────────────────────────────────


def test_record_writes_then_throttles_then_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(ad, "_headroom_last_write_monotonic", 0.0)
    p = tmp_path / "api-headroom.json"
    ad.record_headroom_observation(
        "batch",
        "claude-haiku-4-5",
        _H({"anthropic-ratelimit-requests-remaining": "100"}),
        path=p,
        now_monotonic=100.0,
        now_iso=_iso(),
    )
    # within the 5s window -> throttled, NOT written (org stays just 'batch')
    ad.record_headroom_observation(
        "low_prio",
        "claude-opus-4-8",
        _H({"anthropic-ratelimit-requests-remaining": "20"}),
        path=p,
        now_monotonic=102.0,
        now_iso=_iso(),
    )
    data = json.loads(p.read_text())
    assert [k for k in data if k != "writer_pid"] == ["batch"]
    # past the window -> written
    ad.record_headroom_observation(
        "low_prio",
        "claude-opus-4-8",
        _H({"anthropic-ratelimit-requests-remaining": "20"}),
        path=p,
        now_monotonic=110.0,
        now_iso=_iso(),
    )
    data = json.loads(p.read_text())
    assert sorted(k for k in data if k != "writer_pid") == ["batch", "low_prio"]


def test_record_no_headers_no_write(tmp_path, monkeypatch):
    monkeypatch.setattr(ad, "_headroom_last_write_monotonic", 0.0)
    p = tmp_path / "api-headroom.json"
    ad.record_headroom_observation("batch", "m", _H({}), path=p, now_monotonic=100.0)
    assert not p.exists()


def test_record_is_fail_soft(tmp_path, monkeypatch):
    # A write failure must NEVER propagate into the dispatch loop.
    monkeypatch.setattr(ad, "_headroom_last_write_monotonic", 0.0)

    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(ad, "_atomic_write_json", _boom)
    # Should not raise.
    ad.record_headroom_observation(
        "batch",
        "m",
        _H({"anthropic-ratelimit-requests-remaining": "1"}),
        path=tmp_path / "x.json",
        now_monotonic=100.0,
    )


# ── staleness + status rows ──────────────────────────────────────────────────


def test_staleness_labels():
    now = dt.datetime(2026, 7, 3, 1, 0, 0, tzinfo=dt.UTC)
    assert ad._staleness_label((now - dt.timedelta(seconds=10)).isoformat(), now) == "fresh"
    assert ad._staleness_label((now - dt.timedelta(minutes=10)).isoformat(), now) == "stale"
    assert ad._staleness_label((now - dt.timedelta(hours=3)).isoformat(), now) == "very-stale"
    assert ad._staleness_label("not-a-date", now) == "unknown"


def test_build_status_rows_skips_writer_pid_and_sorts():
    now = dt.datetime(2026, 7, 3, 0, 0, 30, tzinfo=dt.UTC)
    snap = {
        "high_prio": {"claude-sonnet-4-5": {"requests_remaining": 90, "observed_at_iso": _iso()}},
        "writer_pid": 5,
    }
    rows = ad.build_headroom_status_rows(snap, now=now)
    assert len(rows) == 1
    assert rows[0]["org"] == "high_prio" and rows[0]["staleness"] == "fresh"
    assert rows[0]["requests_remaining"] == 90


def test_format_status_text_empty():
    assert "no observations" in ad.format_headroom_status_text([])


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cmd_status_no_file(tmp_path, capsys):
    rc = ad._cmd_status(path=tmp_path / "missing.json", as_json=True)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["rows"] == []


def test_cmd_status_reads_snapshot(tmp_path, capsys):
    p = tmp_path / "api-headroom.json"
    now = dt.datetime(2026, 7, 3, 0, 0, 30, tzinfo=dt.UTC)
    p.write_text(
        json.dumps(
            {
                "high_prio": {
                    "claude-sonnet-4-5": {"requests_remaining": 90, "observed_at_iso": _iso()}
                },
                "writer_pid": 7,
            }
        )
    )
    rc = ad._cmd_status(path=p, as_json=True, now=now)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["writer_pid"] == 7 and out["rows"][0]["staleness"] == "fresh"


def test_main_status_json(tmp_path, monkeypatch, capsys):
    p = tmp_path / "api-headroom.json"
    p.write_text(json.dumps({"writer_pid": 1}))
    monkeypatch.setattr(ad, "HEADROOM_SNAPSHOT_PATH", p)
    rc = ad.main(["--status", "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["rows"] == []
