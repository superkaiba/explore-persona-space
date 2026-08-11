"""Pending-call wedge observer (#2115) — dry-run zero-write pin.

`--pending-call-wedge-only --dry-run` is the sanctioned live smoke (the
root-unstaged `--root-unstaged-audit-only --dry-run` precedent), so
dry_run=True against a would-flag fixture must:

(a) COMPUTE the flag (return True + emit the stdout flag line — the smoke
    proves the predicate fires on real state);
(b) emit NO real push (every `_telegram_push` call carries dry_run=True,
    under which the real implementation prints and returns False with zero
    side effects);
(c) write NOTHING — no sidecar row, no state file (the "[dry-run] would…"
    prints substitute for both).
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402

NOW_AGE_S = 60 * 60.0  # one hour pending — comfortably past the 25-min window


def _would_flag_rows() -> list[dict]:
    ts = datetime.now(tz=UTC).timestamp() - NOW_AGE_S
    iso = datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")
    return [
        {
            "type": "assistant",
            "timestamp": iso,
            "message": {
                "content": [{"type": "tool_use", "id": "toolu_dry1", "name": "Bash", "input": {}}]
            },
        }
    ]


def test_dry_run_computes_flag_but_writes_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("EPM_DISABLE_PENDING_CALL_WEDGE", raising=False)
    monkeypatch.delenv("EPM_PENDING_CALL_WEDGE_MIN", raising=False)
    monkeypatch.setattr(asw, "PROJECT_ROOT", tmp_path / "repo")
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path / "registry")
    monkeypatch.setattr(asw, "_issue_registrations", lambda: {2115: {"sids": {"sid-dry"}}})
    monkeypatch.setattr(
        asw, "_transcript_tail_rows", lambda pid, max_bytes=262144: _would_flag_rows()
    )
    pushes: list[tuple[str, bool]] = []
    real_push = asw._telegram_push

    def _spy_push(msg: str, dry_run: bool) -> bool:
        pushes.append((msg, dry_run))
        return real_push(msg, dry_run)  # dry_run=True: prints only, returns False

    monkeypatch.setattr(asw, "_telegram_push", _spy_push)

    # (a) the flag is COMPUTED under dry_run.
    assert asw.pending_call_wedge_pass(True, pids_by_sid={"sid-dry": 4242}) is True
    out = capsys.readouterr().out
    assert "pending-call-wedge: issue #2115" in out
    assert "toolu_dry1" in out

    # (b) no real push: every call went through with dry_run=True.
    assert len(pushes) == 1
    assert pushes[0][1] is True
    assert "[dry-run] would telegram-push" in out

    # (c) zero writes: no sidecar row, no state file — only the dry-run prints.
    assert not asw._pending_call_sidecar_path().exists()
    assert not asw._pending_call_state_path().exists()
    assert "[dry-run] would append pending-call-wedge sidecar row" in out
    assert "[dry-run] would save pending-call-wedge state" in out
    # Belt: nothing at all was created under the tmp roots.
    assert not (tmp_path / "repo").exists()
    assert not (tmp_path / "registry").exists()
