"""Pin the tmux socket-dir shim contract (#1466).

``scripts/eps_tmux_env.sh`` is the single source of truth for the fleet's
``TMUX_TMPDIR``: durable default ``$HOME/.tmux-sockets`` (0700), with a
legacy pin to ``/tmp`` while ANY socket file remains in ``/tmp/tmux-<uid>``
(single-server invariant during the #1466 migration), and unconditional
respect for a pre-set ``TMUX_TMPDIR``.

All shim tests run the shim in a subprocess under ``bash -euo pipefail``
(the strictest flag set of the shim's real consumers — the mygoat systemd
wrapper runs ``set -euo pipefail``), so every case doubly proves
strict-mode safety. No live tmux server is touched: the "socket" is a
plain AF_UNIX bind in a scratch dir (``EPS_TMUX_LEGACY_DIR`` overrides the
legacy base for tests).

The fifth test pins the two repo cron wrappers' source-line PLACEMENT —
the shim must be sourced before the first tmux-touching invocation, which
a later editor could silently drop or reorder.
"""

from __future__ import annotations

import os
import socket
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "scripts" / "eps_tmux_env.sh"


def _run_shim(
    *,
    legacy_base: Path,
    home: Path,
    preset_tmux_tmpdir: str | None = None,
) -> str:
    """Source the shim under ``bash -euo pipefail``; return $TMUX_TMPDIR."""
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "EPS_TMUX_LEGACY_DIR": str(legacy_base),
    }
    if preset_tmux_tmpdir is not None:
        env["TMUX_TMPDIR"] = preset_tmux_tmpdir
    proc = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            f'. "{SHIM}"; printf %s "${{TMUX_TMPDIR:?shim did not set TMUX_TMPDIR}}"',
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"shim failed under -euo pipefail (rc={proc.returncode}): {proc.stderr}"
    )
    return proc.stdout


def test_shim_prefers_legacy_live_socket() -> None:
    """A real AF_UNIX socket in the legacy dir pins TMUX_TMPDIR to it."""
    # Short prefix under /tmp: pytest tmp_path can exceed the 108-byte
    # AF_UNIX path limit on long usernames.
    with tempfile.TemporaryDirectory(prefix="t1466-") as td:
        root = Path(td)
        legacy = root / "legacy"
        home = root / "home"
        sock_dir = legacy / f"tmux-{os.getuid()}"
        sock_dir.mkdir(parents=True)
        home.mkdir()
        sock = socket.socket(socket.AF_UNIX)
        try:
            sock.bind(str(sock_dir / "default"))
            out = _run_shim(legacy_base=legacy, home=home)
        finally:
            sock.close()
        assert out == str(legacy), f"expected legacy pin to {legacy}, got {out!r}"
        assert not (home / ".tmux-sockets").exists(), "durable dir must not be created on pin"


def test_shim_durable_default_when_no_socket() -> None:
    """No legacy socket: durable $HOME/.tmux-sockets, created mode 0700."""
    with tempfile.TemporaryDirectory(prefix="t1466-") as td:
        root = Path(td)
        legacy = root / "legacy"
        home = root / "home"
        (legacy / f"tmux-{os.getuid()}").mkdir(parents=True)  # exists, empty
        home.mkdir()
        out = _run_shim(legacy_base=legacy, home=home)
        durable = home / ".tmux-sockets"
        assert out == str(durable), f"expected durable default {durable}, got {out!r}"
        assert durable.is_dir(), "shim must create the durable dir"
        assert (durable.stat().st_mode & 0o777) == 0o700, "durable dir must be 0700"


def test_shim_respects_preset_tmux_tmpdir() -> None:
    """A pre-set TMUX_TMPDIR is passed through untouched; no dir is created."""
    with tempfile.TemporaryDirectory(prefix="t1466-") as td:
        root = Path(td)
        legacy = root / "legacy"
        home = root / "home"
        legacy.mkdir()
        home.mkdir()
        out = _run_shim(legacy_base=legacy, home=home, preset_tmux_tmpdir="/custom/x")
        assert out == "/custom/x", f"pre-set TMUX_TMPDIR must be respected, got {out!r}"
        assert not (home / ".tmux-sockets").exists(), "no dir may be created on pass-through"


def test_non_socket_file_does_not_pin() -> None:
    """A REGULAR file named `default` in the legacy dir must not pin /tmp."""
    with tempfile.TemporaryDirectory(prefix="t1466-") as td:
        root = Path(td)
        legacy = root / "legacy"
        home = root / "home"
        sock_dir = legacy / f"tmux-{os.getuid()}"
        sock_dir.mkdir(parents=True)
        home.mkdir()
        (sock_dir / "default").write_text("not a socket\n")
        out = _run_shim(legacy_base=legacy, home=home)
        assert out == str(home / ".tmux-sockets"), (
            f"a stale regular file must not pin the legacy dir (-type s), got {out!r}"
        )


def test_wrappers_source_shim_before_first_tmux() -> None:
    """Both repo cron wrappers source the shim BEFORE their first
    tmux-touching invocation (placement a later editor could drop/reorder).
    """
    # wrapper -> the driver invocation that talks to tmux (the watcher's
    # driver contains no literal `tmux` in its own invocation line, so the
    # tmux-touching line is pinned per wrapper by driver filename).
    wrappers = {
        "scripts/cron_session_summarize.sh": "tmux_window_titles.py",
        "scripts/cron_autonomous_session_watch.sh": "autonomous_session_watch.py",
    }
    for rel, driver in wrappers.items():
        lines = (REPO_ROOT / rel).read_text().splitlines()
        source_line = None
        driver_line = None
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if source_line is None and "eps_tmux_env.sh" in line:
                source_line = i
            if driver_line is None and driver in line:
                driver_line = i
        assert source_line is not None, f"{rel}: no line sourcing eps_tmux_env.sh"
        assert driver_line is not None, f"{rel}: driver invocation {driver} not found"
        assert source_line < driver_line, (
            f"{rel}: shim sourced at line {source_line}, AFTER the first "
            f"tmux-touching invocation ({driver} at line {driver_line})"
        )
