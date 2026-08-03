"""Tests for the ``scripts/bootstrap_pod.sh`` default-PATH tool exposure.

Background: ``ssh pod "uv run ..."`` and ``ssh pod "python ..."`` open a
non-interactive non-login shell that does NOT source ``/root/.bashrc`` /
``/root/.profile`` (they bail on the ``[ -z "$PS1" ] && return`` guard), so
the PATH exports those rc files carry never reach such a shell. The bootstrap
script therefore drops ``uv``/``uvx`` symlinks and a ``python`` exec shim into
``/usr/local/bin`` (which IS on the default PATH for those shells).

These tests are static (no live pod, no network): they assert the script's
syntax stays valid and that the symlink + shim block is present with the
right shape. They guard against the regression where someone removes the
``/usr/local/bin`` exposure and silently breaks non-login SSH tool resolution.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_pod.sh"


def _script_text() -> str:
    return BOOTSTRAP.read_text(encoding="utf-8")


def test_bootstrap_script_exists() -> None:
    assert BOOTSTRAP.is_file(), f"missing {BOOTSTRAP}"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_bootstrap_syntax_is_valid() -> None:
    """`bash -n` must pass — the symlink/shim block must not break parsing."""
    result = subprocess.run(
        ["bash", "-n", str(BOOTSTRAP)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"bash -n failed:\n{result.stderr}"


def test_usr_local_bin_uv_symlink_present() -> None:
    """uv must be symlinked into /usr/local/bin (on the default non-login PATH)."""
    text = _script_text()
    assert "ln -sf" in text, "expected a symlink command for uv"
    assert "/usr/local/bin/uv" in text, "expected /usr/local/bin/uv symlink target"


def test_usr_local_bin_uvx_symlink_present() -> None:
    """uvx ships alongside uv and must be symlinked when present."""
    text = _script_text()
    assert "/usr/local/bin/uvx" in text, "expected /usr/local/bin/uvx symlink target"


def test_python_shim_execs_uv_run_python() -> None:
    """The python shim must forward to the locked project interpreter via uv."""
    text = _script_text()
    assert "/usr/local/bin/python" in text, "expected /usr/local/bin/python shim path"
    assert "chmod +x /usr/local/bin/python" in text, "python shim must be executable"
    assert "exec uv run python" in text, "python shim must exec `uv run python`"


def test_uv_binary_resolution_fails_loud() -> None:
    """If uv is missing after install, bootstrap must error out, not silently skip."""
    text = _script_text()
    # The resolution block must hard-exit when no uv binary is found rather
    # than installing a dangling symlink (fail-loud, per project convention).
    assert "/root/.local/bin/uv" in text, "expected canonical uv install location"
    assert "exit 1" in text, "expected a hard exit on missing uv"


def test_rc_file_exports_retained() -> None:
    """The original rc-file PATH/cache exports must remain (additive change)."""
    text = _script_text()
    assert "/root/.bashrc" in text, "rc-file writes must be preserved"
    assert "WANDB_CACHE_DIR=/workspace/.cache/wandb" in text, "cache exports must remain"


# ---------------------------------------------------------------------------
# issue #1172 — repo-root PYTHONPATH mask (trap #823/#853), three channels
# ---------------------------------------------------------------------------

# The :+ prepend form: repo root first, any inherited PYTHONPATH appended,
# and NO trailing colon when unset/empty (a leading/trailing colon silently
# adds cwd to sys.path — cpython #107353). Nounset-exempt under set -u.
_PYTHONPATH_PREPEND = (
    'export PYTHONPATH="/workspace/explore-persona-space${PYTHONPATH:+:$PYTHONPATH}"'
)


def test_pythonpath_rc_append_separately_guarded() -> None:
    """Channel (a): rc-file append with its OWN grep guard — NOT folded into
    the WANDB_CACHE_DIR-keyed cache-redirect heredoc, so already-bootstrapped
    pods gain the line on any re-bootstrap."""
    text = _script_text()
    # Exactly two :+ prepend occurrences: the rc append heredoc + the shim.
    assert text.count(_PYTHONPATH_PREPEND) == 2, text.count(_PYTHONPATH_PREPEND)
    assert 'grep -q "PYTHONPATH=\\"/workspace/explore-persona-space" "$f"' in text, (
        "rc append must carry its own PYTHONPATH-keyed idempotency guard"
    )


def test_pythonpath_env_file_plain_assignment_presence_guarded() -> None:
    """Channel (b): the pod .env gains a PLAIN no-expansion assignment (read
    both by shell sourcing under set -u and by python-dotenv, which has no
    :+ interpolation) behind a PRESENCE guard (never append if ANY
    PYTHONPATH= line exists, whatever its value)."""
    text = _script_text()
    assert "\nPYTHONPATH=/workspace/explore-persona-space\n" in text, (
        ".env channel must be a plain single-path assignment (no expansion)"
    )
    assert 'grep -q "^PYTHONPATH=" "$ENV_FILE"' in text, (
        ".env append must be behind the anchored PRESENCE guard"
    )


def test_pythonpath_python_shim_exports_before_exec() -> None:
    """Channel (c): the /usr/local/bin/python shim exports the repo-root
    prepend so bare non-login `ssh pod "python ..."` invocations inherit it."""
    text = _script_text()
    shim_start = text.index("cat > /usr/local/bin/python")
    shim_end = text.index("chmod +x /usr/local/bin/python")
    shim_body = text[shim_start:shim_end]
    assert _PYTHONPATH_PREPEND in shim_body, "shim must export PYTHONPATH"
    assert shim_body.index(_PYTHONPATH_PREPEND) < shim_body.index("exec uv run python"), (
        "shim export must precede the exec"
    )


# ---------------------------------------------------------------------------
# issue #1794 — ~/.local/bin PATH export in the pod rc files
# ---------------------------------------------------------------------------


def test_rc_path_append_separately_guarded() -> None:
    """The PATH rc-append block exists, carries its OWN presence guard, and is
    NOT folded into the WANDB_CACHE_DIR-keyed cache-redirect heredoc — so
    already-bootstrapped pods gain the export on any re-bootstrap (#1794;
    mirrors the #1172 PYTHONPATH block).

    Two extra invariants (round-1/round-2 critic hardening):

    1. Guard<->payload consistency: the raw grep pattern, with its backslash
       escapes removed, must be a byte substring of the heredoc-written
       ``export PATH=...`` line — a quoting drift that un-matches the guard
       (silent duplicate appends on every re-bootstrap) goes red here.
    2. The RAW pattern must keep the literal two-character ``\\$HOME``
       sequence: the block lives inside the step-6 ``ssh_cmd '...'``
       single-quoted argument, so an UNescaped ``$HOME`` would expand on the
       REMOTE shell (pattern becomes ``PATH="/root/...``) and never match the
       literal heredoc line — check (1) alone would still pass post-expansion
       source-side, so this pins the escape itself.
    """
    text = _script_text()
    # Own guard, distinct heredoc delimiter (not the RCEOF cache-redirect one).
    assert 'grep -qF "PATH=\\"\\$HOME/.local/bin" "$f"' in text, (
        "PATH rc append must carry its own escaped-literal presence guard"
    )
    assert '<<"RC3EOF"' in text, "PATH rc append must use its own quoted heredoc delimiter"

    # Extract the RAW grep pattern (the only `grep -qF` in the script).
    m = re.search(r'grep -qF "((?:[^"\\]|\\.)*)" "\$f"', text)
    assert m is not None, "could not locate the grep -qF guard pattern"
    raw_pattern = m.group(1)
    assert "\\$HOME" in raw_pattern, (
        "guard pattern must keep the dollar sign backslash-escaped (remote-literal $HOME)"
    )

    # Extract the heredoc payload line the guard must match.
    m2 = re.search(r'<<"RC3EOF"\n(.*?)\nRC3EOF', text, re.DOTALL)
    assert m2 is not None, "could not locate the RC3EOF heredoc body"
    payload_lines = [ln for ln in m2.group(1).split("\n") if ln.startswith("export PATH=")]
    assert len(payload_lines) == 1, payload_lines
    assert payload_lines[0] == 'export PATH="$HOME/.local/bin:$PATH"'

    # Guard<->payload consistency: de-escaped pattern is a byte substring of
    # the written line (what remote grep -F actually compares).
    unescaped = raw_pattern.replace("\\", "")
    assert unescaped in payload_lines[0], (
        f"de-escaped guard pattern {unescaped!r} must byte-match the heredoc "
        f"payload {payload_lines[0]!r} — quoting drift would silently duplicate appends"
    )
