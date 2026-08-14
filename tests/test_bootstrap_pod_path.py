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
POD_LIFECYCLE = REPO_ROOT / "scripts" / "pod_lifecycle.py"


def _script_text() -> str:
    return BOOTSTRAP.read_text(encoding="utf-8")


def _shim_heredoc_bodies() -> dict[str, str]:
    """Extract the /usr/local/bin/python shim heredoc body from BOTH writers.

    The shim has TWO writers — ``bootstrap_pod.sh`` step 6 (fresh provision)
    and ``pod_lifecycle._UV_RESTORE_SNIPPET`` (the ``pod.py resume`` path,
    which reinstalls the shim after RunPod wipes the container overlay). A
    single-writer test is exactly what let the two copies drift before task
    #2278, so every shim-body invariant below scans BOTH heredocs. Exactly
    one PYEOF heredoc is expected per file — a second one would silently
    retarget these extractions.
    """
    bodies: dict[str, str] = {}
    for name, path in {"bootstrap_pod.sh": BOOTSTRAP, "pod_lifecycle.py": POD_LIFECYCLE}.items():
        text = path.read_text(encoding="utf-8")
        matches = re.findall(
            r'cat > /usr/local/bin/python <<"PYEOF"\n(.*?)\nPYEOF', text, re.DOTALL
        )
        assert len(matches) == 1, (
            f"expected exactly one PYEOF shim heredoc in {name}, got {len(matches)}"
        )
        bodies[name] = matches[0]
    return bodies


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


def test_python_shim_execs_project_venv_interpreter() -> None:
    """Each shim body execs the project venv interpreter DIRECTLY (no uv hop),
    with a system-interpreter fallback chain and a loud terminal failure."""
    text = _script_text()
    assert "/usr/local/bin/python" in text, "expected /usr/local/bin/python shim path"
    assert "chmod +x /usr/local/bin/python" in text, "python shim must be executable"
    for name, body in _shim_heredoc_bodies().items():
        assert "exec /workspace/explore-persona-space/.venv/bin/python" in body, (
            f"{name}: shim must exec the project venv interpreter directly"
        )
        assert "/usr/bin/python3.11" in body, f"{name}: expected the system-interpreter fallback"
        assert "exit 1" in body, f"{name}: shim must fail loud when no interpreter resolves"


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
    exec_anchor = "exec /workspace/explore-persona-space/.venv/bin/python"
    assert shim_body.index(_PYTHONPATH_PREPEND) < shim_body.index(exec_anchor), (
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


# ---------------------------------------------------------------------------
# task #2278 — the python shim must never invoke uv (interpreter-discovery
# deadlock), across BOTH shim writers (bootstrap step 6 + the resume path)
# ---------------------------------------------------------------------------


def test_python_shim_does_not_invoke_uv() -> None:
    """Direct regression pin for task #2278: NEITHER shim writer's heredoc
    body may contain a ``uv`` token at all. /usr/local/bin/python is the
    first ``python`` on the default non-interactive-SSH PATH, so uv's
    interpreter discovery executes it as a candidate; a body that re-enters
    uv blocks on the project lock a parent ``uv sync`` holds — a silent
    futex deadlock with stacked get_interpreter_info probes and zero output.
    The ban is total (comments included) so the deadlock stays impossible by
    construction rather than by a fragile am-I-being-probed heuristic.

    ``uvx`` is banned alongside ``uv``: it drives the same uv machinery (and
    the same project lock), so a ``uvx``-invoking shim body deadlocks
    identically while evading a bare ``\\buv\\b`` pin (word boundary fails
    before the ``x``).
    """
    for name, body in _shim_heredoc_bodies().items():
        hits = re.findall(r"\buvx?\b", body)
        assert not hits, (
            f"{name}: the /usr/local/bin/python shim body must not reference uv "
            f"or uvx anywhere (found {len(hits)} token(s)); a uv-invoking shim "
            "deadlocks uv interpreter discovery against a lock-holding uv sync "
            "(task #2278)"
        )


def test_python_shim_preserves_pythonpath_and_cwd() -> None:
    """Both shim bodies carry the repo-root PYTHONPATH export (#1172) and the
    ``cd`` into the repo — pinning closed the pod_lifecycle drift that
    omitted the PYTHONPATH export despite its stays-in-sync docstring."""
    for name, body in _shim_heredoc_bodies().items():
        assert _PYTHONPATH_PREPEND in body, f"{name}: shim must export the repo-root PYTHONPATH"
        assert "cd /workspace/explore-persona-space || exit 1" in body, (
            f"{name}: shim must cd into the repo root before exec"
        )


def test_shim_self_test_present() -> None:
    """Bootstrap runs a post-install shim self-test so a shim that cannot
    resolve an interpreter fails the provision NOW, not hours later as a
    silent hang (the mechanical form of acceptance criterion 3)."""
    text = _script_text()
    assert 'if ! /usr/local/bin/python -c "import sys; print(sys.executable)"' in text, (
        "expected the post-install python shim self-test invocation"
    )


def test_shim_self_test_fails_loud() -> None:
    """The self-test's failure path exits non-zero EXPLICITLY and is not
    swallowed: the step-6 remote payload carries no ``set -e`` and ssh_cmd
    propagates only the LAST command's status, so without an explicit
    ``exit 1`` a failing self-test would be masked by the closing echo
    lines — the same silent-failure class as the bug itself."""
    text = _script_text()
    m = re.search(
        r'if ! /usr/local/bin/python -c "import sys; print\(sys\.executable\)"(.*?)\nfi\n',
        text,
        re.DOTALL,
    )
    assert m is not None, "could not locate the shim self-test block"
    block = m.group(0)
    assert "exit 1" in block, "self-test failure branch must exit 1 explicitly"
    for swallow in ("|| true", "2>/dev/null"):
        assert swallow not in block, (
            f"self-test block must not muffle failures with {swallow!r} — a broken "
            "shim must fail the provision, not pass green"
        )


def test_dangling_venv_symlink_guard_present() -> None:
    """Step 5 clears a DANGLING .venv symlink before ``uv sync``: /root is the
    container overlay (recreated on pod stop/resume), so a .venv symlink into
    /root from the overlay-venv recovery survives the stop pointing nowhere,
    and the sync would otherwise fail on it. The guard must precede the sync."""
    text = _script_text()
    guard = "if [ -L .venv ] && [ ! -e .venv ]; then"
    assert guard in text, "expected the dangling .venv symlink guard"
    guard_block = text[text.index(guard) : text.index(guard) + 400]
    assert "rm -f .venv" in guard_block, "the guard must remove the dangling link"
    # Anchor on the INVOCATION line (the step-5 banner echoes the same phrase
    # earlier in the file); the guard must precede the actual sync command.
    sync_invocation = "uv sync --locked 2>&1 | tail -5"
    assert sync_invocation in text, "expected the step-5 uv sync invocation"
    assert text.index(guard) < text.index(sync_invocation), (
        "the dangling-symlink guard must run BEFORE uv sync"
    )
