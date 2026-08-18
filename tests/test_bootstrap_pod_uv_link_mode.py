"""Pattern-pins for the #2360 ``scripts/bootstrap_pod.sh`` edits.

Sibling of ``test_bootstrap_pod_path.py`` (whose scope is the default-PATH
tool exposure / python shim): these tests pin the uv cache + link-mode
determinism edits and the greppable step-10 preflight-failure line.

Static (no live pod, no network). KNOWN LIMIT, by design: tests (a)-(e) are
pattern PRESENCE only — the single-vs-double-quoting hazard of the step-10
edit (``$?`` must expand REMOTELY, i.e. inside the SINGLE-quoted ssh block) is
NOT testable by substring and is an explicit code-review eyeball item (#2360
plan D4 step 4). Test (f) (#2360 r2, hardened r3) is the exception: it
EXECUTES the extracted step-10 remote payload in a sandboxed bash (fake ``uv``
on PATH; exactly ONE asserted production ``cd`` line replaced by a cd into the
sandbox — never a global ``cd()`` shim) and asserts the ``.env`` assignments
reach the ``uv run`` CHILD process environment — binding the EXPORT semantics
a presence pin structurally cannot (the round-1 bare ``source`` left
UV_LINK_MODE/UV_CACHE_DIR shell-local and every presence test stayed green).
The r3 mutation tests (f-mut-i/ii) pin that the binder itself is falsifiable:
an interior quote after the invocation guard and a second ``cd`` before the
``.env`` sourcing each made the r2 binder stay green (first-quote extraction
truncated silently; the global ``cd()`` shim retargeted EVERY cd) — the same
vacuity theme as the earlier #2360 blockers B5/B1, one level down.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_pod.sh"


def _text() -> str:
    return BOOTSTRAP.read_text(encoding="utf-8")


def _step5_block(text: str) -> str:
    """The step-5 sync ssh block: from the `step 5` line to the sync COMMAND.

    Anchors the end on the line-initial ``uv sync --locked`` command — the
    step-5 TITLE string also contains that phrase, so a bare substring index
    would truncate the block inside the title.
    """
    start = text.index('step 5 "Syncing Python environment')
    end = text.index("\nuv sync --locked", start)
    return text[start:end]


# The step-10 ssh_cmd payload closes on a STANDALONE delimiter line: nothing
# but optional whitespace around the closing single quote.
_STANDALONE_CLOSE_RE = re.compile(r"^[ \t]*'[ \t]*$", re.MULTILINE)


def _step10_payload(text: str) -> str:
    """Extract the step-10 ``ssh_cmd '...'`` single-quoted remote payload.

    Slices to the exact STANDALONE closing-delimiter line and asserts the
    ENTIRE interior is quote-free (#2360 r3). Bash closes a single-quoted
    string at the FIRST interior ``'``, so any quote in the interior means the
    payload bash would hand the remote shell no longer matches the intended
    block — the r2 first-quote extractor instead truncated there SILENTLY,
    leaving the guard green when a quote appeared after its substrings.
    """
    start = text.index('step 10 "Running preflight check"')
    open_q = text.index("ssh_cmd '", start) + len("ssh_cmd '")
    close = _STANDALONE_CLOSE_RE.search(text, open_q)
    assert close is not None, "step-10 payload: standalone closing delimiter line not found"
    payload = text[open_q : close.start()]
    assert "'" not in payload, (
        "step-10 payload interior contains a single quote — bash would close the "
        f"ssh_cmd string early, so extraction is unsound: {payload!r}"
    )
    return payload


def _exec_step10_payload(text: str, tmp_path: Path) -> subprocess.CompletedProcess:
    """Execute the extracted step-10 payload in a sandboxed bash; return proc.

    Replaces exactly ONE asserted production ``cd /workspace/...`` line with a
    cd into the sandbox (#2360 r3). Never a global ``cd()`` shim: that would
    redirect EVERY cd regardless of its production target, so a second ``cd``
    inserted before the ``.env`` sourcing would be silently retargeted back to
    the sandbox and the binder would stay green while the real remote payload
    stopped exporting ``.env`` into the child.
    """
    payload = _step10_payload(text)
    prod_cd = "cd /workspace/explore-persona-space"
    assert payload.count(prod_cd) == 1, payload

    sandbox = tmp_path / "workspace"
    sandbox.mkdir(exist_ok=True)
    (sandbox / ".env").write_text(
        "UV_CACHE_DIR=/workspace/.cache/uv\nUV_LINK_MODE=copy\n", encoding="utf-8"
    )
    # The payload prepends $HOME/.local/bin to PATH; an empty fake HOME keeps
    # the fake `uv` below first on PATH (the real VM ~/.local/bin has real uv).
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        'echo "CHILD_UV_LINK_MODE=${UV_LINK_MODE:-UNSET}"\n'
        'echo "CHILD_UV_CACHE_DIR=${UV_CACHE_DIR:-UNSET}"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    script = payload.replace(prod_cd, f'cd "{sandbox}"', 1)
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(fake_home)},
    )


def test_bash_syntax_still_valid() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(BOOTSTRAP)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr


def test_step5_exports_uv_link_mode_before_sync() -> None:
    """(a) UV_LINK_MODE=copy exported in the step-5 block BEFORE the
    `uv sync --locked` line, so the FIRST venv-populating sync inherits it."""
    block = _step5_block(_text())
    assert "export UV_LINK_MODE=copy" in block


def test_step5_exports_uv_cache_dir_before_sync() -> None:
    """(b) UV_CACHE_DIR on the persistent volume before the first sync (step 6
    only wires it for LATER shells), with the mkdir preceding the export."""
    block = _step5_block(_text())
    assert "export UV_CACHE_DIR=/workspace/.cache/uv" in block
    assert "mkdir -p /workspace/.cache/uv" in block
    assert block.index("mkdir -p /workspace/.cache/uv") < block.index(
        "export UV_CACHE_DIR=/workspace/.cache/uv"
    )


def test_rc_file_uv_link_mode_block_separately_guarded() -> None:
    """(c) a SEPARATELY-guarded rc-file block (own grep guard, the PYTHONPATH
    precedent) appends UV_LINK_MODE so already-bootstrapped pods gain it on
    any re-bootstrap."""
    text = _text()
    assert re.search(r'grep -q "\^export UV_LINK_MODE=" "\$f"', text), (
        "missing the separately-guarded rc-file UV_LINK_MODE grep guard"
    )
    assert re.search(r'<<"RCUVEOF"\n\n[^\n]*\nexport UV_LINK_MODE=copy\nRCUVEOF', text), (
        "missing the rc-file UV_LINK_MODE heredoc body"
    )


def test_env_file_uv_link_mode_append_guarded() -> None:
    """(d) the .env append carries its own ^UV_LINK_MODE= presence guard so
    dotenv-loading subprocesses + `set -a; . .env` launchers inherit it."""
    text = _text()
    assert re.search(r'grep -q "\^UV_LINK_MODE=" "\$ENV_FILE"', text), (
        "missing the .env UV_LINK_MODE grep guard"
    )
    assert "\nUV_LINK_MODE=copy\n" in text, "missing the .env UV_LINK_MODE=copy line"


def test_step10_preflight_failure_is_greppable_not_swallowed() -> None:
    """(e) the step-10 preflight invocation no longer swallows failure with
    `|| true`; the greppable PREFLIGHT-FAILED-AT-BOOTSTRAP rc line replaces
    it (presence-only — quoting context is the review eyeball item)."""
    text = _text()
    preflight_lines = [
        ln
        for ln in text.splitlines()
        if "explore_persona_space.orchestrate.preflight" in ln and "uv run" in ln
    ]
    assert preflight_lines, "step-10 preflight invocation not found"
    for ln in preflight_lines:
        assert "|| true" not in ln, f"preflight failure still swallowed: {ln!r}"
    assert any('|| echo "PREFLIGHT-FAILED-AT-BOOTSTRAP rc=$?"' in ln for ln in preflight_lines), (
        "missing the PREFLIGHT-FAILED-AT-BOOTSTRAP rc line on the preflight invocation"
    )


def test_step10_env_assignments_exported_to_uv_run_child(tmp_path) -> None:
    """(f) EXPORT-SEMANTICS binder (#2360 r2, Codex blocker
    step10-uv-env-not-exported; hardened r3): EXECUTE the step-10 remote
    payload in a sandbox and assert the plain ``.env`` assignments reach the
    ``uv run`` CHILD process — presence of a `source` line is not enough (a
    bare `source` without `set -a` leaves them shell-local, so an implicit
    sync in the preflight invocation would re-emit the hardlink-fallback
    warning into the very log Acceptance 3 greps)."""
    proc = _exec_step10_payload(_text(), tmp_path)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "CHILD_UV_LINK_MODE=copy" in proc.stdout, (proc.stdout, proc.stderr)
    assert "CHILD_UV_CACHE_DIR=/workspace/.cache/uv" in proc.stdout, (proc.stdout, proc.stderr)


def test_step10_payload_is_single_quote_free() -> None:
    """(f-guard, hardened r3) extraction completeness + interior integrity:
    ``_step10_payload`` slices to the STANDALONE closing delimiter and refuses
    any quote in the complete interior; this pin asserts the extracted
    interior carries the full preflight invocation through its final
    load-bearing line."""
    payload = _step10_payload(_text())
    assert "explore_persona_space.orchestrate.preflight" in payload, payload
    assert "PREFLIGHT-FAILED-AT-BOOTSTRAP" in payload, payload


def test_mutation_quote_after_invocation_guard_fails_extraction() -> None:
    """(f-mut-i, #2360 r3) MUTATION: an apostrophe inserted AFTER the
    full-invocation guard substrings must make extraction FAIL LOUD. Under
    the r2 first-quote extractor this exact mutation stayed green — the
    extraction silently truncated past the required substrings while bash
    would close the real remote payload at that quote."""
    text = _text()
    needle = '|| echo "PREFLIGHT-FAILED-AT-BOOTSTRAP rc=$?"'
    idx = text.index(needle) + len(needle)
    mutated = text[:idx] + "\n    echo pods' status probed" + text[idx:]
    with pytest.raises(AssertionError, match="single quote"):
        _step10_payload(mutated)


def test_mutation_second_cd_before_env_sourcing_fails_binder(tmp_path) -> None:
    """(f-mut-ii, #2360 r3) MUTATION: a second ``cd`` inserted before the
    ``.env`` sourcing (production would then source ``.env`` from the wrong
    directory — the exact drift the binder exists to catch) must make the
    binder FAIL. Under the r2 global ``cd()`` shim this mutation stayed
    green: the shim retargeted the second cd back to the sandbox too."""
    text = _text()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    anchor = "set -a; [ -f .env ] && source .env; set +a"
    start = text.index('step 10 "Running preflight check"')
    idx = text.index(anchor, start)  # the payload's own sourcing line
    mutated = text[:idx] + f'cd "{elsewhere}"\n    ' + text[idx:]
    proc = _exec_step10_payload(mutated, tmp_path)
    assert "CHILD_UV_LINK_MODE=copy" not in proc.stdout, (
        "binder stayed green despite a second cd diverting .env sourcing",
        proc.stdout,
    )
    assert "CHILD_UV_LINK_MODE=UNSET" in proc.stdout, (proc.stdout, proc.stderr)
    assert "CHILD_UV_CACHE_DIR=UNSET" in proc.stdout, (proc.stdout, proc.stderr)
