"""Pattern-pins for the #2360 ``scripts/bootstrap_pod.sh`` edits.

Sibling of ``test_bootstrap_pod_path.py`` (whose scope is the default-PATH
tool exposure / python shim): these tests pin the uv cache + link-mode
determinism edits and the greppable step-10 preflight-failure line.

Static (no live pod, no network). KNOWN LIMIT, by design: pattern PRESENCE
only — the single-vs-double-quoting hazard of the step-10 edit (``$?`` must
expand REMOTELY, i.e. inside the SINGLE-quoted ssh block) is NOT testable by
substring and is an explicit code-review eyeball item (#2360 plan D4 step 4).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

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
