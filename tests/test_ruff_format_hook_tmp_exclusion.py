"""Pin tests for the PostToolUse ruff-format hook's ephemeral-root exclusion (#1632).

The PostToolUse ``Edit|Write`` hook in ``.claude/settings.json`` runs
``ruff check --fix`` + ``ruff format`` on every edited ``.py`` path. Scratch
scripts under ``/tmp/``, ``/var/tmp/``, and ``/dev/shm/`` are
written-to-be-executed-once; formatting them buys nothing and invalidates the
writer's in-context file state (incident #1602: an Edit right after a Write
failed with "String to replace not found" because the hook reformatted the
file in between). The hook therefore excludes paths matching
``^/(tmp|var/tmp|dev/shm)/`` while keeping repo-tree formatting unchanged.

These tests pin that behavior (pattern precedent:
``tests/test_guard_trigger_dense_read.py`` — parse settings.json, then drive
the CONFIGURED command end-to-end with synthetic stdin JSON). Registered in
``WORKFLOW_INVARIANT`` (``scripts/select_step9c_tests.py``) because the
selector's stem/literal/dependency arms are .py-only — a settings.json diff
maps to no test otherwise.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

# Resolve against THIS checkout (worktree or main), so the test exercises the
# settings.json of whichever tree it runs in.
CHECKOUT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = CHECKOUT_ROOT / ".claude" / "settings.json"

EXCLUSION_LITERAL = "^/(tmp|var/tmp|dev/shm)/"
EPHEMERAL_ROOTS = ["/tmp", "/var/tmp", "/dev/shm"]
# Both jq extraction arms of the hook command: Edit/Write tool_input carries
# file_path; the tool_response fallback carries filePath.
PAYLOAD_FORMS = ["tool_input", "tool_response"]


def _load_settings() -> dict:
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


def _hook_command() -> str:
    """Locate the Edit|Write PostToolUse hook command that runs ruff format."""
    for entry in _load_settings()["hooks"]["PostToolUse"]:
        if entry.get("matcher") != "Edit|Write":
            continue
        for hook in entry.get("hooks", []):
            command = hook.get("command", "")
            if "ruff format" in command:
                return command
    raise AssertionError(
        "No PostToolUse hook with matcher 'Edit|Write' and a 'ruff format' command "
        f"found in {SETTINGS_PATH}"
    )


def _payload(form: str, file_path: str) -> str:
    if form == "tool_input":
        return json.dumps({"tool_input": {"file_path": file_path}})
    if form == "tool_response":
        return json.dumps({"tool_response": {"filePath": file_path}})
    raise ValueError(f"unknown payload form: {form}")


def _run_hook(command: str, payload: str) -> None:
    subprocess.run(
        ["bash", "-c", command],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )


def test_settings_json_parses():
    """Fleet-wide blast-radius guard: settings.json must stay valid JSON."""
    settings = _load_settings()
    assert isinstance(settings, dict)
    assert settings, "settings.json parsed to an empty object"


def test_ruff_hook_carries_tmp_exclusion():
    """Durability pin: the hook keeps the ephemeral-root exclusion AND the
    original .py filter + ruff invocations (repo-path behavior retained)."""
    command = _hook_command()
    assert EXCLUSION_LITERAL in command
    assert "\\.py$" in command
    assert "ruff check --fix" in command
    assert "ruff format" in command


@pytest.mark.parametrize("payload_form", PAYLOAD_FORMS)
@pytest.mark.parametrize("root", EPHEMERAL_ROOTS)
def test_hook_skips_ephemeral_paths(root: str, payload_form: str):
    """Behavioral: a deliberately-unformatted .py under an ephemeral root is
    left byte-unchanged (pre-fix, the hook reformatted it — the #1602 bug)."""
    root_path = Path(root)
    if not root_path.is_dir() or not os.access(root, os.W_OK):
        pytest.skip(f"{root} absent or unwritable on this host")
    target = root_path / f"eps1632_pin_{os.getpid()}_{payload_form}.py"
    try:
        target.write_text("x=1\n", encoding="utf-8")
        _run_hook(_hook_command(), _payload(payload_form, str(target)))
        assert target.read_text(encoding="utf-8") == "x=1\n"
    finally:
        target.unlink(missing_ok=True)


@pytest.mark.parametrize("payload_form", PAYLOAD_FORMS)
def test_hook_formats_repo_path(payload_form: str):
    """Behavioral control: a repo-tree .py path still gets formatted.

    The leading underscore keeps pytest from ever collecting the scratch file.
    If ruff's normalization of ``x=1`` ever changes, relax the equality below
    to "bytes changed" (content != "x=1\\n").
    """
    target = CHECKOUT_ROOT / "tests" / f"_eps1632_pin_{os.getpid()}_{payload_form}.py"
    try:
        target.write_text("x=1\n", encoding="utf-8")
        _run_hook(_hook_command(), _payload(payload_form, str(target)))
        assert target.read_text(encoding="utf-8") == "x = 1\n"
    finally:
        target.unlink(missing_ok=True)
