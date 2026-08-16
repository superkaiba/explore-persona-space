"""Tests for ``scripts/guard_skill_doc_headroom.sh`` + its settings.json registration (#2325).

PostToolUse(Edit|Write) ADVISORY hook: after any Edit/Write to a
``.claude/skills/**/*.md`` file it runs the edited TREE's
``workflow_lint.py --check-skill-doc-size`` (main checkout's venv; the edited
tree's script + caps, since ``workflow_lint._REPO_ROOT`` resolves from
``__file__``) and relays the lint's own FAIL / low-headroom line via stderr +
exit 2 — PostToolUse fires after the tool ran, so nothing can be blocked.
Every other path exits 0 silently (fail-open); kill switch
``EPM_SKIP_SKILL_DOC_HEADROOM_HOOK=1``; warn threshold
``EPM_SKILL_DOC_HEADROOM_WARN_BYTES`` (default 2000).

WORKFLOW_INVARIANT-registered: the Step 9c selector maps ``.json`` diffs to
no tests (``_DATA_DOC_SUFFIXES``), so the settings.json registration pin
below runs on every gate ONLY via the tuple registration (the
``test_guard_trigger_dense_read.py`` / ``test_ruff_format_hook_tmp_exclusion.py``
precedent).

The trip/quiet pair is deliberate (Codex Should-Fix 1): the forced-trip test
alone also passes against a reversed comparison or an unconditionally-warning
script — only the threshold-0 quiet test constrains the comparison's
direction.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
SCRIPT = _REPO_ROOT / "scripts" / "guard_skill_doc_headroom.sh"
SETTINGS = _REPO_ROOT / ".claude" / "settings.json"

# Deployed-path convention (the test_guard_trigger_dense_read.py precedent):
# settings.json registers hooks by canonical repo-root absolute path; a
# pre-merge worktree run remaps ONLY the canonical-root prefix onto this
# checkout, so wrong-directory / wrong-basename registration typos still fail.
CANONICAL_ROOT = "/home/thomasjiralerspong/explore-persona-space"


def _main_repo_root() -> str | None:
    r = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return str(Path(r.stdout.strip()).parent)


def _run(stdin: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Feed a PostToolUse payload to the hook with a scrubbed env."""
    env = os.environ.copy()
    env.pop("EPM_SKIP_SKILL_DOC_HEADROOM_HOOK", None)
    env.pop("EPM_SKILL_DOC_HEADROOM_WARN_BYTES", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def _hook_json(path: str) -> str:
    return json.dumps({"tool_name": "Edit", "tool_input": {"file_path": path}})


def test_script_exists_and_executable() -> None:
    assert SCRIPT.is_file(), SCRIPT
    assert os.access(SCRIPT, os.X_OK), "hook script must be executable"


def test_registered_in_settings_posttooluse() -> None:
    """A hooks.PostToolUse entry matches Edit AND Write and names the script.

    Should-Fix 2 (#2325): additionally extract the path token from the
    command string and assert it RESOLVES to a file (canonical-root prefix
    remapped onto this checkout for pre-merge worktree runs) — a typo'd
    registration path can no longer pass while the script exists elsewhere.
    """
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    matching = [
        (entry, hook)
        for entry in settings["hooks"]["PostToolUse"]
        for hook in entry.get("hooks", [])
        if "guard_skill_doc_headroom.sh" in hook.get("command", "")
    ]
    assert matching, "guard_skill_doc_headroom.sh not registered under hooks.PostToolUse"
    entry, hook = matching[0]
    matcher = entry.get("matcher", "")
    assert re.fullmatch(matcher, "Edit"), matcher
    assert re.fullmatch(matcher, "Write"), matcher
    token_match = re.search(r"\S*guard_skill_doc_headroom\.sh", hook["command"])
    assert token_match is not None, hook["command"]
    token = token_match.group(0)
    assert os.path.isabs(token), token
    main_root = _main_repo_root()
    if main_root is not None and str(_REPO_ROOT) != main_root:
        prefix = CANONICAL_ROOT.rstrip("/") + "/"
        if token.startswith(prefix):
            token = str(_REPO_ROOT / token[len(prefix) :])
    assert Path(token).is_file(), token


def test_fail_open_on_garbage_stdin() -> None:
    proc = _run("not json")
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)


def test_non_skill_path_silent() -> None:
    proc = _run(_hook_json(str(_REPO_ROOT / "scripts" / "workflow_lint.py")))
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stdout == ""
    assert proc.stderr == ""


def test_kill_switch() -> None:
    proc = _run(
        _hook_json(str(_REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md")),
        env_extra={"EPM_SKIP_SKILL_DOC_HEADROOM_HOOK": "1"},
    )
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stdout == ""
    assert proc.stderr == ""


def test_trips_on_low_headroom() -> None:
    """Force the trip on the live grandfathered file (no tree mutation): a
    1,000,000 B warn threshold makes ANY real headroom read as low."""
    proc = _run(
        _hook_json(str(_REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md")),
        env_extra={"EPM_SKILL_DOC_HEADROOM_WARN_BYTES": "1000000"},
    )
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "issue/SKILL.md" in proc.stderr
    assert "cap" in proc.stderr


def test_quiet_on_eligible_path_with_ample_headroom() -> None:
    """Inverse branch (Codex Should-Fix 1): threshold 0 means no real headroom
    can read as low, so the SAME eligible live file must produce rc 0 and
    EMPTY stdout+stderr — the pair constrains the comparison's direction."""
    proc = _run(
        _hook_json(str(_REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md")),
        env_extra={"EPM_SKILL_DOC_HEADROOM_WARN_BYTES": "0"},
    )
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stdout == ""
    assert proc.stderr == ""
