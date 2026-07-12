"""CI wrapper for the ``.claude/hooks/guard_lessons_edit.sh`` PreToolUse hook (#1279).

The hook blocks Edit/Write tool calls targeting ``.claude/rules/LESSONS.md``
whose PROSPECTIVE post-edit content would FAIL the #1269 byte-budget /
index-parity gates (``scripts/workflow_lint.py::check_lessons_index``,
imported at runtime by ``.claude/hooks/guard_lessons_edit_check.py`` — never
re-implemented), closing the direct-to-main bypass that skips the commit-time
lint.

This file is a THIN wrapper (plan #1279 §6): the in-script ``--self-test``
suite stays the single source of truth for the behavior matrix; here we
(1) run that suite in CI, (2) pin one allow + one block case per check family
via targeted stdin-JSON payloads against synthetic ``tmp_path`` trees,
(3) pin the Edit semantics / fail-open / escape-hatch / path-shape behaviors,
(4) pin the edited-tree constant-resolution contract (a tree carrying its OWN
``scripts/workflow_lint.py`` binds ITS constants — the #1269 same-diff
constant-first ordering honored at edit time), (5) pin day-one safety (a
verbatim rewrite of the live LESSONS.md is allowed), and (6) assert the
settings.json wiring (#965 ``TestSettingsWiring`` convention — closing the
"hook ships green but inert" channel).

Fixture rules (plan §12-9): every fixture size is computed at runtime from
the imported ``workflow_lint`` constants (the banked-slack check FAILs a
too-small allow fixture, so allow totals sit inside ``[ratchet - headroom,
ratchet]``); no synthetic stub is ever named ``gotchas.md`` (the grandfather
hygiene would FAIL a short gotchas row). Deny/allow determinism: every run
scrubs ``EPM_ALLOW_LESSONS_EDIT`` and points the sentinel escape hatch at a
nonexistent path via ``EPM_LESSONS_EDIT_SENTINEL``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = _REPO_ROOT / ".claude" / "hooks" / "guard_lessons_edit.sh"
HELPER = _REPO_ROOT / ".claude" / "hooks" / "guard_lessons_edit_check.py"
SETTINGS = _REPO_ROOT / ".claude" / "settings.json"
HOOK_REL = ".claude/hooks/guard_lessons_edit.sh"

_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import (  # noqa: E402
    _LESSONS_MAX_BYTES,
    _LESSONS_RATCHET_BYTES,
    _LESSONS_RATCHET_MAX_HEADROOM_BYTES,
    _LESSONS_ROW_MAX_BYTES,
)

VALID_TOTAL = _LESSONS_RATCHET_BYTES - _LESSONS_RATCHET_MAX_HEADROOM_BYTES // 2
ROW_A = "- alpha.md — trigger a"
ROW_B = "- beta.md — trigger b"
LONG_TRIGGER = "y" * (_LESSONS_ROW_MAX_BYTES + 20)


def _env(*, sentinel: Path | None = None, allow_env: str | None = None) -> dict[str, str]:
    """Hermetic hook env: hatch scrubbed, sentinel redirected off the live path."""
    env = {k: v for k, v in os.environ.items() if k != "EPM_ALLOW_LESSONS_EDIT"}
    env["EPM_LESSONS_EDIT_SENTINEL"] = str(
        sentinel if sentinel is not None else "/nonexistent/eps-lessons-sentinel"
    )
    if allow_env is not None:
        env["EPM_ALLOW_LESSONS_EDIT"] = allow_env
    return env


def _run(
    payload: dict | str,
    *,
    env: dict[str, str] | None = None,
    script: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Feed an Edit/Write PreToolUse payload to the guard, returning the process."""
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(
        ["bash", str(script or SCRIPT)],
        input=data,
        text=True,
        capture_output=True,
        env=env or _env(),
    )


def _write_payload(fp: Path | str, content: str) -> dict:
    return {"tool_name": "Write", "tool_input": {"file_path": str(fp), "content": content}}


def _edit_payload(fp: Path | str, old: str, new: str, *, replace_all: bool = False) -> dict:
    ti: dict = {"file_path": str(fp), "old_string": old, "new_string": new}
    if replace_all:
        ti["replace_all"] = True
    return {"tool_name": "Edit", "tool_input": ti}


def _mk_tree(root: Path, stems: tuple[str, ...] = ("alpha", "beta")) -> Path:
    """Synthetic tree: rules dir + zero-byte stub rules; returns the LESSONS.md path.

    Never a stub named ``gotchas`` — the grandfather-table hygiene would FAIL
    a short gotchas row (plan §12-9).
    """
    rules = root / ".claude" / "rules"
    rules.mkdir(parents=True)
    for s in stems:
        (rules / f"{s}.md").touch()
    return rules / "LESSONS.md"


def _sized(rows: list[str], total: int) -> str:
    """Index content with the given rows, padded to EXACTLY `total` bytes.

    Padding is a non-row 'x' line; sizes are byte-exact (the em-dash is 3
    bytes in UTF-8, so char counts would drift).
    """
    body = "# Lessons index (synthetic fixture)\n" + "".join(r + "\n" for r in rows)
    pad = total - len(body.encode("utf-8")) - 1
    assert pad >= 0, (len(body.encode("utf-8")), total)
    return body + "x" * pad + "\n"


def test_self_test_passes() -> None:
    r = subprocess.run(
        ["bash", str(SCRIPT), "--self-test"], text=True, capture_output=True, env=_env()
    )
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "self-test: PASS" in r.stdout, r.stdout


def test_bash_syntax() -> None:
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_unrelated_path_allowed(tmp_path: Path) -> None:
    r = _run(_write_payload(tmp_path / "notes.md", "anything"))
    assert r.returncode == 0, (r.returncode, r.stderr)


def test_valid_write_allowed(tmp_path: Path) -> None:
    lessons = _mk_tree(tmp_path)
    r = _run(_write_payload(lessons, _sized([ROW_A, ROW_B], VALID_TOTAL)))
    assert r.returncode == 0, (r.returncode, r.stderr)


@pytest.mark.parametrize(
    ("rows", "total", "want"),
    [
        pytest.param([ROW_A, ROW_B], _LESSONS_MAX_BYTES + 200, "leanness cap", id="total-cap"),
        pytest.param([ROW_A, ROW_B], _LESSONS_RATCHET_BYTES + 100, "grew past", id="ratchet"),
        pytest.param(
            [f"- alpha.md — {LONG_TRIGGER}", ROW_B], VALID_TOTAL, "per-row cap", id="per-row"
        ),
        pytest.param([ROW_A], VALID_TOTAL, "no index row", id="parity-missing-row"),
        pytest.param(
            [ROW_A, ROW_B, "- ghost.md — trigger g"],
            VALID_TOTAL,
            "no matching",
            id="parity-stale-row",
        ),
        pytest.param(
            [ROW_A, ROW_B],
            _LESSONS_RATCHET_BYTES - _LESSONS_RATCHET_MAX_HEADROOM_BYTES - 500,
            "banked slack",
            id="banked-slack",
        ),
    ],
)
def test_block_per_check_family(tmp_path: Path, rows: list[str], total: int, want: str) -> None:
    lessons = _mk_tree(tmp_path)
    r = _run(_write_payload(lessons, _sized(rows, total)))
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "BLOCKED" in r.stderr, r.stderr
    assert want in r.stderr, (want, r.stderr)


def test_block_message_names_recovery_paths(tmp_path: Path) -> None:
    lessons = _mk_tree(tmp_path)
    r = _run(_write_payload(lessons, _sized([ROW_A, ROW_B], _LESSONS_MAX_BYTES + 200)))
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "_LESSONS_RATCHET_BYTES" in r.stderr, r.stderr
    assert "EPM_ALLOW_LESSONS_EDIT" in r.stderr, r.stderr
    # The sentinel hatch is named by its RESOLVED ABSOLUTE path (a worktree-cwd
    # session following a relative touch recipe would touch the wrong file).
    assert "/.claude/cache/allow-lessons-edit" in r.stderr, r.stderr
    assert "--check-lessons-index" in r.stderr, r.stderr


def test_edit_absent_old_string_allowed(tmp_path: Path) -> None:
    lessons = _mk_tree(tmp_path)
    lessons.write_text(_sized([ROW_A, ROW_B], VALID_TOTAL), encoding="utf-8")
    r = _run(_edit_payload(lessons, "ZZZ_NOT_PRESENT", "zzz"))
    assert r.returncode == 0, (r.returncode, r.stderr)


def test_edit_ambiguous_old_string_allowed(tmp_path: Path) -> None:
    lessons = _mk_tree(tmp_path)
    lessons.write_text(_sized([ROW_A, ROW_B], VALID_TOTAL), encoding="utf-8")
    # 'trigger' occurs in both rows; the Edit tool errors on this itself.
    r = _run(_edit_payload(lessons, "trigger", LONG_TRIGGER))
    assert r.returncode == 0, (r.returncode, r.stderr)


def test_edit_replace_all_valid_result_allowed(tmp_path: Path) -> None:
    lessons = _mk_tree(tmp_path)
    lessons.write_text(_sized([ROW_A, ROW_B], VALID_TOTAL), encoding="utf-8")
    r = _run(_edit_payload(lessons, "trigger", "trig", replace_all=True))
    assert r.returncode == 0, (r.returncode, r.stderr)


def test_edit_growing_row_over_cap_blocks(tmp_path: Path) -> None:
    lessons = _mk_tree(tmp_path)
    lessons.write_text(_sized([ROW_A, ROW_B], VALID_TOTAL), encoding="utf-8")
    r = _run(_edit_payload(lessons, "trigger a", LONG_TRIGGER))
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "per-row cap" in r.stderr, r.stderr


def test_malformed_json_fails_open() -> None:
    r = _run("not json {")
    assert r.returncode == 0, (r.returncode, r.stderr)


def test_missing_file_path_fails_open() -> None:
    r = _run({"tool_name": "Write", "tool_input": {"content": "x"}})
    assert r.returncode == 0, (r.returncode, r.stderr)


def test_unimportable_lint_fails_open(tmp_path: Path) -> None:
    """No lint in the edited tree AND none in the hook's own repo -> allow.

    Copies (never symlinks — the helper resolve()s __file__) the hook pair
    into a fake repo with no scripts/workflow_lint.py, so BOTH the edited-tree
    import and the fallback miss.
    """
    fake_hooks = tmp_path / "fakerepo" / ".claude" / "hooks"
    fake_hooks.mkdir(parents=True)
    for src in (SCRIPT, HELPER):
        shutil.copy(src, fake_hooks / src.name)
    fake_script = fake_hooks / SCRIPT.name
    fake_script.chmod(0o755)
    lessons = _mk_tree(tmp_path / "tree")
    payload = _write_payload(lessons, _sized([ROW_A, ROW_B], _LESSONS_MAX_BYTES + 200))
    r = _run(payload, script=fake_script)
    assert r.returncode == 0, (r.returncode, r.stderr)


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_env_hatch_allows(tmp_path: Path, value: str) -> None:
    lessons = _mk_tree(tmp_path)
    payload = _write_payload(lessons, _sized([ROW_A, ROW_B], _LESSONS_MAX_BYTES + 200))
    r = _run(payload, env=_env(allow_env=value))
    assert r.returncode == 0, (r.returncode, r.stderr)


def test_env_hatch_zero_still_blocks(tmp_path: Path) -> None:
    lessons = _mk_tree(tmp_path)
    payload = _write_payload(lessons, _sized([ROW_A, ROW_B], _LESSONS_MAX_BYTES + 200))
    r = _run(payload, env=_env(allow_env="0"))
    assert r.returncode == 2, (r.returncode, r.stderr)


def test_fresh_sentinel_allows(tmp_path: Path) -> None:
    lessons = _mk_tree(tmp_path)
    sentinel = tmp_path / "sentinel"
    sentinel.touch()
    payload = _write_payload(lessons, _sized([ROW_A, ROW_B], _LESSONS_MAX_BYTES + 200))
    r = _run(payload, env=_env(sentinel=sentinel))
    assert r.returncode == 0, (r.returncode, r.stderr)


def test_stale_sentinel_still_blocks(tmp_path: Path) -> None:
    lessons = _mk_tree(tmp_path)
    sentinel = tmp_path / "sentinel"
    sentinel.touch()
    stale = time.time() - 3600
    os.utime(sentinel, (stale, stale))
    payload = _write_payload(lessons, _sized([ROW_A, ROW_B], _LESSONS_MAX_BYTES + 200))
    r = _run(payload, env=_env(sentinel=sentinel))
    assert r.returncode == 2, (r.returncode, r.stderr)


def test_worktree_shaped_path_blocks(tmp_path: Path) -> None:
    wt_root = tmp_path / ".claude" / "worktrees" / "issue-9"
    lessons = _mk_tree(wt_root, stems=("alpha",))
    r = _run(_write_payload(lessons, _sized([ROW_A], _LESSONS_MAX_BYTES + 200)))
    assert r.returncode == 2, (r.returncode, r.stderr)


def test_dotdot_path_normalized_and_blocked(tmp_path: Path) -> None:
    """A `..`-bearing non-canonical spelling still resolves to the guarded file."""
    _mk_tree(tmp_path)
    (tmp_path / "sub").mkdir()
    noncanon = tmp_path / "sub" / ".." / ".claude" / "rules" / "LESSONS.md"
    r = _run(_write_payload(noncanon, _sized([ROW_A, ROW_B], _LESSONS_MAX_BYTES + 200)))
    assert r.returncode == 2, (r.returncode, r.stderr)


_STUB_LINT = '''\
from pathlib import Path

_LESSONS_MAX_BYTES = 200000
_LESSONS_WARN_BYTES = 190000
_LESSONS_RATCHET_BYTES = 50000
_LESSONS_RATCHET_MAX_HEADROOM_BYTES = 200000
_LESSONS_ROW_MAX_BYTES = 100000
_LESSONS_ROW_GRANDFATHER_MAX_BYTES = {}
_LESSONS_ROW_GRANDFATHER_MAX_HEADROOM_BYTES = 40


def check_lessons_index(*, repo_root=None, warn_sink=None,
                        ratchet_bytes=_LESSONS_RATCHET_BYTES,
                        row_max_bytes=_LESSONS_ROW_MAX_BYTES):
    """Toy stand-in honoring only the total-byte ratchet (test double)."""
    lessons = Path(repo_root) / ".claude" / "rules" / "LESSONS.md"
    raw = lessons.read_bytes()
    if len(raw) > ratchet_bytes:
        return [f"stub-ratchet: {len(raw)} grew past {ratchet_bytes}"]
    return []
'''


def _mk_stub_lint_tree(tmp_path: Path) -> Path:
    """Synthetic tree carrying its OWN scripts/workflow_lint.py with a 50k ratchet."""
    lessons = _mk_tree(tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "workflow_lint.py").write_text(_STUB_LINT, encoding="utf-8")
    return lessons


def test_edited_tree_constants_bind_allow_side(tmp_path: Path) -> None:
    """Content over the REAL ratchet but under the edited tree's stub ratchet -> allow.

    Pins the #1269 constant-first ordering at edit time: a same-tree constant
    bump (here a stub lint with ratchet 50000) is honored, so 7000-byte
    content that the hook-repo lint would block passes.
    """
    lessons = _mk_stub_lint_tree(tmp_path)
    content = _sized([ROW_A, ROW_B], _LESSONS_RATCHET_BYTES + 1000)
    r = _run(_write_payload(lessons, content))
    assert r.returncode == 0, (r.returncode, r.stderr)


def test_edited_tree_constants_bind_block_side(tmp_path: Path) -> None:
    """Content over the stub's OWN ratchet -> the stub blocks (not fail-open).

    Companion to the allow-side case: proves the stub actually loaded and
    produced the block (its error string appears), so the allow-side pass
    cannot be a silent always-fail-open regression.
    """
    lessons = _mk_stub_lint_tree(tmp_path)
    content = _sized([ROW_A, ROW_B], 60000)
    r = _run(_write_payload(lessons, content))
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "stub-ratchet" in r.stderr, r.stderr


def test_live_lessons_rewrite_allowed() -> None:
    """Day-one safety: rewriting the live LESSONS.md verbatim is allowed.

    If main's lint ever goes red this fails together with the workflow_lint
    default-run test — no new flake class.
    """
    live = _REPO_ROOT / ".claude" / "rules" / "LESSONS.md"
    r = _run(_write_payload(live, live.read_text(encoding="utf-8")))
    assert r.returncode == 0, (r.returncode, r.stderr)


def _main_repo_root() -> str | None:
    """Canonical main-checkout root (parent of the shared .git common dir)."""
    r = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return str(Path(r.stdout.strip()).parent)


class TestSettingsWiring:
    """Parse .claude/settings.json and exercise the CONFIGURED command.

    Every other test drives ``SCRIPT`` directly; without this class a wrong
    command path / lost +x ships the hook inert with a green suite. Pre-merge
    worktree runs remap ONLY the canonical-root PREFIX of the configured
    absolute path onto this checkout (the #965 convention — wrong-directory /
    wrong-basename / missing-+x bugs still fail under the remap); on the main
    checkout no remap occurs.
    """

    def _configured_command(self) -> Path:
        settings = json.loads(SETTINGS.read_text())
        cmds = [
            h["command"]
            for entry in settings["hooks"]["PreToolUse"]
            if entry.get("matcher") == "Edit|Write"
            for h in entry.get("hooks", [])
            if h.get("type") == "command" and h.get("command", "").endswith(HOOK_REL)
        ]
        assert len(cmds) == 1, cmds
        cmd = cmds[0]
        assert os.path.isabs(cmd), cmd
        main_root = _main_repo_root()
        if main_root is not None and str(_REPO_ROOT) != main_root:
            prefix = main_root.rstrip("/") + "/"
            if cmd.startswith(prefix):
                cmd = str(_REPO_ROOT / cmd[len(prefix) :])
        return Path(cmd)

    def test_registered_under_edit_write_matcher(self) -> None:
        self._configured_command()

    def test_configured_command_exists_and_is_executable(self) -> None:
        cmd = self._configured_command()
        assert cmd.exists(), cmd
        assert os.access(cmd, os.X_OK), cmd

    def test_configured_command_blocks_overcap_write(self, tmp_path: Path) -> None:
        lessons = _mk_tree(tmp_path)
        payload = _write_payload(lessons, _sized([ROW_A, ROW_B], _LESSONS_MAX_BYTES + 200))
        r = _run(payload, script=self._configured_command())
        assert r.returncode == 2, (r.returncode, r.stderr)
        assert "BLOCKED" in r.stderr, r.stderr
