"""Unit tests for scripts/step5a_coupling_check.py (#2327).

Covers the four hard-won spec points from the #2327 plan v3 §4.1:
- `_extract_caps` accepts BOTH `ast.Assign` and `ast.AnnAssign` (the #2303
  raise landed as an AnnAssign; an Assign-only walker silently skips the
  regime) — pinned against the LIVE workflow_lint.py so a lint-side
  node-class migration degrades loudly in CI.
- ONE tree-state basis (worktree-vs-origin/main) — branch-authored regrowth
  is in the divergence set and never WARNs, committed or not.
- Advisory degrade: a missing constant skips its regime with a notice,
  never a crash.
- Parity pins: `_ISSUE_NUMBER_RE` vs the sibling probe; `SIBLING_PATHSPECS`
  vs the Step 5a sibling-arm diff line in 09-step-5.md.

Import hygiene: modules are loaded via `importlib.util.spec_from_file_location`
under test-unique names (never a cached `sys.modules` entry).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_HELPER_PATH = _REPO / "scripts" / "step5a_coupling_check.py"

_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
}


def _import_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    # register under the TEST-UNIQUE name (dataclass decorators resolve
    # cls.__module__ through sys.modules; never the real module name)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


helper = _import_by_path("step5a_coupling_check_2327_under_test", _HELPER_PATH)


# ---------------------------------------------------------------------------
# git fixture plumbing (mkdtemp, not tmp_path: concurrent pytest sessions
# prune /tmp/pytest-of-* numbered roots mid-test — see the family-sync tests)
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(cwd),
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=T",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        env=_ENV,
    )
    assert proc.returncode == 0, f"git {args} failed:\n{proc.stderr}"
    return proc.stdout


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _fs_literal(items: tuple[str, ...]) -> str:
    if not items:
        return "frozenset()"
    inner = ", ".join(repr(i) for i in sorted(items))
    return "frozenset({" + inner + "})"


def _lint_src(
    *,
    grandfather: dict[str, int] | None = None,
    fail: int = 60,
    lessons: int = 10_000,
    agent_fail: int = 40_000,
    exempt_segs: tuple[str, ...] = (),
    gen_exempt: tuple[str, ...] = (),
    omit: tuple[str, ...] = (),
) -> str:
    """Compose a fixture workflow_lint.py carrying the six cap constants."""
    gf = dict(grandfather or {})
    lines = ['"""fixture lint (step5a coupling-check unit tests, #2327)."""']
    if "SKILL_DOC_FAIL_BYTES" not in omit:
        lines.append(f"SKILL_DOC_FAIL_BYTES = {fail}")
    if "SKILL_DOC_GENERATED_EXEMPT" not in omit:
        lines.append(f"SKILL_DOC_GENERATED_EXEMPT: frozenset[str] = {_fs_literal(gen_exempt)}")
    if "SKILL_DOC_EXEMPT_DIR_SEGMENTS" not in omit:
        lines.append(f"SKILL_DOC_EXEMPT_DIR_SEGMENTS: frozenset[str] = {_fs_literal(exempt_segs)}")
    if "SKILL_DOC_SIZE_GRANDFATHER" not in omit:
        lines.append(f"SKILL_DOC_SIZE_GRANDFATHER: dict[str, int] = {gf!r}")
    if "_LESSONS_MAX_BYTES" not in omit:
        lines.append(f"_LESSONS_MAX_BYTES = {lessons}")
    if "AGENT_SPEC_FAIL_BYTES" not in omit:
        lines.append(f"AGENT_SPEC_FAIL_BYTES = {agent_fail}")
    return "\n".join(lines) + "\n"


def _mini_repo(
    tmp: Path,
    fork_files: dict[str, str],
    main_commits: tuple[dict[str, str], ...] = (),
) -> Path:
    """origin bare + seed (fork commit, then per-dict main advances) + wt clone
    on branch issue-9999 with origin/main fetched. wt clones at the FORK tip."""
    origin = tmp / "origin.git"
    _git(tmp, "init", "--bare", "-b", "main", str(origin))
    seed = tmp / "seed"
    _git(tmp, "clone", str(origin), str(seed))
    _write(seed, fork_files)
    _git(seed, "add", "-A")
    _git(seed, "commit", "-m", "fork-era")
    _git(seed, "push", "origin", "main")
    wt = tmp / "wt"
    _git(tmp, "clone", str(origin), str(wt))
    _git(wt, "checkout", "-b", "issue-9999")
    for files in main_commits:
        _write(seed, files)
        _git(seed, "add", "-A")
        _git(seed, "commit", "-m", "main-side advance")
        _git(seed, "push", "origin", "main")
    _git(wt, "fetch", "origin")
    return wt


def _run_main(wt: Path, capsys) -> tuple[int, str, str]:
    mb = _git(wt, "merge-base", "HEAD", "origin/main").strip()
    rc = helper.main(["--worktree", str(wt), "--merge-base", mb, "--own-issue", "9999"])
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


# ---------------------------------------------------------------------------
# 1-2: cap extraction
# ---------------------------------------------------------------------------


def test_cap_extraction_matches_live_lint():
    """All SIX cap names resolve from the LIVE workflow_lint.py via accepted
    Assign|AnnAssign nodes, with values equal to the imported module's — a
    future lint-side node-class migration degrades loudly here."""
    lint_path = _REPO / "scripts" / "workflow_lint.py"
    wl = _import_by_path("workflow_lint_live_2327_under_test", lint_path)
    caps = helper._extract_caps(lint_path.read_text(encoding="utf-8"))
    assert caps.missing == (), f"missing/non-literal on live lint: {caps.missing}"
    assert caps.values["SKILL_DOC_SIZE_GRANDFATHER"] == wl.SKILL_DOC_SIZE_GRANDFATHER
    assert caps.values["_LESSONS_MAX_BYTES"] == wl._LESSONS_MAX_BYTES
    assert caps.values["SKILL_DOC_FAIL_BYTES"] == wl.SKILL_DOC_FAIL_BYTES
    assert caps.values["AGENT_SPEC_FAIL_BYTES"] == wl.AGENT_SPEC_FAIL_BYTES
    assert set(caps.values["SKILL_DOC_EXEMPT_DIR_SEGMENTS"]) == set(
        wl.SKILL_DOC_EXEMPT_DIR_SEGMENTS
    )
    assert set(caps.values["SKILL_DOC_GENERATED_EXEMPT"]) == set(wl.SKILL_DOC_GENERATED_EXEMPT)
    # agent-caps grammar parity: helper parser == live loader on the live file
    caps_file = _REPO / ".claude" / "config" / "agent_spec_size_caps.txt"
    assert helper._parse_agent_caps(caps_file.read_text(encoding="utf-8")) == (
        wl._load_agent_spec_caps()
    )


def test_cap_extraction_both_node_classes():
    """Assign AND AnnAssign both extract; a value-less annotation is skipped
    (never a crash), with the later valued binding winning."""
    src = "\n".join(
        [
            "_LESSONS_MAX_BYTES: int",  # value-less annotation -> skipped
            "SKILL_DOC_FAIL_BYTES = 60_000",  # Assign int
            "_LESSONS_MAX_BYTES = 10492",  # Assign int (after value-less ann)
            "AGENT_SPEC_FAIL_BYTES: int = 40_000",  # AnnAssign int
            'SKILL_DOC_SIZE_GRANDFATHER: dict[str, int] = {"a/SKILL.md": 100_000}',
            'SKILL_DOC_GENERATED_EXEMPT = frozenset({"issue/markers.md"})',  # Assign call-form
            'SKILL_DOC_EXEMPT_DIR_SEGMENTS: frozenset[str] = frozenset({"templates"})',
        ]
    )
    caps = helper._extract_caps(src)
    assert caps.missing == ()
    assert caps.values["SKILL_DOC_FAIL_BYTES"] == 60_000
    assert caps.values["_LESSONS_MAX_BYTES"] == 10492
    assert caps.values["AGENT_SPEC_FAIL_BYTES"] == 40_000
    assert caps.values["SKILL_DOC_SIZE_GRANDFATHER"] == {"a/SKILL.md": 100_000}
    assert caps.values["SKILL_DOC_GENERATED_EXEMPT"] == frozenset({"issue/markers.md"})
    assert caps.values["SKILL_DOC_EXEMPT_DIR_SEGMENTS"] == frozenset({"templates"})


# ---------------------------------------------------------------------------
# 3-4: parity pins
# ---------------------------------------------------------------------------


def test_issue_regex_parity_with_probe():
    probe = _import_by_path(
        "step5a_sibling_probe_2327_under_test", _REPO / "scripts" / "step5a_sibling_probe.py"
    )
    assert helper._ISSUE_NUMBER_RE.pattern == probe._ISSUE_NUMBER_RE.pattern


def test_sibling_pathspec_parity_with_doc():
    """Every helper pathspec appears single-quoted on the Step 5a sibling-arm
    diff line, and the line carries no extra `:(glob)` pathspec the helper
    lacks (count equality => set equality given the containment check)."""
    doc = (_REPO / ".claude" / "skills" / "issue" / "steps" / "09-step-5.md").read_text(
        encoding="utf-8"
    )
    diff_lines = [
        ln
        for ln in doc.splitlines()
        if "diff --name-only origin/main" in ln and ":(glob)scripts/issue[0-9]*_*.py" in ln
    ]
    assert diff_lines, "sibling-arm diff line not found in 09-step-5.md"
    line = diff_lines[0]
    for ps in helper.SIBLING_PATHSPECS:
        assert f"'{ps}'" in line, f"pathspec {ps!r} missing from the doc's sibling diff line"
    assert line.count(":(glob)") == len(helper.SIBLING_PATHSPECS)


# ---------------------------------------------------------------------------
# 5-8: behavior on scratch repos
# ---------------------------------------------------------------------------


@pytest.fixture()
def scratch(tmp_path_factory):
    # mkdtemp outside /tmp/pytest-of-* (concurrent-session prune race)
    d = Path(tempfile.mkdtemp(prefix="step5a-coupling-unit-"))
    yield d
    import shutil

    shutil.rmtree(d, ignore_errors=True)


def test_branch_regrowth_is_silent(scratch, capsys):
    """Branch-authored doc growth (committed AND uncommitted) is in the
    divergence set => never a cap WARN, even over-cap on both sides."""
    fork_lint = _lint_src(grandfather={"issue/SKILL.md": 100}, fail=60)
    wt = _mini_repo(
        scratch,
        fork_files={
            "scripts/workflow_lint.py": fork_lint,
            ".claude/skills/issue/SKILL.md": "d" * 50,
        },
        # advance the lint on main so Arm A engages (lint in divergence set)
        main_commits=({"scripts/workflow_lint.py": fork_lint + "# main tweak\n"},),
    )
    # committed branch-side regrowth over BOTH caps
    _write(wt, {".claude/skills/issue/SKILL.md": "d" * 150})
    _git(wt, "add", "-A")
    _git(wt, "commit", "-m", "issue-9999: deliberate doc growth")
    rc, out, _ = _run_main(wt, capsys)
    assert rc == 0
    assert "cap-skew" not in out and "cap-red-on-main" not in out
    assert "coupling check: clean" in out
    # uncommitted further regrowth: same silence
    _write(wt, {".claude/skills/issue/SKILL.md": "d" * 300})
    rc, out, _ = _run_main(wt, capsys)
    assert rc == 0
    assert "cap-skew" not in out and "cap-red-on-main" not in out


def test_missing_constant_degrades_advisory(scratch, capsys):
    """A lint vintage missing _LESSONS_MAX_BYTES skips the LESSONS regime with
    a printed notice — rc stays 0, no crash, no WARN for LESSONS.md."""
    fork_lint = _lint_src(omit=("_LESSONS_MAX_BYTES",))
    wt = _mini_repo(
        scratch,
        fork_files={"scripts/workflow_lint.py": fork_lint},
        main_commits=(
            {
                "scripts/workflow_lint.py": fork_lint + "# main tweak\n",
                ".claude/rules/LESSONS.md": "L" * 80,
            },
        ),
    )
    # half-sync: LESSONS.md synced from origin/main, lint withheld (not synced)
    _git(wt, "checkout", "origin/main", "--", ".claude/rules/LESSONS.md")
    rc, out, _ = _run_main(wt, capsys)
    assert rc == 0
    assert "LESSONS regime skipped" in out
    assert "WARN" not in out or "LESSONS.md" not in out


def test_exempt_dirs_skipped(scratch, capsys):
    """Docs exempt via SKILL_DOC_EXEMPT_DIR_SEGMENTS or
    SKILL_DOC_GENERATED_EXEMPT never WARN, even over every cap."""
    fork_lint = _lint_src(fail=60, exempt_segs=("templates",), gen_exempt=("issue/markers.md",))
    wt = _mini_repo(
        scratch,
        fork_files={"scripts/workflow_lint.py": fork_lint},
        main_commits=(
            {
                "scripts/workflow_lint.py": fork_lint + "# main tweak\n",
                ".claude/skills/foo/templates/big.md": "x" * 150,
                ".claude/skills/issue/markers.md": "m" * 150,
            },
        ),
    )
    _git(wt, "checkout", "origin/main", "--", ".claude/skills")
    rc, out, _ = _run_main(wt, capsys)
    assert rc == 0
    assert "WARN" not in out
    assert "coupling check: clean" in out


def test_agent_caps_fallback(scratch, capsys):
    """An agent doc with no caps-file entry falls back to
    AGENT_SPEC_FAIL_BYTES on each side; a branch-vintage fallback that rejects
    while main's admits is a cap-skew naming AGENT_SPEC_FAIL_BYTES."""
    fork_lint = _lint_src(agent_fail=60)
    main_lint = _lint_src(agent_fail=200)
    wt = _mini_repo(
        scratch,
        fork_files={
            "scripts/workflow_lint.py": fork_lint,
            ".claude/config/agent_spec_size_caps.txt": "x.md 1_000  # unrelated entry\n",
        },
        main_commits=(
            {
                "scripts/workflow_lint.py": main_lint,
                ".claude/agents/y.md": "y" * 100,
            },
        ),
    )
    # half-sync: the agent doc synced, the lint (cap raise) withheld
    _git(wt, "checkout", "origin/main", "--", ".claude/agents/y.md")
    rc, out, _ = _run_main(wt, capsys)
    assert rc == 0
    skew_lines = [ln for ln in out.splitlines() if "[step5a] WARN: cap-skew:" in ln]
    assert len(skew_lines) == 1, out
    assert ".claude/agents/y.md" in skew_lines[0]
    assert "AGENT_SPEC_FAIL_BYTES" in skew_lines[0]
    assert "= 60" in skew_lines[0] and "200" in skew_lines[0]
    assert "1 cap-skew, 0 sibling-split, 0 cap-red-on-main" in out
