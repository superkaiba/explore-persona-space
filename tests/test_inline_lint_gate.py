"""Unit + subprocess tests for ``scripts/inline_lint_gate.py`` (#1500).

The helper mechanizes the SKILL.md Step 9a-ter § Inline payload lint gate
verdict semantics (#1460) and writes the content-hash-bound certification
lines ``.claude/hooks/guard_root_code_commit.sh`` validates. Exit codes:
0 = PASS (every payload path certified), 1 = BLOCK, 3 = INCONCLUSIVE
(instrument-ran completeness failure / TOCTOU mid-gate edit; no cert).

Subprocess tests drive the CLI hermetically: tmp git repos with a SYNTHETIC
``origin/main`` (``git update-ref refs/remotes/origin/main HEAD``) and the
documented test-only leg-command overrides ``EPM_INLINE_GATE_LINT_CMD`` /
``EPM_INLINE_GATE_MAP_CMD`` / ``EPM_INLINE_GATE_PYTEST_CMD`` (each ``cat``s a
pre-written output file), so no test runs the real multi-minute legs. Direct
function tests cover the TSV pair parser, added-line-range parsing, and the
TOCTOU + trim behavior of ``write_cert``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.issue_skill_source import issue_skill_text

_REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = _REPO_ROOT / "scripts" / "inline_lint_gate.py"

_spec = importlib.util.spec_from_file_location("inline_lint_gate", SCRIPT)
assert _spec and _spec.loader
ilg = importlib.util.module_from_spec(_spec)
# Register BEFORE exec_module: @dataclass resolves cls.__module__ through
# sys.modules at class-creation time and crashes on an unregistered module.
sys.modules["inline_lint_gate"] = ilg
_spec.loader.exec_module(ilg)

LINT_OK = "workflow_lint: PASS\n"
LINT_FAIL_TERMINAL = "workflow_lint: FAIL (1 error(s))\n"


@pytest.fixture(autouse=True)
def _pin_load_guard_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Determinism pin (#2039 T0, round-1 critic blocker 1): disable the load
    guard for EVERY test in this module — including the ones that hand-build
    their subprocess env via ``os.environ.copy()`` and bypass ``_run_gate``
    (e.g. ``test_pytest_leg_env_carries_scan_extra_files``), which inherit
    the pin because it lands in the PROCESS env before the copy. Without it,
    every red-leg test expecting exit 1 would flake into
    INCONCLUSIVE-under-load whenever the REAL VM load1 >= 20 at test time (a
    self-referential load flake in the load-flake fix), and the default
    300 s pre-pytest wait would add real sleeps. FUNCTION-scoped autouse
    (``scope="module"`` with monkeypatch raises ScopeMismatch); the #2039
    load tests re-enable per-test via ``extra_env`` / ``monkeypatch``."""
    monkeypatch.setenv("EPM_GATE_LOAD_MAX", "0")  # kill switch: guard disabled
    monkeypatch.setenv("EPM_GATE_LOAD_WAIT_S", "0")  # belt+braces: never sleep
    monkeypatch.delenv("EPM_GATE_LOAD1_OVERRIDE", raising=False)  # no host leak


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def _make_repo(tmp_path: Path) -> Path:
    """Tmp repo with scripts/mod.py committed and a synthetic origin/main."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    (repo / "scripts").mkdir()
    (repo / "scripts" / "mod.py").write_text("l1\nl2\nl3\nl4\nl5\n", encoding="utf-8")
    _git(repo, "add", "--", "scripts/mod.py")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


def _run_gate(
    repo: Path,
    payload: list[str],
    tmp_path: Path,
    *,
    lint_out: str,
    map_out: str = "",
    pytest_out: str = "",
    lint_cmd_extra: str = "",
    lint_cmd_prefix: str = "",
    payload_name: str = "payload.txt",
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    payload_file = tmp_path / payload_name
    payload_file.write_text("\n".join(payload) + "\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    # Merged AFTER the copy so #2039 load tests can override the module-wide
    # _pin_load_guard_off determinism pin per-test.
    if extra_env:
        env.update(extra_env)
    for name, text in (
        ("EPM_INLINE_GATE_LINT_CMD", lint_out),
        ("EPM_INLINE_GATE_MAP_CMD", map_out),
        ("EPM_INLINE_GATE_PYTEST_CMD", pytest_out),
    ):
        leg_file = tmp_path / f"{name}.txt"
        leg_file.write_text(text, encoding="utf-8")
        env[name] = f"cat {leg_file}"
    if lint_cmd_extra:
        # Runs with shell=True + cwd=repo inside _run_leg — lets a test mutate
        # the repo MID-GATE (after read_payload snapshots) for TOCTOU cases.
        env["EPM_INLINE_GATE_LINT_CMD"] += f" && {lint_cmd_extra}"
    if lint_cmd_prefix:
        # Runs BEFORE the terminal-line cat: a failing prefix suppresses the
        # healthy lint terminal line (-> INCONCLUSIVE), so the LEG's own
        # success can pin gate-side pre-leg ordering (#1950 purge pin).
        env["EPM_INLINE_GATE_LINT_CMD"] = f"{lint_cmd_prefix} && " + env["EPM_INLINE_GATE_LINT_CMD"]
    env["EPM_INLINE_CERT_PATH"] = str(tmp_path / "cert.txt")
    # Zero the #1857 settle-and-re-hash retry delay: subprocess TOCTOU cases
    # stay deterministic-fast (the retry still RUNS — it just doesn't wait).
    env["EPM_CERT_REHASH_DELAY_S"] = "0"
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--issue",
            "9999",
            "--payload-file",
            str(payload_file),
            "--repo-root",
            str(repo),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        env=env,
    )


def _cert_lines(tmp_path: Path) -> list[str]:
    cert = tmp_path / "cert.txt"
    if not cert.exists():
        return []
    return cert.read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------
# INCONCLUSIVE (exit 3, no cert)
# ---------------------------------------------------------------------------
def test_empty_payload_file_inconclusive(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    payload_file = tmp_path / "payload.txt"
    payload_file.write_text("\n   \n\n", encoding="utf-8")  # blank lines only
    env = os.environ.copy()
    env["EPM_INLINE_CERT_PATH"] = str(tmp_path / "cert.txt")
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--issue",
            "9999",
            "--payload-file",
            str(payload_file),
            "--repo-root",
            str(repo),
            "--out-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)
    assert "inline_lint_gate: INCONCLUSIVE" in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


def test_lint_leg_missing_terminal_line_inconclusive(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    r = _run_gate(repo, ["scripts/mod.py"], tmp_path, lint_out="some unrelated output\n")
    assert r.returncode == 3, (r.returncode, r.stdout)
    assert "lint-leg-dead" in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


def test_schema_fail_early_exit_inconclusive(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    r = _run_gate(repo, ["scripts/mod.py"], tmp_path, lint_out="workflow_lint: schema FAIL\nboom\n")
    assert r.returncode == 3, (r.returncode, r.stdout)
    assert _cert_lines(tmp_path) == []


def test_nonempty_map_without_pytest_summary_inconclusive(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out="",  # dead pytest leg
    )
    assert r.returncode == 3, (r.returncode, r.stdout)
    assert "pytest-leg-dead" in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


# ---------------------------------------------------------------------------
# PASS (exit 0) — repo-wide pre-existing red never blocks; cert shape correct
# ---------------------------------------------------------------------------
def test_repo_wide_red_on_non_payload_passes_and_certifies(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out="workflow_lint: scripts/other.py:3: bad\n" + LINT_FAIL_TERMINAL,
    )
    assert r.returncode == 0, (r.returncode, r.stdout)
    assert "inline_lint_gate: PASS" in r.stdout, r.stdout
    lines = _cert_lines(tmp_path)
    assert len(lines) == 1, lines
    tag, epoch, sha, path = lines[0].split(" ", 3)
    assert tag == "v1" and epoch.isdigit(), lines[0]
    assert path == "scripts/mod.py"
    assert sha == _git(repo, "hash-object", "--", str(repo / "scripts" / "mod.py")).strip()


def test_warn_prefixed_payload_lines_never_block(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out="WARN — untested touched file: scripts/mod.py\n" + LINT_OK,
    )
    assert r.returncode == 0, (r.returncode, r.stdout)
    assert len(_cert_lines(tmp_path)) == 1


# ---------------------------------------------------------------------------
# BLOCK (exit 1)
# ---------------------------------------------------------------------------
def test_new_file_with_hit_blocks_but_clean_sibling_certifies(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "scripts" / "new.py").write_text("print(1)\n", encoding="utf-8")  # not on origin/main
    r = _run_gate(
        repo,
        ["scripts/mod.py", "scripts/new.py"],
        tmp_path,
        lint_out="workflow_lint: scripts/new.py:1: bare list_repo_tree\n" + LINT_FAIL_TERMINAL,
    )
    assert r.returncode == 1, (r.returncode, r.stdout)
    assert "inline_lint_gate: BLOCK (scripts/new.py)" in r.stdout, r.stdout
    lines = _cert_lines(tmp_path)
    assert len(lines) == 1 and lines[0].endswith(" scripts/mod.py"), lines


def test_mixed_block_and_toctou_prints_toctou_note_before_block(tmp_path: Path) -> None:
    """Round-2 Minor: in a mixed BLOCK+TOCTOU outcome the mid-gate-edit note
    must print BEFORE the exit-1 BLOCK return — previously the operator only
    learned of the uncertified TOCTOU path on the next hook block."""
    repo = _make_repo(tmp_path)
    (repo / "scripts" / "new.py").write_text("print(1)\n", encoding="utf-8")  # not on origin/main
    r = _run_gate(
        repo,
        ["scripts/mod.py", "scripts/new.py"],
        tmp_path,
        lint_out="workflow_lint: scripts/new.py:1: bad hit\n" + LINT_FAIL_TERMINAL,
        # Edit the PASSING payload path mid-gate (after read_payload snapshots).
        lint_cmd_extra="printf 'edited\\n' >> scripts/mod.py",
    )
    assert r.returncode == 1, (r.returncode, r.stdout)
    toctou_at = r.stdout.find("INCONCLUSIVE (edited during gate — re-run: scripts/mod.py)")
    block_at = r.stdout.find("inline_lint_gate: BLOCK (scripts/new.py)")
    assert toctou_at != -1 and block_at != -1, r.stdout
    assert toctou_at < block_at, r.stdout
    assert _cert_lines(tmp_path) == []  # blocked + toctou: nothing certified


def _repo_with_added_lines(tmp_path: Path) -> Path:
    """scripts/mod.py on origin/main is 5 lines; worktree appends lines 6-7."""
    repo = _make_repo(tmp_path)
    (repo / "scripts" / "mod.py").write_text(
        "l1\nl2\nl3\nl4\nl5\nadded6\nadded7\n", encoding="utf-8"
    )
    return repo


def test_modified_file_hit_outside_added_ranges_passes(tmp_path: Path) -> None:
    repo = _repo_with_added_lines(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out="workflow_lint: scripts/mod.py:2: pre-existing red\n" + LINT_FAIL_TERMINAL,
    )
    assert r.returncode == 0, (r.returncode, r.stdout)
    assert "pre-existing" in r.stdout, r.stdout  # reported, not re-buried
    assert len(_cert_lines(tmp_path)) == 1


def test_modified_file_hit_inside_added_ranges_blocks(tmp_path: Path) -> None:
    repo = _repo_with_added_lines(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out="workflow_lint: scripts/mod.py:6: new red\n" + LINT_FAIL_TERMINAL,
    )
    assert r.returncode == 1, (r.returncode, r.stdout)
    assert _cert_lines(tmp_path) == []


def test_modified_file_hit_without_lineno_blocks_conservatively(tmp_path: Path) -> None:
    repo = _repo_with_added_lines(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out="FAILED tests/test_x.py::test_y - scans scripts/mod.py\n1 failed\n",
    )
    assert r.returncode == 1, (r.returncode, r.stdout)
    assert "without a parseable lineno" in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


# ---------------------------------------------------------------------------
# Warnings-summary attribution rows (#1585; the #1112 false-block incident):
# bare node-id headers / `path: N warnings` aggregates INSIDE pytest's fenced
# "warnings summary" section report instead of blocking; everything outside
# the section (or in the lint leg) keeps the conservative block.
# ---------------------------------------------------------------------------
def _incident_pytest_out(node: str) -> str:
    """Verbatim pytest 9.0.2 ``-q -rA`` shape from the #1585 plan §2 probe:
    fenced warnings-summary header, non-indented bare node-id rows (x2, one
    per warning group — the #1112 incident repetition), indented warning
    bodies pointing at site-packages files, the docs-link line, fenced
    PASSES + short-summary sections, and the UNfenced final summary line."""
    return (
        "..                                                                       [100%]\n"
        "=============================== warnings summary ===============================\n"
        f"{node}\n"
        "  /usr/lib/python3.11/site-packages/torch/utils/_pytree.py:185: "
        "DeprecationWarning: legacy\n"
        '    warnings.warn("legacy", DeprecationWarning)\n'
        f"{node}\n"
        "  /usr/lib/python3.11/site-packages/swig_runtime.py:3: DeprecationWarning: "
        "builtin type swigvarlink has no __module__ attribute\n"
        "-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html\n"
        "==================================== PASSES ====================================\n"
        "=========================== short test summary info ============================\n"
        f"PASSED {node}\n"
        "29 passed, 2 warnings in 3.21s\n"
    )


def test_warnings_summary_attribution_reports_not_blocks(tmp_path: Path) -> None:
    """Incident-shape repro + durability pin (#1112): a 29/29-green suite whose
    warnings summary attributes environmental warnings to a MODIFIED payload
    test file must PASS and certify, with the attribution rows reported."""
    repo = _repo_with_added_lines(tmp_path)
    node = "scripts/mod.py::test_x"
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out=_incident_pytest_out(node),
    )
    assert r.returncode == 0, (r.returncode, r.stdout)
    assert f"[warnings-summary attribution] {node}" in r.stdout, r.stdout
    assert "conservative block" not in r.stdout, r.stdout
    lines = _cert_lines(tmp_path)
    assert len(lines) == 1 and lines[0].endswith(" scripts/mod.py"), lines


def test_new_file_warnings_summary_attribution_passes(tmp_path: Path) -> None:
    """The reclassification covers the NEW-on-origin/main branch too (its
    "any non-WARN hit blocks" rule would otherwise false-block, D1)."""
    repo = _make_repo(tmp_path)
    (repo / "scripts" / "new.py").write_text("print(1)\n", encoding="utf-8")  # not on origin/main
    node = "scripts/new.py::test_x"
    r = _run_gate(
        repo,
        ["scripts/new.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/new.py\n",
        pytest_out=_incident_pytest_out(node),
    )
    assert r.returncode == 0, (r.returncode, r.stdout)
    assert f"[warnings-summary attribution] {node}" in r.stdout, r.stdout
    lines = _cert_lines(tmp_path)
    assert len(lines) == 1 and lines[0].endswith(" scripts/new.py"), lines


def test_short_summary_failed_line_still_blocks(tmp_path: Path) -> None:
    """A FAILED row naming the payload (short-test-summary section) still
    blocks — it carries spaces + tokens, so the attribution row shape never
    matches, and the short-summary fence is not a warnings-summary title."""
    repo = _repo_with_added_lines(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out=(
            "=========================== short test summary info ============================\n"
            "FAILED scripts/mod.py::test_x - AssertionError\n"
            "1 failed, 28 passed in 3.21s\n"
        ),
    )
    assert r.returncode == 1, (r.returncode, r.stdout)
    assert "inline_lint_gate: BLOCK (scripts/mod.py)" in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


def test_bare_node_id_outside_section_still_conservative_blocks(tmp_path: Path) -> None:
    """Double predicate (D3): the bare-row shape alone never whitelists — a
    node id OUTSIDE any warnings-summary section keeps the conservative block."""
    repo = _repo_with_added_lines(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out="scripts/mod.py::test_x\n1 passed in 0.01s\n",  # no fence anywhere before it
    )
    assert r.returncode == 1, (r.returncode, r.stdout)
    assert "conservative block" in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


def test_aggregated_warning_count_row_reports_not_blocks(tmp_path: Path) -> None:
    """pytest's aggregated `<path>: N warnings` row inside the section reports
    (its `path: N` shape carries no parseable `path:<lineno>:` — A4), and the
    section is subsequently CLOSED by a fence (pins window-close placement)."""
    repo = _repo_with_added_lines(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out=(
            "=============================== warnings summary ===============================\n"
            "scripts/mod.py: 3 warnings\n"
            "-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html\n"
            "=========================== short test summary info ============================\n"
            "3 passed, 3 warnings in 0.05s\n"
        ),
    )
    assert r.returncode == 0, (r.returncode, r.stdout)
    assert "[warnings-summary attribution] scripts/mod.py: 3 warnings" in r.stdout, r.stdout
    assert len(_cert_lines(tmp_path)) == 1


def test_lint_leg_fence_never_opens_whitelist(tmp_path: Path) -> None:
    """Section tracking is scoped to the PYTEST leg only (D2): a fenced
    warnings-summary header in the LINT leg opens no whitelist window, so a
    bare node id naming the payload there still conservative-blocks."""
    repo = _make_repo(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=(
            "=============================== warnings summary ===============================\n"
            "scripts/mod.py::test_x\n" + LINT_FAIL_TERMINAL
        ),
    )
    assert r.returncode == 1, (r.returncode, r.stdout)
    assert "conservative block" in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


def test_bare_node_id_after_section_close_still_blocks(tmp_path: Path) -> None:
    """Window-CLOSE transition pin: any non-warnings fence (here short test
    summary info) RESETS the section, so a bare payload node id AFTER the
    close (captured-stdout echo shape) keeps the conservative block. An
    implementation that never closes the window fails this test. (The
    PASSES-fence-close case moved to
    test_passes_section_closed_by_next_fence_still_blocks once #2023 made
    the PASSES section itself a report-class window.)"""
    repo = _repo_with_added_lines(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out=(
            "=============================== warnings summary ===============================\n"
            "tests/test_other.py::test_benign\n"
            "=========================== short test summary info ============================\n"
            "scripts/mod.py::test_x\n"
            "1 passed, 1 warning in 0.02s\n"
        ),
    )
    assert r.returncode == 1, (r.returncode, r.stdout)
    assert "conservative block" in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


# ---------------------------------------------------------------------------
# PASSES-section carve-out (#2023; the #1345 v242 false-block incident):
# EVERY pytest-leg line inside the -rA fenced "PASSES" section is captured
# output of a test pytest reports as PASSED — definitionally not red
# evidence — so the WHOLE section reports ([passing-capture]) instead of
# blocking; the FAILURES section and everything after the PASSES window
# closes keep their existing block behavior.
# ---------------------------------------------------------------------------
def _passes_capture_pytest_out(frame_path: str) -> str:
    """pytest 9.0.2 ``-q -rA`` shape of the #1345 v242 incident: a
    designed-crash test that PASSED echoes its captured stderr traceback
    (naming the payload's absolute path, lineno-less under the gate's
    ``path:<lineno>:`` parse) inside the fenced PASSES section; the terminal
    summary is all-green."""
    return (
        "..                                                                       [100%]\n"
        "==================================== PASSES ====================================\n"
        "_________________________ test_designed_crash_recovery _________________________\n"
        "----------------------------- Captured stderr call -----------------------------\n"
        "Traceback (most recent call last):\n"
        f'  File "/workspace/explore-persona-space/{frame_path}", line 2342, '
        "in _fit_within_cells\n"
        '    raise RuntimeError("designed crash")\n'
        "=========================== short test summary info ============================\n"
        "572 passed in 41.20s\n"
    )


def test_passes_section_traceback_reports_not_blocks(tmp_path: Path) -> None:
    """Fails-pre-fix pin (#2023): a captured traceback frame naming a MODIFIED
    payload inside the PASSES section must PASS, certify, and report the
    line under the [passing-capture] label — not conservative-block."""
    repo = _repo_with_added_lines(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out=_passes_capture_pytest_out("scripts/mod.py"),
    )
    assert r.returncode == 0, (r.returncode, r.stdout)
    assert "[passing-capture]" in r.stdout, r.stdout
    assert "conservative block" not in r.stdout, r.stdout
    lines = _cert_lines(tmp_path)
    assert len(lines) == 1 and lines[0].endswith(" scripts/mod.py"), lines


def test_new_file_passes_section_traceback_passes(tmp_path: Path) -> None:
    """The PASSES carve-out covers the NEW-on-origin/main branch too (mirror
    of test_new_file_warnings_summary_attribution_passes — the "any non-WARN
    hit blocks" rule would otherwise false-block)."""
    repo = _make_repo(tmp_path)
    (repo / "scripts" / "new.py").write_text("print(1)\n", encoding="utf-8")  # not on origin/main
    r = _run_gate(
        repo,
        ["scripts/new.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/new.py\n",
        pytest_out=_passes_capture_pytest_out("scripts/new.py"),
    )
    assert r.returncode == 0, (r.returncode, r.stdout)
    assert "[passing-capture]" in r.stdout, r.stdout
    lines = _cert_lines(tmp_path)
    assert len(lines) == 1 and lines[0].endswith(" scripts/new.py"), lines


def test_failures_section_traceback_still_blocks(tmp_path: Path) -> None:
    """Regression guard: the carve-out is PASSES-only — the SAME traceback
    frame inside a fenced FAILURES section keeps the conservative block."""
    repo = _repo_with_added_lines(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out=(
            "=================================== FAILURES ===================================\n"
            "_________________________ test_designed_crash_recovery _________________________\n"
            '  File "/workspace/explore-persona-space/scripts/mod.py", line 2342, '
            "in _fit_within_cells\n"
            "1 failed in 3.21s\n"
        ),
    )
    assert r.returncode == 1, (r.returncode, r.stdout)
    assert "conservative block" in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


def test_passes_section_closed_by_next_fence_still_blocks(tmp_path: Path) -> None:
    """Window-close guard for the NEW window: a payload-naming lineno-less
    hit AFTER the PASSES section is closed by the next fence (short test
    summary info) keeps the conservative block."""
    repo = _repo_with_added_lines(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out=(
            "==================================== PASSES ====================================\n"
            "tests/test_other.py::test_benign\n"
            "=========================== short test summary info ============================\n"
            "scripts/mod.py::test_x\n"
            "1 passed, 1 warning in 0.02s\n"
        ),
    )
    assert r.returncode == 1, (r.returncode, r.stdout)
    assert "conservative block" in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


def test_lineno_bearing_hit_inside_passes_section_reports(tmp_path: Path) -> None:
    """Pins the deliberate WHOLE-section semantics (#2023 method delta vs
    #1585): a lineno-BEARING captured warning line inside PASSES whose lineno
    (6) sits INSIDE the round's added range still reports instead of blocking
    via the added-lines branch."""
    repo = _repo_with_added_lines(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out=(
            "==================================== PASSES ====================================\n"
            "scripts/mod.py:6: DeprecationWarning: legacy\n"
            "=========================== short test summary info ============================\n"
            "1 passed, 1 warning in 0.02s\n"
        ),
    )
    assert r.returncode == 0, (r.returncode, r.stdout)
    assert "[passing-capture] scripts/mod.py:6: DeprecationWarning: legacy" in r.stdout, r.stdout
    assert len(_cert_lines(tmp_path)) == 1


# ---------------------------------------------------------------------------
# Direct function tests: parser, ranges, TOCTOU, trim
# ---------------------------------------------------------------------------
def test_parse_map_pairs_is_pair_generic() -> None:
    out = "a.py\tb.py\nx\ty\tEXTRA-COL\nmalformed-no-tab\n\n"
    assert ilg.parse_map_pairs(out) == [("a.py", "b.py"), ("x", "y")]


def test_mapped_pytest_timeout_floor_matches_selector(tmp_path: Path) -> None:
    """Round-2 Minor: the mapped-pytest timeout is floored at the CANONICAL
    select_step9c_tests.TIMEOUT_FLOOR_S — 1 mapped non-slow test must get the
    900 s floor, not the bare 150 s formula value."""
    spec = importlib.util.spec_from_file_location(
        "select_step9c_tests_floor_pin", _REPO_ROOT / "scripts" / "select_step9c_tests.py"
    )
    assert spec and spec.loader
    sel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sel)
    assert ilg.PYTEST_TIMEOUT_FLOOR_S == sel.TIMEOUT_FLOOR_S  # parity pin
    # Surcharge parity (#1646): the deliberately-duplicated ilg constant must
    # track the selector's SLOW_TESTS entry — drift here is a silent re-split
    # of the two gates' sizing.
    assert sel.SLOW_TESTS["tests/test_workflow_lint.py"] == ilg.PYTEST_WORKFLOW_LINT_SURCHARGE_S
    # 1 non-slow file: formula 120+30=150 < floor -> floored.
    assert ilg.mapped_pytest_timeout(["tests/test_x.py"]) == sel.TIMEOUT_FLOOR_S
    # Slow-surcharge case stays above the floor (120 + 30 + 2400) and matches
    # the selector's own sizing for the identical selection.
    assert ilg.mapped_pytest_timeout(["tests/test_workflow_lint.py"]) == 2550
    assert ilg.mapped_pytest_timeout(["tests/test_workflow_lint.py"]) == sel.recommended_timeout_s(
        ["tests/test_workflow_lint.py"]
    )


def test_added_line_ranges_parses_u0_hunks(tmp_path: Path) -> None:
    repo = _repo_with_added_lines(tmp_path)
    assert ilg.added_line_ranges(repo, "scripts/mod.py") == [(6, 8)]
    assert ilg.added_line_ranges(repo, "scripts/other.py") == []


def test_write_cert_toctou_refuses_edited_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EPM_CERT_REHASH_DELAY_S", "0")  # keep the #1857 retry instant
    repo = _make_repo(tmp_path)
    (repo / "scripts" / "sib.py").write_text("print(0)\n", encoding="utf-8")
    snapshots = ilg.read_payload(["scripts/mod.py", "scripts/sib.py"], repo)
    # Edit mod.py DURING the (simulated) gate run.
    (repo / "scripts" / "mod.py").write_text("edited mid-gate\n", encoding="utf-8")
    cert = tmp_path / "cert.txt"
    certified, toctou = ilg.write_cert(["scripts/mod.py", "scripts/sib.py"], snapshots, cert, repo)
    assert toctou == ["scripts/mod.py"]
    assert certified == ["scripts/sib.py"]
    lines = cert.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and lines[0].endswith(" scripts/sib.py"), lines


def test_write_cert_transient_flip_recovers_after_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1857 (a): a TRANSIENT worktree flip — mismatch at the first pass,
    settled back to the read_payload snapshot by the retry re-hash — is
    certified as normal (cert line carries the snapshot sha), no TOCTOU."""
    repo = _make_repo(tmp_path)
    target = repo / "scripts" / "mod.py"
    original = target.read_text(encoding="utf-8")
    snapshots = ilg.read_payload(["scripts/mod.py"], repo)
    target.write_text("transient flip\n", encoding="utf-8")

    def _settle(_delay: float) -> None:
        # Deterministic stand-in for the settle window: the concurrent
        # writer restores the snapshot content during the retry delay.
        target.write_text(original, encoding="utf-8")

    monkeypatch.setattr(ilg.time, "sleep", _settle)
    cert = tmp_path / "cert.txt"
    certified, toctou = ilg.write_cert(["scripts/mod.py"], snapshots, cert, repo)
    assert certified == ["scripts/mod.py"] and toctou == [], (certified, toctou)
    lines = cert.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, lines
    assert lines[0].split()[2] == snapshots["scripts/mod.py"], lines


def test_write_cert_stable_mismatch_sleeps_once_and_stays_toctou(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1857 (b): a STABLE mismatch takes exactly ONE settle sleep, the
    re-hash still mismatches, and the verdict stays TOCTOU (no cert line)."""
    repo = _make_repo(tmp_path)
    snapshots = ilg.read_payload(["scripts/mod.py"], repo)
    (repo / "scripts" / "mod.py").write_text("edited mid-gate\n", encoding="utf-8")
    slept: list[float] = []
    monkeypatch.setattr(ilg.time, "sleep", lambda s: slept.append(s))
    cert = tmp_path / "cert.txt"
    certified, toctou = ilg.write_cert(["scripts/mod.py"], snapshots, cert, repo)
    assert toctou == ["scripts/mod.py"] and certified == [], (certified, toctou)
    # #1992: filter out interpreter-internal backoff sleeps (<=0.05s each,
    # from subprocess.Popen.wait(timeout) under load) captured by the
    # process-global time.sleep patch; only settle-scale sleeps pin #1857.
    settle = [s for s in slept if s >= 1.0]
    assert len(settle) == 1, slept
    assert not cert.exists(), "stable mismatch must not write a cert line"


def test_write_cert_malformed_rehash_delay_falls_back_and_still_toctous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1857: a malformed EPM_CERT_REHASH_DELAY_S never crashes write_cert —
    the default delay is used and a stable mismatch still refuses the cert
    (fail toward TOCTOU/block, never a skipped re-check)."""
    repo = _make_repo(tmp_path)
    snapshots = ilg.read_payload(["scripts/mod.py"], repo)
    (repo / "scripts" / "mod.py").write_text("edited mid-gate\n", encoding="utf-8")
    monkeypatch.setenv("EPM_CERT_REHASH_DELAY_S", "not-a-number")
    slept: list[float] = []
    monkeypatch.setattr(ilg.time, "sleep", lambda s: slept.append(s))
    cert = tmp_path / "cert.txt"
    certified, toctou = ilg.write_cert(["scripts/mod.py"], snapshots, cert, repo)
    assert toctou == ["scripts/mod.py"] and certified == [], (certified, toctou)
    settle = [s for s in slept if s >= 1.0]  # #1992: see sibling test above
    assert settle == [2.0], slept


def test_write_cert_trims_to_last_500_lines_atomically(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    cert = tmp_path / "cert.txt"
    cert.write_text("".join(f"v1 1 {'0' * 40} scripts/old{i}.py\n" for i in range(600)))
    snapshots = ilg.read_payload(["scripts/mod.py"], repo)
    certified, toctou = ilg.write_cert(["scripts/mod.py"], snapshots, cert, repo)
    assert certified == ["scripts/mod.py"] and toctou == []
    lines = cert.read_text(encoding="utf-8").splitlines()
    assert len(lines) == ilg.CERT_TRIM_LINES, len(lines)
    assert lines[-1].endswith(" scripts/mod.py"), lines[-1]
    # tmp+rename leaves no orphaned temp files beside the cert
    strays = [p for p in cert.parent.iterdir() if p.name.startswith(cert.name + ".")]
    assert strays == [], strays


def test_write_cert_relocks_after_concurrent_trim_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1620 fix (d): a concurrent trim's os.replace swaps the cert path to a
    NEW inode while a second writer waits on the OLD inode's flock; without
    the inode re-check the append lands on the orphaned inode and never
    appears at the path. Monkeypatched so ONLY THE FIRST flock call simulates
    the winning trim (later calls delegate untouched — otherwise the test
    exercises the retry-exhaustion path instead of the re-check)."""
    import fcntl as _fcntl
    import tempfile as _tempfile

    repo = _make_repo(tmp_path)
    snapshots = ilg.read_payload(["scripts/mod.py"], repo)
    cert = tmp_path / "cert.txt"
    cert.write_text("v1 1 " + "0" * 40 + " scripts/old.py\n", encoding="utf-8")

    real_flock = _fcntl.flock
    calls = {"n": 0}

    def fake_flock(fd: object, op: int) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            # Concurrent trim wins the race: replace the path with a NEW
            # inode BEFORE the first lock is granted.
            tfd, tmp = _tempfile.mkstemp(dir=str(cert.parent), prefix=cert.name + ".")
            with os.fdopen(tfd, "w", encoding="utf-8") as tf:
                tf.write("v1 2 " + "1" * 40 + " scripts/other.py\n")
            os.replace(tmp, cert)
        real_flock(fd, op)

    monkeypatch.setattr(ilg.fcntl, "flock", fake_flock)
    certified, toctou = ilg.write_cert(["scripts/mod.py"], snapshots, cert, repo)
    assert certified == ["scripts/mod.py"] and toctou == []
    # The appended line must be visible AT THE PATH (not on the orphaned inode).
    content = cert.read_text(encoding="utf-8")
    assert content.rstrip("\n").endswith(" scripts/mod.py"), content
    assert calls["n"] >= 2, calls  # the re-check re-locked at least once


def test_read_payload_missing_path_inconclusive(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    with pytest.raises(ilg.Inconclusive):
        ilg.read_payload(["scripts/does_not_exist.py"], repo)


# ---------------------------------------------------------------------------
# EPM_SCAN_EXTRA_FILES payload threading + untracked-payload note (#1889)
# ---------------------------------------------------------------------------
def test_pytest_leg_env_carries_scan_extra_files(tmp_path: Path) -> None:
    """The mapped-pytest leg's CHILD env carries the os.pathsep-joined payload
    list as EPM_SCAN_EXTRA_FILES (#1889), observable through the hermetic
    EPM_INLINE_GATE_PYTEST_CMD override. The echo line is WARN-prefixed so it
    classifies as a report line (never a verdict input), and the `1 passed`
    tail satisfies PYTEST_SUMMARY_RE."""
    repo = _make_repo(tmp_path)
    (repo / "scripts" / "sib.py").write_text("print(0)\n", encoding="utf-8")  # untracked sibling
    payload = ["scripts/mod.py", "scripts/sib.py"]
    payload_file = tmp_path / "payload.txt"
    payload_file.write_text("\n".join(payload) + "\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    lint_file = tmp_path / "lint.txt"
    lint_file.write_text(LINT_OK, encoding="utf-8")
    env["EPM_INLINE_GATE_LINT_CMD"] = f"cat {lint_file}"
    map_file = tmp_path / "map.txt"
    map_file.write_text("tests/test_x.py\tscripts/mod.py\n", encoding="utf-8")
    env["EPM_INLINE_GATE_MAP_CMD"] = f"cat {map_file}"
    # Shell override runs with the merged child env: $EPM_SCAN_EXTRA_FILES expands there.
    env["EPM_INLINE_GATE_PYTEST_CMD"] = (
        'echo "WARN scan-extra=$EPM_SCAN_EXTRA_FILES" && echo "1 passed in 0.01s"'
    )
    env.pop("EPM_SCAN_EXTRA_FILES", None)  # prove the value comes from the gate, not the host
    env["EPM_INLINE_CERT_PATH"] = str(tmp_path / "cert.txt")
    env["EPM_CERT_REHASH_DELAY_S"] = "0"
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--issue",
            "9999",
            "--payload-file",
            str(payload_file),
            "--repo-root",
            str(repo),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    joined = os.pathsep.join(sorted(payload))
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    # The payload paths reached the pytest child's env (echoed + audit-persisted).
    assert f"WARN scan-extra={joined}" in r.stdout, r.stdout
    audit = (out_dir / "issue-9999-inline-lint.txt").read_text(encoding="utf-8")
    assert f"WARN scan-extra={joined}" in audit, audit


def test_untracked_payload_note_printed(tmp_path: Path) -> None:
    """An UNTRACKED payload path gets the stderr audit note (#1889);
    a tracked payload path prints no note. Report-only: verdict unchanged."""
    repo = _make_repo(tmp_path)
    (repo / "scripts" / "new.py").write_text("print(1)\n", encoding="utf-8")  # not tracked
    r = _run_gate(repo, ["scripts/mod.py", "scripts/new.py"], tmp_path, lint_out=LINT_OK)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "inline_lint_gate: note: payload scripts/new.py is untracked" in r.stderr, r.stderr
    assert "payload scripts/mod.py is untracked" not in r.stderr, r.stderr


# ---------------------------------------------------------------------------
# Round-unique payload path contract (#1948): the bare issue-keyed legacy
# basename is refused BEFORE any leg runs; round-unique + arbitrary names are
# accepted; the map leg consumes a PRIVATE mkstemp copy (never the caller's
# file); a payload-binding audit line prints before the verdict line.
# ---------------------------------------------------------------------------
def test_legacy_payload_basename_refused_before_any_leg(tmp_path: Path) -> None:
    """#1948 criterion 1: the bare issue-keyed payload name is refused
    (exit 3, Inconclusive) BEFORE any leg subprocess runs — concurrent
    same-issue rounds clobber the shared path (cross-certification, #1768).
    The leg-override seams write a sentinel; its absence proves no leg ran."""
    repo = _make_repo(tmp_path)
    payload_file = tmp_path / "issue-9999-inline-payload.txt"
    payload_file.write_text("scripts/mod.py\n", encoding="utf-8")
    sentinel = tmp_path / "leg-ran"
    env = os.environ.copy()
    for name in (
        "EPM_INLINE_GATE_LINT_CMD",
        "EPM_INLINE_GATE_MAP_CMD",
        "EPM_INLINE_GATE_PYTEST_CMD",
    ):
        env[name] = f"touch {sentinel}"
    env["EPM_INLINE_CERT_PATH"] = str(tmp_path / "cert.txt")
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--issue",
            "9999",
            "--payload-file",
            str(payload_file),
            "--repo-root",
            str(repo),
            "--out-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)
    assert "legacy shared payload path refused (#1948)" in r.stdout, r.stdout
    assert "round-unique" in r.stdout, r.stdout
    assert not sentinel.exists(), "a leg subprocess ran despite the legacy-path refusal"
    assert _cert_lines(tmp_path) == []


def test_round_unique_payload_name_accepted(tmp_path: Path) -> None:
    """#1948 criterion 2: a round-unique payload name gates normally (the
    arbitrary-name case is covered by every other test's ``payload.txt``)."""
    repo = _make_repo(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        payload_name="issue-9999-r2-fu1-inline-payload.txt",
    )
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "inline_lint_gate: PASS" in r.stdout, r.stdout
    assert len(_cert_lines(tmp_path)) == 1


def test_map_leg_receives_private_copy_not_caller_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1948 criterion 3 (regression: fails pre-fix): main() hands run_legs a
    PRIVATE mkstemp copy of the resolved payload list — never the caller's
    file — so a mid-run overwrite of the caller's path cannot redirect the
    mapped-test set. run_legs is faked signature-conformantly at the
    subprocess boundary; the real run_legs body is exercised end-to-end by
    the subprocess tests above via the documented leg-override seams."""
    repo = _make_repo(tmp_path)
    caller = tmp_path / "issue-9999-r1-inline-payload.txt"
    caller.write_text("scripts/mod.py\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_run_legs(payload_file, issue, repo, out_dir, payload=None):
        seen["payload_file"] = Path(payload_file)
        seen["payload"] = list(payload or [])
        return ilg.LegResults(lint_output="workflow_lint: PASS\n", map_pairs=[])

    monkeypatch.setattr(ilg, "run_legs", fake_run_legs)
    monkeypatch.setenv("EPM_INLINE_CERT_PATH", str(tmp_path / "cert.txt"))
    rc = ilg.main(
        [
            "--issue",
            "9999",
            "--payload-file",
            str(caller),
            "--repo-root",
            str(repo),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    private = seen["payload_file"]
    assert isinstance(private, Path)
    assert private.resolve() != caller.resolve(), "map leg still consumes the caller's file"
    assert private.read_text(encoding="utf-8") == "scripts/mod.py\n"
    assert seen["payload"] == ["scripts/mod.py"]
    # The private mkstemp name (dot-suffixed) never matches the legacy regex.
    assert not ilg.LEGACY_PAYLOAD_BASENAME_RE.match(private.name)
    private.unlink(missing_ok=True)


def test_payload_binding_audit_line_before_verdict(tmp_path: Path) -> None:
    """#1948 criterion 4: ONE payload-binding audit line — source path, n,
    and the 12-hex sha256 of the sorted payload list — prints BEFORE the
    (byte-stable) terminal verdict line."""
    repo = _make_repo(tmp_path)
    (repo / "scripts" / "b.py").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "--", "scripts/b.py")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "b")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    r = _run_gate(repo, ["scripts/mod.py", "scripts/b.py"], tmp_path, lint_out=LINT_OK)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    payload_sorted = sorted(["scripts/mod.py", "scripts/b.py"])
    sha = hashlib.sha256(("\n".join(payload_sorted) + "\n").encode("utf-8")).hexdigest()[:12]
    audit = f"inline_lint_gate: payload-source {tmp_path / 'payload.txt'} n=2 list-sha256={sha}"
    assert audit in r.stdout, r.stdout
    assert r.stdout.index(audit) < r.stdout.index("inline_lint_gate: PASS"), r.stdout


def test_inline_paths_audit_line_source_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """#1948 criterion 4 (--paths branch): the audit line's source field reads
    ``inline-paths`` when no payload file was given."""
    repo = _make_repo(tmp_path)

    def fake_run_legs(payload_file, issue, repo, out_dir, payload=None):
        return ilg.LegResults(lint_output="workflow_lint: PASS\n", map_pairs=[])

    monkeypatch.setattr(ilg, "run_legs", fake_run_legs)
    monkeypatch.setenv("EPM_INLINE_CERT_PATH", str(tmp_path / "cert.txt"))
    rc = ilg.main(["--issue", "9999", "--paths", "scripts/mod.py", "--repo-root", str(repo)])
    assert rc == 0
    out = capsys.readouterr().out
    sha = hashlib.sha256(b"scripts/mod.py\n").hexdigest()[:12]
    assert f"inline_lint_gate: payload-source inline-paths n=1 list-sha256={sha}" in out, out


# ---------------------------------------------------------------------------
# Bytecode determinism (#1950): pre-leg __pycache__ purge of the editable code
# roots + PYTHONDONTWRITEBYTECODE=1 on every leg's CHILD env.
# ---------------------------------------------------------------------------
def _plant_pyc(repo: Path, rel_dir: str, name: str = "mod.cpython-311.pyc") -> Path:
    """Plant a fake stale ``.pyc`` under ``<repo>/<rel_dir>/__pycache__/``."""
    d = repo / rel_dir / "__pycache__"
    d.mkdir(parents=True, exist_ok=True)
    pyc = d / name
    pyc.write_bytes(b"stale-bytecode")
    return pyc


def test_purge_repo_bytecode_removes_code_root_pyc_only(tmp_path: Path) -> None:
    """#1950 criteria 1+3 (direct function test): pyc under the three editable
    code roots' __pycache__ (scripts/, nested src/**, tests/) are removed and
    counted; pyc under .venv/, external/, and data/ are NEVER touched."""
    repo = _make_repo(tmp_path)
    removed_targets = [
        _plant_pyc(repo, "scripts"),
        _plant_pyc(repo, "src/pkg"),  # nested: rglob must reach sub-packages
        _plant_pyc(repo, "tests"),
    ]
    kept_targets = [
        _plant_pyc(repo, ".venv/lib/python3.11/site-packages/x"),
        _plant_pyc(repo, "external/dep"),
        _plant_pyc(repo, "data/issue_1"),
    ]
    assert ilg.purge_repo_bytecode(repo) == 3
    for p in removed_targets:
        assert not p.exists(), f"code-root pyc survived the purge: {p}"
    for p in kept_targets:
        assert p.exists(), f"out-of-scope pyc was deleted: {p}"


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory write permissions")
def test_purge_repo_bytecode_warns_on_unremovable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """#1950 criterion 1 best-effort branch: an unremovable pyc (read-only
    parent dir) WARNs on stderr and never crashes the purge."""
    repo = _make_repo(tmp_path)
    pyc = _plant_pyc(repo, "scripts")
    locked_dir = pyc.parent
    locked_dir.chmod(0o555)  # unlink needs write on the parent dir
    try:
        removed = ilg.purge_repo_bytecode(repo)
    finally:
        locked_dir.chmod(0o755)
    err = capsys.readouterr().err
    assert removed == 0, removed
    assert pyc.exists()
    assert "could not be removed" in err, err
    assert "purged 0 stale-candidate bytecode" in err, err


def test_run_legs_purges_before_legs(tmp_path: Path) -> None:
    """#1950 criterion 1 ordering pin (plan-review Should-Fix 1): the lint
    leg's OWN command asserts the planted pyc is already gone (`test ! -f`)
    before emitting the healthy terminal line — a purge that ran after the
    legs (or not at all) yields no terminal line -> INCONCLUSIVE, failing the
    exit-0 assert. Also pins the audit split: the purge line prints on the
    GATE's stderr and never enters the leg-captured audit file."""
    repo = _make_repo(tmp_path)
    pyc = _plant_pyc(repo, "scripts")
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        lint_cmd_prefix=f"test ! -f {pyc}",
    )
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert not pyc.exists()
    assert "inline_lint_gate: purged 1 stale-candidate bytecode" in r.stderr, r.stderr
    audit = (tmp_path / "out" / "issue-9999-inline-lint.txt").read_text(encoding="utf-8")
    assert "stale-candidate bytecode" not in audit, audit


def test_legs_child_env_carries_dont_write_bytecode(tmp_path: Path) -> None:
    """#1950 criterion 2: all THREE legs (lint, map, pytest) observe
    PYTHONDONTWRITEBYTECODE=1 in their CHILD env through the hermetic
    override seams; the pytest leg's merge additionally still carries the
    #1889 EPM_SCAN_EXTRA_FILES threading (neither clobbers the other)."""
    repo = _make_repo(tmp_path)
    payload = ["scripts/mod.py"]
    payload_file = tmp_path / "payload.txt"
    payload_file.write_text("\n".join(payload) + "\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    probe = tmp_path / "leg-env.txt"
    env = os.environ.copy()
    lint_file = tmp_path / "lint.txt"
    lint_file.write_text(LINT_OK, encoding="utf-8")
    env["EPM_INLINE_GATE_LINT_CMD"] = (
        f'echo "lint-pdb=$PYTHONDONTWRITEBYTECODE" >> {probe} && cat {lint_file}'
    )
    map_file = tmp_path / "map.txt"
    map_file.write_text("tests/test_x.py\tscripts/mod.py\n", encoding="utf-8")
    env["EPM_INLINE_GATE_MAP_CMD"] = (
        f'echo "map-pdb=$PYTHONDONTWRITEBYTECODE" >> {probe} && cat {map_file}'
    )
    env["EPM_INLINE_GATE_PYTEST_CMD"] = (
        f'echo "pytest-pdb=$PYTHONDONTWRITEBYTECODE scan=$EPM_SCAN_EXTRA_FILES" >> {probe}'
        ' && echo "1 passed in 0.01s"'
    )
    # Prove the values come from the gate's extra_env merge, not the host env.
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env.pop("EPM_SCAN_EXTRA_FILES", None)
    env["EPM_INLINE_CERT_PATH"] = str(tmp_path / "cert.txt")
    env["EPM_CERT_REHASH_DELAY_S"] = "0"
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--issue",
            "9999",
            "--payload-file",
            str(payload_file),
            "--repo-root",
            str(repo),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    observed = probe.read_text(encoding="utf-8")
    assert "lint-pdb=1" in observed, observed
    assert "map-pdb=1" in observed, observed
    assert "pytest-pdb=1 scan=scripts/mod.py" in observed, observed


# ---------------------------------------------------------------------------
# Load awareness (#2039): EPM_GATE_LOAD_MAX / EPM_GATE_LOAD_WAIT_S /
# EPM_GATE_LOAD1_OVERRIDE. A would-be BLOCK whose payload-naming hits are ALL
# pytest-leg under hot load downgrades to a DISTINCT INCONCLUSIVE (exit 3,
# no cert); lint-leg hits, would-be-PASS paths, and below-threshold runs are
# byte-untouched. Fail-direction invariant: never a new exit-0, never a
# suppressed lint finding, never a would-be PASS converted (T8).
# ---------------------------------------------------------------------------
# Every HOT-load subprocess env zeroes EPM_GATE_LOAD_WAIT_S: a static
# override of 31 never drops below threshold, so the default 300 s wait
# would otherwise add pure sleep to the suite (plan note 2).
HOT_LOAD_ENV = {
    "EPM_GATE_LOAD_MAX": "20",
    "EPM_GATE_LOAD1_OVERRIDE": "31",
    "EPM_GATE_LOAD_WAIT_S": "0",
}
# Lineno-less payload-naming red pytest line: the conservative-block shape
# (rc 1 pre-#2039) that the #2039 incident's false BLOCK rode.
RED_PYTEST_OUT = "FAILED tests/test_x.py::test_y - scans scripts/mod.py\n1 failed in 0.5s\n"


def test_pytest_red_under_hot_load_defers_inconclusive(tmp_path: Path) -> None:
    """T1 (regression anchor — the #2039 / session a0400dd4 incident shape):
    a red mapped-pytest leg naming a MODIFIED payload at load1=31 >=
    threshold 20 is a DISTINCT INCONCLUSIVE (exit 3) naming the path — not a
    false BLOCK — and writes NO cert."""
    repo = _repo_with_added_lines(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out=RED_PYTEST_OUT,
        extra_env=dict(HOT_LOAD_ENV),
    )
    assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)
    assert "pytest-leg red under load" in r.stdout, r.stdout
    assert "re-run when load drops: scripts/mod.py" in r.stdout, r.stdout
    assert "inline_lint_gate: BLOCK" not in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


def test_pytest_red_below_threshold_blocks_as_today(tmp_path: Path) -> None:
    """T2 (no-behavior-change witness): the SAME red leg at load1=5 < 20
    keeps the pre-#2039 conservative BLOCK, byte-identical terminal shape."""
    repo = _repo_with_added_lines(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out=RED_PYTEST_OUT,
        extra_env={**HOT_LOAD_ENV, "EPM_GATE_LOAD1_OVERRIDE": "5"},
    )
    assert r.returncode == 1, (r.returncode, r.stdout)
    assert "inline_lint_gate: BLOCK (scripts/mod.py)" in r.stdout, r.stdout
    assert "pytest-leg red under load" not in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


def test_lint_leg_hit_never_load_downgraded(tmp_path: Path) -> None:
    """T3: a lint-leg finding naming the payload BLOCKs under hot load —
    lint findings are deterministic, never timing-sensitive. The mapped
    pytest leg runs green so the hot endpoint samples genuinely exist."""
    repo = _repo_with_added_lines(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out="workflow_lint: scripts/mod.py:6: new red\n" + LINT_FAIL_TERMINAL,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out="1 passed in 0.01s\n",
        extra_env=dict(HOT_LOAD_ENV),
    )
    assert r.returncode == 1, (r.returncode, r.stdout)
    assert "inline_lint_gate: BLOCK (scripts/mod.py)" in r.stdout, r.stdout
    assert "pytest-leg red under load" not in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


def test_green_legs_under_hot_load_pass_and_certify(tmp_path: Path) -> None:
    """T4: green legs at load1=31 PASS and certify — load cannot make a
    failing test pass, so a PASS under load certifies unchanged."""
    repo = _make_repo(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out="1 passed in 0.01s\n",
        extra_env=dict(HOT_LOAD_ENV),
    )
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "inline_lint_gate: PASS" in r.stdout, r.stdout
    assert len(_cert_lines(tmp_path)) == 1


def test_load_max_zero_kill_switch_disables_guard(tmp_path: Path) -> None:
    """T5a: EPM_GATE_LOAD_MAX=0 disables the guard — a red pytest leg at
    override 99 keeps the pre-#2039 BLOCK exactly (the one-line kill
    switch)."""
    repo = _repo_with_added_lines(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out=RED_PYTEST_OUT,
        extra_env={
            "EPM_GATE_LOAD_MAX": "0",
            "EPM_GATE_LOAD1_OVERRIDE": "99",
            "EPM_GATE_LOAD_WAIT_S": "0",
        },
    )
    assert r.returncode == 1, (r.returncode, r.stdout)
    assert "inline_lint_gate: BLOCK (scripts/mod.py)" in r.stdout, r.stdout
    assert "pytest-leg red under load" not in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


def test_load_max_malformed_falls_back_to_default(tmp_path: Path) -> None:
    """T5b: a malformed EPM_GATE_LOAD_MAX falls back to the ACTIVE default 20
    (fail toward guarded, the EPM_CERT_REHASH_DELAY_S precedent): a red leg
    at override 31 defers (exit 3)."""
    repo = _repo_with_added_lines(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out=RED_PYTEST_OUT,
        extra_env={**HOT_LOAD_ENV, "EPM_GATE_LOAD_MAX": "abc"},
    )
    assert r.returncode == 3, (r.returncode, r.stdout)
    assert "pytest-leg red under load" in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


def test_new_file_pytest_hits_under_hot_load_defer(tmp_path: Path) -> None:
    """D5 NEW-branch parity: the NEW-on-origin/main branch keys off the same
    hit lines — a NEW payload whose only hits are pytest-leg defers under
    hot load too (no cert, no BLOCK)."""
    repo = _make_repo(tmp_path)
    (repo / "scripts" / "new.py").write_text("print(1)\n", encoding="utf-8")  # not on origin/main
    r = _run_gate(
        repo,
        ["scripts/new.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/new.py\n",
        pytest_out="FAILED tests/test_x.py::test_y - scans scripts/new.py\n1 failed in 0.5s\n",
        extra_env=dict(HOT_LOAD_ENV),
    )
    assert r.returncode == 3, (r.returncode, r.stdout)
    assert "pytest-leg red under load" in r.stdout, r.stdout
    assert "re-run when load drops: scripts/new.py" in r.stdout, r.stdout
    assert "inline_lint_gate: BLOCK" not in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


def test_mixed_lint_block_and_pytest_defer_precedence(tmp_path: Path) -> None:
    """T7 (mixed precedence): BLOCK (1) beats load-deferred (3) — the
    lint-hit NEW path lands in the BLOCK list, the pytest-only path rides
    the INCONCLUSIVE-under-load line (printed BEFORE the BLOCK terminal, the
    TOCTOU-ordering precedent), and NEITHER certifies."""
    repo = _repo_with_added_lines(tmp_path)
    (repo / "scripts" / "new.py").write_text("print(1)\n", encoding="utf-8")  # not on origin/main
    r = _run_gate(
        repo,
        ["scripts/mod.py", "scripts/new.py"],
        tmp_path,
        lint_out="workflow_lint: scripts/new.py:1: bad hit\n" + LINT_FAIL_TERMINAL,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out=RED_PYTEST_OUT,
        extra_env=dict(HOT_LOAD_ENV),
    )
    assert r.returncode == 1, (r.returncode, r.stdout)
    deferred_at = r.stdout.find("pytest-leg red under load")
    block_at = r.stdout.find("inline_lint_gate: BLOCK (scripts/new.py)")
    assert deferred_at != -1 and block_at != -1, r.stdout
    assert deferred_at < block_at, r.stdout
    assert "re-run when load drops: scripts/mod.py" in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


def test_hot_load_pass_preserving_boundary(tmp_path: Path) -> None:
    """T8 (round-1 critic blocker 2 witness — the over-blocking direction): a
    MODIFIED payload whose red pytest hits all sit OUTSIDE the round's added
    ranges keeps rc 0 + PASS + a WRITTEN cert under hot load. The downgrade
    keys on the would-be-blocked outcome, never on mere pytest-hit presence:
    an implementation deferring every hot path with pytest hits converts
    hot-round PASSes into a NEW false-block class at default-ON (no cert =>
    guard_root_code_commit.sh blocks the commit) and fails here."""
    repo = _repo_with_added_lines(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out="tests/test_x.py\tscripts/mod.py\n",
        pytest_out=(
            "FAILED tests/test_x.py::test_y\n"
            "scripts/mod.py:2: pre-existing red context\n"
            "1 failed in 0.5s\n"
        ),
        extra_env=dict(HOT_LOAD_ENV),
    )
    assert r.returncode == 0, (r.returncode, r.stdout)
    assert "inline_lint_gate: PASS" in r.stdout, r.stdout
    assert "pytest-leg red under load" not in r.stdout, r.stdout
    lines = _cert_lines(tmp_path)
    assert len(lines) == 1 and lines[0].endswith(" scripts/mod.py"), lines


def test_load_wait_bounded_and_proceeds_on_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    """T6a: the pre-pytest wait polls at 15 s and stops the moment load
    drops; recorded sleeps stay within the EPM_GATE_LOAD_WAIT_S budget."""
    monkeypatch.setenv("EPM_GATE_LOAD_WAIT_S", "60")
    samples = iter([31.0, 31.0, 5.0])
    monkeypatch.setattr(ilg, "_sample_load1", lambda: next(samples))
    slept: list[float] = []
    monkeypatch.setattr(ilg.time, "sleep", lambda s: slept.append(s))
    ilg._wait_for_load_drop(20.0)
    assert slept == [15.0, 15.0], slept
    assert sum(slept) <= 60.0


def test_load_wait_budget_exhaustion_proceeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """T6b: a load that never drops exhausts the budget and PROCEEDS — the
    wait is hard-bounded, never a refusal to run (quick green payloads must
    not starve)."""
    monkeypatch.setenv("EPM_GATE_LOAD_WAIT_S", "40")
    monkeypatch.setattr(ilg, "_sample_load1", lambda: 99.0)
    slept: list[float] = []
    monkeypatch.setattr(ilg.time, "sleep", lambda s: slept.append(s))
    ilg._wait_for_load_drop(20.0)
    assert slept == [15.0, 15.0, 10.0], slept  # capped at the 40 s budget


def test_load_wait_zero_budget_no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """T6c: EPM_GATE_LOAD_WAIT_S=0 -> zero sleeps, immediate proceed."""
    monkeypatch.setenv("EPM_GATE_LOAD_WAIT_S", "0")
    monkeypatch.setattr(ilg, "_sample_load1", lambda: 99.0)
    slept: list[float] = []
    monkeypatch.setattr(ilg.time, "sleep", lambda s: slept.append(s))
    ilg._wait_for_load_drop(20.0)
    assert slept == [], slept


def test_run_legs_waits_and_samples_around_pytest_leg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Wiring pin (production run_legs body): the bounded wait fires exactly
    once, BEFORE the pytest leg, when the mapping is non-empty and the
    threshold is enabled; the pre/post endpoint samples land on LegResults;
    the at-start diagnostic + pre/post audit lines print on the gate's
    stderr."""
    repo = _make_repo(tmp_path)
    monkeypatch.setenv("EPM_GATE_LOAD_MAX", "20")
    monkeypatch.setenv("EPM_GATE_LOAD_WAIT_S", "0")
    monkeypatch.setenv("EPM_GATE_LOAD1_OVERRIDE", "31")
    lint_file = tmp_path / "lint.txt"
    lint_file.write_text(LINT_OK, encoding="utf-8")
    monkeypatch.setenv("EPM_INLINE_GATE_LINT_CMD", f"cat {lint_file}")
    map_file = tmp_path / "map.txt"
    map_file.write_text("tests/test_x.py\tscripts/mod.py\n", encoding="utf-8")
    monkeypatch.setenv("EPM_INLINE_GATE_MAP_CMD", f"cat {map_file}")
    monkeypatch.setenv("EPM_INLINE_GATE_PYTEST_CMD", 'echo "1 passed in 0.01s"')
    waits: list[float] = []
    monkeypatch.setattr(ilg, "_wait_for_load_drop", lambda thr: waits.append(thr))
    payload_file = tmp_path / "payload.txt"
    payload_file.write_text("scripts/mod.py\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    legs = ilg.run_legs(payload_file, 9999, repo, out_dir, payload=["scripts/mod.py"])
    assert waits == [20.0], waits
    assert legs.load1_pre == 31.0 and legs.load1_post == 31.0, legs
    err = capsys.readouterr().err
    assert "inline_lint_gate: load1 at-start=31.00" in err, err
    assert "inline_lint_gate: load1 pre-pytest=31.00 post-pytest=31.00 threshold=20" in err, err


def test_run_legs_no_wait_or_samples_when_mapping_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wiring pin: an EMPTY test mapping runs no pytest leg — no wait call,
    endpoint samples stay None (an empty-mapping run can never read hot)."""
    repo = _make_repo(tmp_path)
    monkeypatch.setenv("EPM_GATE_LOAD_MAX", "20")
    monkeypatch.setenv("EPM_GATE_LOAD1_OVERRIDE", "31")
    lint_file = tmp_path / "lint.txt"
    lint_file.write_text(LINT_OK, encoding="utf-8")
    monkeypatch.setenv("EPM_INLINE_GATE_LINT_CMD", f"cat {lint_file}")
    map_file = tmp_path / "map.txt"
    map_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("EPM_INLINE_GATE_MAP_CMD", f"cat {map_file}")
    waits: list[float] = []
    monkeypatch.setattr(ilg, "_wait_for_load_drop", lambda thr: waits.append(thr))
    payload_file = tmp_path / "payload.txt"
    payload_file.write_text("scripts/mod.py\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    legs = ilg.run_legs(payload_file, 9999, repo, out_dir, payload=["scripts/mod.py"])
    assert waits == [], waits
    assert legs.load1_pre is None and legs.load1_post is None, legs


def test_load_knob_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Knob grammar (D1/D2): defaults, explicit-zero disable, negatives,
    malformed fallbacks, and the test-support load1 override."""
    monkeypatch.delenv("EPM_GATE_LOAD_MAX", raising=False)
    assert ilg._load_max() == ilg.GATE_LOAD_MAX_DEFAULT == 20.0
    monkeypatch.setenv("EPM_GATE_LOAD_MAX", "0")
    assert ilg._load_max() is None  # the kill switch
    monkeypatch.setenv("EPM_GATE_LOAD_MAX", "-5")
    assert ilg._load_max() is None
    monkeypatch.setenv("EPM_GATE_LOAD_MAX", "abc")
    assert ilg._load_max() == 20.0  # malformed -> default, guard stays ACTIVE
    monkeypatch.setenv("EPM_GATE_LOAD_MAX", "33.5")
    assert ilg._load_max() == 33.5

    monkeypatch.delenv("EPM_GATE_LOAD_WAIT_S", raising=False)
    assert ilg._load_wait_s() == ilg.GATE_LOAD_WAIT_DEFAULT_S == 300.0
    monkeypatch.setenv("EPM_GATE_LOAD_WAIT_S", "junk")
    assert ilg._load_wait_s() == 300.0
    monkeypatch.setenv("EPM_GATE_LOAD_WAIT_S", "-9")
    assert ilg._load_wait_s() == 0.0

    monkeypatch.setenv("EPM_GATE_LOAD1_OVERRIDE", "31")
    assert ilg._sample_load1() == 31.0
    monkeypatch.setenv("EPM_GATE_LOAD1_OVERRIDE", "bogus")
    real = ilg._sample_load1()  # malformed override ignored -> real read
    assert real is not None and real >= 0.0, real


def test_load_hot_uses_max_available_endpoint_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """D4: load_hot = max of the AVAILABLE pre/post samples >= threshold; no
    samples or a disabled threshold is never hot."""
    monkeypatch.setenv("EPM_GATE_LOAD_MAX", "20")
    hot = ilg.LegResults(lint_output="", map_pairs=[], load1_pre=5.0, load1_post=25.0)
    assert ilg._load_hot(hot) is True
    cool = ilg.LegResults(lint_output="", map_pairs=[], load1_pre=5.0, load1_post=None)
    assert ilg._load_hot(cool) is False
    unsampled = ilg.LegResults(lint_output="", map_pairs=[])
    assert ilg._load_hot(unsampled) is False
    monkeypatch.setenv("EPM_GATE_LOAD_MAX", "0")
    disabled = ilg.LegResults(lint_output="", map_pairs=[], load1_pre=99.0, load1_post=99.0)
    assert ilg._load_hot(disabled) is False  # kill switch beats any sample


# ---------------------------------------------------------------------------
# #2235 payload-scoping TDD surface (plan .claude/plans/issue-2235.md § TDD,
# cases 8-12): gate-side --files argv composition (Phase B6), the
# FILES-MODE-REFUSED -> bare-rerun fallback, the step9c-baseline ledger layer
# (Phase A), the cert-grammar and env-override-seam invariants, and the
# LINT_TIMEOUT_S 900 -> 1200 fence bump. The --files / ledger tests are RED
# until Phases A/B land; the seam/cert pins are green guards the change must
# not regress.
# ---------------------------------------------------------------------------
def _noop_pre_leg_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence run_legs' pre-leg side work (choom / fetch / bytecode purge)
    for hermetic in-process wiring tests — none of it bears on argv
    composition, and the fixture repo has no origin remote to fetch."""
    monkeypatch.setattr(ilg, "_best_effort_choom", lambda: None)
    monkeypatch.setattr(ilg, "_bounded_fetch", lambda repo: None)
    monkeypatch.setattr(ilg, "purge_repo_bytecode", lambda repo: 0)


def _run_legs_with_fake(
    tmp_path: Path,
    repo: Path,
    payload: list[str],
    fake_run_leg,
) -> ilg.LegResults:
    """Drive run_legs in-process with a substituted _run_leg."""
    payload_file = tmp_path / "wiring-payload.txt"
    payload_file.write_text("\n".join(payload) + "\n", encoding="utf-8")
    out_dir = tmp_path / "wiring-out"
    out_dir.mkdir(exist_ok=True)
    return ilg.run_legs(payload_file, 9999, repo, out_dir, payload=payload)


def test_workflow_surface_prefixes_pinned() -> None:
    """Plan §4 B6: the gate's eligibility tuple names every workflow-surface
    prefix whose payloads must take the bare full run (a payload editing any
    of these can change GLOBAL check outcomes, so scoping would be unsound).
    Superset assert: adding a prefix is fine, dropping one is a regression."""
    required = {
        ".claude/",
        "workflow.yaml",
        "scripts/workflow_lint.py",
        "scripts/inline_lint_gate.py",
        "scripts/select_step9c_tests.py",
        "scripts/step9c_baseline.py",
        "tests/",
    }
    assert required.issubset(set(ilg.WORKFLOW_SURFACE_PREFIXES)), ilg.WORKFLOW_SURFACE_PREFIXES


def test_gate_scopes_eligible_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Case 8a: a payload with no workflow-surface path composes the SCOPED
    lint invocation — the default argv gains `--files <payload...>`."""
    repo = _make_repo(tmp_path)
    _noop_pre_leg_work(monkeypatch)
    calls: list[tuple[list[str], str]] = []

    def fake_run_leg(default_argv, override_env, repo_, timeout, extra_env=None):
        calls.append((list(default_argv), override_env))
        if override_env == "EPM_INLINE_GATE_LINT_CMD":
            return (LINT_OK, 0)
        return ("", 0)

    monkeypatch.setattr(ilg, "_run_leg", fake_run_leg)
    legs = _run_legs_with_fake(tmp_path, repo, ["scripts/mod.py"], fake_run_leg)
    lint_calls = [argv for argv, env in calls if env == "EPM_INLINE_GATE_LINT_CMD"]
    assert len(lint_calls) == 1, calls
    argv = lint_calls[0]
    assert any("workflow_lint.py" in str(part) for part in argv), argv
    assert argv[-2:] == ["--files", "scripts/mod.py"], argv
    assert ilg.LINT_TERMINAL_RE.search(legs.lint_output), legs.lint_output


@pytest.mark.parametrize(
    "surface_path",
    [
        "tests/test_something.py",
        ".claude/skills/issue/SKILL.md",
        "scripts/workflow_lint.py",
        "scripts/inline_lint_gate.py",
        "scripts/select_step9c_tests.py",
        "scripts/step9c_baseline.py",
    ],
)
def test_gate_full_run_on_workflow_surface_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface_path: str
) -> None:
    """Case 8b: ANY workflow-surface path in the payload disables scoping —
    the lint leg runs the bare (unscoped) default argv, today's exact
    behavior. `tests/` is included conservatively: a payload editing tests
    can change global check outcomes and mapped-test verdicts."""
    repo = _make_repo(tmp_path)
    _noop_pre_leg_work(monkeypatch)
    calls: list[tuple[list[str], str]] = []

    def fake_run_leg(default_argv, override_env, repo_, timeout, extra_env=None):
        calls.append((list(default_argv), override_env))
        if override_env == "EPM_INLINE_GATE_LINT_CMD":
            return (LINT_OK, 0)
        return ("", 0)

    monkeypatch.setattr(ilg, "_run_leg", fake_run_leg)
    _run_legs_with_fake(tmp_path, repo, ["scripts/mod.py", surface_path], fake_run_leg)
    lint_calls = [argv for argv, env in calls if env == "EPM_INLINE_GATE_LINT_CMD"]
    assert len(lint_calls) == 1, calls
    assert "--files" not in lint_calls[0], lint_calls[0]


def test_gate_falls_back_on_files_mode_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Case 9: the FILES-MODE-REFUSED sentinel in the scoped leg's output
    triggers exactly ONE bare re-run; the verdict input (legs.lint_output)
    is the re-run's output (slow-but-correct, never silently-skipped)."""
    repo = _make_repo(tmp_path)
    _noop_pre_leg_work(monkeypatch)
    calls: list[tuple[list[str], str]] = []
    refusal = "workflow_lint: FILES-MODE-REFUSED (unclassified check check_foo)\n"
    bare_out = "unscoped-rerun-output\n" + LINT_OK

    def fake_run_leg(default_argv, override_env, repo_, timeout, extra_env=None):
        calls.append((list(default_argv), override_env))
        if override_env == "EPM_INLINE_GATE_LINT_CMD":
            if "--files" in default_argv:
                return (refusal, 2)
            return (bare_out, 0)
        return ("", 0)

    monkeypatch.setattr(ilg, "_run_leg", fake_run_leg)
    legs = _run_legs_with_fake(tmp_path, repo, ["scripts/mod.py"], fake_run_leg)
    lint_calls = [argv for argv, env in calls if env == "EPM_INLINE_GATE_LINT_CMD"]
    assert len(lint_calls) == 2, calls
    assert "--files" in lint_calls[0], lint_calls[0]
    assert "--files" not in lint_calls[1], lint_calls[1]
    assert "unscoped-rerun-output" in legs.lint_output, legs.lint_output
    assert ilg.LINT_TERMINAL_RE.search(legs.lint_output), legs.lint_output


def test_refused_scoped_leg_never_push_clean(tmp_path: Path) -> None:
    """Case 9 companion negative control (named in plan §4 B2's fail-loud
    pin): a lint leg that ends on the refusal sentinel with NO healthy
    terminal line — the env-override seam replays the sentinel on the
    fallback attempt too — is INCONCLUSIVE (exit 3, no cert), never a
    silent PASS. GREEN under today's dead-leg rule by design; pins the
    fail-closed direction the refusal path must preserve."""
    repo = _make_repo(tmp_path)
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out="workflow_lint: FILES-MODE-REFUSED (unclassified check check_foo)\n",
    )
    assert r.returncode == 3, (r.returncode, r.stdout)
    assert "inline_lint_gate: INCONCLUSIVE" in r.stdout, r.stdout
    assert _cert_lines(tmp_path) == []


# ---------------------------------------------------------------------------
# Case 10 — Phase A step9c-baseline ledger layer: loader arms + the
# subtractive-only evaluate() consumption.
# ---------------------------------------------------------------------------
def _write_ledger(repo: Path, **overrides: object) -> Path:
    """Fixture ledger at the canonical .claude/cache/step9c-baseline.json,
    sha-matched to the fixture repo's origin/main unless overridden."""
    ledger: dict[str, object] = {
        "schema_version": 1,
        "main_sha": _git(repo, "rev-parse", "origin/main").strip(),
        "failing_tests": [],
        "refreshed_at": "2026-08-11T00:00:00Z",
    }
    ledger.update(overrides)
    path = repo / ".claude" / "cache" / "step9c-baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger), encoding="utf-8")
    return path


_LEDGER_PYTEST_OUT = "FAILED tests/test_foo.py::test_bar - touches scripts/mod.py\n1 failed\n"


def _ledger_legs() -> ilg.LegResults:
    """A pytest-leg hit naming the payload with NO parseable lineno — the
    conservative-block shape a pre-existing failing test produces."""
    return ilg.LegResults(
        lint_output=LINT_OK,
        map_pairs=[("tests/test_foo.py", "scripts/mod.py")],
        pytest_output=_LEDGER_PYTEST_OUT,
    )


def test_load_baseline_ledger_fresh_sha_returns_dict(tmp_path: Path) -> None:
    """A parsing, schema-accepted, sha-matched ledger loads; the consumed
    failing_tests field round-trips."""
    repo = _make_repo(tmp_path)
    _write_ledger(repo, failing_tests=["tests/test_foo.py::test_bar"])
    ledger = ilg.load_baseline_ledger(repo)
    assert ledger is not None
    assert ledger["failing_tests"] == ["tests/test_foo.py::test_bar"]


@pytest.mark.parametrize("variant", ["mismatched-sha", "absent", "corrupt", "wrong-schema"])
def test_load_baseline_ledger_degrades_to_none(tmp_path: Path, variant: str) -> None:
    """A stale / absent / corrupt / schema-drifted ledger returns None: the
    subtraction layer disengages and evaluate() behaves bit-identically to
    today (fail-closed to current semantics — a bad ledger NEVER blocks and
    NEVER passes anything)."""
    repo = _make_repo(tmp_path)
    if variant == "mismatched-sha":
        _write_ledger(repo, main_sha="0" * 40)
    elif variant == "corrupt":
        path = repo / ".claude" / "cache" / "step9c-baseline.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
    elif variant == "wrong-schema":
        _write_ledger(repo, schema_version=999)
    # variant == "absent": no file written at all.
    assert ilg.load_baseline_ledger(repo) is None


def test_evaluate_ledger_none_is_todays_behavior(tmp_path: Path) -> None:
    """Baseline arm: with no ledger the payload-naming FAILED line is a
    conservative block (non-new path, no parseable lineno) — exactly today's
    semantics. The ledger parameter must be additive with default None, so
    every existing call site binds unchanged."""
    repo = _make_repo(tmp_path)
    verdict = ilg.evaluate(["scripts/mod.py"], _ledger_legs(), repo)
    assert "scripts/mod.py" in verdict.blocked, verdict


def test_evaluate_ledger_subtracts_preexisting_failing_test(tmp_path: Path) -> None:
    """Phase A subtraction arm: a pytest-leg hit whose FAILED node id is in
    ledger["failing_tests"] (and whose test file is not itself in the
    payload) reclassifies to reported — labeled pre-existing-on-main
    (ledger) — and the path certifies. Subtractive ONLY: block -> reported,
    never the reverse."""
    repo = _make_repo(tmp_path)
    ledger = {
        "schema_version": 1,
        "main_sha": _git(repo, "rev-parse", "origin/main").strip(),
        "failing_tests": ["tests/test_foo.py::test_bar"],
    }
    verdict = ilg.evaluate(["scripts/mod.py"], _ledger_legs(), repo, ledger=ledger)
    assert verdict.blocked == {}, verdict.blocked
    assert "scripts/mod.py" in verdict.passing, verdict
    assert any("pre-existing-on-main (ledger)" in line for line in verdict.reported), (
        verdict.reported
    )


def test_evaluate_ledger_ignores_nonlisted_failures(tmp_path: Path) -> None:
    """Keyed subtraction, never a blanket pytest-leg waiver: a FAILED node id
    NOT in the ledger keeps today's classification (blocked) even with a
    fresh ledger present."""
    repo = _make_repo(tmp_path)
    ledger = {
        "schema_version": 1,
        "main_sha": _git(repo, "rev-parse", "origin/main").strip(),
        "failing_tests": ["tests/test_other.py::test_unrelated"],
    }
    verdict = ilg.evaluate(["scripts/mod.py"], _ledger_legs(), repo, ledger=ledger)
    assert "scripts/mod.py" in verdict.blocked, verdict


def test_evaluate_ledger_never_subtracts_when_payload_is_the_test_file(tmp_path: Path) -> None:
    """Own-file condition (plan §4 Phase A arm 1): when the payload ITSELF is
    the failing test's file, the round is editing that test — its failure is
    payload-attributable and must still block, ledger-listed or not."""
    repo = _make_repo(tmp_path)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_foo.py").write_text(
        "def test_bar():\n    assert False\n", encoding="utf-8"
    )
    _git(repo, "add", "--", "tests/test_foo.py")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "add test")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    ledger = {
        "schema_version": 1,
        "main_sha": _git(repo, "rev-parse", "origin/main").strip(),
        "failing_tests": ["tests/test_foo.py::test_bar"],
    }
    legs = ilg.LegResults(
        lint_output=LINT_OK,
        map_pairs=[("tests/test_foo.py", "tests/test_foo.py")],
        pytest_output="FAILED tests/test_foo.py::test_bar - broken\n1 failed\n",
    )
    verdict = ilg.evaluate(["tests/test_foo.py"], legs, repo, ledger=ledger)
    assert "tests/test_foo.py" in verdict.blocked, verdict


def test_gate_prints_ledger_provenance_line(tmp_path: Path) -> None:
    """Phase A audit line (plan §4 Phase A arm 2): the gate prints its ledger
    engagement state — `inline_lint_gate: ledger=<fresh sha12|stale|absent>`
    — so the round's audit trail shows whether the subtraction layer was
    armed (mirrors the #1948 payload-source binding line)."""
    repo = _make_repo(tmp_path)  # no ledger file in the fixture -> absent
    r = _run_gate(repo, ["scripts/mod.py"], tmp_path, lint_out=LINT_OK)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    combined = r.stdout + "\n" + r.stderr
    assert "inline_lint_gate: ledger=absent" in combined, combined


# ---------------------------------------------------------------------------
# Cases 11-12 + the fence bump.
# ---------------------------------------------------------------------------
def test_cert_line_shape_unchanged(tmp_path: Path) -> None:
    """Case 11 (acceptance 6): a cert written through the new path still
    parses under the hook's `v1 <epoch> <sha> <path>` grammar
    (guard_root_code_commit.sh check_certified ~1139-1151: tag == v1,
    numeric epoch, exact-path + exact-sha field equality). Complements the
    hook-side fixture and the path-identity pins in
    tests/test_issue_skill_inline_gate_pin.py."""
    repo = _make_repo(tmp_path)
    r = _run_gate(repo, ["scripts/mod.py"], tmp_path, lint_out=LINT_OK)
    assert r.returncode == 0, (r.returncode, r.stdout)
    lines = _cert_lines(tmp_path)
    assert len(lines) == 1, lines
    m = re.match(r"^v1 (\d+) ([0-9a-f]{40}) (\S+)$", lines[0])
    assert m, lines[0]
    assert m.group(3) == "scripts/mod.py"
    assert m.group(2) == _git(repo, "hash-object", "--", str(repo / "scripts" / "mod.py")).strip()


def test_lint_cmd_override_runs_verbatim_unscoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Case 12 (acceptance 7): an EPM_INLINE_GATE_LINT_CMD override is
    executed VERBATIM via shell=True even for a scoped-eligible payload —
    the gate never appends --files into (or around) the override string, so
    an override runs UNSCOPED unless the override string itself carries
    --files (the fail-safe direction: full run). GREEN today; guards the
    seam through the Phase B rewiring."""
    repo = _make_repo(tmp_path)
    _noop_pre_leg_work(monkeypatch)
    lint_cmd = "printf 'workflow_lint: PASS\\n'"
    monkeypatch.setenv("EPM_INLINE_GATE_LINT_CMD", lint_cmd)
    monkeypatch.setenv("EPM_INLINE_GATE_MAP_CMD", "printf ''")
    monkeypatch.setenv("EPM_INLINE_GATE_PYTEST_CMD", "printf ''")
    recorded: list[tuple[tuple, dict]] = []
    real_run = subprocess.run

    def recording(*args, **kwargs):
        recorded.append((args, kwargs))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(ilg.subprocess, "run", recording)
    payload_file = tmp_path / "seam-payload.txt"
    payload_file.write_text("scripts/mod.py\n", encoding="utf-8")
    out_dir = tmp_path / "seam-out"
    out_dir.mkdir()
    legs = ilg.run_legs(payload_file, 9999, repo, out_dir, payload=["scripts/mod.py"])
    shell_cmds = [a[0] for a, kw in recorded if kw.get("shell")]
    assert lint_cmd in shell_cmds, shell_cmds  # byte-verbatim, unmodified
    for a, kw in recorded:
        cmd = a[0] if a else kw.get("args")
        parts = cmd if isinstance(cmd, list) else [str(cmd)]
        assert all("--files" not in str(part) for part in parts), cmd
    assert ilg.LINT_TERMINAL_RE.search(legs.lint_output), legs.lint_output


def test_lint_timeout_is_1200() -> None:
    """Plan §11 item 1: the measured full no-flags wall (547.9 s, 2026-08-11,
    task #2235 body) already exceeds half the old 900 s fence — one contended
    run from rc=-1 INCONCLUSIVE on the fallback path. The fence must be
    >= 2x the measured wall: 1200 s = 2.19x. RED until Phase B lands."""
    assert ilg.LINT_TIMEOUT_S == 1200


# ---------------------------------------------------------------------------
# #2318 — violation-grain classification for whole-repo scan nodes (plan §5
# rows C1-C9 + the §6 named pins). Rows drive evaluate() directly with
# synthetic LegResults (the zero-side-effect harness the attribution probe
# used); the registry + extractor are the REAL imported step9c_baseline
# symbols except where a row's docstring says otherwise.
# ---------------------------------------------------------------------------
_SCAN_NODE = "tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints"
_SCAN_FILE, _SCAN_NAME = _SCAN_NODE.split("::")


def _scan_ledger(repo: Path, nodes: list[str] | None = None) -> dict:
    """Fresh dict-row ledger (the LIVE failing_tests row shape) listing
    *nodes* (default: the thread-caps scan node)."""
    rows = []
    for n in nodes if nodes is not None else [_SCAN_NODE]:
        file, _, name = n.partition("::")
        rows.append(
            {
                "classname": file.removesuffix(".py").replace("/", "."),
                "file": file,
                "name": name,
            }
        )
    return {
        "schema_version": 1,
        "main_sha": _git(repo, "rev-parse", "origin/main").strip(),
        "failing_tests": rows,
    }


def _scan_pytest_out(
    *, offender_lines: list[str], node: str = _SCAN_NODE, failed_line: str | None = None
) -> str:
    """pytest -q -rA rendering of ONE failing scan node: underscore-fenced
    header inside the equals-fenced FAILURES section, offender rows, the
    trailing `<file>:<line>: <Exc>` locus, then the short-summary FAILED row
    (the shape measured on the installed pytest 9.0.2 — plan assumption 3)."""
    file, _, name = node.partition("::")
    body = "\n".join(offender_lines)
    failed = failed_line if failed_line is not None else f"FAILED {node} - AssertionError"
    return (
        "F                                                        [100%]\n"
        "=================================== FAILURES ===================================\n"
        f"_______________________________ {name} _______________________________\n"
        "\n"
        f"    def {name}():\n"
        ">       assert not violations, header\n"
        "E       AssertionError: module-top heavy imports before load_dotenv():\n"
        f"{body}\n"
        f"{file}:1005: AssertionError\n"
        "=========================== short test summary info ============================\n"
        f"{failed}\n"
        "1 failed in 0.42s\n"
    )


def _scan_legs(pytest_out: str) -> ilg.LegResults:
    return ilg.LegResults(
        lint_output=LINT_OK,
        map_pairs=[(_SCAN_FILE, "scripts/mod.py")],
        pytest_output=pytest_out,
    )


def test_scan_node_payload_added_violation_blocks(tmp_path: Path) -> None:
    """C1 (the headline; plan kill criterion (a)): a payload path IN the
    extracted violation set means the round ADDED a violation to a registry
    node red on pristine main — BLOCK, never a node-identity demotion. If
    this test ever demotes, the criterion is unsound: stop, do not tune a
    threshold."""
    repo = _make_repo(tmp_path)
    out = _scan_pytest_out(
        offender_lines=[
            "E         scripts/mod.py (module-top heavy import at line 2, import torch)",
        ]
    )
    mod = ilg._step9c_baseline_module()
    assert mod is not None
    # Precondition: the payload IS in the extracted set (the ADDED-violation shape).
    assert "scripts/mod.py" in mod.extract_violation_paths(out)
    verdict = ilg.evaluate(["scripts/mod.py"], _scan_legs(out), repo, ledger=_scan_ledger(repo))
    assert "scripts/mod.py" in verdict.blocked, verdict
    assert not any("violation-set" in ln for ln in verdict.reported), verdict.reported


def test_scan_node_same_set_red_demotes_and_certifies(tmp_path: Path) -> None:
    """C2 (criterion 3, the #1388-class non-regression): the only reachable
    demote residue — the payload path appears in a block line but its
    matched token carries the `...` saferepr elision marker, so the
    extracted set does NOT contain it. The fixture's own precondition is
    asserted FIRST: an unsatisfiable fixture must fail loud here, never be
    rescued by narrowing the criterion (plan §4 single-sidedness)."""
    repo = _make_repo(tmp_path)
    out = _scan_pytest_out(
        offender_lines=[
            "E         scripts/other_offender.py (module-top heavy import at line 3, import torch)",
            "E       assert not ['scripts/mod.py...ort torch)']",
        ]
    )
    mod = ilg._step9c_baseline_module()
    assert mod is not None
    extracted = mod.extract_violation_paths(out)
    assert "scripts/mod.py" not in extracted, extracted  # the C2 precondition
    assert "scripts/mod.py" in out  # ...yet a block line NAMES the payload
    verdict = ilg.evaluate(["scripts/mod.py"], _scan_legs(out), repo, ledger=_scan_ledger(repo))
    assert verdict.blocked == {}, verdict.blocked
    assert "scripts/mod.py" in verdict.passing, verdict
    assert any("pre-existing-on-main (ledger, violation-set)" in ln for ln in verdict.reported), (
        verdict.reported
    )


def test_scan_node_not_in_ledger_blocks(tmp_path: Path) -> None:
    """C3: a registry node red on the gate run but NOT ledger-listed is not
    pre-existing — today's conservative block stands unchanged."""
    repo = _make_repo(tmp_path)
    out = _scan_pytest_out(offender_lines=["E       assert not ['scripts/mod.py...ort torch)']"])
    ledger = _scan_ledger(repo, nodes=["tests/test_other.py::test_unrelated"])
    verdict = ilg.evaluate(["scripts/mod.py"], _scan_legs(out), repo, ledger=ledger)
    assert "scripts/mod.py" in verdict.blocked, verdict


def test_nonregistry_ledger_node_keeps_failed_line_grain(tmp_path: Path) -> None:
    """C4 (no-change control): a ledger-listed node OUTSIDE the registry
    keeps today's ^FAILED-line node-grain path byte-for-byte — the FAILED
    summary row demotes on node identity (D1 now engages for the live dict
    rows) while a payload-naming traceback line still blocks (a non-scanning
    failure has no finer sound grain to decide at)."""
    repo = _make_repo(tmp_path)
    node = "tests/test_foo.py::test_bar"
    out = _scan_pytest_out(
        offender_lines=["E       boom touching scripts/mod.py"],
        node=node,
        failed_line=f"FAILED {node} - touches scripts/mod.py",
    )
    verdict = ilg.evaluate(
        ["scripts/mod.py"], _scan_legs(out), repo, ledger=_scan_ledger(repo, nodes=[node])
    )
    assert "scripts/mod.py" in verdict.blocked, verdict  # traceback line still blocks
    assert any(
        ln.startswith("[pre-existing-on-main (ledger)]") and "FAILED" in ln
        for ln in verdict.reported
    ), verdict.reported
    assert not any("violation-set" in ln for ln in verdict.reported), verdict.reported


def test_scan_node_empty_extraction_blocks_with_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """C5 (criterion 4): an empty extracted set routes to a loud warn + NO
    demotion (the conservative block). Unreachable through the natural path
    on a well-formed block — the trailing `<file>:<line>:` locus always
    contributes the test-file token — so the extractor seam is stubbed to
    frozenset() for THIS row only (the real extractor body runs in C1/C2 and
    the subprocess E2E)."""
    repo = _make_repo(tmp_path)
    out = _scan_pytest_out(offender_lines=["E       assert not ['scripts/mod.py...x)']"])
    mod = ilg._step9c_baseline_module()
    assert mod is not None
    monkeypatch.setattr(mod, "extract_violation_paths", lambda text: frozenset())
    verdict = ilg.evaluate(["scripts/mod.py"], _scan_legs(out), repo, ledger=_scan_ledger(repo))
    assert "scripts/mod.py" in verdict.blocked, verdict
    assert "empty violation-path extraction" in capsys.readouterr().err


def test_scan_block_without_ledger_blocks(tmp_path: Path) -> None:
    """C6 (no-change control): ledger=None (absent / sha-stale / corrupt) is
    bit-identical to the pre-#2235 gate — no demotion at any grain, even for
    a registry-member block that would demote under a fresh ledger (the C2
    fixture minus the ledger)."""
    repo = _make_repo(tmp_path)
    out = _scan_pytest_out(
        offender_lines=[
            "E         scripts/other_offender.py (module-top heavy import at line 3, import torch)",
            "E       assert not ['scripts/mod.py...ort torch)']",
        ]
    )
    verdict = ilg.evaluate(["scripts/mod.py"], _scan_legs(out), repo)
    assert "scripts/mod.py" in verdict.blocked, verdict
    assert verdict.reported == [], verdict.reported


def test_scan_node_own_file_payload_still_blocks(tmp_path: Path) -> None:
    """C7 (#2235 rule preserved): when the payload IS the failing scan
    test's own file, the round is editing that test — payload-attributable,
    still blocks, ledger-listed + registry-member or not."""
    repo = _make_repo(tmp_path)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_shared_vm_thread_caps.py").write_text(
        f"def {_SCAN_NAME}():\n    assert False\n", encoding="utf-8"
    )
    _git(repo, "add", "--", _SCAN_FILE)
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "add test")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    out = _scan_pytest_out(
        offender_lines=["E       broken tests/test_shared_vm_thread_caps.py here"]
    )
    legs = ilg.LegResults(
        lint_output=LINT_OK, map_pairs=[(_SCAN_FILE, _SCAN_FILE)], pytest_output=out
    )
    verdict = ilg.evaluate([_SCAN_FILE], legs, repo, ledger=_scan_ledger(repo))
    assert _SCAN_FILE in verdict.blocked, verdict


def test_registry_import_failure_degrades_to_node_grain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """C8: an unavailable registry import degrades LOUDLY to today's
    node-grain behavior — the C2 demote fixture BLOCKS instead, with one
    stderr warn; never a crash (INCONCLUSIVE wedge), never a new blocking
    class."""
    repo = _make_repo(tmp_path)
    out = _scan_pytest_out(offender_lines=["E       assert not ['scripts/mod.py...x)']"])
    monkeypatch.setattr(ilg, "_STEP9C_CACHE", {})
    monkeypatch.delitem(sys.modules, "step9c_baseline", raising=False)
    monkeypatch.setattr(ilg, "_STEP9C_BASELINE_PATH", tmp_path / "nonexistent.py")
    verdict = ilg.evaluate(["scripts/mod.py"], _scan_legs(out), repo, ledger=_scan_ledger(repo))
    assert "scripts/mod.py" in verdict.blocked, verdict
    assert "registry import failed" in capsys.readouterr().err


def test_block_summary_nodeid_mismatch_drops_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """C9: a FAILURES block whose derived nodeid is absent from the FAILED
    short-summary set is DROPPED (fail-closed) with one warn — no demotion,
    today's behavior; parser brittleness across pytest versions degrades to
    a block, never to a waiver."""
    repo = _make_repo(tmp_path)
    out = _scan_pytest_out(
        offender_lines=["E       assert not ['scripts/mod.py...x)']"],
        failed_line="FAILED tests/test_shared_vm_thread_caps.py::test_other - AssertionError",
    )
    verdict = ilg.evaluate(["scripts/mod.py"], _scan_legs(out), repo, ledger=_scan_ledger(repo))
    assert "scripts/mod.py" in verdict.blocked, verdict
    assert "dropped FAILURES block" in capsys.readouterr().err


def test_ledger_nodeids_reads_live_ledger_dict_rows() -> None:
    """D1 live-ledger regression (plan §6): the LIVE step9c ledger's
    failing_tests rows are DICTS ({classname, file, name}); _ledger_nodeids
    must map each to file::name — the pre-#2318 str() coercion put dict
    reprs in the set, so membership was always False and the #2235 Phase A
    layer was dead code. Read-only; skips cleanly when the live ledger is
    absent (fresh clones; the cache lives at the MAIN checkout)."""
    r = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    main_root = Path(r.stdout.strip()).parent if r.returncode == 0 else _REPO_ROOT
    candidates = [
        _REPO_ROOT / ".claude" / "cache" / "step9c-baseline.json",
        main_root / ".claude" / "cache" / "step9c-baseline.json",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        pytest.skip("live step9c-baseline.json absent")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("failing_tests") or []
    nodeids = ilg._ledger_nodeids(data)
    for row in rows:
        if (
            isinstance(row, dict)
            and isinstance(row.get("file"), str)
            and isinstance(row.get("name"), str)
        ):
            assert f"{row['file']}::{row['name']}" in nodeids, (row, nodeids)
        elif isinstance(row, str):
            assert row in nodeids
    # No stringified dict reprs may survive (the D1 defect signature).
    assert not any(n.startswith("{") for n in nodeids), nodeids


def test_ledger_nodeids_skips_unexpected_rows_with_warn(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Edit 1's shape contract: str rows pass through (forward-compat), dict
    rows map to file::name, anything else is SKIPPED with one aggregate
    stderr warn naming the row types — never a raise, never a silent
    no-op."""
    ledger = {
        "failing_tests": [
            "tests/test_a.py::test_s",
            {"classname": "c", "file": "tests/test_b.py", "name": "test_d"},
            {"file": "tests/test_c.py"},  # dict missing name -> skipped
            42,
            None,
        ]
    }
    out = ilg._ledger_nodeids(ledger)
    assert out == {"tests/test_a.py::test_s", "tests/test_b.py::test_d"}
    err = capsys.readouterr().err
    assert "unexpected shape" in err
    assert "int" in err and "dict" in err and "NoneType" in err


def test_ledger_dirty_code_paths_is_not_a_refusal(tmp_path: Path) -> None:
    """Documented residual (task #2318 body, ## Scope decision): the gate
    reads the ledger WITHOUT a dirty_code_paths refusal — refusing a
    dirty_code_paths: true ledger would disable the subtraction whenever the
    fleet has dirty paths (nearly always) and thereby newly BLOCK payloads
    on main's own pre-existing reds, the #1388 fleet-wedge class. Pin
    today's behavior: a dirty ledger still loads and still maps its rows."""
    repo = _make_repo(tmp_path)
    _write_ledger(
        repo,
        dirty_code_paths=True,
        dirty_paths=["scripts/x.py"],
        failing_tests=[{"classname": "c", "file": _SCAN_FILE, "name": _SCAN_NAME}],
    )
    ledger = ilg.load_baseline_ledger(repo)
    assert ledger is not None
    assert ilg._ledger_nodeids(ledger) == {_SCAN_NODE}


def test_violation_registry_import_is_wired_live_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 5's live-tree pin: the gate consults step9c_baseline's OWN
    registry object AT CALL TIME — identity against sys.modules plus a
    behavioral wiring probe (a registry mutation on the live module IS
    visible to the demotion path; a future drift copy inside the gate would
    not see the mutation and this test would fail), plus a source scan for a
    copy-shaped local definition."""
    mod = ilg._step9c_baseline_module()
    assert mod is not None
    assert mod is sys.modules["step9c_baseline"]
    assert mod.VIOLATION_SET_SCAN_NODES is sys.modules["step9c_baseline"].VIOLATION_SET_SCAN_NODES
    assert mod.VIOLATION_SET_SCAN_NODES  # live registry non-empty
    src = SCRIPT.read_text(encoding="utf-8")
    assert "VIOLATION_SET_SCAN_NODES: frozenset" not in src  # no drift copy (annotated)
    assert "VIOLATION_SET_SCAN_NODES = frozenset" not in src  # no drift copy (bare)
    repo = _make_repo(tmp_path)
    fake_node = "tests/test_fake.py::test_fake"
    monkeypatch.setattr(mod, "VIOLATION_SET_SCAN_NODES", frozenset({fake_node}))
    out = _scan_pytest_out(
        offender_lines=[
            "E         scripts/unrelated_offender.py (offender row)",
            "E       assert not ['scripts/mod.py...x)']",
        ],
        node=fake_node,
    )
    verdict = ilg.evaluate(
        ["scripts/mod.py"], _scan_legs(out), repo, ledger=_scan_ledger(repo, nodes=[fake_node])
    )
    assert "scripts/mod.py" in verdict.passing, verdict
    assert any("violation-set" in ln for ln in verdict.reported), verdict.reported


def test_skill_md_step9a_ter_demotion_prose_names_violation_grain() -> None:
    """Criterion 6's prose pin: the Step 9a-ter inline-gate verdict prose
    names the violation-grain demotion contract (the #2235 Phase A demotion
    sentence carries the #2318 clause) — durability for the SKILL side of
    the contract."""
    skill = issue_skill_text()
    anchor = "Phase A ledger demotion (#2235)"
    assert anchor in skill, "SKILL.md lost the Phase A ledger demotion sentence"
    window = skill[skill.index(anchor) : skill.index(anchor) + 1200]
    assert "VIOLATION grain" in window, window
    assert "VIOLATION_SET_SCAN_NODES" in window, window
    assert "extract_violation_paths" in window, window
    assert "#2318" in window, window


def test_gate_subprocess_demotes_scan_node_same_set_red(tmp_path: Path) -> None:
    """End-to-end (subprocess): a fresh dict-row ledger + a registry-member
    FAILURES block with no payload path in the extracted set -> exit 0, the
    payload certifies, and the run prints the fresh-ledger provenance line.
    Exercises the gate's OWN lazy step9c_baseline import in a fresh process
    (no test-harness sys.modules reuse)."""
    repo = _make_repo(tmp_path)
    _write_ledger(
        repo,
        failing_tests=[
            {
                "classname": "tests.test_shared_vm_thread_caps",
                "file": _SCAN_FILE,
                "name": _SCAN_NAME,
            }
        ],
    )
    out = _scan_pytest_out(
        offender_lines=[
            "E         scripts/other_offender.py (module-top heavy import at line 3, import torch)",
            "E       assert not ['scripts/mod.py...ort torch)']",
        ]
    )
    r = _run_gate(
        repo,
        ["scripts/mod.py"],
        tmp_path,
        lint_out=LINT_OK,
        map_out=f"{_SCAN_FILE}\tscripts/mod.py\n",
        pytest_out=out,
    )
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert len(_cert_lines(tmp_path)) == 1
    assert "inline_lint_gate: ledger=fresh" in r.stdout + r.stderr
