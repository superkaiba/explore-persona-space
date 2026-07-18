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

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

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
) -> subprocess.CompletedProcess[str]:
    payload_file = tmp_path / "payload.txt"
    payload_file.write_text("\n".join(payload) + "\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
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
    env["EPM_INLINE_CERT_PATH"] = str(tmp_path / "cert.txt")
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
    # 1 non-slow file: formula 120+30=150 < floor -> floored.
    assert ilg.mapped_pytest_timeout(["tests/test_x.py"]) == sel.TIMEOUT_FLOOR_S
    # Slow-surcharge case stays above the floor (120 + 30 + 900).
    assert ilg.mapped_pytest_timeout(["tests/test_workflow_lint.py"]) == 1050


def test_added_line_ranges_parses_u0_hunks(tmp_path: Path) -> None:
    repo = _repo_with_added_lines(tmp_path)
    assert ilg.added_line_ranges(repo, "scripts/mod.py") == [(6, 8)]
    assert ilg.added_line_ranges(repo, "scripts/other.py") == []


def test_write_cert_toctou_refuses_edited_path(tmp_path: Path) -> None:
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


def test_read_payload_missing_path_inconclusive(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    with pytest.raises(ilg.Inconclusive):
        ilg.read_payload(["scripts/does_not_exist.py"], repo)
