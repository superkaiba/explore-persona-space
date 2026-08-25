"""Pin tests for ``scripts/step10d_guards.sh`` (task #1978).

The script extracts the Step 10d PRELUDE + Guard 0 + Guard 4 shell fragments
from ``.claude/skills/issue/SKILL.md`` § Merge safety guards into ONE tested,
checked-in executable so sessions invoke it instead of re-typing prose per
merge. Each guard subcommand is byte-equivalent-in-effect to the current
SKILL.md fence; these tests exercise every behavioral arm to catch drift.

Fixture strategy: subprocess against throwaway ``tmp_path`` git repos (the
``tests/test_guard_*.py`` family pattern). The script derives ``REPO_ROOT``
from ``git rev-parse --path-format=absolute --git-common-dir``; each fixture
sets up a scratch repo (or a scratch repo + issue worktree) and invokes the
script with cwd inside it, so the script's own derivation resolves to the
scratch, not the live repo.

TODO(unit-b): add ``test_skill_md_invokes_script`` -- a SKILL.md prose pin
asserting ``.claude/skills/issue/SKILL.md`` § Merge safety guards contains the
``scripts/step10d_guards.sh`` invocation for guards prelude / 0 / 4 (per
``tests/test_issue_skill_workload_cmd_script_pin.py`` precedent). Deferred to
Unit B (the SKILL.md fence-replacement diff) so the pin is written against
the actual invocation shape landing there.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "step10d_guards.sh"

# The scoped file set the Guard 4 fence protects. The fence's REAL case glob
# is ``.claude/skills/*`` (broader than the plan's paraphrase
# ``.claude/skills/**/SKILL.md``); pick a stem inside ``.claude/rules/`` for
# the pass / refuse fixtures so we cover the case-glob preservation.
GUARD4_SCOPED_FILE = ".claude/rules/step10d-guard4-fixture.md"


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("GIT_AUTHOR_NAME", "eps-test")
    env.setdefault("GIT_AUTHOR_EMAIL", "eps-test@example.invalid")
    env.setdefault("GIT_COMMITTER_NAME", "eps-test")
    env.setdefault("GIT_COMMITTER_EMAIL", "eps-test@example.invalid")
    # Scrub the caller's GIT_* env so scratch repos don't leak into the live tree.
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(k, None)
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} in {cwd} failed rc={result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


def _run_script(
    cwd: Path,
    *args: str,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke ``scripts/step10d_guards.sh`` with cwd inside a scratch repo.

    ``cwd`` MUST be inside a git repo (scratch or worktree) -- the script
    derives ``REPO_ROOT`` from ``git rev-parse --path-format=absolute
    --git-common-dir`` and fails without one.
    """
    env = os.environ.copy()
    # Scrub the caller's GIT_* env so the script's own git resolution uses cwd.
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(k, None)
    env.setdefault("GIT_AUTHOR_NAME", "eps-test")
    env.setdefault("GIT_AUTHOR_EMAIL", "eps-test@example.invalid")
    env.setdefault("GIT_COMMITTER_NAME", "eps-test")
    env.setdefault("GIT_COMMITTER_EMAIL", "eps-test@example.invalid")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def _make_scratch_repo(tmp_path: Path) -> Path:
    """Build a scratch git repo at ``tmp_path/scratch``.

    Returns the scratch repo root. The script's PRELUDE derivation resolves
    ``REPO_ROOT`` to this path when invoked with cwd inside it.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _git(scratch, "init", "-q", "-b", "main")
    (scratch / "README.md").write_text("scratch\n")
    _git(scratch, "add", "README.md")
    _git(scratch, "commit", "-q", "-m", "c1")
    return scratch


def _make_scratch_repo_with_worktree(tmp_path: Path, issue: int) -> tuple[Path, Path]:
    """Build a scratch repo + a real ``.claude/worktrees/issue-<N>`` worktree.

    Returns (repo_root, worktree_path). The script's derivation lands
    ``$WT = repo_root/.claude/worktrees/issue-<N>``.

    Uses ``git worktree add`` so ``$WT`` is a real linked worktree that git
    accepts under ``git -C``. A plain ``mkdir`` would fail the script's
    ``git -C "$WT" status`` calls.
    """
    scratch = _make_scratch_repo(tmp_path)
    _git(scratch, "checkout", "-q", "-b", f"issue-{issue}")
    _git(scratch, "checkout", "-q", "main")
    worktree_dir = scratch / ".claude" / "worktrees" / f"issue-{issue}"
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)
    _git(scratch, "worktree", "add", "-q", str(worktree_dir), f"issue-{issue}")
    return scratch, worktree_dir


def _parse_kv(stdout: str) -> dict[str, str]:
    """Parse the script's ``KEY=VALUE`` stdout lines into a dict.

    Values are the ON-WIRE encoding (``printf %q``-escaped for path-bearing
    keys like ``LOST_UPDATE_PATHS`` — r2, #2428); assertions about the
    LOGICAL values the caller sees go through ``_run_guard4_eval_consumer``,
    which exercises the documented ``eval`` form in a real bash child.
    """
    d: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        d[k] = v
    return d


def _run_guard4_eval_consumer(
    cwd: Path,
    issue: int,
    main_sha: str,
) -> subprocess.CompletedProcess[str]:
    """Consume Guard 4 via the DOCUMENTED two-step caller form in real bash.

    Mirrors the script header's caller contract verbatim::

        GUARD4_OUT=$(bash scripts/step10d_guards.sh <N> --guard 4 \
            --main-sha "$MAIN_SHA"); GUARD4_RC=$?
        eval "$GUARD4_OUT"

    then prints the eval-recovered LOGICAL values as ``EVAL_*=...`` lines so
    tests assert on what the production caller actually sees — not on the
    on-wire encoding (the splitter-only parse is exactly how the unescaped
    ``(count)`` emission survived r1 review; #2428 r2). Script path / issue /
    main-sha travel via env vars, never string interpolation.
    """
    consumer = (
        'GUARD4_OUT=$(bash "$STEP10D_SCRIPT" "$STEP10D_ISSUE" --guard 4 '
        '--main-sha "$STEP10D_MAIN_SHA"); GUARD4_RC=$?\n'
        'eval "$GUARD4_OUT"\n'
        "printf 'EVAL_GUARD4_RC=%s\\n' \"$GUARD4_RC\"\n"
        "printf 'EVAL_GUARD4=%s\\n' \"${GUARD4:-}\"\n"
        "printf 'EVAL_GUARD4_MERGE_BASE=%s\\n' \"${GUARD4_MERGE_BASE:-}\"\n"
        "printf 'EVAL_LOST_UPDATE_PATHS=%s\\n' \"${LOST_UPDATE_PATHS:-}\"\n"
    )
    env = os.environ.copy()
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(k, None)
    env.update(
        STEP10D_SCRIPT=str(SCRIPT),
        STEP10D_ISSUE=str(issue),
        STEP10D_MAIN_SHA=main_sha,
    )
    return subprocess.run(
        ["bash", "-c", consumer],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# PRELUDE
# ---------------------------------------------------------------------------


def test_prelude_emits_repo_root_and_wt(tmp_path):
    scratch = _make_scratch_repo(tmp_path)
    proc = _run_script(scratch, "1978", "--guard", "prelude")
    assert proc.returncode == 0, proc.stderr
    kv = _parse_kv(proc.stdout)
    assert "REPO_ROOT" in kv, f"prelude did not emit REPO_ROOT: {proc.stdout!r}"
    assert "WT" in kv, f"prelude did not emit WT: {proc.stdout!r}"
    # REPO_ROOT resolves to the scratch repo root (path-format=absolute,
    # git-common-dir), NOT the live EPS repo.
    assert Path(kv["REPO_ROOT"]).resolve() == scratch.resolve()
    assert kv["WT"] == f"{scratch}/.claude/worktrees/issue-1978"


# ---------------------------------------------------------------------------
# Guard 0
# ---------------------------------------------------------------------------


def test_guard0_noop_on_clean_tree(tmp_path):
    scratch, worktree = _make_scratch_repo_with_worktree(tmp_path, 1978)
    # No dirty agent-memory files -> Guard 0 is a no-op.
    proc = _run_script(scratch, "1978", "--guard", "0")
    assert proc.returncode == 0, proc.stderr
    kv = _parse_kv(proc.stdout)
    assert kv.get("MEM_COMMITTED") == "no", proc.stdout
    # No new commit on the branch.
    log_before = _git(worktree, "log", "--oneline").stdout.strip().splitlines()
    assert len(log_before) == 1, log_before


def test_guard0_commits_dirty_agent_memory(tmp_path):
    scratch, worktree = _make_scratch_repo_with_worktree(tmp_path, 1978)
    # Add a dirty agent-memory file.
    mem_dir = worktree / ".claude" / "agent-memory" / "test-agent"
    mem_dir.mkdir(parents=True, exist_ok=True)
    mem_file = mem_dir / "note.md"
    mem_file.write_text("test memory entry\n")
    # First run -> commits, MEM_COMMITTED=yes.
    proc = _run_script(scratch, "1978", "--guard", "0")
    assert proc.returncode == 0, proc.stderr
    kv = _parse_kv(proc.stdout)
    assert kv.get("MEM_COMMITTED") == "yes", proc.stdout
    # Confirm commit exists and contains the memory file.
    log = _git(worktree, "log", "--oneline").stdout
    assert "persist agent-memory writes before Step-10d merge" in log, log
    show = _git(worktree, "show", "--stat", "HEAD").stdout
    assert ".claude/agent-memory/test-agent/note.md" in show, show
    # Second run -> idempotent no-op, MEM_COMMITTED=no.
    proc2 = _run_script(scratch, "1978", "--guard", "0")
    assert proc2.returncode == 0, proc2.stderr
    kv2 = _parse_kv(proc2.stdout)
    assert kv2.get("MEM_COMMITTED") == "no", proc2.stdout


def test_guard0_worktree_missing_exits_2(tmp_path):
    scratch = _make_scratch_repo(tmp_path)
    # No worktree at .claude/worktrees/issue-1978 -> exit 2 + ERROR=worktree-missing.
    proc = _run_script(scratch, "1978", "--guard", "0")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("ERROR") == "worktree-missing", proc.stdout


# ---------------------------------------------------------------------------
# Guard 4
# ---------------------------------------------------------------------------


def _seed_guard4_fixture(
    tmp_path: Path,
    issue: int,
    *,
    drop_main_lines: bool,
    scoped_file: str = GUARD4_SCOPED_FILE,
) -> tuple[Path, Path]:
    """Scratch + worktree with a scoped file that origin/main added lines to.

    Layout after this returns:
    - scratch repo at ``tmp_path/scratch`` (bare-origin at
      ``tmp_path/origin.git`` pushed to)
    - main has 2 commits: c1 seed + c2 adding new lines to the scoped file
    - worktree ``issue-<N>`` forked from c1 (before c2)
    - if drop_main_lines=True: worktree DROPS the c2-added lines by taking
      a snapshot at c1's content (whole-file overwrite); this is the exact
      lost-update shape Guard 4 refuses
    - if drop_main_lines=False: worktree PRESERVES c2's added lines
      (fetches origin/main into origin/main ref, keeps identical content)
    """
    origin_dir = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin_dir))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _git(scratch, "init", "-q", "-b", "main")
    _git(scratch, "remote", "add", "origin", str(origin_dir))
    # c1: seed with a scoped file that has original lines.
    scoped = scratch / scoped_file
    scoped.parent.mkdir(parents=True, exist_ok=True)
    scoped.write_text("original line 1\noriginal line 2\n")
    _git(scratch, "add", str(scoped.relative_to(scratch)))
    _git(scratch, "commit", "-q", "-m", "c1")
    _git(scratch, "push", "-q", "-u", "origin", "main")
    # Fork the branch at c1.
    _git(scratch, "checkout", "-q", "-b", f"issue-{issue}")
    _git(scratch, "checkout", "-q", "main")
    # c2: add lines on main.
    scoped.write_text("original line 1\noriginal line 2\nmain-added line A\nmain-added line B\n")
    _git(scratch, "add", str(scoped.relative_to(scratch)))
    _git(scratch, "commit", "-q", "-m", "c2: main adds lines")
    _git(scratch, "push", "-q", "origin", "main")
    # Set up the worktree at issue-<N> (branch was forked at c1).
    worktree_dir = scratch / ".claude" / "worktrees" / f"issue-{issue}"
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)
    _git(scratch, "worktree", "add", "-q", str(worktree_dir), f"issue-{issue}")
    # Branch adds its OWN line to the scoped file.
    scoped_wt = worktree_dir / scoped_file
    if drop_main_lines:
        # Whole-file snapshot that omits main's added lines -- the exact
        # lost-update shape.
        scoped_wt.write_text("original line 1\noriginal line 2\nbranch-added line X\n")
    else:
        # Preserve main's added lines AND add the branch's own.
        scoped_wt.write_text(
            "original line 1\n"
            "original line 2\n"
            "main-added line A\n"
            "main-added line B\n"
            "branch-added line X\n"
        )
    _git(worktree_dir, "add", scoped_file)
    _git(worktree_dir, "commit", "-q", "-m", "branch adds line X")
    return scratch, worktree_dir


def test_guard4_pass_when_no_dropped_lines(tmp_path):
    scratch, _ = _seed_guard4_fixture(tmp_path, 1978, drop_main_lines=False)
    proc = _run_script(scratch, "1978", "--guard", "4")
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("GUARD4") == "pass", proc.stdout
    assert "LOST_UPDATE_PATHS" not in kv, proc.stdout


def test_guard4_refuses_on_dropped_mainline_line(tmp_path):
    scratch, _ = _seed_guard4_fixture(tmp_path, 1978, drop_main_lines=True)
    proc = _run_script(scratch, "1978", "--guard", "4")
    # Refusal semantics: exit 1, GUARD4=refused, LOST_UPDATE_PATHS names the
    # scoped file, stderr carries the verbatim refusal banner.
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("GUARD4") == "refused", proc.stdout
    assert "LOST_UPDATE_PATHS" in kv, proc.stdout
    assert GUARD4_SCOPED_FILE in kv["LOST_UPDATE_PATHS"], kv["LOST_UPDATE_PATHS"]
    # The dropped-count is embedded as ``path(N)``; assert on that shape.
    assert "(" in kv["LOST_UPDATE_PATHS"] and ")" in kv["LOST_UPDATE_PATHS"]
    # Verbatim refusal banner on stderr.
    assert "LOST-UPDATE REFUSAL (Guard 4, #1713)" in proc.stderr, proc.stderr


def test_guard4_kill_switch(tmp_path):
    scratch, _ = _seed_guard4_fixture(tmp_path, 1978, drop_main_lines=True)
    # Same drop-lines fixture BUT with the kill switch on -> skipped, exit 0.
    proc = _run_script(
        scratch,
        "1978",
        "--guard",
        "4",
        env_extra={"EPM_SKIP_LOST_UPDATE_GUARD": "1"},
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("GUARD4") == "skipped", proc.stdout
    # No refusal banner on stderr when skipped.
    assert "LOST-UPDATE REFUSAL" not in proc.stderr, proc.stderr


def test_guard4_worktree_missing_exits_2(tmp_path):
    # A scratch repo with NO issue worktree -> Guard 4 fails infra with
    # ERROR=worktree-missing, exit 2. (Kill switch is checked FIRST but is
    # OFF here; the derivation is next.)
    scratch = _make_scratch_repo(tmp_path)
    proc = _run_script(scratch, "1978", "--guard", "4")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("ERROR") == "worktree-missing", proc.stdout


# ---------------------------------------------------------------------------
# Guard 4 --main-sha tip contract (#2428)
#
# The Step 10d caller passes the PINNED origin/main TIP (its Guard-1 capture)
# via --main-sha; the helper derives the merge-base FROM it. Pre-#2428 the
# helper consumed the flag value AS the merge-base, so the caller's tip made
# every main-side diff tip..tip (empty) and Guard 4 passed vacuously on every
# branch. T1-T9 pin the corrected contract; T7 is the discriminator that
# kills the flag-IGNORING implementation class.
# ---------------------------------------------------------------------------


def test_guard4_refuses_with_caller_tip_flag_form(tmp_path):
    """#2428 criterion 2's pin: the caller's literal flag form REFUSES.

    Invokes Guard 4 exactly as ``steps/18-step-10d.md`` does — ``--main-sha
    $(git rev-parse origin/main)`` — on the lost-update fixture arm. Pre-fix
    the helper read the tip as the MERGE-BASE, the tip..tip add-set was
    empty, and this vacuously passed (the #2212 incident shape).
    """
    scratch, wt = _seed_guard4_fixture(tmp_path, 2428, drop_main_lines=True)
    tip = _git(wt, "rev-parse", "origin/main").stdout.strip()
    proc = _run_script(scratch, "2428", "--guard", "4", "--main-sha", tip)
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("GUARD4") == "refused", proc.stdout
    assert GUARD4_SCOPED_FILE in kv.get("LOST_UPDATE_PATHS", ""), proc.stdout
    assert "LOST-UPDATE REFUSAL (Guard 4, #1713)" in proc.stderr, proc.stderr


def test_guard4_passes_with_caller_tip_flag_form_when_no_drop(tmp_path):
    """Converged branch + the caller's flag form: pass (no over-correction).

    Guards against a "fix" that refuses unconditionally — such an
    implementation would satisfy the refusal pin alone.
    """
    scratch, wt = _seed_guard4_fixture(tmp_path, 2428, drop_main_lines=False)
    tip = _git(wt, "rev-parse", "origin/main").stdout.strip()
    proc = _run_script(scratch, "2428", "--guard", "4", "--main-sha", tip)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("GUARD4") == "pass", proc.stdout
    assert "LOST_UPDATE_PATHS" not in kv, proc.stdout


def test_guard4_flag_and_noflag_verdicts_agree(tmp_path):
    """#2428 criterion 1: pinned and no-flag forms are verdict-equivalent.

    Runs BOTH fixture arms through both forms and asserts identical rc,
    ``GUARD4``, and ``LOST_UPDATE_PATHS``. NOTE: this does NOT by itself kill
    the vacuity class — the fixture never advances origin/main between
    capture and invocation, so a flag-IGNORING implementation also satisfies
    it; the discriminator is ``test_guard4_honors_pin_older_than_live_main``.
    """
    for arm, drop in (("drop", True), ("keep", False)):
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        scratch, wt = _seed_guard4_fixture(arm_dir, 2428, drop_main_lines=drop)
        tip = _git(wt, "rev-parse", "origin/main").stdout.strip()
        pinned = _run_script(scratch, "2428", "--guard", "4", "--main-sha", tip)
        noflag = _run_script(scratch, "2428", "--guard", "4")
        assert pinned.returncode == noflag.returncode, (arm, pinned.stdout, noflag.stdout)
        kv_pinned = _parse_kv(pinned.stdout)
        kv_noflag = _parse_kv(noflag.stdout)
        assert kv_pinned.get("GUARD4") == kv_noflag.get("GUARD4"), (
            arm,
            pinned.stdout,
            noflag.stdout,
        )
        assert kv_pinned.get("LOST_UPDATE_PATHS") == kv_noflag.get("LOST_UPDATE_PATHS"), (
            arm,
            pinned.stdout,
            noflag.stdout,
        )


def test_guard4_emits_merge_base_not_the_passed_tip(tmp_path):
    """The helper DERIVES the base — ``GUARD4_MERGE_BASE`` != the passed tip.

    On the drop fixture the derived base must equal the worktree's own
    ``git merge-base HEAD origin/main`` and differ from the tip passed in.
    """
    scratch, wt = _seed_guard4_fixture(tmp_path, 2428, drop_main_lines=True)
    tip = _git(wt, "rev-parse", "origin/main").stdout.strip()
    mb = _git(wt, "merge-base", "HEAD", "origin/main").stdout.strip()
    # Fixture invariant: the branch forked BEFORE main's c2, so mb != tip.
    assert mb != tip, (mb, tip)
    proc = _run_script(scratch, "2428", "--guard", "4", "--main-sha", tip)
    kv = _parse_kv(proc.stdout)
    assert kv.get("GUARD4_MERGE_BASE") == mb, proc.stdout
    assert kv.get("GUARD4_MERGE_BASE") != tip, proc.stdout


def test_main_sha_rejected_for_non_guard4(tmp_path):
    """#2428 D5b: ``--main-sha`` with any non-4 guard fails loud (rc 2).

    ``divergence`` pins its OWN main snapshot and would silently shadow a
    passed value; ``prelude``/``0`` never read it. The parser rejects the
    inert flag instead of accepting it.
    """
    scratch = _make_scratch_repo(tmp_path)
    sha = _git(scratch, "rev-parse", "HEAD").stdout.strip()
    for guard in ("prelude", "0", "divergence"):
        proc = _run_script(scratch, "1978", "--guard", guard, "--main-sha", sha)
        assert proc.returncode == 2, (guard, proc.stdout, proc.stderr)
        kv = _parse_kv(proc.stdout)
        assert kv.get("ERROR", "").startswith("main-sha-not-supported"), (guard, proc.stdout)


def test_step10d_guard4_contract_prose_agrees():
    """#2428 criterion 3's mechanical pin over BOTH contract surfaces.

    The helper's ``--help`` text and the ``steps/18-step-10d.md`` Guard-4
    block each state the TIP semantics of ``--main-sha``, and NEITHER
    describes the flag as a "pinned merge-base" — the exact stale wording
    that let the caller/helper mismatch ship.
    """
    helper_text = SCRIPT.read_text()
    caller_text = (
        REPO_ROOT / ".claude" / "skills" / "issue" / "steps" / "18-step-10d.md"
    ).read_text()
    assert "pinned merge-base" not in helper_text, "stale --main-sha contract in helper"
    assert "pinned merge-base" not in caller_text, "stale --main-sha contract in caller spec"
    assert "pinned origin/main TIP" in helper_text
    assert "derives the merge-base from it" in helper_text
    assert "NOT the merge-base" in helper_text
    assert "`origin/main` TIP" in caller_text
    assert "NOT the merge-base" in caller_text
    assert "verdict-equivalent by construction" in caller_text


def test_guard4_honors_pin_older_than_live_main(tmp_path):
    """THE discriminator (#2428): a pin OLDER than live origin/main is HONORED.

    The #1128 scenario the flag exists for: a concurrent session's fetch
    advances ``origin/main`` mid-guard, so the caller's captured tip (here
    ``c1``, the fork commit) is STRICTLY OLDER than the live tip (``c2``).
    A pin-honoring implementation derives ``MB = merge-base(HEAD, c1) = c1``,
    evaluates the EMPTY ``c1..c1`` add-set, and PASSES; a pin-IGNORING
    implementation reads live ``origin/main`` (= ``c2``), finds c2's added
    lines missing from the dropped snapshot, and REFUSES (rc 1).
    """
    scratch, wt = _seed_guard4_fixture(tmp_path, 2428, drop_main_lines=True)
    c1 = _git(wt, "rev-parse", "origin/main~1").stdout.strip()
    live_tip = _git(wt, "rev-parse", "origin/main").stdout.strip()
    assert c1 != live_tip, (c1, live_tip)
    proc = _run_script(scratch, "2428", "--guard", "4", "--main-sha", c1)
    # PASS is CORRECT here even though drop_main_lines=True: the pin (c1)
    # PREDATES main's added lines, so the pinned add-set (c1..c1) is empty —
    # exactly what honoring the caller's snapshot means. A refusal (rc 1)
    # here is the SIGNATURE of a flag-IGNORING implementation that read live
    # origin/main (c2) instead of the pin. Do NOT "correct" this expectation
    # to refused — that would re-bless the very implementation this test
    # exists to kill (#2428 plan §4 T7 mis-maintenance inoculation).
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("GUARD4") == "pass", proc.stdout
    # Second line of defense: the derived base must equal the PIN itself.
    assert kv.get("GUARD4_MERGE_BASE") == c1, proc.stdout


def test_guard4_unresolvable_main_sha_fails_loud(tmp_path):
    """An unresolvable ``--main-sha`` is rc 2 ``merge-base-failed``.

    Pins a hole the #2428 fix newly CLOSES: pre-fix, ``MB=deadbeef`` flowed
    into the path enumeration (``diff --name-only deadbeef...HEAD`` under
    ``2>/dev/null`` — unresolvable revision, EMPTY output), the scan loop
    never executed, and the guard silently emitted ``GUARD4=pass`` exit 0 —
    a second, independent vacuity route. Post-fix the merge-base derivation
    fails loud before any enumeration.
    """
    scratch, _ = _seed_guard4_fixture(tmp_path, 2428, drop_main_lines=True)
    proc = _run_script(scratch, "2428", "--guard", "4", "--main-sha", "deadbeef")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("ERROR") == "merge-base-failed", proc.stdout


def test_guard4_passes_on_legitimate_prefork_deletion(tmp_path):
    """False-positive direction: deleting a PRE-FORK line is NOT this class.

    Behaviorally pins the carve-out the caller prose states ("a legitimate
    branch DELETION of a pre-existing function is NOT this class, because
    those lines were never main-side additions past the merge-base"): the
    branch deletes one pre-fork original line while preserving every
    main-added line — Guard 4 passes under BOTH invocation forms.
    """
    scratch, wt = _seed_guard4_fixture(tmp_path, 2428, drop_main_lines=False)
    scoped_wt = wt / GUARD4_SCOPED_FILE
    scoped_wt.write_text(
        "original line 1\nmain-added line A\nmain-added line B\nbranch-added line X\n"
    )
    _git(wt, "add", GUARD4_SCOPED_FILE)
    _git(wt, "commit", "-q", "-m", "branch deletes a pre-fork original line")
    tip = _git(wt, "rev-parse", "origin/main").stdout.strip()
    for args in (("--main-sha", tip), ()):
        proc = _run_script(scratch, "2428", "--guard", "4", *args)
        assert proc.returncode == 0, (args, proc.stdout, proc.stderr)
        kv = _parse_kv(proc.stdout)
        assert kv.get("GUARD4") == "pass", (args, proc.stdout)


# ---------------------------------------------------------------------------
# Guard 4 eval-consumer conformance (r2, #2428:
# concern guard4-eval-output-not-shell-safe)
#
# The tests above parse stdout with the Python splitter, never the production
# ``eval`` form — which is how the unescaped refusal emission
# (``LOST_UPDATE_PATHS=<path>(<count>)``: bare parens are a bash syntax error
# inside ``eval "$GUARD4_OUT"``) survived r1. These run the DOCUMENTED caller
# form in a real bash child and assert exact recovery of the logical values.
# ---------------------------------------------------------------------------


def test_guard4_eval_consumer_recovers_metachar_path(tmp_path):
    """The documented eval consumer exactly recovers a metachar-bearing path.

    The scoped path carries spaces AND parentheses. Pre-fix (``%s``
    emission) the ``eval`` dies with a bash syntax error and
    ``LOST_UPDATE_PATHS`` is never assigned; post-fix (``%q``, the
    ``DIVERGED_FILE`` pattern) every value round-trips exactly.
    """
    scoped = ".claude/rules/step10d guard4 (metachars).md"
    scratch, wt = _seed_guard4_fixture(tmp_path, 2428, drop_main_lines=True, scoped_file=scoped)
    tip = _git(wt, "rev-parse", "origin/main").stdout.strip()
    mb = _git(wt, "merge-base", "HEAD", "origin/main").stdout.strip()
    proc = _run_guard4_eval_consumer(scratch, 2428, tip)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    # The eval'd payload must not be a bash parse error; the refusal banner
    # (stderr) confirms the refusal path actually executed.
    assert "syntax error" not in proc.stderr, proc.stderr
    assert "LOST-UPDATE REFUSAL (Guard 4, #1713)" in proc.stderr, proc.stderr
    kv = _parse_kv(proc.stdout)
    # Refusal rc, captured via the caller form's own $? two-step.
    assert kv.get("EVAL_GUARD4_RC") == "1", proc.stdout
    assert kv.get("EVAL_GUARD4") == "refused", proc.stdout
    assert kv.get("EVAL_GUARD4_MERGE_BASE") == mb, proc.stdout
    # Exact logical recovery: both main-added lines dropped -> (2).
    assert kv.get("EVAL_LOST_UPDATE_PATHS") == f"{scoped}(2)", proc.stdout


def test_guard4_eval_consumer_on_plain_refusal_fixture(tmp_path):
    """Production repro: even an ORDINARY path broke the eval consumer.

    The ``(count)`` suffix alone made the pre-fix ``%s`` emission a bash
    syntax error under ``eval`` — every real refusal, metachars or not.
    Post-fix the plain fixture round-trips exactly like the metachar one.
    """
    scratch, wt = _seed_guard4_fixture(tmp_path, 2428, drop_main_lines=True)
    tip = _git(wt, "rev-parse", "origin/main").stdout.strip()
    mb = _git(wt, "merge-base", "HEAD", "origin/main").stdout.strip()
    proc = _run_guard4_eval_consumer(scratch, 2428, tip)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "syntax error" not in proc.stderr, proc.stderr
    assert "LOST-UPDATE REFUSAL (Guard 4, #1713)" in proc.stderr, proc.stderr
    kv = _parse_kv(proc.stdout)
    assert kv.get("EVAL_GUARD4_RC") == "1", proc.stdout
    assert kv.get("EVAL_GUARD4") == "refused", proc.stdout
    assert kv.get("EVAL_GUARD4_MERGE_BASE") == mb, proc.stdout
    assert kv.get("EVAL_LOST_UPDATE_PATHS") == f"{GUARD4_SCOPED_FILE}(2)", proc.stdout


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------


def test_unknown_guard_exits_2(tmp_path):
    scratch = _make_scratch_repo(tmp_path)
    proc = _run_script(scratch, "1978", "--guard", "99")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("ERROR", "").startswith("unknown-guard"), proc.stdout


def test_missing_issue_number_exits_2(tmp_path):
    scratch = _make_scratch_repo(tmp_path)
    # Argument-order: without a numeric issue arg, the script rejects.
    proc = _run_script(scratch, "not-a-number", "--guard", "prelude")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("ERROR") == "bad-issue", proc.stdout


# ---------------------------------------------------------------------------
# --guard divergence (#1771 -> #2201)
# ---------------------------------------------------------------------------


def _seed_divergence_fixture(
    tmp_path: Path,
    issue: int,
    *,
    diverge: bool = True,
) -> tuple[Path, Path]:
    """Scratch + worktree covering every divergence-probe filter class.

    Layout after this returns (bare origin at ``tmp_path/origin.git``):

    - main c1 seeds five files; branch ``issue-<N>`` forks at c1.
    - main c2 edits ALL five files (pushed -> ``refs/remotes/origin/main``).
    - The worktree commits branch-side edits so the raw branch-AND-main
      intersection holds all five, and the refined set keeps exactly ONE:

      * ``scripts/collide.py`` — both sides edit DIFFERENTLY, non-sync
        subject -> the semantic-collision SURVIVOR (skipped when
        ``diverge=False`` — the clean-path fixture for pin (k)).
      * ``.claude/rules/synconly.md`` — branch side touched ONLY by a
        commit whose subject carries the spec-freshness sync anchor ->
        dropped by the subject-scoped exclusion (pin (b)).
      * ``docs/converged.md`` — branch content ends byte-equal to the main
        tip -> dropped by the content-identical filter (pin (c)).
      * ``tasks/running/9/note.md`` + ``.claude/agent-memory/a/m.md`` ->
        dropped by the carve-outs (pin (d)).
    """
    origin_dir = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin_dir))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _git(scratch, "init", "-q", "-b", "main")
    _git(scratch, "remote", "add", "origin", str(origin_dir))
    seeds = {
        "scripts/collide.py": "line1\nline2\n",
        ".claude/rules/synconly.md": "sync base\n",
        "docs/converged.md": "converged base\n",
        "tasks/running/9/note.md": "task base\n",
        ".claude/agent-memory/a/m.md": "mem base\n",
    }
    for rel, content in seeds.items():
        p = scratch / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    _git(scratch, "add", *seeds.keys())
    _git(scratch, "commit", "-q", "-m", "c1")
    _git(scratch, "push", "-q", "-u", "origin", "main")
    _git(scratch, "checkout", "-q", "-b", f"issue-{issue}")
    _git(scratch, "checkout", "-q", "main")
    # c2 on main: edit every seeded file.
    (scratch / "scripts/collide.py").write_text("line1-main-edit\nline2\n")
    (scratch / ".claude/rules/synconly.md").write_text("sync base\nmain synconly edit\n")
    (scratch / "docs/converged.md").write_text("converged target\n")
    (scratch / "tasks/running/9/note.md").write_text("task main edit\n")
    (scratch / ".claude/agent-memory/a/m.md").write_text("mem main edit\n")
    _git(scratch, "add", *seeds.keys())
    _git(scratch, "commit", "-q", "-m", "c2: main edits every file")
    _git(scratch, "push", "-q", "origin", "main")
    # Worktree on the branch (forked at c1).
    worktree_dir = scratch / ".claude" / "worktrees" / f"issue-{issue}"
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)
    _git(scratch, "worktree", "add", "-q", str(worktree_dir), f"issue-{issue}")
    if diverge:
        (worktree_dir / "scripts/collide.py").write_text("line1\nline2-branch-edit\n")
        _git(worktree_dir, "add", "scripts/collide.py")
        _git(worktree_dir, "commit", "-q", "-m", "branch edits collide.py")
    (worktree_dir / ".claude/rules/synconly.md").write_text("sync base\nbranch sync import\n")
    _git(worktree_dir, "add", ".claude/rules/synconly.md")
    _git(
        worktree_dir,
        "commit",
        "-q",
        "-m",
        "sync workflow-surface specs from origin/main @ deadbeef",
    )
    (worktree_dir / "docs/converged.md").write_text("converged target\n")
    _git(worktree_dir, "add", "docs/converged.md")
    _git(worktree_dir, "commit", "-q", "-m", "branch converges docs/converged.md")
    (worktree_dir / "tasks/running/9/note.md").write_text("task branch edit\n")
    (worktree_dir / ".claude/agent-memory/a/m.md").write_text("mem branch edit\n")
    _git(worktree_dir, "add", "tasks/running/9/note.md", ".claude/agent-memory/a/m.md")
    _git(worktree_dir, "commit", "-q", "-m", "branch edits task-state paths")
    return scratch, worktree_dir


def _run_divergence(
    scratch: Path, issue: int, out: Path
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    proc = _run_script(scratch, str(issue), "--guard", "divergence", "--out", str(out))
    return proc, _parse_kv(proc.stdout)


def test_divergence_lists_semantic_collision(tmp_path):
    """Pin (a): branch edits F + main edits F differently -> F listed, diverged."""
    scratch, _ = _seed_divergence_fixture(tmp_path, 2201)
    out = tmp_path / "div.txt"
    proc, kv = _run_divergence(scratch, 2201, out)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert kv.get("DIVERGENCE") == "diverged", proc.stdout
    assert kv.get("DIVERGED_COUNT") == "1", proc.stdout
    assert kv.get("DIVERGED_FILE") == str(out), proc.stdout
    assert out.read_text().splitlines() == ["scripts/collide.py"]


def test_divergence_excludes_sync_only_paths(tmp_path):
    """Pin (b): a path whose ONLY branch-side commit is a sync import is dropped."""
    scratch, _ = _seed_divergence_fixture(tmp_path, 2201)
    out = tmp_path / "div.txt"
    _proc, _kv = _run_divergence(scratch, 2201, out)
    assert ".claude/rules/synconly.md" not in out.read_text().splitlines()


def test_divergence_excludes_content_identical(tmp_path):
    """Pin (c): branch content byte-equal to the main tip is dropped."""
    scratch, _ = _seed_divergence_fixture(tmp_path, 2201)
    out = tmp_path / "div.txt"
    _proc, _kv = _run_divergence(scratch, 2201, out)
    assert "docs/converged.md" not in out.read_text().splitlines()


def test_divergence_excludes_tasks_and_agent_memory(tmp_path):
    """Pin (d): ``tasks/`` + ``.claude/agent-memory/`` carve-outs."""
    scratch, _ = _seed_divergence_fixture(tmp_path, 2201)
    out = tmp_path / "div.txt"
    _proc, _kv = _run_divergence(scratch, 2201, out)
    listed = out.read_text().splitlines()
    assert "tasks/running/9/note.md" not in listed
    assert ".claude/agent-memory/a/m.md" not in listed


def test_divergence_skips_on_main_checkout(tmp_path):
    """Pin (e): no issue worktree -> the CURRENT (main) checkout self-skips."""
    scratch = _make_scratch_repo(tmp_path)
    proc = _run_script(scratch, "2201", "--guard", "divergence")
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("DIVERGENCE") == "skipped", proc.stdout
    assert kv.get("DIVERGED_COUNT") == "0", proc.stdout
    assert kv.get("DIVERGED_FILE") == "", proc.stdout


def test_divergence_no_origin_main_exits_2(tmp_path):
    """Pin (f), arm 1: unresolvable remote-main ref -> exit 2 + ERROR=."""
    scratch, _ = _make_scratch_repo_with_worktree(tmp_path, 2201)
    proc = _run_script(scratch, "2201", "--guard", "divergence")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("ERROR") == "no-origin-main", proc.stdout


def test_divergence_no_merge_base_exits_2(tmp_path):
    """Pin (f), arm 2: origin/main with UNRELATED history -> ERROR=no-merge-base."""
    scratch, _ = _make_scratch_repo_with_worktree(tmp_path, 2201)
    empty_tree = _git(scratch, "hash-object", "-t", "tree", "/dev/null").stdout.strip()
    root_sha = _git(scratch, "commit-tree", empty_tree, "-m", "unrelated root").stdout.strip()
    _git(scratch, "update-ref", "refs/remotes/origin/main", root_sha)
    proc = _run_script(scratch, "2201", "--guard", "divergence")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("ERROR") == "no-merge-base", proc.stdout


def test_divergence_main_sha_pin(tmp_path):
    """Pin (g): MAIN_SHA= equals the fixture origin/main tip; the emitted list
    equals the refined diff computed AT that sha."""
    scratch, worktree = _seed_divergence_fixture(tmp_path, 2201)
    out = tmp_path / "div.txt"
    proc, kv = _run_divergence(scratch, 2201, out)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    tip = _git(worktree, "rev-parse", "refs/remotes/origin/main").stdout.strip()
    assert kv.get("MAIN_SHA") == tip, proc.stdout
    # Recompute the expected refined set at the emitted sha: for this fixture
    # exactly the semantic-collision file (branch vs pinned tip still differ).
    diff = _git(worktree, "diff", "--name-only", "HEAD", kv["MAIN_SHA"], "--", "scripts/collide.py")
    assert diff.stdout.strip() == "scripts/collide.py"
    assert out.read_text().splitlines() == ["scripts/collide.py"]


def test_divergence_static_capture_order():
    """Pin (g2): STATIC capture-order pin — the divergence arm captures
    ``MAIN_SHA=$(... rev-parse origin/main)`` BEFORE the merge-base line and
    contains NO literal ``origin/main`` read after the capture (the round-2
    reconciler descoped the behavioral mid-probe ref-advance test — the
    static pin + (g) suffice)."""
    text = SCRIPT.read_text()
    assert "divergence)" in text, "divergence arm missing from the script"
    block = text.split("divergence)", 1)[1]
    # The arm ends at the fall-through unknown-guard case.
    block = block.split('_die_usage "unknown-guard-', 1)[0]
    lines = block.splitlines()
    capture_idx = next(
        i
        for i, ln in enumerate(lines)
        if "MAIN_SHA=$(" in ln and "rev-parse" in ln and "origin/main" in ln
    )
    mb_idx = next(i for i, ln in enumerate(lines) if "merge-base" in ln and "MB=" in ln)
    assert capture_idx < mb_idx, "MAIN_SHA capture must precede the merge-base line"
    tail = "\n".join(lines[capture_idx + 1 :])
    assert "origin/main" not in tail, (
        "literal origin/main read AFTER the pinned capture in the divergence arm:\n" + tail
    )


# --- MF1/MF2 mechanization pins: the Step 10d delta-computation caller -------
#
# The D4 fenced block reads the review-time note via ``scripts/task.py view``
# (unrunnable in a scratch repo -- task.py branch-guards to the live main), so
# these pins run a token-pinned faithful TRANSCRIPTION of the fenced
# delta-computation lines against fixture notes.
# ``test_delta_transcription_fragments_pinned_in_spec`` ties the transcription
# to the composed spec text so the two cannot drift silently.

_DELTA_TRANSCRIPTION = r"""
LASTNOTE=$(cat "$LASTNOTE_FILE")
printf '%s' "$LASTNOTE" | sed -n 's/.*files=//p' | tr ',' '\n' | sed '/^$/d' | sort -u > "$REVSET"
REV_MAIN=$(printf '%s' "$LASTNOTE" | grep -oE 'main=[0-9a-f]+' | head -1 | cut -d= -f2)
sort -u "$DIVOUT" > "$CUR"
if [ -z "$LASTNOTE" ] || printf '%s' "$LASTNOTE" | grep -q ' ERROR ' \
   || [ -z "$REV_MAIN" ] || ! git -C "$WT" cat-file -e "$REV_MAIN^{commit}" 2>/dev/null; then
  cp "$CUR" "$NEWLIST"
else
  comm -13 "$REVSET" "$CUR" > "$AOUT"
  if git -C "$WT" -c core.quotePath=false diff --name-only "$REV_MAIN" "$MAIN_SHA" \
      > "$XYOUT"; then
    sort -u "$XYOUT" \
      | comm -12 - "$CUR" > "$BOUT"
    sort -u "$AOUT" "$BOUT" > "$NEWLIST"
  else
    cp "$CUR" "$NEWLIST"
  fi
fi
"""

#: Load-bearing pipeline fragments shared verbatim by the transcription above
#: and the composed-spec fenced block (asserted below).
_DELTA_FRAGMENTS = (
    "sed -n 's/.*files=//p'",
    "grep -oE 'main=[0-9a-f]+'",
    'comm -13 "$REVSET"',
    "comm -12 -",
    'cat-file -e "$REV_MAIN^{commit}"',
    # Review r1 MF-1b + MF-2: the reviewed->current main diff is MATERIALIZED
    # with an rc check (never a bare pipeline) and quotePath-disabled.
    'git -C "$WT" -c core.quotePath=false diff --name-only "$REV_MAIN" "$MAIN_SHA"',
)


def test_delta_transcription_fragments_pinned_in_spec():
    """The transcription's load-bearing fragments appear verbatim in the
    composed /issue spec (the Step 10d delta gate's fenced block)."""
    from tests.issue_skill_source import issue_skill_text

    text = issue_skill_text()
    for frag in _DELTA_FRAGMENTS:
        assert frag in text, f"delta-gate fragment missing from composed spec: {frag!r}"


def _run_delta(
    tmp_path: Path,
    worktree: Path,
    divout: Path,
    main_sha: str,
    lastnote: str,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    """Run the transcribed Step 10d delta computation; returns NEWLIST lines."""
    workdir = tmp_path / "delta"
    workdir.mkdir(exist_ok=True)
    lastnote_file = workdir / "lastnote.txt"
    lastnote_file.write_text(lastnote)
    newlist = workdir / "newlist.txt"
    env = os.environ.copy()
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(k, None)
    env.update(
        {
            "LASTNOTE_FILE": str(lastnote_file),
            "REVSET": str(workdir / "reviewed.txt"),
            "DIVOUT": str(divout),
            "CUR": str(workdir / "cur.txt"),
            "AOUT": str(workdir / "a.txt"),
            "BOUT": str(workdir / "b.txt"),
            "XYOUT": str(workdir / "xy.txt"),
            "NEWLIST": str(newlist),
            "WT": str(worktree),
            "MAIN_SHA": main_sha,
        }
    )
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        ["bash", "-c", _DELTA_TRANSCRIPTION],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    return newlist.read_text().splitlines()


def test_delta_content_keyed_retouch(tmp_path):
    """Pin (h): a file DISCLOSED at ``main=<sha1>`` that main re-touches after
    ``<sha1>`` lands in the NEW list (content-keyed, never pathname-only) —
    and an empty review record fail-closes to the FULL probe set."""
    scratch, worktree = _seed_divergence_fixture(tmp_path, 2201)
    sha_c2 = _git(worktree, "rev-parse", "refs/remotes/origin/main").stdout.strip()
    # main c3: re-touch the collision file AFTER the reviewed sha.
    (scratch / "scripts/collide.py").write_text("line1-main-edit-2\nline2\n")
    _git(scratch, "add", "scripts/collide.py")
    _git(scratch, "commit", "-q", "-m", "c3: main re-touches collide.py")
    _git(scratch, "push", "-q", "origin", "main")
    out = tmp_path / "div.txt"
    proc, kv = _run_divergence(scratch, 2201, out)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert kv["MAIN_SHA"] != sha_c2, "fixture must advance main past the reviewed sha"
    lastnote = f"[divergence-probe] r1 count=1 main={sha_c2} files=scripts/collide.py"
    newlist = _run_delta(tmp_path, worktree, out, kv["MAIN_SHA"], lastnote)
    assert "scripts/collide.py" in newlist, (
        "re-touched disclosed path must be UNREVIEWED under the content key"
    )
    # FAIL-CLOSED arm: no review record -> the full probe set is unreviewed.
    assert _run_delta(tmp_path, worktree, out, kv["MAIN_SHA"], "") == out.read_text().splitlines()


def _make_git_shim(tmp_path: Path, refuse: str) -> Path:
    """Write a PATH-shim ``git`` that fails on subcommand ``refuse`` and
    delegates everything else to the real git; returns the shim dir."""
    real_git = shutil.which("git")
    assert real_git, "git not on PATH"
    shim_dir = tmp_path / "git-shim"
    shim_dir.mkdir(exist_ok=True)
    shim = shim_dir / "git"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        'for a in "$@"; do\n'
        f'  if [ "$a" = "{refuse}" ]; then\n'
        f'    echo "shim: {refuse} refused" >&2\n'
        "    exit 128\n"
        "  fi\n"
        "done\n"
        f'exec "{real_git}" "$@"\n'
    )
    shim.chmod(0o755)
    return shim_dir


def test_delta_masked_diff_failure_fails_closed(tmp_path):
    """MF-1(b) (review r1): a FAILED reviewed->current main diff at the merge
    site fail-closes to the FULL current probe set -- a bare pipeline would
    exit through sort|comm rc 0, read set B as EMPTY, and let a
    previously-disclosed re-touched file merge as reviewed."""
    scratch, worktree = _seed_divergence_fixture(tmp_path, 2201)
    out = tmp_path / "div.txt"
    proc, kv = _run_divergence(scratch, 2201, out)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    lastnote = f"[divergence-probe] r1 count=1 main={kv['MAIN_SHA']} files=scripts/collide.py"
    shim_dir = _make_git_shim(tmp_path, "diff")
    newlist = _run_delta(
        tmp_path,
        worktree,
        out,
        kv["MAIN_SHA"],
        lastnote,
        extra_env={"PATH": f"{shim_dir}:{os.environ['PATH']}"},
    )
    assert newlist == out.read_text().splitlines(), (
        "masked diff failure must fail CLOSED to the full current probe set"
    )


def test_delta_quotepath_nonascii_retouch(tmp_path):
    """MF-2 (review r1): a DISCLOSED non-ASCII path that main re-touches
    after the reviewed sha lands in NEW -- without quotePath=false the
    merge-site diff lists it C-escaped, misses ``comm -12`` against the raw
    current set, and the unreviewed re-touch merges as reviewed."""
    scratch, worktree = _seed_divergence_fixture(tmp_path, 2201)
    nonascii = "scripts/café.py"
    (scratch / nonascii).write_text("v1\n")
    _git(scratch, "add", nonascii)
    _git(scratch, "commit", "-q", "-m", "c3: add non-ascii path")
    _git(scratch, "push", "-q", "origin", "main")
    rev_main = _git(scratch, "rev-parse", "HEAD").stdout.strip()  # disclosure sha
    (scratch / nonascii).write_text("v2\n")
    _git(scratch, "add", nonascii)
    _git(scratch, "commit", "-q", "-m", "c4: main re-touches non-ascii path")
    _git(scratch, "push", "-q", "origin", "main")
    main_sha = _git(scratch, "rev-parse", "HEAD").stdout.strip()
    divout = tmp_path / "div.txt"
    divout.write_text(f"{nonascii}\n", encoding="utf-8")  # raw current probe set
    lastnote = f"[divergence-probe] r1 count=1 main={rev_main} files={nonascii}"
    newlist = _run_delta(tmp_path, worktree, divout, main_sha, lastnote)
    assert nonascii in newlist, "re-touched non-ASCII disclosed path must land in NEW"


def test_divergence_git_log_failure_exits_2(tmp_path):
    """MF-1(a) (review r1): a failed per-path history probe exits 2 with
    ``ERROR=log-failed`` -- the masked-pipeline form read a failed log as
    "sync-only" (empty) and silently DROPPED a real divergence."""
    scratch, _ = _seed_divergence_fixture(tmp_path, 2201)
    shim_dir = _make_git_shim(tmp_path, "log")
    proc = _run_script(
        scratch,
        "2201",
        "--guard",
        "divergence",
        "--out",
        str(tmp_path / "div.txt"),
        env_extra={"PATH": f"{shim_dir}:{os.environ['PATH']}"},
    )
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("ERROR") == "log-failed", proc.stdout


def test_divergence_out_path_quoted_for_eval(tmp_path):
    """Should-fix (review r1): ``DIVERGED_FILE=%q`` round-trips a
    space-bearing --out path through the two-step caller's ``eval``."""
    scratch, _ = _seed_divergence_fixture(tmp_path, 2201)
    out = tmp_path / "di v.txt"
    proc, _kv = _run_divergence(scratch, 2201, out)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    env = os.environ.copy()
    env["DIV_OUT"] = proc.stdout
    check = subprocess.run(
        ["bash", "-c", 'eval "$DIV_OUT"; printf "%s" "$DIVERGED_FILE"'],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert check.returncode == 0, (check.stdout, check.stderr)
    assert check.stdout == str(out)
    assert out.read_text().splitlines() == ["scripts/collide.py"]


def test_delta_cap_durability_and_prefix_split(tmp_path):
    """Pin (i): a ``step10d new=`` note NEWER than the latest per-round note
    spends the one reconciliation dispatch (proceed-after-cap, no second
    dispatch); a FRESH per-round note re-arms it. Also pins that the two
    note prefixes differ, so the ``startswith("[divergence-probe] r")``
    filter excludes the step10d notes by construction."""

    def cap_spent(notes: list[str]) -> bool:
        per_round = [i for i, n in enumerate(notes) if n.startswith("[divergence-probe] r")]
        new_notes = [
            i for i, n in enumerate(notes) if n.startswith("[divergence-probe] step10d new=")
        ]
        if not new_notes:
            return False
        if not per_round:
            return True
        return max(new_notes) > max(per_round)

    r1 = "[divergence-probe] r1 count=1 main=abc123 files=scripts/collide.py"
    r2 = "[divergence-probe] r2 count=1 main=def456 files=scripts/collide.py"
    step10d_new = "[divergence-probe] step10d new=scripts/collide.py"
    assert cap_spent([r1, step10d_new]), "newer step10d new= note must spend the cap"
    assert not cap_spent([r1, step10d_new, r2]), "a fresh review round re-arms the cap"
    assert not cap_spent([r1]), "no step10d note -> cap unspent"
    # The per-round filter must not match the step10d notes.
    assert not step10d_new.startswith("[divergence-probe] r")
    assert "[divergence-probe] step10d ERROR rc=2".startswith("[divergence-probe] r") is False
    # Should-fix (review r1): bind the cap clause's fragments to the
    # MERGE-SITE region of the composed spec, not just anywhere in the doc.
    from tests.issue_skill_source import issue_skill_text

    text = issue_skill_text()
    start = text.index("#### Pre-merge divergence delta gate")
    region = text[start : text.index("\n#### ", start + 1)]
    for frag in (
        "[divergence-probe] step10d new=",
        "NEWER than the latest per-round",
        "disposition=proceed-after-cap",
    ):
        assert frag in region, f"cap-clause fragment missing from the merge-site region: {frag!r}"


def test_caller_rc_nonzero_hygiene(tmp_path):
    """Pin (j): helper exit 2 with a STALE pre-seeded --out file -> the caller
    posts an ERROR-shaped note (never ``count=``), computes NO new-count, and
    takes the probe-error disposition; the stale list is gone (rm -f)."""
    fake = tmp_path / "fake_helper.sh"
    fake.write_text("#!/usr/bin/env bash\nprintf 'ERROR=%s\\n' fixture-fail\nexit 2\n")
    divout = tmp_path / "div.txt"
    divout.write_text("stale/path.py\n")  # STALE list a prior invocation left
    caller = """
rm -f "$DIVOUT"
DIV_OUT=$(bash "$FAKE" 2201 --guard divergence --out "$DIVOUT"); DIV_RC=$?
eval "$DIV_OUT"
if [ "$DIV_RC" -ne 0 ]; then
  DIV_NOTE="[divergence-probe] r1 ERROR rc=$DIV_RC ${ERROR:-probe-failed}"
  printf 'DELTA:skipped disposition=probe-error\\n'
else
  DIV_NOTE="[divergence-probe] r1 count=$DIVERGED_COUNT main=$MAIN_SHA files="
  printf 'DELTA:computed\\n'
fi
printf 'NOTE:%s\\n' "$DIV_NOTE"
[ -e "$DIVOUT" ] && printf 'OUTFILE:present\\n' || printf 'OUTFILE:absent\\n'
"""
    env = os.environ.copy()
    env.update({"FAKE": str(fake), "DIVOUT": str(divout)})
    proc = subprocess.run(
        ["bash", "-c", caller], capture_output=True, text=True, env=env, timeout=60
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    note = next(ln for ln in proc.stdout.splitlines() if ln.startswith("NOTE:"))
    assert note.startswith("NOTE:[divergence-probe] r1 ERROR rc=2"), note
    assert "count=" not in note, f"a failed probe must never post a clean count= note: {note}"
    assert "DELTA:skipped disposition=probe-error" in proc.stdout, proc.stdout
    assert "OUTFILE:absent" in proc.stdout, "stale --out list must be removed, never computed from"


def test_divergence_clean_path_note_shape(tmp_path):
    """Pin (k): rc 0 with no divergence yields the ``count=0 main=<sha>
    files=`` note shape (count=0 included — the every-round contract)."""
    import re

    scratch, _ = _seed_divergence_fixture(tmp_path, 2201, diverge=False)
    out = tmp_path / "div.txt"
    proc, kv = _run_divergence(scratch, 2201, out)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert kv.get("DIVERGENCE") == "clean", proc.stdout
    files_part = ",".join(out.read_text().split())
    note = (
        f"[divergence-probe] r1 count={kv['DIVERGED_COUNT']} "
        f"main={kv['MAIN_SHA']} files={files_part}"
    )
    assert re.fullmatch(r"\[divergence-probe\] r1 count=0 main=[0-9a-f]{40} files=", note), note


def test_script_is_executable():
    """The extracted script should be executable.

    Callers can then invoke it as ``bash <path>`` OR ``<path>`` directly.
    """
    assert SCRIPT.exists(), f"script not found at {SCRIPT}"
    # The script's PRELUDE recipe is transcribed byte-close to the SKILL.md
    # fence -- pin the ``--path-format=absolute`` token to catch a typo drift
    # (the #1867 incident-triggering fragment).
    text = SCRIPT.read_text()
    assert "--path-format=absolute" in text, "script must use --path-format=absolute"
    # The refusal banner must be the exact SKILL.md fence text.
    assert "LOST-UPDATE REFUSAL (Guard 4, #1713)" in text, "refusal banner drift"
    # The kill switch env var must be honored.
    assert "EPM_SKIP_LOST_UPDATE_GUARD" in text, "kill switch env var not honored"
