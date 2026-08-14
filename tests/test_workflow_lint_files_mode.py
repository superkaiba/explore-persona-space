"""TDD surface for the ``workflow_lint.py --files`` payload-scoping mode (task #2235).

Written TESTS-FIRST (plan ``.claude/plans/issue-2235.md`` section "TDD", cases
1-7): every ``--files``-exercising test here is RED until Phase B lands. The
contract under test:

* ``--files <path> [<path>...]`` (repo-relative) runs the NO-FLAGS check set
  with per-check scoping via a module-level ``CHECK_SCOPES`` registry
  (``kind`` in {"path-local", "global"}): path-local checks run with their
  file enumeration filtered to payload + per-issue import closure; global
  checks run only when the payload intersects their surfaces, else emit one
  SKIP line. Combining ``--files`` with any ``--check-*`` flag is an argparse
  error (exit 2).
* A deliberately-red payload still exits FAIL naming the file — the #1388
  fleet-integrity property (acceptance criterion 2).
* Corpus-global findings (allowlist staleness, findings on non-payload files)
  never leak into a scoped verdict: a clean payload PASSes even in a tree
  where the bare no-flags run is red (probe 2026-08-11: a minimal tmp tree
  bare run exits FAIL with ~100 allowlist-staleness errors from live
  allowlists naming absent files — the #2079 class).
* The per-issue import closure (bare ``issue*``-stem imports resolving to
  ``scripts/<name>.py``) is scanned; an unresolvable ``issue*``-stem import
  FAILs naming the PAYLOAD file; a non-``issue*`` bare import (numpy, stdlib)
  is NEVER treated as unresolvable (plan section 4 B4 — the load-bearing
  stem restriction).
* SCOPE/SKIP informational lines carry counts + check NAMES only — never a
  payload path string: ``inline_lint_gate.evaluate`` treats any leg line
  containing a payload path and not prefixed WARN/PASSED/SKIPPED as a red
  hit, so a payload-naming informational line would self-inflict a false
  block (plan section 4 B5).
* Without ``--files``, behavior is byte-identical to today: unscoped
  whole-tree enumeration, no SCOPE line (acceptance: Step 9c unaffected).
* The terminal line shapes ``workflow_lint: PASS`` / ``workflow_lint: FAIL
  (N error(s))`` are preserved in files-mode — pinned against the REAL
  ``inline_lint_gate.LINT_TERMINAL_RE`` so a drift breaks here, not in a
  fleet gate.

Fixture strategy: hermetic TMP TREES — ``workflow_lint.py`` resolves
``_REPO_ROOT`` from ``__file__``, so a copy at ``<tree>/scripts/`` scans only
the tree's fixture files; ``workflow.yaml`` is copied in because the
unconditional schema load reads ``.claude/workflow.yaml`` relative to CWD.

Deliberately a NEW file (NOT ``tests/test_workflow_lint.py``): that file
carries the mapped-pytest slow surcharge (``inline_lint_gate.py`` ~528-536),
while the Step 9c selector's broad stem glob ``tests/test_*workflow_lint*.py``
still auto-selects this file for future ``workflow_lint.py`` diffs (plan
section 11 items 5-6).
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LINT = _REPO_ROOT / "scripts" / "workflow_lint.py"
_GATE = _REPO_ROOT / "scripts" / "inline_lint_gate.py"

_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint as wl  # noqa: E402

# The gate's terminal-line + non-red-prefix contracts are pinned against the
# REAL inline_lint_gate module (not a hard-coded copy) so producer/consumer
# drift fails here. Reuse an already-registered module instance (the sibling
# test_inline_lint_gate.py registers the same name) to avoid a re-exec.
if "inline_lint_gate" in sys.modules:
    ilg = sys.modules["inline_lint_gate"]
else:
    _spec = importlib.util.spec_from_file_location("inline_lint_gate", _GATE)
    assert _spec and _spec.loader
    ilg = importlib.util.module_from_spec(_spec)
    sys.modules["inline_lint_gate"] = ilg
    _spec.loader.exec_module(ilg)


# --------------------------------------------------------------------------
# Fixture payload contents.
#
# The judge-pin fixture must be red while THIS test file stays lint-clean:
# check_judge_model_pins scans tests/**/*.py too, and a hit only requires the
# forbidden substring on a judge-named line — so the forbidden literal is
# SPLIT here and only ever contiguous inside the written fixture file.
# --------------------------------------------------------------------------
_FORBIDDEN_PIN = "claude-haiku" + "-4-5"
JUDGE_PIN_RED = f'judge_model = "{_FORBIDDEN_PIN}"\n'

# check_upload_return_discard (#2087): an Expr-statement (discarded-return)
# call to the imported fail-soft hub upload helper.
UPLOAD_DISCARD_RED = (
    "from explore_persona_space.orchestrate.hub import _upload\n"
    "\n"
    '_upload("local_dir", "repo/prefix")\n'
)

# check_jsonl_splitlines (#950), arm (b): bare receiver Name matching /jsonl/i.
JSONL_RED = (
    "def load(jsonl_text: str) -> list[str]:\n"
    '    """Shred-prone JSONL split (fixture: trips check_jsonl_splitlines)."""\n'
    "    return jsonl_text.splitlines()\n"
)

CLEAN_FIG = (
    '"""Clean per-issue fig-script fixture: no workflow_lint findings."""\n'
    "\n"
    "import os\n"
    "\n"
    "\n"
    "def main() -> None:\n"
    '    """Print cwd (stand-in for a plotting entrypoint)."""\n'
    "    print(os.getcwd())\n"
    "\n"
    "\n"
    'if __name__ == "__main__":\n'
    "    main()\n"
)

_FAIL_COUNT_RE = re.compile(r"^workflow_lint: FAIL \((\d+) error\(s\)\)$", re.MULTILINE)
_SCOPE_LINE_RE = re.compile(
    r"^workflow_lint: SCOPE files=\d+ closure=\+\d+ checks_ran=\d+ checks_skipped=\d+$",
    re.MULTILINE,
)


def _make_tree(tmp_path: Path) -> Path:
    """Hermetic tmp tree the copied linter scans as its own repo root."""
    tree = tmp_path / "tree"
    (tree / "scripts").mkdir(parents=True)
    (tree / ".claude").mkdir()
    shutil.copy2(_LINT, tree / "scripts" / "workflow_lint.py")
    shutil.copy2(_REPO_ROOT / ".claude" / "workflow.yaml", tree / ".claude" / "workflow.yaml")
    # The copied linter reads its agent-spec grandfather caps from this data
    # file at IMPORT time (#1718 moved them out of a Python dict literal so
    # concurrent cap raises edit different lines and merge cleanly). A hermetic
    # tree must supply it for the same reason it supplies workflow.yaml: the
    # module hard-fails without it BY DESIGN — the fail-loud posture is pinned
    # by test_workflow_lint_agent_spec_caps.py (a silent empty caps map would
    # un-grandfather every spec and flip WARN-under-cap into FAIL-uncapped
    # fleet-wide), so the fixture models the dependency rather than the loader
    # tolerating its absence.
    (tree / ".claude" / "config").mkdir()
    shutil.copy2(
        _REPO_ROOT / ".claude" / "config" / "agent_spec_size_caps.txt",
        tree / ".claude" / "config" / "agent_spec_size_caps.txt",
    )
    return tree


def _run_lint(tree: Path, *args: str) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run the tree's linter copy; return (proc, combined stdout+stderr).

    The combined form mirrors ``inline_lint_gate._run_leg``, which is the
    production consumer of this output.
    """
    r = subprocess.run(
        [sys.executable, str(tree / "scripts" / "workflow_lint.py"), *args],
        cwd=str(tree),
        capture_output=True,
        text=True,
        check=False,
    )
    return r, r.stdout + "\n" + r.stderr


def _error_lines(out: str) -> list[str]:
    """The ``workflow_lint: <err>`` finding lines (terminal + SCOPE/SKIP
    informational lines excluded)."""
    lines = []
    for ln in out.splitlines():
        if not ln.startswith("workflow_lint: "):
            continue
        if ln.startswith(("workflow_lint: PASS", "workflow_lint: FAIL (")):
            continue
        if ln.startswith(("workflow_lint: SCOPE", "workflow_lint: SKIP")):
            continue
        lines.append(ln)
    return lines


# ---------------------------------------------------------------------------
# Case 1 — THE deliberately-red #1388 regression test (acceptance 2).
# ---------------------------------------------------------------------------
def test_files_mode_new_red_blocks(tmp_path: Path) -> None:
    """A payload tripping a path-local correctness check exits FAIL naming the
    file. If payload-scoping ever lets NEW red through, this is the test that
    catches it (#1388: two inline-landed lint-red scripts broke the Step 9c
    gate fleet-wide). Also pins that the scoped verdict contains ONLY the
    payload's own finding — no corpus-global (allowlist-staleness) leakage,
    which the bare run in this same tree class demonstrably emits."""
    tree = _make_tree(tmp_path)
    payload = "scripts/issue9990_red.py"
    (tree / payload).write_text(JUDGE_PIN_RED, encoding="utf-8")
    r, out = _run_lint(tree, "--files", payload)
    assert r.returncode == 1, out
    terminal = ilg.LINT_TERMINAL_RE.search(out)
    assert terminal is not None and terminal.group(1).startswith("FAIL"), out
    named = [ln for ln in _error_lines(out) if "issue9990_red.py" in ln]
    assert named, f"no error line names the red payload:\n{out}"
    m = _FAIL_COUNT_RE.search(out)
    assert m is not None, out
    assert int(m.group(1)) == 1, f"expected ONLY the payload's own finding:\n{out}"


# ---------------------------------------------------------------------------
# Case 2 — correctness-class checks still fire under --files (acceptance 4).
# The user's scope decision: messy per-issue plotting is acceptable, but
# messy != wrong — these checks must never be scoped away.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("stem", "content"),
    [
        pytest.param("issue9990_judge", JUDGE_PIN_RED, id="check_judge_model_pins"),
        pytest.param("issue9991_upload", UPLOAD_DISCARD_RED, id="check_upload_return_discard"),
        pytest.param("issue9992_jsonl", JSONL_RED, id="check_jsonl_splitlines"),
    ],
)
def test_files_mode_correctness_checks_fire(tmp_path: Path, stem: str, content: str) -> None:
    tree = _make_tree(tmp_path)
    payload = f"scripts/{stem}.py"
    (tree / payload).write_text(content, encoding="utf-8")
    r, out = _run_lint(tree, "--files", payload)
    assert r.returncode == 1, out
    assert any(f"{stem}.py" in ln for ln in _error_lines(out)), (
        f"correctness check did not fire on the payload under --files:\n{out}"
    )


def test_files_mode_clean_payload_passes(tmp_path: Path) -> None:
    """A clean fig-script payload PASSes under --files — in a tree where the
    BARE no-flags run exits FAIL with allowlist-staleness errors (live
    allowlists naming files absent from the tree; probe 2026-08-11: ~100
    errors). Pins that scoping never leaks corpus-global findings into a
    scoped verdict — in production this is exactly the pre-existing-red
    class that must never block an unrelated payload (acceptance 3)."""
    tree = _make_tree(tmp_path)
    payload = "scripts/issue9993_clean.py"
    (tree / payload).write_text(CLEAN_FIG, encoding="utf-8")
    r, out = _run_lint(tree, "--files", payload)
    assert r.returncode == 0, out
    terminal = ilg.LINT_TERMINAL_RE.search(out)
    assert terminal is not None and terminal.group(1) == "PASS", out


# ---------------------------------------------------------------------------
# Case 3 — per-issue import closure (acceptance 5) + the issue*-stem
# restriction's negative control (plan section 4 B4).
# ---------------------------------------------------------------------------
def test_files_mode_import_closure_scanned(tmp_path: Path) -> None:
    """A payload importing an issue9998_common-style sibling by bare module
    name pulls the sibling into scope: the SIBLING's violation line appears
    in the output (per-issue scripts are not leaf code — 725 cross-imports
    at plan time)."""
    tree = _make_tree(tmp_path)
    payload = "scripts/issue9999_payload.py"
    (tree / payload).write_text(
        "from issue9998_common import helper\n\nprint(helper)\n", encoding="utf-8"
    )
    (tree / "scripts" / "issue9998_common.py").write_text(
        f'helper = 1\njudge_model = "{_FORBIDDEN_PIN}"\n', encoding="utf-8"
    )
    r, out = _run_lint(tree, "--files", payload)
    assert r.returncode == 1, out
    assert any("issue9998_common.py" in ln for ln in _error_lines(out)), (
        f"closure sibling's violation not surfaced under --files:\n{out}"
    )


def test_files_mode_unresolvable_issue_import_fails_payload(tmp_path: Path) -> None:
    """A bare issue*-stem import that does NOT resolve to scripts/<name>.py is
    NEW red attributed to the PAYLOAD file (payload-caused by construction:
    the payload references a per-issue module that does not exist)."""
    tree = _make_tree(tmp_path)
    payload = "scripts/issue9999_payload.py"
    (tree / payload).write_text("import issue9997_missing\n", encoding="utf-8")
    r, out = _run_lint(tree, "--files", payload)
    assert r.returncode == 1, out
    hits = [
        ln for ln in _error_lines(out) if "issue9999_payload.py" in ln and "issue9997_missing" in ln
    ]
    assert hits, f"unresolvable issue*-stem import did not FAIL naming the payload:\n{out}"


def test_files_mode_nonissue_stem_imports_never_fail(tmp_path: Path) -> None:
    """NEGATIVE CONTROL (critic round-1 concern (a) — the load-bearing stem
    restriction): 'does not resolve to scripts/<name>.py' is equally true of
    numpy, torch, and every stdlib import, so a literal reading of the
    unresolvable-import rule would FAIL every payload. A payload whose only
    bare imports are non-issue*-stem must PASS with no line blaming it."""
    tree = _make_tree(tmp_path)
    payload = "scripts/issue9996_negctl.py"
    (tree / payload).write_text(
        "import os\n\nimport numpy\n\nprint(numpy.__name__, os.name)\n", encoding="utf-8"
    )
    r, out = _run_lint(tree, "--files", payload)
    assert r.returncode == 0, out
    assert not any("issue9996_negctl.py" in ln for ln in _error_lines(out)), out


# ---------------------------------------------------------------------------
# Case 4 — fail-closed registry completeness (plan section 4 B2.i).
# ---------------------------------------------------------------------------
def test_files_mode_registry_covers_dispatch_chain() -> None:
    """Every ``args.check_* or no_flags`` dispatch site in workflow_lint.main
    has a CHECK_SCOPES entry, every entry's kind is a known value, and no
    entry has an empty surfaces tuple. A future check added without a
    classification turns test-red at its author's own gate."""
    src = _LINT.read_text(encoding="utf-8")
    sites = set(re.findall(r"args\.(check_\w+)\s+or\s+no_flags", src))
    # 68 dispatch sites at plan time; the floor guards against pattern rot,
    # not the exact count (which grows).
    assert len(sites) >= 60, f"dispatch-site scan looks broken: {sorted(sites)}"
    scopes = wl.CHECK_SCOPES
    missing = sites - set(scopes)
    assert not missing, f"dispatch-chain checks missing a CHECK_SCOPES entry: {sorted(missing)}"
    for name, scope in scopes.items():
        assert scope.kind in ("path-local", "global"), (name, scope.kind)
        assert scope.surfaces, f"{name}: empty surfaces tuple"


def test_files_mode_correctness_floor_is_path_local() -> None:
    """The task-body 'Scope decision' hard floor: these correctness-class
    checks are path-local BY REQUIREMENT — they run (scoped) on every
    eligible payload and are never classifiable as skippable-global."""
    floor = (
        "check_judge_model_pins",
        "check_upload_or_true",
        "check_upload_return_discard",
        "check_upload_file_in_loop",
        "check_upload_prefix_clobber",
        "check_upload_as_file",
        "check_dotenv_before_hf_import",
        "check_jsonl_splitlines",
        "check_batch_judge_client",
    )
    for name in floor:
        assert name in wl.CHECK_SCOPES, f"{name} missing from CHECK_SCOPES"
        assert wl.CHECK_SCOPES[name].kind == "path-local", (
            f"{name} must be path-local (user scope decision, task #2235 body)"
        )


# ---------------------------------------------------------------------------
# Case 5 — fail-closed runtime refusal (plan section 4 B2.ii).
# ---------------------------------------------------------------------------
def test_files_mode_unclassified_check_refuses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A registry miss at runtime prints the FILES-MODE-REFUSED sentinel and
    exits 2 with NO terminal PASS/FAIL line — never a silent skip, never a
    swallowed PASS. (The gate detects the sentinel and falls back to the
    bare full run: tests/test_inline_lint_gate.py case 9.)"""
    monkeypatch.chdir(wl._REPO_ROOT)
    monkeypatch.delitem(wl.CHECK_SCOPES, "check_judge_model_pins")
    rc = wl.main(["--files", "scripts/task.py"])
    cap = capsys.readouterr()
    out = cap.out + "\n" + cap.err
    assert rc == 2, out
    assert "FILES-MODE-REFUSED (unclassified check check_judge_model_pins)" in out, out
    assert ilg.LINT_TERMINAL_RE.search(out) is None, (
        f"refusal must not emit a terminal PASS/FAIL line:\n{out}"
    )


# ---------------------------------------------------------------------------
# Case 6 — SCOPE/SKIP output hygiene (plan section 4 B5): counts and check
# names only, never payload path strings.
# ---------------------------------------------------------------------------
def test_files_mode_scope_lines_never_name_payload_paths(tmp_path: Path) -> None:
    """``inline_lint_gate.evaluate`` treats any leg line containing a payload
    path and not prefixed WARN/PASSED/SKIPPED as a red hit — an informational
    SCOPE/SKIP line naming a payload path would self-inflict a false block.
    Pinned against the REAL gate constants so the two files cannot drift."""
    tree = _make_tree(tmp_path)
    payload = "scripts/issue9993_clean.py"
    (tree / payload).write_text(CLEAN_FIG, encoding="utf-8")
    r, out = _run_lint(tree, "--files", payload)
    assert r.returncode == 0, out
    # The completeness line (exact shape) prints BEFORE the terminal line.
    m = _SCOPE_LINE_RE.search(out)
    assert m is not None, f"missing/misshapen SCOPE completeness line:\n{out}"
    assert m.start() < out.rindex("workflow_lint: PASS"), out
    # Global checks were skipped for a scripts-only payload; some checks ran.
    counts = re.search(r"checks_ran=(\d+) checks_skipped=(\d+)", out)
    assert counts is not None, out
    assert int(counts.group(1)) >= 1, out
    assert int(counts.group(2)) >= 1, out
    # No SCOPE/SKIP line names the payload.
    for ln in out.splitlines():
        if ln.startswith(("workflow_lint: SCOPE", "workflow_lint: SKIP")):
            assert payload not in ln and "issue9993_clean" not in ln, ln
    # The gate-side red-hit rule, applied verbatim: every payload-naming line
    # in the combined output must carry a non-red prefix.
    for ln in out.splitlines():
        if payload in ln:
            assert ln.strip().startswith(ilg.NON_RED_PREFIXES), (
                f"payload-naming line would read as a red hit to the gate: {ln!r}"
            )


# ---------------------------------------------------------------------------
# Case 7 — no --files => byte-identical no-flags behavior (Step 9c unaffected).
# ---------------------------------------------------------------------------
def test_no_files_flag_is_unscoped_no_scope_line(tmp_path: Path) -> None:
    """The bare no-flags run scans the WHOLE tree (both planted red fixtures
    named — no payload notion, no scoping) and emits NO SCOPE/SKIP files-mode
    lines. This is what protects every other session's Step 9c gate: the
    no-flags instrument is unchanged."""
    tree = _make_tree(tmp_path)
    (tree / "scripts" / "issue9994_red_a.py").write_text(JUDGE_PIN_RED, encoding="utf-8")
    (tree / "scripts" / "issue9995_red_b.py").write_text(JSONL_RED, encoding="utf-8")
    r, out = _run_lint(tree)
    assert r.returncode == 1, out
    assert any("issue9994_red_a.py" in ln for ln in _error_lines(out)), out
    assert any("issue9995_red_b.py" in ln for ln in _error_lines(out)), out
    assert "workflow_lint: SCOPE" not in out, out
    assert "files-mode" not in out, out
    assert _FAIL_COUNT_RE.search(out) is not None, out


def test_files_mode_rejects_check_flag_combination(tmp_path: Path) -> None:
    """--files combined with any --check-* flag is an argparse error (one
    mode at a time, plan section 4 B1): exit 2, no terminal PASS/FAIL line."""
    tree = _make_tree(tmp_path)
    payload = "scripts/issue9993_clean.py"
    (tree / payload).write_text(CLEAN_FIG, encoding="utf-8")
    r, out = _run_lint(tree, "--files", payload, "--check-judge-model-pins")
    assert r.returncode == 2, out
    assert ilg.LINT_TERMINAL_RE.search(out) is None, out


# ---------------------------------------------------------------------------
# Code-review round 1 findings: cross-file wrapper completeness + absolute-path
# payload normalization.
# ---------------------------------------------------------------------------
SHARED_UPLOADER = '''"""Out-of-scope shared uploader (non-issue stem, so the
issue*-restricted import closure never pulls it into scope)."""

from huggingface_hub import HfApi


def push_artifacts(repo_id: str, dest: str) -> None:
    """Wrapper whose `dest` param feeds upload_folder's path_in_repo."""
    HfApi().upload_folder(repo_id=repo_id, path_in_repo=dest, folder_path=".")
'''

WRAPPER_CALLER_RED = '''"""Payload calling the out-of-scope wrapper with ANOTHER issue's prefix."""

import shared_uploader


def main() -> None:
    shared_uploader.push_artifacts(repo_id="superkaiba1/d", dest="issue9991_other/tensors")
'''


def test_files_mode_cross_file_wrapper_is_a_known_residual(tmp_path: Path) -> None:
    """PINS A DELIBERATE RESIDUAL, not desired behavior (code-review round 1).

    Pass 1 (wrapper inference) keys wrappers by BARE NAME over the files it is
    given, and files-mode gives it the FILTERED set — so a payload calling an
    upload wrapper whose def lives in an out-of-scope module (non-`issue*`-stem
    bare import, which the issue*-restricted closure never pulls in) is not
    flagged by the SCOPED run. The companion test below pins that the bare run
    — the actual Step 9c merge gate — does flag it, making this delayed
    detection rather than an escape to main.

    Do NOT "fix" this by widening pass 1's input without re-measuring: measured
    2026-08-11 on a 2-file fig payload, the whole walked set costs +15.1 s, an
    UPLOAD_DEST_FUNCS-mention gate +6.2 s, and a "defines a name the payload
    calls" gate +32.8 s (generic names like `main` defeat its selectivity) —
    and the legs run sequentially, so against the recorded 52.5 s median (i) and
    (iii) break the <60 s bar outright while (ii) lands ~58.7 s — under it, but
    on ~1.3 s of headroom against a median that swings 35.9 -> 52.5 s on VM load
    alone. The real fix is to key pass 1 on import provenance instead of bare
    name, in the #1452 check.
    """
    tree = _make_tree(tmp_path)
    (tree / "scripts" / "shared_uploader.py").write_text(SHARED_UPLOADER, encoding="utf-8")
    payload = "scripts/issue9990_uses_wrapper.py"
    (tree / payload).write_text(WRAPPER_CALLER_RED, encoding="utf-8")
    r, out = _run_lint(tree, "--files", payload)
    assert r.returncode == 0, out
    assert not any("issue9990_uses_wrapper.py" in ln for ln in _error_lines(out)), out


def test_bare_run_catches_cross_file_wrapper_the_scoped_run_misses(tmp_path: Path) -> None:
    """The backstop for the residual above is REAL, not assumed: the bare
    no-flags run (the Step 9c gate instrument) flags the same fixture the scoped
    run lets through. If this ever goes green-and-silent, the residual stops
    being delayed detection and becomes an escape to main."""
    tree = _make_tree(tmp_path)
    (tree / "scripts" / "shared_uploader.py").write_text(SHARED_UPLOADER, encoding="utf-8")
    (tree / "scripts" / "issue9990_uses_wrapper.py").write_text(
        WRAPPER_CALLER_RED, encoding="utf-8"
    )
    r, out = _run_lint(tree)
    assert r.returncode == 1, out
    assert any("issue9990_uses_wrapper.py" in ln for ln in _error_lines(out)), out
    assert any("9991" in ln for ln in _error_lines(out)), out


def test_files_mode_refuses_out_of_repo_absolute_path(tmp_path: Path) -> None:
    """An absolute payload path OUTSIDE the repo can never match a
    repo-relative enumeration entry, so scoping it would certify a vacuous
    near-empty PASS. Refuse it loudly instead: exit 2, no terminal line, so the
    gate falls back to one bare full run (code-review round 1)."""
    tree = _make_tree(tmp_path)
    outsider = tmp_path / "outside.py"
    outsider.write_text(CLEAN_FIG, encoding="utf-8")
    r, out = _run_lint(tree, "--files", str(outsider))
    assert r.returncode == 2, out
    assert "FILES-MODE-REFUSED (payload path outside repo:" in out, out
    assert ilg.LINT_TERMINAL_RE.search(out) is None, out


def test_files_mode_normalizes_in_repo_dotdot_path(tmp_path: Path) -> None:
    """`Path.relative_to` is lexical and leaves `..` in place, so a dotdot form
    produced a key no repo-relative enumeration entry could equal — silently
    scoping the payload out of its own run (review round 2). It now normalizes,
    so the payload's own red is still reported."""
    tree = _make_tree(tmp_path)
    rel = "scripts/issue9997_red_dotdot.py"
    (tree / rel).write_text(JUDGE_PIN_RED, encoding="utf-8")
    r, out = _run_lint(tree, "--files", f"scripts/../{rel}")
    assert r.returncode == 1, out
    assert any("issue9997_red_dotdot.py" in ln for ln in _error_lines(out)), out


def test_files_mode_refuses_dotdot_path_escaping_repo(tmp_path: Path) -> None:
    """A dotdot form that normalizes to OUTSIDE the repo is refused, not
    silently scoped to nothing (review round 2)."""
    tree = _make_tree(tmp_path)
    r, out = _run_lint(tree, "--files", "scripts/../../../../etc/passwd")
    assert r.returncode == 2, out
    assert "FILES-MODE-REFUSED (payload path escapes repo:" in out, out
    assert ilg.LINT_TERMINAL_RE.search(out) is None, out


def test_files_mode_refuses_empty_payload(tmp_path: Path) -> None:
    """An all-whitespace --files list scopes to nothing, which would certify a
    vacuous PASS over an empty payload (review round 2)."""
    tree = _make_tree(tmp_path)
    r, out = _run_lint(tree, "--files", "   ")
    assert r.returncode == 2, out
    assert "FILES-MODE-REFUSED (empty payload)" in out, out
    assert ilg.LINT_TERMINAL_RE.search(out) is None, out


def test_files_mode_normalizes_in_repo_absolute_path(tmp_path: Path) -> None:
    """An absolute IN-repo payload path relativizes into its own scope. Before
    the fix it stayed absolute, so the payload was scoped OUT of its own run and
    its red went unreported as a near-empty PASS (code-review round 1)."""
    tree = _make_tree(tmp_path)
    rel = "scripts/issue9992_red_abs.py"
    (tree / rel).write_text(JUDGE_PIN_RED, encoding="utf-8")
    r, out = _run_lint(tree, "--files", str(tree / rel))
    assert r.returncode == 1, out
    assert any("issue9992_red_abs.py" in ln for ln in _error_lines(out)), out
