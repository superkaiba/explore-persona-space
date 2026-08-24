"""Tests for the #2158 pre-split review-guard surface pin in
``scripts/workflow_lint.py``.

One check under test: ``check_pre_split_review_guard``
(``--check-pre-split-review-guard``, bundled into the no-flags default run) —
the pre-split review guard (#2158; incident #1336 r4: the Step 5 ensemble
was dispatched against a Unit-A-only intermediate commit of a pre-split
multi-unit round, costing 2 subagent deaths + a ~2-day park) must stay
present across its SEVEN surfaces spanning EIGHT files:

(1) ``scripts/pre_split_review_guard.py`` naming ``pre_split_review_gate``
    AND the #2294 ``IMPLEMENTER-MARKER-MISSING`` exit-4 verdict;
(2) ``task_workflow.py``: ``def pre_split_review_gate`` +
    ``PRE-SPLIT-INCOMPLETE``;
(3) 09-step-5.md: the ``**Pre-split completeness guard`` region (incl. the
    #2294 ``IMPLEMENTER-MARKER-MISSING`` token);
(4) 08-step-4.md: breadcrumb-grammar tokens + the ``unit=<k>`` emitter
    mandate;
(5) 08-step-4.md: the shared-worktree arbitration note;
(6) ``.claude/rules/cross-session-writer-arbitration.md``;
(7) the ``Read-pinning under external churn`` bullet in BOTH implementer
    specs — experiment-implementer.md AND implementer.md, each reported
    independently (a lint implementation checking only ONE of the two files
    fails the other file's drop param).

Tests (mirroring ``tests/test_workflow_lint_smoke_blind_spots.py``):

1. ``test_passes_on_complete_corpus`` — tmp corpus with all 7 surfaces /
   8 files.
2. ``test_fails_per_missing_surface`` — 10 parametrized drops: one per FILE
   (surfaces 1-6 one file each; surface 7 TWO independent per-file drops)
   plus the two #2294 ``IMPLEMENTER-MARKER-MISSING`` token drops (CLI +
   step-5 region).
3. ``test_review_lens_passes_on_live_tree`` — binds the landed #2158 edits.
4. ``test_bundled_in_no_flags`` — the two-part behavioral bundling pin
   (scoped-flag subprocess against a drifted corpus via
   ``EPS_WORKFLOW_LINT_REPO_ROOT``, plus OR-chain + dispatch-ladder source
   evidence).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import check_pre_split_review_guard  # noqa: E402

_VERDICT = "PRE-SPLIT-INCOMPLETE"


def _write_guard_corpus(root: Path, *, drop: str | None = None) -> Path:
    """Build a minimal seven-surface / eight-file corpus under ``root``;
    ``drop`` removes exactly one surface/token to exercise each per-surface
    (and, for surface 7, per-FILE) error."""
    scripts = root / "scripts"
    src = root / "src" / "explore_persona_space"
    steps = root / ".claude" / "skills" / "issue" / "steps"
    rules = root / ".claude" / "rules"
    agents = root / ".claude" / "agents"
    for d in (scripts, src, steps, rules, agents):
        d.mkdir(parents=True, exist_ok=True)

    # (1) the thin CLI naming the library entry + the #2294 exit-4 verdict.
    impl_verdict_line = '_EXIT_FOR_VERDICT_4 = "IMPLEMENTER-MARKER-MISSING"  # exit 4 (#2294)\n'
    cli_body = (
        '"""Thin CLI for the #2158 pre-split review gate."""\n\n'
        "from explore_persona_space.task_workflow import pre_split_review_gate\n"
        + impl_verdict_line
    )
    if drop == "cli-gate-token":
        cli_body = '"""Thin CLI placeholder (library entry stripped)."""\n' + impl_verdict_line
    elif drop == "cli-impl-marker-token":
        cli_body = (
            '"""Thin CLI for the #2158 pre-split review gate."""\n\n'
            "from explore_persona_space.task_workflow import pre_split_review_gate\n"
        )
    (scripts / "pre_split_review_guard.py").write_text(cli_body, encoding="utf-8")

    # (2) the library predicate + verdict token.
    fn_name = "pre_split_gate_renamed" if drop == "predicate-def" else "pre_split_review_gate"
    (src / "task_workflow.py").write_text(
        f"def {fn_name}(events):\n"
        f'    """Two-arm pre-split predicate (#2158)."""\n'
        f'    return ("REVIEW-OK", None)  # nonzero verdict: {_VERDICT}\n',
        encoding="utf-8",
    )

    # (3) 09-step-5.md: the guard block region (bounded by the next
    # paragraph opening ``**``).
    step5_verdict = "INCOMPLETE-VERDICT" if drop == "step5-region-token" else _VERDICT
    step5_impl_missing = (
        ""
        if drop == "step5-impl-marker-token"
        else " `IMPLEMENTER-MARKER-MISSING` (exit 4) means no "
        "implementation-class marker exists on canonical events — post the "
        "implementer marker FIRST, then re-run the guard (#2294)."
    )
    (steps / "09-step-5.md").write_text(
        "# Step 5\n\n"
        "Only if status is `running` and the implementation marker is "
        "present.\n\n"
        "**Pre-split completeness guard (#2158).** Before ANY reviewer "
        "dispatch run `uv run python scripts/pre_split_review_guard.py "
        f"<N>`. `{step5_verdict}` (exit 2) means a breadcrumb with a "
        "non-empty `remaining:` list has no later implementation marker — "
        f"do NOT dispatch either reviewer.{step5_impl_missing}\n\n"
        "**Per-commit split-review dispatch.** Placeholder paragraph.\n",
        encoding="utf-8",
    )

    # (4)+(5) 08-step-4.md: grammar + emitter tokens, shared-worktree note.
    emitter = (
        ""
        if drop == "step4-unit-token"
        else " Emitter convention: each unit's stage-dispatch note carries `unit=<k>`."
    )
    arbitration = (
        "the sequencing rule"
        if drop == "step4-arbitration-note"
        else "`.claude/rules/cross-session-writer-arbitration.md`"
    )
    (steps / "08-step-4.md").write_text(
        "# Step 4\n\n"
        "Pre-split breadcrumb grammar: `pre-split unit k/M complete: "
        "<SHAs>; remaining: <deliverables>`.\n\n"
        "Shared-worktree note (#2158): a shared worktree is the EXPECTED "
        f"shape for a multi-unit split; see {arbitration}.{emitter}\n",
        encoding="utf-8",
    )

    # (6) the arbitration rule file.
    if drop != "rule-file":
        (rules / "cross-session-writer-arbitration.md").write_text(
            "# Cross-session writer arbitration\n\n"
            "Overlapping live claim or live-writer probe hit means never "
            "dispatch a concurrent writer. Read-pinning: pin reads to "
            "`git show <BASE_SHA>:<path>`.\n",
            encoding="utf-8",
        )

    # (7) both implementer specs: the read-pinning bullet, per FILE.
    read_pin_bullet = (
        "- **Read-pinning under external churn (#2158; #1336 death #9).** "
        "Record BASE_SHA at round start; pin reads to "
        "`git show <BASE_SHA>:<path>`, one bounded provenance probe, "
        "reconcile at commit time.\n"
    )
    for agent_name, drop_key in (
        ("experiment-implementer.md", "expimpl-readpin"),
        ("implementer.md", "impl-readpin"),
    ):
        bullet = "" if drop == drop_key else read_pin_bullet
        (agents / agent_name).write_text(
            f"# {agent_name.removesuffix('.md')}\n\n"
            "- **Work only inside the worktree.** Placeholder.\n"
            + bullet
            + "- **No silent failures.** Placeholder.\n",
            encoding="utf-8",
        )
    return root


def test_passes_on_complete_corpus(tmp_path: Path) -> None:
    _write_guard_corpus(tmp_path)
    errors = check_pre_split_review_guard(repo_root=tmp_path)
    assert errors == [], f"complete corpus should pass; got: {errors}"


_DROP_CASES: list[tuple[str, str, str]] = [
    ("cli-gate-token", "pre_split_review_gate", "scripts/pre_split_review_guard.py"),
    ("cli-impl-marker-token", "IMPLEMENTER-MARKER-MISSING", "scripts/pre_split_review_guard.py"),
    ("predicate-def", "def pre_split_review_gate", "task_workflow.py"),
    ("step5-region-token", _VERDICT, "09-step-5.md"),
    ("step5-impl-marker-token", "IMPLEMENTER-MARKER-MISSING", "09-step-5.md"),
    ("step4-unit-token", "unit=<k>", "08-step-4.md"),
    ("step4-arbitration-note", "cross-session-writer-arbitration.md", "08-step-4.md"),
    ("rule-file", "missing", "rules/cross-session-writer-arbitration.md"),
    ("expimpl-readpin", "Read-pinning under external churn", "agents/experiment-implementer.md"),
    ("impl-readpin", "Read-pinning under external churn", "agents/implementer.md"),
]


@pytest.mark.parametrize(("drop", "token", "path_frag"), _DROP_CASES)
def test_fails_per_missing_surface(tmp_path: Path, drop: str, token: str, path_frag: str) -> None:
    _write_guard_corpus(tmp_path, drop=drop)
    errors = check_pre_split_review_guard(repo_root=tmp_path)
    assert errors, f"drop={drop}: expected >=1 error"
    assert any(token in e and path_frag in e for e in errors), (
        f"drop={drop}: no error carries both {token!r} and {path_frag!r}; got: {errors}"
    )


def test_review_lens_passes_on_live_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binds the landed #2158 edits; the standing regression guard for
    future refactors of any of the seven surfaces / eight files."""
    monkeypatch.delenv("EPS_WORKFLOW_LINT_REPO_ROOT", raising=False)
    errors = check_pre_split_review_guard(repo_root=None)
    assert errors == [], f"live tree should carry all seven surfaces; got: {errors}"


def test_bundled_in_no_flags(tmp_path: Path) -> None:
    """Two-part behavioral bundling pin (the #1701/#2165 precedent shape).

    Part A — scoped-flag subprocess against a DRIFTED corpus (arbitration
    rule file dropped), rooted via ``EPS_WORKFLOW_LINT_REPO_ROOT``: proves
    the flag exists, the dispatch calls the function, and it emits its
    uniquely-named error (nonzero exit).

    Part B — no-flags OR-chain + dispatch-ladder evidence: ``main()``'s
    source names ``args.check_pre_split_review_guard`` in BOTH the
    ``no_flags = not (...)`` OR-chain and the ``or no_flags`` dispatch
    ladder.
    """
    # Part A — scoped-flag subprocess against a drifted corpus.
    _write_guard_corpus(tmp_path, drop="rule-file")
    workflow_yaml_src = _REPO_ROOT / ".claude" / "workflow.yaml"
    workflow_yaml_dst = tmp_path / ".claude" / "workflow.yaml"
    workflow_yaml_dst.parent.mkdir(parents=True, exist_ok=True)
    workflow_yaml_dst.write_bytes(workflow_yaml_src.read_bytes())
    lint_script = _REPO_ROOT / "scripts" / "workflow_lint.py"
    env = {**os.environ, "EPS_WORKFLOW_LINT_REPO_ROOT": str(tmp_path)}
    result = subprocess.run(
        [
            sys.executable,
            str(lint_script),
            "--check-pre-split-review-guard",
            "--file",
            str(workflow_yaml_dst),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    combined = result.stdout + result.stderr
    assert "cross-session-writer-arbitration" in combined, (
        "cross-session-writer-arbitration error token missing from output — "
        "the CLI flag does not dispatch the check. "
        f"exit={result.returncode}, combined output:\n{combined}"
    )
    assert result.returncode != 0, (
        f"expected nonzero exit under drifted corpus; got exit="
        f"{result.returncode}, combined output:\n{combined}"
    )

    # Part B — OR-chain + dispatch ladder evidence.
    lint_src = lint_script.read_text(encoding="utf-8")
    main_start = lint_src.find("def main(")
    assert main_start >= 0, "could not locate def main( in workflow_lint.py"
    main_end = lint_src.find('if __name__ == "__main__":', main_start)
    assert main_end > main_start, "could not locate main() end sentinel"
    main_src = lint_src[main_start:main_end]
    or_chain_start = main_src.find("no_flags = not (")
    assert or_chain_start >= 0, "no_flags OR-chain not found in main()"
    or_chain_end = main_src.find(")", or_chain_start)
    or_chain_src = main_src[or_chain_start:or_chain_end]
    assert "args.check_pre_split_review_guard" in or_chain_src, (
        "args.check_pre_split_review_guard is NOT in the no_flags OR-chain "
        "— a bare workflow_lint.py invocation will not fire this check. "
        f"OR-chain source:\n{or_chain_src}"
    )
    assert "args.check_pre_split_review_guard or no_flags" in main_src, (
        "args.check_pre_split_review_guard is NOT dispatched under "
        "`or no_flags` — the flag is defined but not bundled into the "
        "no-flags default run."
    )
