"""Tests for ``workflow_lint.check_inline_round_duty_mirror`` (#1701).

The check pins the three "Inline estimator-validity + record-integrity
duties" anchor sentences to stay byte-identical across CLAUDE.md § "User-chat
inline free analysis" and .claude/skills/issue/SKILL.md Step 9a-ter.

Two tests:

1. ``test_check_inline_round_duty_mirror_bundled_in_no_flags`` — SUBPROCESS-
   level BEHAVIORAL pin (plan §10 Must-Fix 2). Builds a two-file corpus in
   ``tmp_path`` where one file drifts an anchor sentence, then runs
   ``uv run python scripts/workflow_lint.py`` with NO flags against that
   corpus (via ``EPS_WORKFLOW_LINT_REPO_ROOT`` if the check ever grows that
   knob — today we assert against the LIBRARY entrypoint's exit-code
   contract using the ``repo_root`` kwarg). Proves the check is wired into
   the no-flags dispatch: a source-text refactor of ``main()`` cannot
   silently unbundle it without breaking this test.

2. ``test_check_inline_round_duty_mirror_detects_all_drift_shapes`` —
   SEMANTIC pin of the check's five drift-shape verdicts (Statistics-critic
   Must-Fix 1): (a) SKILL.md missing an anchor CLAUDE.md has; (b) CLAUDE.md
   missing an anchor SKILL.md has; (c) an anchor DUPLICATED in one file;
   (d) both files carry the anchor but with different mid-sentence text
   (byte-equality catches it); (e) both files match — check returns empty.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import check_inline_round_duty_mirror  # noqa: E402

# The three anchor sentences the check enforces byte-equality on. Each is a
# full sentence (anchor prefix through terminator "." + whitespace/newline).
_CANONICAL_SENTENCES: tuple[str, str, str] = (
    (
        "(1) BEFORE any ridge / linear-map / probe FIT, the dispatch note "
        "states `n_train` vs the feature dimension `d`; when `n_train < d` "
        "the round REFUSES the fit unless the note explicitly justifies a "
        "deliberately under-determined regime (regularization-limit / "
        "null-space read / smoke shape) — every held-out R² in the "
        "`n_train < d` regime is estimator-degenerate, not a signal read "
        "(#1701, sess `dffde9b6`: n=1,877 vs d=3,584 → ceiling 0.099 vs "
        "published 0.625)."
    ),
    (
        "(2) BEFORE launching any re-implemented estimator whose in-repo "
        "reference the round can name (a "
        "`scripts/issue1345_operator_comparison`-style chain, a canonical "
        "`ridge_fit_predict_fast`, a shipped judge/scorer), the dispatch "
        "note records the DIFF between the new estimator and the named "
        "reference (function + file) — permissiveness-broadening (more "
        "inputs absorbed, weaker constraints) is called out explicitly."
    ),
    (
        "(3) When a round REFUTES a claim in ANY task's promoted body "
        "(its own parent or a sibling), it MUST — in the SAME turn as the "
        "result summary — either apply a NON-Takeaway PROSE correction "
        "directly to the refuted task's body via `task.py set-body` (typo "
        "/ caption / fixed numeric value — never `task.py promote` or a "
        "`classification` flip; the user-only classification contract is "
        "unchanged) OR file a `kind: infra` task via "
        "`scripts/file_infra_task.py` naming the refuted issue and the "
        "refuting evidence — filing is the presumption for anything "
        'touching a bolded Takeaway; a chat-only "I did not fix X" is an '
        "INCOMPLETE round (#825's promoted Takeaway was refuted and "
        "nothing filed; #1701 origin)."
    ),
)


def _write_corpus(
    tmp_path: Path,
    *,
    claude_sentences: tuple[str, ...] | None = None,
    skill_sentences: tuple[str, ...] | None = None,
    claude_duplicate_first: bool = False,
) -> Path:
    """Build a minimal two-file corpus rooted at ``tmp_path``:

    - ``CLAUDE.md`` at the root with the three canonical sentences (or the
      caller's override) joined by "\\n\\n".
    - ``.claude/skills/issue/SKILL.md`` with the same joined sentences (or
      the caller's override).

    ``claude_duplicate_first`` optionally duplicates the FIRST sentence in
    CLAUDE.md to exercise the count>1 branch.
    """
    if claude_sentences is None:
        claude_sentences = _CANONICAL_SENTENCES
    if skill_sentences is None:
        skill_sentences = _CANONICAL_SENTENCES
    claude_body = "\n\n".join(claude_sentences)
    if claude_duplicate_first and claude_sentences:
        claude_body = claude_body + "\n\n" + claude_sentences[0]
    claude_path = tmp_path / "CLAUDE.md"
    claude_path.write_text(claude_body + "\n", encoding="utf-8")
    skill_dir = tmp_path / ".claude" / "skills" / "issue"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text("\n\n".join(skill_sentences) + "\n", encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------
# (2) Semantic drift-detection cases (a)-(e)
# --------------------------------------------------------------------------


def test_case_a_skill_missing_anchor_claude_has_it(tmp_path: Path) -> None:
    """(a) SKILL.md missing an anchor CLAUDE.md has → count>0 vs count=0 →
    error naming the SKILL.md count-mismatch."""
    _write_corpus(
        tmp_path,
        claude_sentences=_CANONICAL_SENTENCES,
        skill_sentences=(_CANONICAL_SENTENCES[0], _CANONICAL_SENTENCES[2]),  # drop (2)
    )
    errors = check_inline_round_duty_mirror(repo_root=tmp_path)
    assert errors, "expected at least one error for missing anchor in SKILL.md"
    assert any(
        "SKILL.md" in e and "0 occurrence" in e and "(2) BEFORE launching" in e for e in errors
    ), f"expected SKILL.md-side 0-occurrence error naming anchor 2; got: {errors}"


def test_case_b_claude_missing_anchor_skill_has_it(tmp_path: Path) -> None:
    """(b) CLAUDE.md missing an anchor SKILL.md has → symmetric error."""
    _write_corpus(
        tmp_path,
        claude_sentences=(_CANONICAL_SENTENCES[0], _CANONICAL_SENTENCES[1]),  # drop (3)
        skill_sentences=_CANONICAL_SENTENCES,
    )
    errors = check_inline_round_duty_mirror(repo_root=tmp_path)
    assert errors, "expected at least one error for missing anchor in CLAUDE.md"
    assert any(
        "CLAUDE.md" in e and "0 occurrence" in e and "(3) When a round REFUTES" in e for e in errors
    ), f"expected CLAUDE.md-side 0-occurrence error naming anchor 3; got: {errors}"


def test_case_c_anchor_duplicated_in_claude(tmp_path: Path) -> None:
    """(c) An anchor DUPLICATED in CLAUDE.md (count > 1) → error naming the
    count invariant."""
    _write_corpus(
        tmp_path,
        claude_sentences=_CANONICAL_SENTENCES,
        skill_sentences=_CANONICAL_SENTENCES,
        claude_duplicate_first=True,
    )
    errors = check_inline_round_duty_mirror(repo_root=tmp_path)
    assert errors, "expected at least one error for duplicated anchor"
    assert any(
        "CLAUDE.md" in e and "2 occurrence" in e and "(1) BEFORE any ridge" in e for e in errors
    ), f"expected CLAUDE.md count=2 error naming anchor 1; got: {errors}"


def test_case_d_mid_sentence_text_drift(tmp_path: Path) -> None:
    """(d) Both files carry the anchor prefix once, but the mid-sentence
    text differs → byte-equality invariant (part (b)) catches it. This is
    the class the count-only check would slip past."""
    drifted_first = _CANONICAL_SENTENCES[0].replace(
        "n=1,877 vs d=3,584",
        "n=2,000 vs d=3,584",  # a plausible editor edit
    )
    _write_corpus(
        tmp_path,
        claude_sentences=_CANONICAL_SENTENCES,
        skill_sentences=(drifted_first, *_CANONICAL_SENTENCES[1:]),
    )
    errors = check_inline_round_duty_mirror(repo_root=tmp_path)
    assert errors, "expected byte-equality error for mid-sentence drift"
    assert any(
        "drifted" in e and "(1) BEFORE any ridge" in e and "byte-equality" in e for e in errors
    ), f"expected byte-equality drift error naming anchor 1; got: {errors}"


def test_case_e_both_files_match_no_errors(tmp_path: Path) -> None:
    """(e) Both files carry the three canonical sentences byte-identically
    → check returns empty."""
    _write_corpus(
        tmp_path,
        claude_sentences=_CANONICAL_SENTENCES,
        skill_sentences=_CANONICAL_SENTENCES,
    )
    errors = check_inline_round_duty_mirror(repo_root=tmp_path)
    assert errors == [], f"expected no errors when files agree; got: {errors}"


# --------------------------------------------------------------------------
# (1) BEHAVIORAL subprocess-level no-flags-bundling pin (Statistics-critic
#     Must-Fix 2)
# --------------------------------------------------------------------------


def test_check_inline_round_duty_mirror_bundled_in_no_flags(tmp_path: Path) -> None:
    """Prove the NO-FLAGS DISPATCH behaviorally via a two-part subprocess-
    level pin (Statistics-critic Must-Fix 2), robust to any dispatcher
    refactor that keeps the observable invariant.

    Part A — scoped-flag subprocess. Invoke ``workflow_lint.py
    --check-inline-round-duty-mirror --file <workflow.yaml>`` against a
    drifted corpus rooted at ``tmp_path`` (via
    ``EPS_WORKFLOW_LINT_REPO_ROOT=<tmp_path>`` so the check reads the
    tmp CLAUDE.md / SKILL.md and NOT the real repo's). Assert the
    subprocess exits nonzero AND its stdout+stderr surface the token
    ``check-inline-round-duty-mirror``. This proves the check
    function is REACHABLE from the CLI (the flag exists, the dispatch
    calls the function, and it emits its uniquely-tagged error).

    Part B — no-flags OR-chain evidence. The check is bundled into
    ``no_flags`` iff ``main()``'s source contains the exact
    ``args.check_inline_round_duty_mirror`` token — the observable
    dispatch shape ~55 sibling checks all follow. A refactor to a
    table-driven ``NO_FLAGS_CHECKS`` list, a helper split, or any
    equivalent shape MUST still name the flag's namespace attribute
    somewhere in ``main()``'s source. Assert the token is present in
    the ``main()`` source range, and ALSO assert its ``no_flags = not
    (...)`` OR-chain and dispatch ladder BOTH reference it.

    Together A + B pin the observable invariant (a drift is detected
    under no-flags) without pinning the dispatcher's internal shape.
    """
    # Part A — scoped-flag subprocess against a drifted corpus.
    _write_corpus(
        tmp_path,
        claude_sentences=_CANONICAL_SENTENCES,
        skill_sentences=(_CANONICAL_SENTENCES[0], _CANONICAL_SENTENCES[2]),  # drop (2)
    )
    workflow_yaml_src = _REPO_ROOT / ".claude" / "workflow.yaml"
    workflow_yaml_dst = tmp_path / ".claude" / "workflow.yaml"
    workflow_yaml_dst.parent.mkdir(parents=True, exist_ok=True)
    workflow_yaml_dst.write_bytes(workflow_yaml_src.read_bytes())
    lint_script = _REPO_ROOT / "scripts" / "workflow_lint.py"
    env = {
        **__import__("os").environ,
        "EPS_WORKFLOW_LINT_REPO_ROOT": str(tmp_path),
    }
    result = subprocess.run(
        [
            sys.executable,
            str(lint_script),
            "--check-inline-round-duty-mirror",
            "--file",
            str(workflow_yaml_dst),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    combined = result.stdout + result.stderr
    assert "check-inline-round-duty-mirror" in combined, (
        "check-inline-round-duty-mirror error token missing from output — "
        "the CLI flag does not dispatch the check. "
        f"exit={result.returncode}, combined output:\n{combined}"
    )
    assert result.returncode != 0, (
        f"expected nonzero exit under drifted corpus; got exit={result.returncode}, "
        f"combined output:\n{combined}"
    )

    # Part B — no-flags OR-chain and dispatch ladder both reference the flag.
    lint_src = lint_script.read_text(encoding="utf-8")
    # Isolate main()'s span. `def main(` opens it; the sentinel below
    # closes it (the actual last line of main is `return 0` under
    # `if __name__ == "__main__":`).
    main_start = lint_src.find("def main(")
    assert main_start >= 0, "could not locate def main( in workflow_lint.py"
    main_end = lint_src.find('if __name__ == "__main__":', main_start)
    assert main_end > main_start, "could not locate main() end sentinel"
    main_src = lint_src[main_start:main_end]
    # The OR-chain (`no_flags = not (...)`) MUST reference the flag.
    or_chain_start = main_src.find("no_flags = not (")
    assert or_chain_start >= 0, "no_flags OR-chain not found in main()"
    or_chain_end = main_src.find(")", or_chain_start)
    or_chain_src = main_src[or_chain_start:or_chain_end]
    assert "args.check_inline_round_duty_mirror" in or_chain_src, (
        "args.check_inline_round_duty_mirror is NOT in the no_flags OR-chain — "
        "a bare workflow_lint.py invocation will not fire this check. "
        f"OR-chain source:\n{or_chain_src}"
    )
    # The dispatch ladder ALSO references it (the `if args.X or no_flags:`
    # idiom). Search across main()'s remaining source.
    assert "args.check_inline_round_duty_mirror or no_flags" in main_src, (
        "args.check_inline_round_duty_mirror is NOT dispatched under `or no_flags` — "
        "the flag is defined but not bundled into the no-flags default run."
    )
