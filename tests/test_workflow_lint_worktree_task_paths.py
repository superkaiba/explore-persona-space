"""Tests for the #2422 worktree-safe task-state brief check in
``scripts/workflow_lint.py``.

Check under test: ``check_worktree_task_state_briefs``
(``--check-worktree-task-state-briefs``, bundled into the no-flags default
run) — the region-anchored six-surface token ladder pinning the contract
that subagent briefs hand ABSOLUTE canonical main-checkout task-state paths
(``$(uv run python "$REPO_ROOT"/scripts/task.py find <N>)/plans/plan.md``)
plus a compose-time ``plan_version=v<K>`` pin, and that subagents never
read ``tasks/`` from inside a worktree (frozen at its base commit —
incidents #2329/#823: stale plan/manifest served with NO error).

Tests (plan #2422 v3 §4.3 Edit 11 enumeration; test 6 of the enumeration —
the Edit-7 wait-mechanism pin — lives in
``tests/test_teammate_coordination_pins.py``):

1. ``test_prefix_corpus_fails`` — the criterion-4 red-before-fix evidence:
   a corpus carrying the PRE-fix verbatim region text of each of the six
   surfaces (extracted from base commit ``5ee4f650e5``, the tree BEFORE
   unit 1's prose edits) makes the check emit >=1 error per surface. If
   this corpus ever reads green, the ladder is hollow (plan §6
   kill-criterion 3).
2. ``test_passes_on_complete_corpus`` + ``test_fails_per_missing_surface``
   — fixed-shape corpus, one surface/anchor/token dropped per case (the
   smoke-blind-spots ``_DROP_CASES`` pattern). The v4 ``s1-events`` case
   RETAINS the bare ``events.jsonl`` instruction while OMITTING the
   canonical-fetch token ``task.py view`` (round-1 concern
   ``worktree-events-path-gap`` — a deletion-only fixture would not
   discriminate that defect).
3. ``test_worktree_task_state_briefs_passes_on_live_tree`` — binds the
   landed unit-1 edits; the standing Durability pin.
4. ``test_check_worktree_task_state_briefs_bundled_in_no_flags`` — the
   two-part behavioral bundling pin (scoped-flag subprocess against a
   drifted corpus + main() OR-chain/dispatch-ladder source evidence).
5. ``test_worktree_freeze_reproduction`` — the criterion-1 reproduction of
   the #2329/#823 freeze mechanism in a throwaway git repo: a worktree cut
   BEFORE a plan revision serves the stale ``plan.md -> v1.md`` symlink and
   stale manifest bytes while the canonical main tree serves v2; the
   read-time mismatch predicate (extensionless ``readlink`` comparison /
   ``diff -q``) fires; and the #550 ABSENT shape (worktree cut before the
   task folder existed) fails ``test -f`` loud.
6. ``test_surface6_end_anchor_ordered_fallback`` — surface 6's ordered end
   anchors (gate-scope duty bullet, then ``Move status to ``): missing only
   the preferred anchor is absorbed without an error (the v3 round-1 anchor
   correction); missing BOTH fails CLOSED with the descriptive
   missing-end-anchor error, never a silent EOF-widened region (v4,
   round-1 concern ``fail-open-surface6-region``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import check_worktree_task_state_briefs  # noqa: E402

_S1 = ".claude/skills/issue/steps/09-step-5.md"
_S2 = ".claude/agents/code-reviewer.md"
_S3 = ".claude/skills/issue/steps/04-step-2.md"
_S4 = ".claude/skills/issue-v2/SKILL.md"
_S5 = "CLAUDE.md"
_S6 = ".claude/skills/issue/steps/08-step-4.md"

_CANONICAL = (
    'the ABSOLUTE canonical path `$(uv run python "$REPO_ROOT"/scripts/'
    "task.py find <N>)/plans/plan.md`"
)

# --------------------------------------------------------------------------
# PRE-FIX corpus — verbatim region excerpts from base commit 5ee4f650e5
# (the pre-#2422 tree; unit 1's prose edits had NOT landed). Baked as
# literals so the red-before-fix evidence survives forever (a runtime
# `git show` would be fragile). Each region carries its live start/end
# anchors and NONE of the pinned tokens.
# --------------------------------------------------------------------------

_PREFIX_S1 = """\
# Step 5

Both reviewers see the same brief:

- `issue_number` — the task number (`<N>`)
- `target_marker_kind` — exactly one of `experiment-implementation` (for
  `experiment`) or `results` (for `infra` / `batch` / `analysis` /
  `survey`). The reviewers read the highest-version row with this kind
  from `events.jsonl` as the implementer's report.
- `revision_round` — 1-indexed integer. `1` on first review; loops up to
  `10`. The cap is **per reviewer** — reconcile invocations are free.
- `previous_critique_summaries` — one-line summaries of every prior
  `epm:code-review` AND `epm:code-review-codex` event on this task
  (empty on round 1). Lets each reviewer notice patterns.
- The diff vs `main`, the approved plan (via the `plans/plan.md`
  symlink), the existing codebase.

The Claude reviewer additionally receives:
- `worktree` path, `base` ref (typically fetched `origin/main` — #1289).

The Codex twin additionally receives:
- `worktree`, `base`, `plan_marker_path` (no `implementation_marker_path`
  — the composer fetches the marker from canonical main state and INLINES
  it; likewise an absent (#550 r1) or STALE (#546 follow-up r1) worktree
  plan — the composer inlines the canonical plan, Step 2-pre-b) — see
  `.claude/agents/codex-code-reviewer.md`.

**Neutral gate vocabulary in EVERY brief — first-pass AND revision
rounds.** Trailing content.
"""

_PREFIX_S2 = """\
# code-reviewer

## Context budget (READ FIRST)

Your spec + the project CLAUDE.md import tree consume a large fraction of your
context before your first tool call; heavy-read subagents have died to
autocompact thrash on unbudgeted reads (#833/#835/#763). Read hygiene bounds
the VARIABLE half of that load — it does not cure fixed-overhead window
pressure (#1090) — so every read below is mandatory IN CONTENT but
budgeted IN FORM:

- **Grep-then-slice.** Never pull a >40 KB file (or a file of unknown size)
  into context in one unchunked `Read`: locate the span with Grep (`-n`,
  bounded `head_limit`), then `Read` only that span with `offset`/`limit` in
  ≤300-line chunks. Material mandated "IN FULL" is still read in full — just
  chunked.
- **Never bare `task.py view <N>`** — it dumps the full event log. Task body:
  `--json | jq -r '.body'`; single fields via jq; plans via `Read` on
  `tasks/<status>/<N>/plans/v<K>.md` (or the path in your brief), sliced.
- **Results are digests.** Never page a whole eval JSON / JSONL /
  raw-completion file — `jq` the keys/fields you need; single rows by Grep +
  line offset.
- **Don't re-read what you just wrote.** `Write`/`Edit` error on failure.

Other sections name WHAT to read; this one governs HOW. On conflict, this
section wins on invocation form.

## Your Responsibilities

Other content.
"""

_PREFIX_S3 = """\
# Step 2

Subagent briefs always pass the symlink path (`plans/plan.md`) so they
read the freshest version — sound because every persisted version is
SELF-CONTAINED by contract: `new-plan-version` refuses thin
amendment-shaped deltas (#2255). After a deliberate `--allow-amendment`
persist the symlink points at a PARTIAL document, so every brief must hand
BOTH paths (the amendment `v<K>.md` AND its base `v<J>.md`);
`verify_plan.py --issue` composes them automatically.

Also include estimated cost prominently in the `epm:plan` note, with a
cost breakdown.
"""

_PREFIX_S4 = """\
# issue-v2

### Step 4: Implement + review

Dispatch the implementer per task kind, exactly as v1
(`.claude/skills/issue/SKILL.md` § "Step 4: Worktree + dispatch implementer"
applies verbatim for worktree creation via `new_worktree.sh`, the spec-freshness
sync, and the implementer/experiment-implementer split):

- `kind: experiment` → `experiment-implementer` (training/eval/data code for the
  one variable this experiment changes).
- `kind: infra|batch|analysis|survey` → `implementer`.

**The implementer's spec (v2-specific, baked in — not just checked by critics):**
launch commands MUST shard across every provisioned GPU by default (no serial
per-cell loop, no single-GPU vLLM on an N-GPU pod), and all API calls route
through `src/explore_persona_space/llm/api_dispatch.py`. These are authoring
obligations; the panel below VERIFIES them.

**Review panel per round** (ONE spawn batch, staggered):

- `plan-adherence-critic` (Claude-only) — diff vs the approved plan + manifest;
  deviations need stated reasons.
- `efficiency-critic` (implementation mode) — the Claude side of the same
  efficiency checks.

### Step 5: Run

Other content.
"""

# The pre-fix plan-handoff line is 308 chars — concatenated parts keep the
# STRING byte-identical to the base-commit line while staying ruff-clean.
_PREFIX_S5 = (
    "# CLAUDE.md\n"
    "\n"
    "- **Plan handoff convention:** pass the PATH to `.claude/plans/issue-<N>.md`, "
    "never the body. Every persisted `plans/v{K}.md` is self-contained — "
    "`new-plan-version` refuses thin amendments (#2255; `--allow-amendment` escape "
    "⇒ hand base+delta together in briefs; `verify_plan --issue` composes "
    "automatically).\n"
)

_PREFIX_S6 = """\
# Step 4

Brief passed to the implementer:
- The plan path (the `plans/plan.md` symlink, NOT the body text)
- Task number + worktree path + branch name
- Code-review history if this is a revision round (`epm:code-review v<m>`)
- Required `report-back` contract — the canonical 4-H3 marker shape from
  `.claude/agents/experiment-implementer.md` Report Format + the matching
  `## Smoke run` H2 from `.claude/agents/code-reviewer.md` Steps 0.5/0.6.
  Canonical labels (use VERBATIM in the brief):
  - `### (a) What was done`
  - `### (b) Considered but not done`
  - `### (c) How to verify`
  - `### (d) Needs human eyeball`
- The brief MUST also carry the gate-scope verification duty (#1288):

Move status to `running`.
"""

_PREFIX_SURFACES: dict[str, str] = {
    _S1: _PREFIX_S1,
    _S2: _PREFIX_S2,
    _S3: _PREFIX_S3,
    _S4: _PREFIX_S4,
    _S5: _PREFIX_S5,
    _S6: _PREFIX_S6,
}


def _write_prefix_corpus(root: Path) -> None:
    """Write the six surfaces carrying their PRE-#2422 verbatim region text."""
    for rel, body in _PREFIX_SURFACES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")


def test_prefix_corpus_fails(tmp_path: Path) -> None:
    """Criterion 4's red-before-fix evidence: the check is RED on the
    pre-fix verbatim text of every surface. A green read here means the
    ladder is mis-anchored/hollow (plan §6 kill-criterion 3) — rework the
    check, never ship it."""
    _write_prefix_corpus(tmp_path)
    errors = check_worktree_task_state_briefs(repo_root=tmp_path)
    assert errors, "check is GREEN on pre-fix text — the ladder is hollow"
    for frag in (
        "09-step-5.md",
        "code-reviewer.md",
        "04-step-2.md",
        "issue-v2",
        "CLAUDE.md",
        "08-step-4.md",
    ):
        assert any(frag in e for e in errors), (
            f"no error names pre-fix surface {frag!r}; got: {errors}"
        )


# --------------------------------------------------------------------------
# Fixed-shape corpus + per-surface drop cases
# --------------------------------------------------------------------------


def _write_corpus(root: Path, *, drop: str | None = None) -> Path:
    """Build a minimal six-surface corpus under ``root``; ``drop`` removes
    exactly one surface/anchor/token to exercise each per-surface error."""
    steps = root / ".claude" / "skills" / "issue" / "steps"
    agents = root / ".claude" / "agents"
    v2 = root / ".claude" / "skills" / "issue-v2"
    steps.mkdir(parents=True, exist_ok=True)
    agents.mkdir(parents=True, exist_ok=True)
    v2.mkdir(parents=True, exist_ok=True)

    # (1) 09-step-5.md — the shared review-brief region.
    if drop != "s1-file":
        start1 = "brief header\n" if drop == "s1-start" else "Both reviewers see the same brief:\n"
        find1 = "" if drop == "s1-find" else f"- The approved plan at {_CANONICAL}\n"
        ver1 = "" if drop == "s1-version" else "- `plan_version=v<K>` stated at compose time\n"
        man1 = (
            ""
            if drop == "s1-manifest"
            else "- the manifest `$TASK_DIR/artifacts/planned_manifest.json`\n"
        )
        bar1 = (
            ""
            if drop == "s1-readbar"
            else "- never read `tasks/` from inside the worktree (frozen at base)\n"
        )
        # v4 Edit 14 (round-1 concern worktree-events-path-gap): the negative
        # case RETAINS the bare `events.jsonl` instruction while OMITTING the
        # canonical-fetch token — a deletion-only fixture would not
        # discriminate the defect (the very sentence being removed carries
        # `events.jsonl`).
        ev1 = (
            "- reviewers read the highest-version row from `events.jsonl`\n"
            if drop == "s1-events"
            else (
                "- implementer report fetched at compose time via `uv run "
                'python "$REPO_ROOT"/scripts/task.py view <N> --json` from '
                "canonical main state\n"
            )
        )
        end1 = (
            "Closing prose without the bolded vocabulary header.\n"
            if drop == "s1-end"
            else "**Neutral gate vocabulary in EVERY brief**\n\nOther content.\n"
        )
        (steps / "09-step-5.md").write_text(
            "# Step 5\n\n" + start1 + "\n" + find1 + ver1 + man1 + bar1 + ev1 + "\n" + end1,
            encoding="utf-8",
        )

    # (2) code-reviewer.md — the Context-budget section.
    find2 = "" if drop == "s2-find" else f"- Plans via {_CANONICAL}\n"
    ver2 = (
        ""
        if drop == "s2-version"
        else "- `plan_version=` pinned by the brief; re-run readlink at read time\n"
    )
    bar2 = "" if drop == "s2-readbar" else "- never read `tasks/` from inside the worktree.\n"
    (agents / "code-reviewer.md").write_text(
        "# code-reviewer\n\n## Context budget (READ FIRST)\n\n"
        + find2
        + ver2
        + bar2
        + "\n## Your Responsibilities\n\nOther content.\n",
        encoding="utf-8",
    )

    # (3) 04-step-2.md — the plan-handoff paragraph.
    find3 = "" if drop == "s3-find" else f"via {_CANONICAL}\n"
    frozen3 = (
        ""
        if drop == "s3-frozen"
        else "— a worktree's `tasks/` tree is frozen at its base commit (#2422).\n"
    )
    (steps / "04-step-2.md").write_text(
        "# Step 2\n\nSubagent briefs always pass the symlink path in ABSOLUTE canonical form\n"
        + find3
        + frozen3
        + "\nAlso include estimated cost prominently in the `epm:plan` note.\n",
        encoding="utf-8",
    )

    # (4) issue-v2/SKILL.md — the Step 4 region.
    find4 = "" if drop == "s4-find" else f"Briefs hand the plan at {_CANONICAL}\n"
    ver4 = "" if drop == "s4-version" else "with `plan_version=v<K>`\n"
    man4 = (
        ""
        if drop == "s4-manifest"
        else "and the `planned_manifest.json` at the same canonical root.\n"
    )
    (v2 / "SKILL.md").write_text(
        "# issue-v2\n\n### Step 4: Implement + review\n\n"
        + find4
        + ver4
        + man4
        + "\n### Step 5: Run\n\nOther content.\n",
        encoding="utf-8",
    )

    # (5) CLAUDE.md — the single plan-handoff convention line.
    if drop == "s5-line":
        (root / "CLAUDE.md").write_text("# CLAUDE.md\n\nOther content.\n", encoding="utf-8")
    else:
        find5 = (
            ""
            if drop == "s5-find"
            else " a worktree consumer gets `$(uv run python scripts/task.py find"
            " <N>)/plans/plan.md`;"
        )
        frozen5 = (
            ""
            if drop == "s5-frozen"
            else " a worktree's `tasks/` tree is frozen at its base commit (#2422)."
        )
        (root / "CLAUDE.md").write_text(
            "# CLAUDE.md\n\n- **Plan handoff convention:** pass the PATH, never the body;"
            + find5
            + frozen5
            + "\n",
            encoding="utf-8",
        )

    # (6) 08-step-4.md — the implementer-brief checklist (ordered end anchors).
    find6 = "" if drop == "s6-find" else f"- The plan path at {_CANONICAL}\n"
    ver6 = "" if drop == "s6-version" else "  with `plan_version=v<K>` stated at compose time\n"
    if drop == "s6-end-both":
        tail6 = "closing prose with no recognizable end anchor.\n"
    elif drop == "s6-end-preferred":
        tail6 = "\nMove status to `running`.\n\ntrailing content.\n"
    else:
        tail6 = (
            "- The brief MUST also carry the gate-scope verification duty (#1288)\n"
            "\nMove status to `running`.\n\ntrailing content after the preferred anchor.\n"
        )
    (steps / "08-step-4.md").write_text(
        "# Step 4\n\nBrief passed to the implementer:\n" + find6 + ver6 + tail6,
        encoding="utf-8",
    )
    return root


def test_passes_on_complete_corpus(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    errors = check_worktree_task_state_briefs(repo_root=tmp_path)
    assert errors == [], f"complete corpus should pass; got: {errors}"


_DROP_CASES: list[tuple[str, str, str]] = [
    ("s1-file", "missing", "09-step-5.md"),
    ("s1-start", "start anchor", "09-step-5.md"),
    ("s1-find", "task.py find", "09-step-5.md"),
    ("s1-version", "plan_version=", "09-step-5.md"),
    ("s1-manifest", "planned_manifest.json", "09-step-5.md"),
    ("s1-readbar", "never read", "09-step-5.md"),
    ("s1-events", "task.py view", "09-step-5.md"),
    ("s1-end", "no end anchor", "09-step-5.md"),
    ("s2-find", "task.py find", "code-reviewer.md"),
    ("s2-version", "plan_version=", "code-reviewer.md"),
    ("s2-readbar", "never read", "code-reviewer.md"),
    ("s3-find", "task.py find", "04-step-2.md"),
    ("s3-frozen", "frozen", "04-step-2.md"),
    ("s4-find", "task.py find", "issue-v2"),
    ("s4-version", "plan_version=", "issue-v2"),
    ("s4-manifest", "planned_manifest.json", "issue-v2"),
    ("s5-line", "Plan handoff convention", "CLAUDE.md"),
    ("s5-find", "task.py find", "CLAUDE.md"),
    ("s5-frozen", "frozen", "CLAUDE.md"),
    ("s6-find", "task.py find", "08-step-4.md"),
    ("s6-version", "plan_version=", "08-step-4.md"),
]


@pytest.mark.parametrize(("drop", "token", "path_frag"), _DROP_CASES)
def test_fails_per_missing_surface(tmp_path: Path, drop: str, token: str, path_frag: str) -> None:
    _write_corpus(tmp_path, drop=drop)
    errors = check_worktree_task_state_briefs(repo_root=tmp_path)
    assert errors, f"drop={drop}: expected >=1 error"
    assert any(token in e and path_frag in e for e in errors), (
        f"drop={drop}: no error carries both {token!r} and {path_frag!r}; got: {errors}"
    )


@pytest.mark.parametrize("drop", ["s6-end-preferred", "s6-end-both"])
def test_surface6_end_anchor_ordered_fallback(tmp_path: Path, drop: str) -> None:
    """Surface 6's end anchor is an ordered fallback (v3 round-1
    correction): the preferred gate-scope bullet, then ``Move status to ``.
    Missing only the preferred anchor is ordinary churn the fallback
    absorbs — no error. Missing BOTH fails CLOSED with the descriptive
    missing-end-anchor error, never a silent EOF-widened region (v4,
    round-1 concern ``fail-open-surface6-region``)."""
    _write_corpus(tmp_path, drop=drop)
    errors = check_worktree_task_state_briefs(repo_root=tmp_path)
    if drop == "s6-end-both":
        assert any("no end anchor" in e and "08-step-4.md" in e for e in errors), (
            f"drop={drop}: expected the descriptive missing-end-anchor error; got: {errors}"
        )
    else:
        assert errors == [], f"drop={drop}: fallback end anchor should pass; got: {errors}"


def test_worktree_task_state_briefs_passes_on_live_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binds the landed #2422 unit-1 edits (plan Edits 1-6); the standing
    Durability pin for future refactors of any of the six surfaces."""
    monkeypatch.delenv("EPS_WORKFLOW_LINT_REPO_ROOT", raising=False)
    errors = check_worktree_task_state_briefs(repo_root=None)
    assert errors == [], f"live tree should carry all six surfaces; got: {errors}"


def test_check_worktree_task_state_briefs_bundled_in_no_flags(tmp_path: Path) -> None:
    """Two-part behavioral bundling pin (the smoke-blind-spots test's shape).

    Part A — scoped-flag subprocess against a DRIFTED corpus (surface 1's
    file dropped), rooted via ``EPS_WORKFLOW_LINT_REPO_ROOT``: proves the
    flag exists, the dispatch calls the function, and it emits its
    #2422-tagged error (nonzero exit).

    Part B — no-flags OR-chain + dispatch-ladder evidence: ``main()``'s
    source names ``args.check_worktree_task_state_briefs`` in BOTH the
    ``no_flags = not (...)`` OR-chain and the ``or no_flags`` dispatch
    ladder.
    """
    # Part A — scoped-flag subprocess against a drifted corpus.
    _write_corpus(tmp_path, drop="s1-file")
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
            "--check-worktree-task-state-briefs",
            "--file",
            str(workflow_yaml_dst),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    combined = result.stdout + result.stderr
    assert "#2422" in combined, (
        "#2422 error token missing from output — the CLI flag does not "
        f"dispatch the check. exit={result.returncode}, combined output:\n{combined}"
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
    assert "args.check_worktree_task_state_briefs" in or_chain_src, (
        "args.check_worktree_task_state_briefs is NOT in the no_flags "
        "OR-chain — a bare workflow_lint.py invocation will not fire this "
        f"check. OR-chain source:\n{or_chain_src}"
    )
    assert "args.check_worktree_task_state_briefs or no_flags" in main_src, (
        "args.check_worktree_task_state_briefs is NOT dispatched under "
        "`or no_flags` — the flag is defined but not bundled into the "
        "no-flags default run."
    )


# --------------------------------------------------------------------------
# Criterion-1 reproduction: the worktree tasks/ freeze mechanism itself
# --------------------------------------------------------------------------


def _run_git(repo: Path, *args: str) -> str:
    """Run git in the throwaway repo with global/system config neutralized."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env={
            **os.environ,
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        },
    )
    return proc.stdout.strip()


def _normalize_plan_version(readlink_target: str) -> str:
    """The contract's extensionless compose-time pin: ``v2.md`` -> ``v2``
    (09-step-5.md: 'extensionless compose-time `readlink`')."""
    return readlink_target.removesuffix(".md")


def test_worktree_freeze_reproduction() -> None:
    """The #2329/#823 mechanism, byte-demonstrated in a throwaway git repo
    (mutates NO live task or worktree): a worktree cut BEFORE a plan
    revision serves the stale symlink + stale manifest silently; the
    canonical (main-tree) resolution reads CURRENT; the read-time mismatch
    predicate fires; the #550 ABSENT shape fails ``test -f`` loud.

    Scratch lives in a ``mkdtemp`` dir (not pytest ``tmp_path``): concurrent
    pytest sessions prune ``/tmp/pytest-of-*`` roots mid-test, and this test
    is subprocess-heavy (the known prune race)."""
    scratch = Path(tempfile.mkdtemp(prefix="eps2422-worktree-freeze-"))
    try:
        repo = scratch / "repo"
        repo.mkdir()
        _run_git(repo, "init", "--initial-branch=main")
        _run_git(repo, "config", "user.name", "eps-test")
        _run_git(repo, "config", "user.email", "eps-test@example.com")

        # Initial EMPTY commit BEFORE the task folder exists (#550 ABSENT).
        _run_git(repo, "commit", "--allow-empty", "-m", "pre-task")
        pretask_sha = _run_git(repo, "rev-parse", "HEAD")

        # Commit A: task 999 at plan v1 (+ relative plan.md symlink + manifest).
        plans = repo / "tasks" / "planning" / "999" / "plans"
        plans.mkdir(parents=True)
        (plans / "v1.md").write_text("plan v1\n", encoding="utf-8")
        os.symlink("v1.md", plans / "plan.md")
        artifacts = repo / "tasks" / "planning" / "999" / "artifacts"
        artifacts.mkdir()
        (artifacts / "planned_manifest.json").write_text('{"v": 1}\n', encoding="utf-8")
        _run_git(repo, "add", "tasks")
        _run_git(repo, "commit", "-m", "commit A: task 999 at v1")

        # Worktrees: STALE (cut at commit A, before the revision) + ABSENT
        # (cut at the pre-task commit).
        wt_stale = scratch / "wt_stale"
        _run_git(repo, "worktree", "add", "--detach", str(wt_stale), "HEAD")
        wt_absent = scratch / "wt_absent"
        _run_git(repo, "worktree", "add", "--detach", str(wt_absent), pretask_sha)

        # Commit B on main: revise to v2 (re-point symlink, change manifest).
        (plans / "v2.md").write_text("plan v2\n", encoding="utf-8")
        (plans / "plan.md").unlink()
        os.symlink("v2.md", plans / "plan.md")
        (artifacts / "planned_manifest.json").write_text('{"v": 2}\n', encoding="utf-8")
        _run_git(repo, "add", "tasks")
        _run_git(repo, "commit", "-m", "commit B: revise to v2")

        stale_plan = wt_stale / "tasks" / "planning" / "999" / "plans" / "plan.md"
        main_plan = repo / "tasks" / "planning" / "999" / "plans" / "plan.md"
        stale_manifest = (
            wt_stale / "tasks" / "planning" / "999" / "artifacts" / "planned_manifest.json"
        )
        main_manifest = artifacts / "planned_manifest.json"

        # STALE shape: the worktree silently serves v1 while main serves v2.
        assert os.readlink(stale_plan) == "v1.md"
        assert os.readlink(main_plan) == "v2.md"
        assert stale_plan.resolve().read_text(encoding="utf-8") == "plan v1\n"
        # The stale read raises NO error — that silence IS the #2329 defect.
        assert stale_manifest.read_bytes() != main_manifest.read_bytes()

        # Contract: the canonical (main-tree absolute) resolution — the
        # tmp-repo stand-in for `task.py find` — reads CURRENT.
        assert main_plan.resolve().read_text(encoding="utf-8") == "plan v2\n"

        # Read-time mismatch predicate (the brief's fail-loud branch):
        # normalization maps v2.md -> v2, AND a genuine v1-vs-v2 difference
        # still trips — the normalization cannot disable the assertion.
        assert _normalize_plan_version("v2.md") == "v2"
        brief_pin = "v2"  # compose-time `plan_version=` from the CURRENT tree
        assert _normalize_plan_version(os.readlink(main_plan)) == brief_pin
        assert _normalize_plan_version(os.readlink(stale_plan)) != brief_pin
        diff = subprocess.run(
            ["diff", "-q", str(stale_manifest), str(main_manifest)],
            capture_output=True,
        )
        assert diff.returncode != 0, "diff -q must be non-zero on the stale manifest"

        # ABSENT shape (#550): the pre-task worktree has NO task folder —
        # a `test -f` read fails loud rather than serving anything.
        absent_plan = wt_absent / "tasks" / "planning" / "999" / "plans" / "plan.md"
        assert not absent_plan.exists()
        tf = subprocess.run(["test", "-f", str(absent_plan)], capture_output=True)
        assert tf.returncode != 0, "test -f must fail loud on the ABSENT shape"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
