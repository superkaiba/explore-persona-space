"""Pin the #2312 Step 10d rewritten-branch arm in .claude/skills/issue/SKILL.md.

Task #2312 (2026-08-14) closes the gap #2296 measured live: a mid-flight
rebase of `issue-<N>` onto a fresher `origin/main` (itself prescribed for
reconciling a sibling landing) leaves `origin/issue-<N>` on pre-rebase
history. The old safe-case COUNT predicate (`rev-list --count
origin/issue-<N>..HEAD > 0`) is then satisfied trivially, the push is
rejected non-fast-forward, and the refspec-less `pull --rebase=merges
--autostash` fallback rebases HEAD onto the STALE remote branch — replaying
hundreds of main commits as new objects (the #1128 shape; #2296 measured
`[ahead 363, behind 1]`). Post-#2312:

- every Step 10d push/pull copy site (canonical snippet (1), the safe-case
  push, the post-gate re-sync push, the shape-2 retry push) sits inside a
  MUTUAL-non-ancestry descendancy guard (two `merge-base --is-ancestor`
  legs, the Step-4a root-divergence probe's own shape) whose else arm keeps
  the original pair byte-identical;
- the guard's firing state is a TWO-state signal — (a) history REWRITTEN,
  (b) remote genuinely DIVERGED (the documented pod/GCE result-push channel,
  pod-side-reporting.md § Result-push verification contract, #1205/#1880) —
  discriminated BY INSPECTION, with the pull-retry self-heal named for the
  all-foreign reading (state (b)) and the § Rewritten-branch landing route
  for state (a);
- the #2240 zero-PR origin precondition gains a stale-ref arm (never open a
  PR on a mutually-non-ancestral ref);
- the #1657 head-sync pre-check is extended into a fail-closed PR-head
  parity gate before `gh pr ready` / `gh pr merge`;
- a force-free landing route (scratch-worktree merge into detached
  `origin/main` + `push HEAD:main`) is documented as the state-(a) landing
  (force-push policy is task #2313, user-decided, never here).

NOTE for future editors: the EXACTLY-5-per-leg count pins in
``test_all_copy_sites_guarded`` (and the companion `pull --rebase=merges
--autostash` == 5 pin) are DELIBERATE FRICTION — a future SKILL.md edit that
adds a push/pull copy site must either adopt the same descendancy guard (and
bump both leg counts) or consciously update the pin in the same diff. A FAIL
here is a routing prompt, not a stale count.

Prose pins are whitespace-normalized substring checks on stable invariants
(the pin-family convention); behavioral fixtures rebuild the #2296 rewritten
state and the #1880 diverged state in throwaway ``tmp_path`` git repos (the
tests/test_step10d_guards.py fixture pattern).

Paths resolve via ``Path(__file__)`` — NEVER ``task_workflow.repo_root()``,
which reads the MAIN checkout and would miss worktree edits pre-merge.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

SNIPPETS_START = "#### Bare push / merge snippets"
AUTOMERGE_START = "#### The auto-merge procedure"
LANDING_START = "#### Rewritten-branch landing route"

# The three-conjunct guard predicate (placeholder form), as it normalizes.
# Fixture test 7 instantiates it with issue-99 in a synthetic repo; this
# constant binds the fixture's semantics to the surface's exact predicate.
GUARD_PREDICATE = (
    'if git -C "$WT" rev-parse --quiet --verify origin/issue-<N> >/dev/null 2>&1 \\ '
    '&& ! git -C "$WT" merge-base --is-ancestor origin/issue-<N> HEAD \\ '
    '&& ! git -C "$WT" merge-base --is-ancestor HEAD origin/issue-<N>; then'
)
LEG_FORWARD = "merge-base --is-ancestor origin/issue-<N> HEAD"
LEG_REVERSE = "merge-base --is-ancestor HEAD origin/issue-<N>"
PULL_PAIR = "pull --rebase=merges --autostash"

PAYLOAD_SUBJECT = "issue-99: payload"
FOREIGN_SUBJECT = "pod: foreign-F1"
LOCAL_SUBJECT = "issue-99: local-L1"


def _text() -> str:
    return issue_skill_text()


def _normalized(text: str) -> str:
    """Collapse all whitespace to single spaces (wrap-tolerant substring checks)."""
    return " ".join(text.split())


def _raw_region(start_marker: str) -> str:
    """The raw SKILL.md span from ``start_marker`` to the next ``#### `` heading."""
    text = _text()
    start = text.find(start_marker)
    assert start != -1, (
        f"SKILL.md lost the {start_marker!r} heading; if the subsection was "
        "renamed, update this pin alongside it."
    )
    end = text.find("\n#### ", start + len(start_marker))
    assert end != -1, (
        f"SKILL.md has no later '#### ' heading after {start_marker!r}; if the "
        "section ordering changed, update this pin alongside it."
    )
    return text[start:end]


def _snippets_region() -> str:
    return _normalized(_raw_region(SNIPPETS_START))


def _automerge_span() -> str:
    return _normalized(_raw_region(AUTOMERGE_START))


def _landing_region_raw() -> str:
    return _raw_region(LANDING_START)


def _fenced_code(region: str) -> list[str]:
    """The bodies of the region's ``` fenced code blocks (fence lines excluded)."""
    blocks: list[str] = []
    cur: list[str] = []
    in_fence = False
    for line in region.split("\n"):
        if line.strip().startswith("```"):
            if in_fence:
                blocks.append("\n".join(cur))
                cur = []
            in_fence = not in_fence
            continue
        if in_fence:
            cur.append(line)
    return blocks


# ---------------------------------------------------------------------------
# git fixture helpers (tests/test_step10d_guards.py conventions)
# ---------------------------------------------------------------------------


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


def _commit(repo: Path, fname: str, subject: str) -> None:
    (repo / fname).write_text(f"{fname}\n")
    _git(repo, "add", fname)
    _git(repo, "commit", "-m", subject)


def _build_pushed_branch(tmp_path: Path) -> tuple[Path, Path]:
    """Bare origin + work clone; 2 base commits on main pushed; ``issue-99``
    + 1 payload commit pushed with ``-u`` (remote tip == payload).

    Returns (origin, work) with ``issue-99`` checked out in the work clone.
    """
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin))
    work = tmp_path / "work"
    _git(tmp_path, "clone", str(origin), str(work))
    _git(work, "checkout", "-B", "main")
    _commit(work, "base1.txt", "base-1")
    _commit(work, "base2.txt", "base-2")
    _git(work, "push", "-u", "origin", "main")
    _git(work, "checkout", "-b", "issue-99")
    _commit(work, "payload.txt", PAYLOAD_SUBJECT)
    _git(work, "push", "-u", "origin", "issue-99")
    return origin, work


def _advance_main_and_rebase(work: Path) -> None:
    """3 more commits on main, pushed; then ``issue-99`` is REWRITTEN by the
    prescribed mid-flight ``rebase origin/main`` (the #2296 state: local tip
    holds replayed history; ``origin/issue-99`` keeps the pre-rebase payload)."""
    _git(work, "checkout", "main")
    for i in (1, 2, 3):
        _commit(work, f"adv{i}.txt", f"advance-{i}")
    _git(work, "push", "origin", "main")
    _git(work, "checkout", "issue-99")
    _git(work, "fetch", "origin")
    _git(work, "rebase", "origin/main")


# ---------------------------------------------------------------------------
# Prose pins (1-6)
# ---------------------------------------------------------------------------


def test_canonical_snippet_form1_descendancy_guarded():
    """Pin 1: canonical snippet (1) carries the two-leg descendancy guard,
    keeps the pull-retry pair in its else arm, names BOTH residual states +
    the state-(b) contract, carries the two inspection commands, and carries
    the MID-RUN leave-unpushed instruction."""
    region = _snippets_region()
    assert "DESCENDANCY-GUARDED (#2312)" in region
    assert GUARD_PREDICATE in region, (
        "Snippet (1) lost the three-conjunct mutual-non-ancestry predicate "
        "(rev-parse --quiet --verify + BOTH merge-base legs)."
    )
    guard_idx = region.find('if git -C "$WT" rev-parse --quiet --verify origin/issue-<N>')
    pull_idx = region.find(PULL_PAIR)
    assert guard_idx != -1 and pull_idx != -1
    assert guard_idx < pull_idx, (
        "The guard must precede the push/pull pair — the pair belongs INSIDE "
        "the guard's else arm (#2312)."
    )
    assert PULL_PAIR in region, "The else arm must KEEP the rebase-retry pair verbatim."
    # Two-state signal + the state-(b) contract citation.
    assert "REWRITTEN" in region and "DIVERGED" in region
    assert "Result-push verification contract" in region or "#1880" in region
    # The discrimination-by-inspection commands.
    assert "log --oneline HEAD..origin/issue-<N>" in region
    assert "cherry HEAD origin/issue-<N>" in region
    # The MID-RUN copy-site instruction (Step-5 round pushes have no verdict).
    assert "leave the branch UNPUSHED" in region
    assert "land it" in region


def test_safe_case_push_guarded():
    """Pin 2: the Step 10d safe-case push sits inside the rewritten-or-diverged
    guard, ordered count-predicate < guard < push pair < `gh pr ready`."""
    span = _automerge_span()
    guard_tag = "REWRITTEN-OR-DIVERGED GUARD (#2312)"
    assert guard_tag in span
    count_idx = span.find(
        '|| [ "$STRIPPED_FOREIGN" = "yes" ] || [ "$MEM_COMMITTED" = "yes" ]; then'
    )
    guard_idx = span.find(guard_tag)
    push_idx = span.find("# Run every push / gh pr command BARE")
    ready_idx = span.find('gh pr ready "$PR"')
    assert -1 not in (count_idx, guard_idx, push_idx, ready_idx), (
        f"lost an ordering anchor: count@{count_idx} guard@{guard_idx} "
        f"push@{push_idx} ready@{ready_idx}"
    )
    assert count_idx < guard_idx < push_idx < ready_idx, (
        "The guard must sit between the count predicate and the wrapped push "
        f"pair, all before the merge (count@{count_idx} guard@{guard_idx} "
        f"push@{push_idx} ready@{ready_idx})."
    )
    # The BLOCKED echo instructs the discrimination + names the self-heal.
    assert "all-foreign" in span
    assert "pull-retry pair by hand" in span


def test_zero_pr_prelude_stale_ref_arm():
    """Pin 3: the #2240 zero-PR origin precondition gained the stale-ref arm —
    an existing mutually-non-ancestral ref never gets a PR opened on it."""
    span = _automerge_span()
    arm_idx = span.find("STALE-REF ARM (#2312)")
    create_idx = span.find("gh pr create --draft --head issue-<N>")
    assert arm_idx != -1, "SKILL.md lost the zero-PR stale-ref arm (#2312)."
    assert create_idx != -1, "SKILL.md lost the #2240 rc-gated create."
    assert arm_idx < create_idx, (
        "The stale-ref arm must gate BEFORE the create — a PR on a stale ref "
        "is content the sha-bound lint verdict never certified (#2296)."
    )
    # The #2240 machinery the arm wraps is retained byte-identical.
    assert "ls-remote --heads origin issue-<N>" in span
    assert "push -u origin issue-<N>" in span


def test_all_copy_sites_guarded():
    """Pin 4: EXACTLY 5 copies of each merge-base leg (hunks A/B/C/D/E — the
    canonical snippet + safe-case + zero-PR + re-sync + shape-2 sites), and
    the pull-retry literal stays at EXACTLY 5 (every pair survives inside an
    else arm; the guards' prose never writes the counted literal).

    DELIBERATE FRICTION: a future push-site addition must adopt the guard
    (bumping both leg counts) or consciously update these pins — read a FAIL
    here as a routing prompt, not a stale count. (Hunk G's landing
    verification uses the distinct `"$LOCAL_TIP" origin/main` form and counts
    toward neither leg.)"""
    text = _text()
    assert text.count(LEG_FORWARD) == 5, (
        f"expected exactly 5 forward legs, got {text.count(LEG_FORWARD)}"
    )
    assert text.count(LEG_REVERSE) == 5, (
        f"expected exactly 5 reverse legs, got {text.count(LEG_REVERSE)}"
    )
    assert text.count(PULL_PAIR) == 5, (
        f"expected exactly 5 pull-retry literals, got {text.count(PULL_PAIR)}"
    )


def test_pr_head_parity_gate_fails_closed():
    """Pin 5: the PR-head parity gate (#2312, extending #1657) fails CLOSED
    before `gh pr ready` / `gh pr merge`; remote-REF parity alone (pure
    PR-object lag) still proceeds; the single `gh pr ready` call is kept."""
    span = _automerge_span()
    assert "PR-HEAD PARITY GATE (#2312" in span
    assert "BLOCKED: PR-head parity (#2312)" in span
    gate_idx = span.find("PR-HEAD PARITY GATE (#2312")
    ready_idx = span.find('gh pr ready "$PR"')
    assert gate_idx != -1 and ready_idx != -1
    assert gate_idx < ready_idx, "The parity gate must precede `gh pr ready`."
    assert span.count('gh pr ready "$PR"') == 1, (
        "Exactly ONE `gh pr ready` call (the #2240 draft-merge precondition "
        "pin) — the parity gate must not add a second."
    )
    # The object-lag proceed arm discriminates by the REMOTE REF.
    assert "Discriminate by the REMOTE REF" in span
    # The new echo keeps the deliberate LOWERCASE form — the uppercase
    # `Verdict NOT consumed` is a counted literal
    # (tests/test_issue_skill_pr_state_probe.py).
    assert "Do NOT merge; verdict NOT consumed." in span


def test_landing_route_subsection_force_free():
    """Pin 6: the § Rewritten-branch landing route exists, is force-free
    (no force flag on any push line, no `gh pr merge`, no global worktree
    prune), and points the policy question at task #2313."""
    region = _landing_region_raw()
    assert region.startswith(LANDING_START)
    assert "push origin HEAD:main" in region
    assert "worktree add --detach -f" in region
    assert 'merge-base --is-ancestor "$LOCAL_TIP" origin/main' in region
    assert "cherry HEAD origin/issue-<N>" in region
    assert "gh pr close" in region
    assert "#2313" in region, "The force-push policy pointer (task #2313) is load-bearing."
    # Concern-1 pin: NO global worktree prune anywhere in the route (a global
    # prune unregisters ANY momentarily-missing peer worktree).
    assert "git worktree prune" not in region
    fences = _fenced_code(region)
    assert fences, "The landing route lost its fenced landing/verification blocks."
    fence_text = "\n".join(fences)
    assert "gh pr merge" not in fence_text, (
        "The landing route must never merge the PR — the head ref holds "
        "superseded pre-rebase history (close it instead)."
    )
    assert "--force-with-lease" not in fence_text, (
        "--force-with-lease may appear ONLY in the policy prose (task #2313), "
        "never in executable route code."
    )
    for line in fence_text.split("\n"):
        assert not ("push" in line and "force" in line), (
            f"force-free route violated — a fenced push line carries a force flag: {line!r}"
        )


# ---------------------------------------------------------------------------
# Behavioral fixtures (7-11)
# ---------------------------------------------------------------------------


def test_synthetic_2296_state_routes_to_rewritten_arm(tmp_path):
    """Fixture 7: rebuild the #2296 state synthetically (branch rebased onto
    an advanced main; remote ref keeps the pre-rebase payload) and assert the
    OLD count predicate is fooled while the NEW two-leg predicate routes to
    the guard arm. The fixture instantiates SKILL.md's placeholder predicate
    with ``issue-<N>`` ↔ ``issue-99``; pin (d) binds fixture to surface."""
    _, work = _build_pushed_branch(tmp_path)
    _advance_main_and_rebase(work)
    # (a) OLD predicate trivially satisfied: 3 replayed-main-side commits + 1
    # rebased payload are "ahead" of the stale remote ref.
    count = int(_git(work, "rev-list", "--count", "origin/issue-99..HEAD").stdout.strip())
    assert count == 4
    assert count > 0  # the defect this task documents: count > 0 read as "safe to push"
    # (b) the bare push would be rejected non-fast-forward.
    push = _git(work, "push", "--dry-run", "origin", "issue-99", check=False)
    assert push.returncode != 0, "a rewritten branch's push must be rejected"
    # (c) NEW predicate: ref exists + MUTUAL non-ancestry => guard arm.
    assert (
        _git(work, "rev-parse", "--quiet", "--verify", "origin/issue-99", check=False).returncode
        == 0
    )
    assert (
        _git(work, "merge-base", "--is-ancestor", "origin/issue-99", "HEAD", check=False).returncode
        == 1
    )
    assert (
        _git(work, "merge-base", "--is-ancestor", "HEAD", "origin/issue-99", check=False).returncode
        == 1
    )
    # (d) predicate-sync pin: the surface carries the exact placeholder-form
    # predicate this fixture instantiated.
    assert GUARD_PREDICATE in _normalized(_text())


def test_ordinary_missing_and_strictly_ahead_take_push_arm(tmp_path):
    """Fixture 8 (acceptance-4 evidence, three arms): the guard is FALSE on
    (1) an ordinary fast-forward branch, (2) a missing remote ref, and (3) a
    STRICTLY-AHEAD remote — where the full measured pair holds (A14): the
    bare push is REJECTED non-fast-forward (never "Everything up-to-date"),
    and the UNCHANGED refspec-less pull-retry rebases onto the correct
    same-branch upstream and re-pushes with no foreign-content loss.

    Arm 3 therefore ALSO proves the refspec-less pull is CORRECT in this
    state — the branch's configured upstream IS the right rebase target —
    which is exactly the asymmetry that makes the same pull catastrophic
    only in the rewritten state (a)."""
    origin, work = _build_pushed_branch(tmp_path)
    # Arm 1 — ordinary: remote tip == payload == ancestor of HEAD => leg 1
    # false => guard false => push arm.
    assert (
        _git(work, "merge-base", "--is-ancestor", "origin/issue-99", "HEAD", check=False).returncode
        == 0
    )
    # Arm 2 — missing remote ref: rev-parse --verify fails => guard false
    # (the #2240 push -u case unchanged; fails toward pushing).
    assert (
        _git(work, "rev-parse", "--quiet", "--verify", "origin/issue-77", check=False).returncode
        != 0
    )
    # Arm 3 — strictly-ahead remote: a second clone pushes foreign F1 while
    # the local branch holds NO novel commits.
    pod = tmp_path / "pod"
    _git(tmp_path, "clone", str(origin), str(pod))
    _git(pod, "checkout", "issue-99")
    _commit(pod, "f1.txt", FOREIGN_SUBJECT)
    _git(pod, "push", "origin", "issue-99")
    _git(work, "fetch", "origin")
    # (i) direction asymmetry: forward leg non-ancestor, reverse leg ANCESTOR
    # => guard false => push arm (never the judgment bucket).
    assert (
        _git(work, "merge-base", "--is-ancestor", "origin/issue-99", "HEAD", check=False).returncode
        == 1
    )
    assert (
        _git(work, "merge-base", "--is-ancestor", "HEAD", "origin/issue-99", check=False).returncode
        == 0
    )
    # (ii) the bare push is REJECTED non-fast-forward (rc!=0) — a
    # strictly-behind local ref is a refused update request, NOT
    # "Everything up-to-date" (git reports that only when the remote already
    # contains the local tip as its tip or a descendant).
    push = _git(work, "push", "origin", "issue-99", check=False)
    assert push.returncode != 0
    combined = push.stdout + push.stderr
    assert "non-fast-forward" in combined, combined
    assert "Everything up-to-date" not in combined
    # (iii) the UNCHANGED refspec-less pull-retry self-heals (the upstream
    # set by `push -u` is the correct same-branch rebase target here).
    pull = _git(work, "pull", "--rebase=merges", "--autostash", check=False)
    assert pull.returncode == 0, pull.stderr
    # (iv) the follow-up push exits 0.
    assert _git(work, "push", "origin", "issue-99", check=False).returncode == 0
    # (v) F1 survives exactly once on the remote branch (no foreign loss).
    _git(work, "fetch", "origin")
    subjects = _git(work, "log", "--format=%s", "origin/issue-99").stdout.strip().split("\n")
    assert subjects.count(FOREIGN_SUBJECT) == 1


def test_landing_route_lands_without_replay(tmp_path):
    """Fixture 9: the landing route (scratch worktree detached at origin/main,
    merge the certified tip, push HEAD:main) lands the rewritten branch with
    NO duplicated history and NO force flag — fast-forward topology."""
    _, work = _build_pushed_branch(tmp_path)
    _advance_main_and_rebase(work)
    local_tip = _git(work, "rev-parse", "HEAD").stdout.strip()
    scratch = tmp_path / "scratch"
    _git(work, "worktree", "add", "--detach", "-f", str(scratch), "origin/main")
    _git(scratch, "merge", "--no-edit", local_tip)  # ff: tip descends from origin/main
    _git(scratch, "push", "origin", "HEAD:main")
    _git(work, "fetch", "origin")
    # Ancestry-based landing verification (the #1897 verify-then-consume
    # posture with ancestry instead of PR state).
    assert (
        _git(work, "merge-base", "--is-ancestor", local_tip, "origin/main", check=False).returncode
        == 0
    )
    # Exact commit arithmetic: 2 base + 3 advance + 1 payload; NO merge
    # commit in the ff case and NO replayed duplicates.
    assert int(_git(work, "rev-list", "--count", "origin/main").stdout.strip()) == 6
    subjects = _git(work, "log", "--format=%s", "origin/main").stdout.strip().split("\n")
    assert subjects.count(PAYLOAD_SUBJECT) == 1


def test_landing_route_merge_commit_topology(tmp_path):
    """Fixture 10: the NON-fast-forward landing (main advanced again after the
    rebase) — the scratch merge produces a merge commit, ancestry verification
    still passes, and the payload appears exactly once (critic concern 3)."""
    origin, work = _build_pushed_branch(tmp_path)
    _advance_main_and_rebase(work)
    # main advances by 1 further commit, pushed from a second clone.
    adv = tmp_path / "adv"
    _git(tmp_path, "clone", str(origin), str(adv))
    _git(adv, "checkout", "main")
    _commit(adv, "post1.txt", "post-rebase-advance-1")
    _git(adv, "push", "origin", "main")
    local_tip = _git(work, "rev-parse", "HEAD").stdout.strip()
    _git(work, "fetch", "origin")
    scratch = tmp_path / "scratch"
    _git(work, "worktree", "add", "--detach", "-f", str(scratch), "origin/main")
    _git(scratch, "merge", "--no-edit", local_tip)  # non-ff => merge commit
    _git(scratch, "push", "origin", "HEAD:main")
    _git(work, "fetch", "origin")
    assert (
        _git(work, "merge-base", "--is-ancestor", local_tip, "origin/main", check=False).returncode
        == 0
    )
    # 2 base + 3 advance + 1 payload + 1 post-rebase advance + 1 merge commit.
    assert int(_git(work, "rev-list", "--count", "origin/main").stdout.strip()) == 8
    subjects = _git(work, "log", "--format=%s", "origin/main").stdout.strip().split("\n")
    assert subjects.count(PAYLOAD_SUBJECT) == 1


def test_diverged_remote_state_self_heals_by_instruction(tmp_path):
    """Fixture 11 (the Must-Fix state (b)): a genuinely DIVERGED remote — a
    pod pushed foreign F1 (the documented #1205 result-push channel) while
    the local branch holds unpushed L1. The guard fires (fail-closed for BOTH
    residual states), the inspection reads all-foreign, and the INSTRUCTED
    pull-retry self-heal recovers with no foreign-content loss.

    Step (c) uses the explicit-refspec ``pull --rebase=merges --autostash
    origin issue-99``; that is equivalent here to the refspec-less pair the
    guard prose instructs, because ``push -u`` set ``origin/issue-99`` as the
    branch's upstream — the refspec-less form would resolve to exactly this
    remote + branch, so the fixture provably exercises the same semantics.
    The self-heal deliberately routes through the pull FIRST: the bare push
    here would likewise be rejected non-fast-forward (L1 makes the clone
    genuinely divergent, not strictly behind — no exit-0 bare-push assumption
    anywhere in this test)."""
    origin, work = _build_pushed_branch(tmp_path)
    pod = tmp_path / "pod"
    _git(tmp_path, "clone", str(origin), str(pod))
    _git(pod, "checkout", "issue-99")
    _commit(pod, "f1.txt", FOREIGN_SUBJECT)
    _git(pod, "push", "origin", "issue-99")
    # Back in the work clone: a local unpushed commit, then fetch.
    _commit(work, "l1.txt", LOCAL_SUBJECT)
    _git(work, "fetch", "origin")
    # (a) the guard predicate FIRES on both legs (mutual non-ancestry).
    assert (
        _git(work, "rev-parse", "--quiet", "--verify", "origin/issue-99", check=False).returncode
        == 0
    )
    assert (
        _git(work, "merge-base", "--is-ancestor", "origin/issue-99", "HEAD", check=False).returncode
        == 1
    )
    assert (
        _git(work, "merge-base", "--is-ancestor", "HEAD", "origin/issue-99", check=False).returncode
        == 1
    )
    # (b) the discrimination reads all-foreign: exactly one remote-only
    # commit, marked `+` (novel) by git cherry.
    cherry = _git(work, "cherry", "HEAD", "origin/issue-99").stdout.strip().split("\n")
    assert len(cherry) == 1 and cherry[0].startswith("+ "), cherry
    log_lines = [
        line
        for line in _git(work, "log", "--oneline", "HEAD..origin/issue-99")
        .stdout.strip()
        .split("\n")
        if line
    ]
    assert len(log_lines) == 1, log_lines
    # (c) the INSTRUCTED self-heal works (see docstring for refspec parity).
    pull = _git(work, "pull", "--rebase=merges", "--autostash", "origin", "issue-99", check=False)
    assert pull.returncode == 0, pull.stderr
    assert (
        _git(work, "merge-base", "--is-ancestor", "origin/issue-99", "HEAD", check=False).returncode
        == 0
    )
    assert _git(work, "push", "origin", "issue-99", check=False).returncode == 0
    _git(work, "fetch", "origin")
    subjects = _git(work, "log", "--format=%s", "origin/issue-99").stdout.strip().split("\n")
    assert subjects.count(FOREIGN_SUBJECT) == 1
    assert subjects.count(LOCAL_SUBJECT) == 1
    # (d) routing-text pin: the auto-merge span names the self-heal for the
    # all-foreign reading (binds whichever routing wording lands to state (b)).
    span = _automerge_span()
    assert "all-foreign" in span
    assert "self-heal" in span
