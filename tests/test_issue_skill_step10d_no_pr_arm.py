"""Pin the #2240 Step 10d payload-aware no-PR arm in .claude/skills/issue/SKILL.md.

Task #2240 (2026-08-12) restructured the Step 10d safe-case PR routing:

- The pre-#2240 ``-z "$PR"`` arm skipped UNCONDITIONALLY ("No PR for
  issue-<N>; nothing to merge.", no marker), so a code-bearing branch whose
  Step 4a draft-PR create never fired (Step 4a runs BEFORE the implementer's
  first commit, so its else arm fires by construction) was left permanently
  unmerged with the durable record reading clean — the #456->#466
  stranded-shared-module class, invisible to the completed_unmerged_pass
  watcher flag (it keys on a marker the skip arm never posted).
- Post-#2240, BOTH no-usable-PR cases (terminal PR, #1897; no PR object at
  all, #2240/#2235) route through one shared payload-aware prelude gated on
  ``USABLE_PR``: #1897's layered fail-safe novel-payload predicate decides
  between "create a fresh PR and merge" and "genuinely nothing to merge",
  the zero-PR create carries an origin-precondition push + an rc-gated
  ``gh pr create``, the anomaly note is composed from the REALIZED outcome,
  and a novel-payload branch that cannot obtain a usable PR fails loud
  (``epm:merge-failed``) instead of printing a false nothing-to-merge line.

These tests fail the suite if a later SKILL.md editor re-introduces the
silent skip, narrows the predicate back to a ``PR_STATE != OPEN``-only arm,
drops a fail-safe layer, or removes the loud-failure routing (plan #2240
durability pins; 8 pins per plan section (D)).

Task #2241 (2026-08-20) moved PRIMARY draft-PR creation to Step 5 round
entry: the "Draft-PR ensure (#2241)" block probes any-state PR existence
and runs an rc-gated, timeout-fenced, fail-open ``gh pr create --draft``
at the first review round — the first point where commits exist — so
Step 10d's payload-aware arm is now the pinned merge-time BACKSTOP, not
the normal path. Pins 9-12 below bind the ensure: presence + rc-gated
create + pinned ordering after Round-push-hygiene (9), fail-open echo
sites + timeout fences (10), the no-executable-git-push INVARIANT over
the block's fenced Bash (11 — a regex scan with in-test mutation
controls, not one literal), and the Step 4a else-arm routing prose (12).

Round 2 (#2241, concerns task-title-shell-injection +
step5-probe-output-validation) added pins 13-14: the PRIMARY create sites
(Step 4a + the Step 5 ensure) resolve the PR title AS DATA via command
substitution — never splicing it into shell source (13), and a hostile
title (backtick / $(...) / double quote / backslash) reaches gh's argv
boundary LITERALLY, with an in-test negative control proving the old
spliced form evaluates the same payload (14). Step 10d's trunk splice
sites are deliberately out of #2241's fence (plan must-ask list) and are
NOT pinned here.

Round 3 (#2241, binding blocker title-resolution-failure-masking) fenced
the title RESOLVER itself at both primary sites: a separately rc-gated,
timeout-bounded resolution step. Every task.py invocation — reads
included — pays the branch-guard resolution, whose #996 bounded rebase
wait (EPM_TASKPY_REBASE_WAIT_SECONDS, default 120 s) can precede a
RuntimeError (detached HEAD / husk timeout); round 2's plain assignment
let jq exit 0 on empty input and mask that failure into a
created-and-memoized degraded ``issue-<N>: `` prefix-only PR title. Pin
13 gains static shape asserts for BOTH sites (the r2 plain-assignment
form is banned); pin 15 EXECUTES the ensure with a selectable failing
``uv`` stub (nonzero exit / malformed JSON / missing title) and asserts
no ``gh pr create`` runs, the inconclusive telemetry is emitted, and the
block stays fail-open (exit 0).

Round 4 (#2241, binding blocker step5-repo-root-uninitialized): round 3
unified the resolver to ``"$REPO_ROOT"/scripts/task.py`` but 09-step-5.md
carried ZERO ``REPO_ROOT=`` assignments — fenced blocks run in separate
shells and the orchestrator's Bash cwd resets, so ``$REPO_ROOT`` expanded
empty, the resolver ran the root-anchored ``/scripts/task.py``, and the
r3 skip arm fired at EVERY round entry: behaviorally the zero-PR
generator this task exists to eliminate. The r3 ``uv`` stub was a
``case "$UV_STUB_MODE"`` that never inspected its argv, so green pins
certified nothing about the invocation path. r4 adds the canonical
in-fence resolve (``REPO_ROOT=$(dirname "$(git rev-parse
--path-format=absolute --git-common-dir)")``, the #506-safe form) and
closes the masking: the ``uv`` stub is now argv-RECORDING (mirroring the
``gh`` stub), the tmp fixture is ``git init``-ed so the recipe resolves
deterministically, and the success pin (14) asserts the invoked script
path equals ``<resolved-root>/scripts/task.py`` with an explicit
rejection of the root-anchored ``/scripts/task.py`` shape. NIT
whitespace-only-pr-title closed the same round: ``[ -z "$RAW_TITLE" ]``
passed a whitespace-only stored title (set_title stores input
unstripped) into a degraded ``issue-<N>:   `` PR — both sites now gate
on ``[ -z "${RAW_TITLE//[[:space:]]/}" ]`` (>=1 non-whitespace char) and
pin 15 gains a ``whitespace-title`` stub parametrization.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.issue_skill_source import issue_skill_text

SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "issue" / "SKILL.md"


def _text() -> str:
    return issue_skill_text()


def _step10d_span() -> str:
    """Return the SKILL.md text from the (unique) `### Step 10d` heading onward."""
    text = _text()
    return text[text.index("### Step 10d") :]


GUARDS_COMMENT = "# Run guards 1-3 above first."
USABLE_GATE = 'if [ "$USABLE_PR" != yes ]; then'
NOVEL_GATE = 'if [ "$NOVEL_PAYLOAD" = "yes" ]; then'

#: Every executable git-push form the workflow's own snippets use: bare
#: `git push`, the `-C`-form `git -C "$WT" push`, a `timeout`-prefixed
#: wrapper of either, and `push -u`. Exotic wrappers (`env VAR=1 git push`,
#: `command git push`, `git -c cfg push`, `$GIT push`) escape the anchored
#: regex by design — the block's prose ban + code review catch those.
_GIT_PUSH_LINE = re.compile(r"^\s*(?:timeout\b.*\s+)?git(?:\s+-C\s+\S+)?\s+push\b")


def _step5_span() -> str:
    """The composed text between the unique `### Step 5:` and `### Step 6:` anchors."""
    text = _text()
    start = text.index("### Step 5:")
    return text[start : text.index("### Step 6:", start)]


def _ensure_block() -> str:
    """The #2241 Draft-PR ensure block (heading to the split-review right anchor)."""
    span = _step5_span()
    start = span.index("Draft-PR ensure (#2241")
    return span[start : span.index("Per-commit split-review dispatch", start)]


def _ensure_block_bash() -> str:
    """The ensure block's fenced Bash snippet (```bash ... ```)."""
    block = _ensure_block()
    start = block.index("```bash")
    return block[start : block.index("\n```", start + len("```bash"))]


def test_no_pr_arm_is_payload_aware():
    """Pin 1: the unconditional silent skip is GONE — from Step 10d and the
    Step 4a prose that promised a follow-up step which never existed."""
    text = _text()
    # The exact pre-#2240 silent-skip line (echo + post nothing) is absent.
    assert "No PR for issue-<N>; nothing to merge." not in text
    # Step 4a's false promise is gone; its else arm now names the Step 10d
    # payload-aware backstop instead (hunk B durability).
    assert "open it after the implementer commits" not in text
    step4a_region = text[: text.index("### Step 10d")]
    assert "This arm fires by construction on a fresh branch" in step4a_region
    assert "payload-aware arm (#2240) opens it at merge time" in step4a_region


def test_both_skip_arms_route_through_usable_pr_gate():
    """Pin 2: the USABLE_PR gate exists, and BOTH skip arms (loud novel-payload
    failure + quiet nothing-to-merge) live inside the routing gate, before the
    guards/merge body."""
    span = _step10d_span()
    assert span.count(USABLE_GATE) == 2  # prelude + routing gate
    idx_routing = span.index(USABLE_GATE, span.index(USABLE_GATE) + 1)
    idx_guards = span.index(GUARDS_COMMENT)
    idx_merge_failed = span.index("post-marker <N> epm:merge-failed")
    idx_echo_prior = span.index("has no novel payload vs origin/main — nothing to merge")
    idx_echo_no_pr = span.index("has no PR and no novel payload vs origin/main — nothing to merge")
    assert idx_routing < idx_merge_failed < idx_guards
    assert idx_routing < idx_echo_prior < idx_guards
    assert idx_routing < idx_echo_no_pr < idx_guards


def test_predicate_in_shared_no_usable_pr_prelude():
    """Pin 3: the novel-payload predicate lives inside the shared no-usable-PR
    prelude, NOT inside a `PR_STATE != OPEN`-only arm; the USABLE_PR
    resolution precedes the predicate's defensive NOVEL_PAYLOAD init."""
    span = _step10d_span()
    # The old scoping is gone entirely.
    assert 'if [ "$PR_STATE" != "OPEN" ]; then' not in span
    # The positive resolution replaces it.
    assert 'if [ -n "$PR" ] && [ "$PR_STATE" = "OPEN" ]; then' in span
    # Ordering: USABLE_PR is assigned before the defensive NOVEL_PAYLOAD=yes
    # init, and the bounded fetch + predicate run inside the prelude (after
    # the first USABLE_PR gate).
    assert span.index("USABLE_PR=no") < span.index("NOVEL_PAYLOAD=yes")
    idx_prelude = span.index(USABLE_GATE)
    idx_fetch = span.index('timeout --kill-after=30s 120s git -C "$REPO_ROOT" fetch origin main')
    assert idx_prelude < idx_fetch < span.index(GUARDS_COMMENT)


def test_fail_safe_predicate_layers_verbatim():
    """Pin 4: all four #1897 fail-safe layers survive verbatim, including
    NOVEL_PAYLOAD=yes as the default and the fail-safe comments on the
    git-error paths."""
    span = _step10d_span()
    assert "NOVEL_PAYLOAD=yes" in span
    assert 'rev-list --count origin/main..issue-<N>)" -eq 0 ]' in span  # (1)
    assert 'elif CHERRY=$(git -C "$WT" cherry origin/main issue-<N>)' in span  # (2)
    assert "(a cherry FAILURE falls through — fail-safe)" in span
    assert 'OWN_FILES=$(git -C "$WT" diff --name-only origin/main...issue-<N>)' in span  # (3)
    assert 'git -C "$WT" diff --quiet origin/main issue-<N> -- $OWN_FILES' in span
    assert "(a diff ERROR keeps 'yes' — fail-safe)" in span
    flat = " ".join(span.split())
    assert "(4) else -> novel payload" in flat


def test_no_pr_anomaly_marker_present():
    """Pin 5: the zero-PR create arm exists (HAD_PRIOR_PR branch + #2240 PR
    body) and posts the [step10d-no-pr-anomaly] note composed from the
    REALIZED outcome (opened-and-proceeding vs recovery-FAILED)."""
    span = _step10d_span()
    assert "HAD_PRIOR_PR=no" in span
    assert "no PR object exists (#2240 probe)" in span
    assert span.count("[step10d-no-pr-anomaly]") == 2  # success + failure notes
    flat = " ".join(span.split())
    assert "Step 10d opened PR #$PR and is proceeding with the auto-merge (#2240)." in flat
    assert "the recovery FAILED: gh pr create did not yield an OPEN PR" in flat


def test_pr_ready_precedes_merge():
    """Pin 6: `gh pr ready "$PR"` still immediately precedes the safe-case
    merge — the draft-merge precondition for PRs created by EITHER fresh-PR
    arm — and no second ready call was added (hunk C durability)."""
    span = _step10d_span()
    assert span.count('gh pr ready "$PR"') == 1
    idx_ready = span.index('gh pr ready "$PR"')
    idx_merge = span.index('gh pr merge "$PR" $MERGE_FORM --delete-branch=false')
    assert idx_ready < idx_merge
    flat = " ".join(span.split())
    assert "Draft-merge precondition (#2240 pin)" in flat
    assert "do NOT add a second ready call elsewhere" in flat


def test_origin_precondition_precedes_rc_gated_create():
    """Pin 7: the origin-precondition (ls-remote probe + push -u) runs BEFORE
    `gh pr create`, and the create is rc-gated so a failed create can never
    fall through into the nothing-to-merge arm."""
    span = _step10d_span()
    idx_lsremote = span.index('git -C "$WT" ls-remote --heads origin issue-<N>')
    idx_push_u = span.index('git -C "$WT" push -u origin issue-<N>')
    idx_create = span.index("gh pr create --draft --head issue-<N>")
    assert idx_lsremote < idx_create
    assert idx_push_u < idx_create
    # rc-gated: the create is the condition of an `if`, never a bare command.
    assert (
        'if gh pr create --draft --head issue-<N> --title "$PR_TITLE" --body "$PR_BODY"; then'
        in span
    )
    # The fresh PR re-resolve only flips USABLE_PR on an OPEN resolve.
    assert '[ -n "$PR" ] && [ "$PR_STATE" = "OPEN" ] && USABLE_PR=yes' in span


def test_nothing_to_merge_guarded_on_novel_payload():
    """Pin 8: the nothing-to-merge echoes sit in the NOVEL_PAYLOAD else arm of
    the routing gate, and the novel-payload-but-no-usable-PR path fails loud
    with epm:merge-failed before them."""
    span = _step10d_span()
    idx_routing = span.index(USABLE_GATE, span.index(USABLE_GATE) + 1)
    idx_novel_routing = span.index(NOVEL_GATE, idx_routing)  # NOVEL conjunct inside the gate
    idx_merge_failed = span.index("post-marker <N> epm:merge-failed")
    idx_echo_prior = span.index("has no novel payload vs origin/main — nothing to merge")
    idx_echo_no_pr = span.index("has no PR and no novel payload vs origin/main — nothing to merge")
    idx_guards = span.index(GUARDS_COMMENT)
    # Routing gate -> NOVEL_PAYLOAD conjunct -> loud failure -> quiet echoes.
    assert idx_routing < idx_novel_routing < idx_merge_failed < idx_echo_prior < idx_guards
    assert idx_merge_failed < idx_echo_no_pr < idx_guards
    flat = " ".join(span.split())
    assert "NOVEL PAYLOAD ON issue-<N> COULD NOT BE MERGED" in flat
    assert "this is a stranding risk, not a no-op" in flat


def test_step5_draft_pr_ensure_present_and_rc_gated():
    """Pin 9 (#2241): the ensure lives in Step 5 AFTER the Round-push-hygiene
    block — a PINNED ordering/stability choice (conservative distance from
    the pre-split-guard lint region, stable _ensure_block() anchoring,
    readable order: push hygiene -> PR ensure -> split-review dispatch), NOT
    lint protection (the check_pre_split_review_guard region was verified
    byte-identical under the adjacent slots — #2241 v4 replay) — probes
    ANY-state existence deterministically, and the create is rc-gated +
    timeout-bounded."""
    span = _step5_span()
    block = _ensure_block()
    assert span.index("**Round push hygiene.**") < span.index("Draft-PR ensure (#2241")
    assert "gh pr list --head issue-<N> --state all --json number --jq length" in block
    assert (
        "if timeout --kill-after=30s 120s gh pr create --draft --head issue-<N>" in block
    )  # rc-gated: the create is an `if` condition, never bare
    assert block.index("gh pr list") < block.index("gh pr create")
    assert (
        block.count("gh pr create --draft --head issue-<N>") == 1
    )  # 1 in insert + 0 pre-existing in Step 5


def test_step5_ensure_fail_open_and_bounded():
    """Pin 10 (#2241; r2 telemetry; r3 resolver fence): every arm logs +
    proceeds (6 echo sites), all three commands (probe / title resolver /
    create) are timeout-fenced, and probe failure never blind-creates.
    r2 (concern step5-probe-output-validation) added TELEMETRY-ONLY echoes
    for the confirmed (>=1) and unexpected/malformed probe outputs; both
    fall through to 5a exactly as before — the tri-state ROUTING
    (probe-failed / zero / no-create-fall-through) is unchanged. r3
    (concern title-resolution-failure-masking) added the resolver fence +
    its inconclusive skip-create echo."""
    block = _ensure_block()
    assert (
        block.count("[step5-pr-ensure]") == 6
    )  # title-fail + create-ok + create-fail + probe-fail + confirmed + unexpected
    assert block.count("timeout --kill-after=") == 3  # probe + resolver + create
    assert "N_PR=probe-failed" in block and '"$N_PR" = "0"' in block


def test_step5_ensure_adds_no_push_site():
    """Pin 11 (#2241, v4 — the INVARIANT, not one literal; MF4): the ensure
    block's fenced Bash carries NO executable git-push form. v3 pinned only
    `push -u origin issue-<N>`; the workflow's canonical bare form
    `git -C "$WT" push origin issue-<N>` would have evaded that literal while
    leaving the #2312 count-5 pins green — silently creating the sixth
    unguarded push site the design forbids. Also: no ancestry legs. Any
    future pre-ensure push site must live OUTSIDE _ensure_block()."""
    bash = _ensure_block_bash()
    for line in bash.splitlines():
        assert not _GIT_PUSH_LINE.match(line), f"executable git-push form in ensure block: {line!r}"
    block = _ensure_block()
    assert "merge-base --is-ancestor" not in block
    # Mutation controls: the regex itself MUST catch every canonical form,
    # so it can never silently rot into matching nothing.
    assert _GIT_PUSH_LINE.match('git -C "$WT" push origin issue-<N>')
    assert _GIT_PUSH_LINE.match("timeout --kill-after=30s 120s git push origin issue-<N>")
    assert _GIT_PUSH_LINE.match("git push -u origin issue-<N>")


def test_step4a_else_arm_names_step5_ensure():
    """Pin 12 (#2241): Step 4a's else arm routes the reader to the Step 5
    ensure as the primary site (backstop naming from #2240 retained — the
    two #2240 literals stay pinned by test_no_pr_arm_is_payload_aware).
    Right boundary tightened to `### Step 5:` (v4, non-blocking item) so the
    Step-5 insert itself can never satisfy the pin."""
    text = _text()
    step4a_region = text[: text.index("### Step 5:")]
    assert "Step 5 draft-PR ensure (#2241)" in step4a_region


def test_pr_create_sites_use_title_transport():
    """Pin 13 (#2241 r2, concern task-title-shell-injection): both PRIMARY
    create sites (Step 4a + the Step 5 ensure) resolve the PR title AS DATA
    — command substitution over `task.py view --json | jq -r` — instead of
    splicing `<task title>` into shell source. A splice is injectable at
    ASSIGNMENT, not just at use (`PR_TITLE="issue-<N>: <task title>"` is
    equally injectable), so the pin binds on the TRANSPORT: the placeholder
    must be absent from the bash and the title must arrive via `$(...)`.
    Region-scoped: Step 10d's trunk splice is out of #2241's fence.

    r3 (concern title-resolution-failure-masking): BOTH sites resolve the
    title in a separately rc-gated, timeout-fenced step and SKIP creation
    on resolver failure / jq failure / empty title. The r2 plain-assignment
    form — the pipeline inside the PR_TITLE assignment, whose task.py
    failure jq masks (RuntimeError -> jq reads empty input -> exit 0 ->
    degraded `issue-<N>: ` title created and memoized) — is BANNED.

    r4 (blocker step5-repo-root-uninitialized): BOTH regions carry the
    canonical #506-safe REPO_ROOT resolve — Step 4a resolves it in-step
    (08-step-4.md:18); the Step 5 ensure resolves it IN-FENCE, because
    fenced blocks run in separate shells and never inherit the file-wide
    idiom's value (r3 shipped the "$REPO_ROOT" idiom with zero
    assignments: /scripts/task.py at every round entry). The bare r3 skip
    gate `[ -z "$RAW_TITLE" ]` is BANNED at both sites (NIT
    whitespace-only-pr-title): set_title stores input unstripped, so a
    whitespace-only stored title passed -z and composed a degraded
    `issue-<N>:   ` PR that the >=1 probe then memoized."""
    bash = _ensure_block_bash()
    assert "<task title>" not in bash
    assert '--title "$PR_TITLE"' in bash
    assert "jq -r '.frontmatter.title // empty'" in bash
    text = _text()
    step4a_region = text[: text.index("### Step 5:")]
    assert "<task title>" not in step4a_region
    assert step4a_region.count('--title "$PR_TITLE"') == 1
    assert "jq -r '.frontmatter.title // empty'" in step4a_region
    root_resolve = 'REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")'
    for region in (bash, step4a_region):
        assert 'PR_TITLE="issue-<N>: $(' not in region  # the r2 masked form
        assert "|| TITLE_RC=$?" in region  # rc-gated resolver
        assert "timeout --kill-after=30s 150s" in region  # resolver fence
        assert 'PR_TITLE="issue-<N>: $RAW_TITLE"' in region  # data-only compose
        assert root_resolve in region  # r4: in-fence / in-step root resolve
        # r4 skip gate: >=1 non-whitespace char required before composing.
        assert '[ "$TITLE_RC" -ne 0 ] || [ -z "${RAW_TITLE//[[:space:]]/}" ]' in region
        assert '[ -z "$RAW_TITLE" ]' not in region  # the whitespace-blind r3 gate


def _title_transport_env(tmp_path, hostile_title: str, uv_mode: str = "ok"):
    """PATH-stubbed env for executing the ensure template: `gh` records its
    create argv to $ARGV_OUT (probe returns 0 = the create arm), `uv` is
    argv-RECORDING to $UV_ARGV_OUT (r4 — the r3 stub never inspected its
    argv, so it succeeded identically for the broken root-anchored
    /scripts/task.py invocation and green pins certified nothing about the
    invocation path) and SELECTABLE via $UV_STUB_MODE (r3 — an
    always-succeed stub pins nothing): ok = emit the task-view JSON
    carrying the hostile title; fail = nonzero exit with no stdout (the
    task.py RuntimeError class: detached HEAD / husk timeout after the
    #996 bounded rebase wait — the r3 stub's message misattributed this to
    repo-root resolution, ironically the exact failure class the r3
    template guaranteed); malformed = non-JSON stdout, exit 0 (jq parse
    error downstream); missing-title = valid JSON whose frontmatter
    carries no title; whitespace-title = valid JSON whose title is
    whitespace-only (r4, concern whitespace-only-pr-title: set_title
    stores input unstripped). jq/timeout/git are real.

    r4: the fixture is ``git init``-ed so the template's in-fence
    ``REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute
    --git-common-dir)")`` resolves deterministically to the fixture root;
    any inherited $REPO_ROOT is popped so the mutation control (the
    uninitialized r3 form) cannot accidentally pass off the test
    environment. Returns (env, gh_argv_out, uv_argv_out, resolved_root).
    """
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    common_dir = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    resolved_root = os.path.dirname(common_dir)  # the fence recipe, verbatim
    stub = tmp_path / "bin"
    stub.mkdir(exist_ok=True)
    argv_out = tmp_path / "create-argv.txt"
    uv_argv_out = tmp_path / "uv-argv.txt"
    title_json = tmp_path / "view.json"
    title_json.write_text(json.dumps({"frontmatter": {"title": hostile_title}}), encoding="utf-8")
    (stub / "gh").write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1 $2" = "pr list" ]; then echo 0; exit 0; fi\n'
        'if [ "$1 $2" = "pr create" ]; then shift 2; '
        'printf \'%s\\n\' "$@" > "$ARGV_OUT"; exit 0; fi\n'
        "exit 1\n",
        encoding="utf-8",
    )
    (stub / "uv").write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$@" > "$UV_ARGV_OUT"\n'
        'case "$UV_STUB_MODE" in\n'
        '  fail) echo "RuntimeError: task.py branch-guard failure'
        ' (detached HEAD / husk timeout)" >&2; exit 3 ;;\n'
        "  malformed) echo 'this is not json' ;;\n"
        "  missing-title) echo '{\"frontmatter\": {}}' ;;\n"
        '  whitespace-title) echo \'{"frontmatter": {"title": "   "}}\' ;;\n'
        '  *) cat "$TITLE_JSON" ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    for name in ("gh", "uv"):
        (stub / name).chmod(0o755)
    env = os.environ.copy()
    env.pop("REPO_ROOT", None)
    env["PATH"] = f"{stub}{os.pathsep}{env['PATH']}"
    env["ARGV_OUT"] = str(argv_out)
    env["UV_ARGV_OUT"] = str(uv_argv_out)
    env["TITLE_JSON"] = str(title_json)
    env["UV_STUB_MODE"] = uv_mode
    return env, argv_out, uv_argv_out, resolved_root


@pytest.mark.parametrize("uv_mode", ["fail", "malformed", "missing-title", "whitespace-title"])
def test_step5_ensure_title_resolution_failure_skips_create(tmp_path, uv_mode):
    """Pin 15 (#2241 r3, concern title-resolution-failure-masking; r4,
    concern whitespace-only-pr-title): EXECUTE the ensure template with the
    `uv` stub forced into each failure mode — nonzero exit (task.py
    RuntimeError after the #996 bounded rebase wait), malformed JSON (jq
    parse error), missing/empty title, and a whitespace-only stored title
    (r4: set_title stores input unstripped, so "   " passed the bare -z
    gate). Each must: (a) NEVER run `gh pr create` — no degraded
    `issue-<N>: ` PR is opened or memoized; (b) emit the inconclusive
    [step5-pr-ensure] telemetry; (c) exit 0 (fail-open — the round
    proceeds, retry at next round entry). Mutation check (run at authoring
    time, r3): against round 2's unfenced plain-assignment resolver, all
    three r3 modes FAIL — the create fires with the degraded title — so
    this pin demonstrably binds the fix rather than passing vacuously.
    Mutation check (r4): against the r3 `[ -z "$RAW_TITLE" ]`-only gate,
    the whitespace-title mode FAILS — the create fires with the degraded
    `issue-9999:   ` title."""
    assert shutil.which("jq"), "jq is required by the Step 5 ensure title transport"
    env, argv_out, _uv_argv_out, _root = _title_transport_env(
        tmp_path, "unused benign title", uv_mode=uv_mode
    )

    bash = _ensure_block_bash()
    script = bash.split("\n", 1)[1].replace("<N>", "9999")  # drop the ```bash fence
    res = subprocess.run(
        ["bash", "-c", script],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)  # fail-open: round proceeds
    assert not argv_out.exists(), (
        f"gh pr create RAN despite title-resolution failure ({uv_mode}): "
        f"{argv_out.read_text(encoding='utf-8')}"
    )
    assert "[step5-pr-ensure] title resolution failed or empty" in res.stdout
    assert "opened draft PR" not in res.stdout


def test_step5_ensure_hostile_title_reaches_argv_literally(tmp_path):
    """Pin 14 (#2241 r2, concern task-title-shell-injection): EXECUTE the
    ensure template with a hostile title carrying a backtick command, a
    $(...) command, a double quote, and a backslash. The payload must reach
    gh's argv boundary LITERALLY — never evaluated (no canary side effect).
    Negative control (in-test mutation check): the pre-r2 SPLICED form —
    the title substituted into shell SOURCE — evaluates the same payload
    class, firing the canaries, so this pin demonstrably fails against the
    old shape rather than passing vacuously.

    r4 (blocker step5-repo-root-uninitialized): the success pin ALSO
    asserts the resolver invoked task.py at the IN-FENCE-resolved repo
    root. Round 3's uninitialized "$REPO_ROOT" expanded empty in the
    separate-shell fence, so the resolver ran the root-anchored
    /scripts/task.py, python exited non-zero, and the skip arm fired at
    every round entry — while the argv-blind r3 uv stub kept these pins
    green. Mutation check (run at authoring time, r4): with the in-fence
    `REPO_ROOT=$(...)` line deleted (the r3 form), the recorded uv argv is
    exactly /scripts/task.py and this pin FAILS."""
    assert shutil.which("jq"), "jq is required by the Step 5 ensure title transport"
    canary = tmp_path / "canary"
    hostile = f'pwn `touch {canary}-bt` and $(touch {canary}-sub) with "quote" and \\ backslash'
    env, argv_out, uv_argv_out, resolved_root = _title_transport_env(tmp_path, hostile)

    bash = _ensure_block_bash()
    script = bash.split("\n", 1)[1].replace("<N>", "9999")  # drop the ```bash fence
    res = subprocess.run(
        ["bash", "-c", script],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "[step5-pr-ensure] opened draft PR" in res.stdout
    assert not (tmp_path / "canary-bt").exists(), "backtick payload was EVALUATED"
    assert not (tmp_path / "canary-sub").exists(), "$(...) payload was EVALUATED"
    argv = argv_out.read_text(encoding="utf-8").splitlines()
    assert f"issue-9999: {hostile}" in argv  # one argv entry, payload verbatim
    # r4: the resolver must run task.py at the in-fence-resolved root —
    # never the root-anchored /scripts/task.py of an empty $REPO_ROOT.
    uv_argv = uv_argv_out.read_text(encoding="utf-8").splitlines()
    assert uv_argv[:2] == ["run", "python"], uv_argv
    script_path = uv_argv[2]
    assert script_path == f"{resolved_root}/scripts/task.py", uv_argv
    assert script_path != "/scripts/task.py", (
        "uninitialized $REPO_ROOT: root-anchored /scripts/task.py invocation"
    )

    # Negative control: the old spliced shape (title in shell SOURCE).
    # Double quote omitted — a raw quote breaks parsing before evaluation,
    # which would mask the substitution firing this control exists to show.
    spliced_title = f"pwn `touch {canary}-old-bt` and $(touch {canary}-old-sub)"
    spliced = (
        "if timeout --kill-after=30s 120s gh pr create --draft --head issue-9999 \\\n"
        f'     --title "issue-9999: {spliced_title}" --body "Closes task #9999."; then\n'
        "  echo created\nfi\n"
    )
    res2 = subprocess.run(
        ["bash", "-c", spliced],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res2.returncode == 0, (res2.stdout, res2.stderr)
    assert (tmp_path / "canary-old-bt").exists(), "negative control lost its teeth (backtick)"
    assert (tmp_path / "canary-old-sub").exists(), "negative control lost its teeth ($(...))"
