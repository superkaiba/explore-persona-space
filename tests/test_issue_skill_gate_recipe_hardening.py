"""Pin the #2126 gate-recipe hardenings (D1-D5) in `/issue` SKILL.md.

Task #2126 fixed five probed defects across the two gate-recipe surfaces
(Step 9c test-verdict gate + Step 10d pre-push lint gate). Without these
pins, any later SKILL.md edit silently re-opens exactly the defects the
task closed (the #884/#1045/#1134 lineage of silently-regressed recipe
prose). One test per fix:

- D1 (#1992): the 1b launch block cross-checks the substituted `<files>`
  set against the selector's LIVE count (`--files-only`) BEFORE the
  launch, and the 1a prose pins the `--json` list key (`tests`, not
  `files`).
- D2 (#2006): the pre-gate origin/main re-sync binds EVERY gate
  (re)launch — the #1742 paragraph carries the every-relaunch scope
  clause and the Step 10d single-flight paragraph carries the companion
  sentence.
- D3 (measured, `tasks/<status>/2126/artifacts/probe-detached-fd.txt`):
  the detached-launch recipe states the stdout-redirect-inside-`$( )`
  mechanism, the `timeout`-does-not-bound-the-hang caveat, and BOTH
  no-pid recovery arms (probe match => adopt; probe empty => never
  started, relaunch).
- D4 (#2006): both pre-merge verdict conditionals carry the re-compose
  ban (the improvised `grep -qxE ... <(sed ...)` form) plus the
  whole-file-grep consequence note, with the pinned `sed -n 2p`
  conjuncts byte-identical.
- D5 (#2087, measured `probe-guard1-restore.txt`): the Guard-1 retry
  restore is PER-DISPOSITION (`R_IN`/`R_GONE`) — the batched
  `checkout HEAD -- "${FOREIGN[@]}"` form (which aborts entirely on one
  unmatched path and restores NOTHING) is gone; the in-HEAD batch keeps
  the `checkout HEAD --` verb (the #897 hook-admitted form —
  `test_step10d_guard3.py` bans documenting the one-shot
  restore-with-worktree-flag form anywhere in the SKILL, so the plan's
  named fallback verb is the shipped form: the per-disposition split,
  not the verb, is the load-bearing part).
- #2317 (#2314): the 1b block additionally guards splice SHAPE — the D1
  count check validates cardinality only, and a newline-separated
  `--files-only` list PASSES it while the launcher's inner `bash -c`
  parses paths 2..N as their own commands (pytest ran 1 of 120 files,
  no junit XML, rc=126). Pins: the `tr`-to-spaces source normalization,
  the `wc -l` embedded-newline assertion at the splice (with a shell
  self-test), a file-wide ban on the broken match-everything case/glob
  form, plus a BEHAVIORAL extract-and-execute test proving the shipped
  assertion actually discriminates.

Regions are anchor-to-anchor spans (never fixed-length windows), so
additive neighboring edits cannot silently push a pinned token out of
scope (the fixed-window fragility named in #2126's plan deferral 5).

#2296 extends the file with one further gate-recipe pin (same surface,
same span convention): the Step 10d mapped-invariant BASELINE pytest legs
run on a detached sparse scratch tree via `step9c_baseline.py
mapped-baseline` — never with cwd inside the shared repo root, which the
#2015 pre-commit stash cycle reverts repo-wide (measured kill: #2288 RUN1).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

SKILL = Path(__file__).resolve().parents[1] / ".claude/skills/issue/SKILL.md"


def _text() -> str:
    return issue_skill_text()


def _span(text: str, start_anchor: str, end_anchor: str) -> str:
    start = text.index(start_anchor)
    end = text.index(end_anchor, start)
    return text[start:end]


def _norm(text: str) -> str:
    """Collapse whitespace: the SKILL wraps prose mid-phrase, so a required
    phrase can span lines (same convention as the sibling harvest pin)."""
    return re.sub(r"\s+", " ", text)


def _section_9c(text: str) -> str:
    return _span(text, "9c. Test-verdict gate", "### Step 10: Auto-complete")


# --- D1: 1b gate-set cross-check + 1a selector-key pin ------------------------


def test_step9c_launch_cross_checks_gate_set():
    """The 1b block compares the substituted <files> count against the
    selector's live count and REFUSES on mismatch, BEFORE the launcher —
    catching the empty, whitespace-only, unsubstituted-placeholder, AND
    short/stale substitution shapes of the #1992 wrong-key channel."""
    sec = _section_9c(_text())
    # The selector-side live count, keyless:
    assert "S9C_N=$(uv run python scripts/select_step9c_tests.py --files-only" in sec
    # The substituted-set count:
    assert "S9C_GOT=$(printf" in sec
    # The count-equality refusal, naming the whole-suite consequence:
    assert '[ "${S9C_GOT:-0}" -eq "${S9C_N:-0}" ]' in sec
    assert "a bare pytest collects the WHOLE suite" in sec
    # A failed selector call refuses the launch too (never a silent 0):
    assert "FATAL: selector failed — do NOT launch the gate" in sec
    # The refusal PRECEDES the launcher line (the junitxml token is
    # launcher-specific; the 1a prose mention of `uv run pytest <files>`
    # carries no junit path):
    cross_idx = sec.index('[ "${S9C_GOT:-0}" -eq "${S9C_N:-0}" ]')
    launcher_idx = sec.index("--junitxml=/tmp/step9c-junit-issue-<N>.xml")
    assert cross_idx < launcher_idx, "gate-set cross-check must precede the launcher"
    # The launcher literal `test_ensemble_review_cap.py` pins is UNTOUCHED —
    # the two pins agree instead of conflicting (#2126 plan blocker 1):
    assert "uv run pytest <files>" in sec


def test_step9c_selector_json_key_pinned():
    """The 1a prose names the `--json` list key (`tests`, count `n_tests`)
    and points a self-composing launcher at `--files-only` — no key to
    guess (#1992)."""
    text = _text()
    sec_1a = _norm(_span(text, "9c. Test-verdict gate", "b. Run the printed command"))
    assert "under key **`tests`**" in sec_1a, "1a must pin the --json list key"
    assert "`n_tests`" in sec_1a
    assert "--files-only" in sec_1a
    assert "#1992" in sec_1a
    # NEGATIVE: nothing may name `files` as the list key (the wrong key the
    # #1992 launcher guessed; the prose may only mention it as the wrong key):
    assert "under key **`files`**" not in sec_1a
    # The 1b refusal names the correct key inline, where the operator reads it:
    sec = _section_9c(text)
    assert "the list key is 'tests'" in sec


# --- D2: the pre-gate re-sync binds every (re)launch ---------------------------


def test_pregate_resync_binds_every_relaunch():
    """The #1742 paragraph carries the every-relaunch scope clause; the four
    operator-facing next-action messages say re-sync-then-re-run; the Step
    10d single-flight paragraph carries the companion sentence."""
    text = _text()
    para = _norm(
        _span(
            text,
            "Pre-gate spec-freshness re-sync (#1742): AFTER",
            "uv run python scripts/select_step9c_tests.py",
        )
    )
    assert "before EVERY gate re-launch" in para
    assert "re-sync-then-relaunch, never a bare relaunch" in para
    assert "#2006" in para
    assert "9c and Step 10d alike" in para
    # The four operator-facing FATAL/BLOCKED messages name the re-sync as the
    # next action (2x rc/verdict-missing FATAL + 2x BLOCKED merge/push; these
    # are single-line echo strings — no wrap, so no _norm):
    assert text.count("re-sync (§ pre-gate re-sync), then re-run the gate ONCE") >= 4
    # Step 10d companion sentence, on the single-flight paragraph:
    tend = _norm(
        _span(
            text,
            "**Single-flight probe (#1606) — before (re)launching this gate",
            "```bash",
        )
    )
    assert "re-run the same Step 5a family-atomic block" in tend
    assert "never a third inlined `FAMILY_OF` copy" in tend


# --- D3: detached stdout-redirect rule + no-pid adopt recovery -----------------


def test_detached_stdout_redirect_rule_and_adopt_recovery():
    """The detached-launch recipe states the measured fd-1 mechanism, the
    timeout caveat, carries the bounded wrapper + no-pid warn, and names
    BOTH recovery arms."""
    raw = _span(
        _text(),
        "**Detached VM-side long compute phases",
        "**Probe-bracket rule",
    )
    block = _norm(raw)
    # The hardened launcher: timeout-bounded wrapper (the thread-cap prefix
    # stays intact for the check_vm_thread_cap_guidance count floor — assert
    # on the RAW span: the prefix must stay unsplit on one physical line):
    assert "timeout 60 bash -c 'setsid nohup env OMP_NUM_THREADS=8" in raw
    # The pid-emptiness check replaces silent trust:
    assert '[ -n "$PHASE_PID" ] ||' in raw
    assert "[warn] launcher returned NO pid" in raw
    # The measured mechanism: stdout inside the $( ) holds the substitution:
    assert "Stdout-redirect rule (#2126" in block
    assert "holds the substitution open" in block
    # The caveat: timeout bounds only a pre-fork wedge, not the fd hang:
    assert "does NOT bound the inherited-fd hang" in block
    assert "wedges BEFORE forking" in block
    # BOTH recovery arms (probe match => adopt; probe empty => relaunch):
    assert "ADOPT the pid and re-emit the breadcrumb" in block
    assert "the job never started — relaunching is then correct" in block
    assert "#1491" in block


# --- D4: verdict-conditional re-compose ban ------------------------------------

# The pinned conjuncts (byte-identical to test_step10d_guard3.py's #1097 pins;
# the #2126 additions are comment-prose ONLY — these literals must survive):
_SHA_CHECK = (
    '[ "$(sed -n 2p /tmp/issue-<N>-lint-verdict.txt 2>/dev/null)"'
    ' = "$(git -C "$WT" rev-parse HEAD)" ]'
)
_NONEMPTY_SHA_CHECK = '[ -n "$(sed -n 2p /tmp/issue-<N>-lint-verdict.txt 2>/dev/null)" ]'
_VERDICT_PROBE = "grep -qxE 'pass|skip-artifact-only' /tmp/issue-<N>-lint-verdict.txt"


def test_verdict_conditional_carries_no_recompose_ban():
    """BOTH pre-merge verdict conditionals (safe case + conflict recovery)
    carry the #2006 re-compose ban and the whole-file-grep consequence note
    in the comment block immediately above the conditional."""
    text = _text()
    first = text.index(_VERDICT_PROBE)
    second = text.index(_VERDICT_PROBE, first + 1)
    for name, idx in (("safe-case", first), ("recovery", second)):
        preamble = text[max(0, idx - 1500) : idx]
        assert "Read this conditional VERBATIM (#2006)" in preamble, (
            f"[{name}] verdict conditional lacks the #2006 re-compose ban"
        )
        assert "Do NOT re-compose it" in preamble, name
        assert "process-substitution form" in preamble, name
        assert "scans the WHOLE file" in preamble, (
            f"[{name}] missing the whole-file-grep consequence note"
        )
    # The pinned conjuncts survive byte-identical at both consumers:
    assert text.count(_SHA_CHECK) >= 2
    assert text.count(_NONEMPTY_SHA_CHECK) >= 2


# --- D5: Guard-1 per-disposition retry restore ---------------------------------


def test_guard1_retry_restore_is_per_disposition():
    """The Guard-1 retry restore splits in-HEAD vs absent-from-HEAD paths
    (a batched pathspec op aborts ENTIRELY on one unmatched path, #2087 —
    measured: the old form restored NOTHING)."""
    text = _text()
    region = _span(text, "GUARD1_STATE=strip-failed", "GUARD1_STATE=ok")
    assert "PER-DISPOSITION (#2126)" in region
    assert "R_IN=(); R_GONE=()" in region
    assert 'cat-file -e "HEAD:$p"' in region, "the split keys on HEAD existence"
    # The in-HEAD batch restores via checkout HEAD (index AND working tree;
    # the #897 hook-admitted form) — never an unmatched pathspec, because
    # the split gated every member on HEAD existence:
    assert 'checkout HEAD -- "${R_IN[@]}"' in region
    assert 'rm -f -q --ignore-unmatch -- "${R_GONE[@]}"' in region
    # The untracked-litter sweep for branch-deleted paths (the FORM C
    # `?? vanished.txt` residue measured in probe-guard1-restore.txt part A):
    assert 'for p in "${R_GONE[@]}"; do rm -f -- "$WT/$p"; done' in region
    # NEGATIVE: the batched abort-prone form is gone from the whole file:
    assert 'checkout HEAD -- "${FOREIGN[@]}"' not in text


# --- #2317: 1b splice-shape guard (source-normalize + wc -l at the splice) ------


def _extract_splice_shape_assertion(sec: str) -> str:
    """Pull the LIVE splice-shape assertion out of the 9c section — the
    physical line probing $S9C_FILES for embedded newlines, plus its
    backslash-continuation lines. Extract-and-execute is load-bearing, not a
    style choice: a hardcoded copy of the assertion shape would pass against
    a PRE-fix SKILL.md, making the fails-before/passes-after criterion
    (#2317 §6.2) vacuous — extracting ties the test to the SHIPPED guard."""
    lines = sec.splitlines()
    for i, ln in enumerate(lines):
        if '"$S9C_FILES" | wc -l' in ln:
            snippet = [ln]
            j = i
            while lines[j].rstrip().endswith("\\") and j + 1 < len(lines):
                j += 1
                snippet.append(lines[j])
            return "\n".join(snippet)
    raise AssertionError(
        "splice-shape assertion not found in the 9c section — the 1b block "
        "does not probe the substituted $S9C_FILES for embedded newlines"
    )


def _run_bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)


def test_step9c_launch_asserts_splice_shape():
    """The 1b block normalizes the --files-only sourcing route to spaces at
    the SOURCE and asserts the substituted <files> carries no embedded
    newline at the SPLICE (#2314: a newline-separated 120-path list PASSED
    the D1 cardinality cross-check — IFS includes newline — while the
    launcher's inner bash -c parsed paths 2..120 as their own COMMANDS;
    pytest ran 1 of 120 files, the trailing flags incl. --junitxml attached
    to the last bogus line, rc=126 landed in the sentinel)."""
    text = _text()
    sec = _section_9c(text)
    norm = _norm(sec)
    # (a) the source normalization on the --files-only route:
    assert "--files-only | tr '\\n' ' '" in sec
    # (b) the splice-shape assertion — embedded-newline count must be 0:
    shape_probe = '[ "$(printf \'%s\' "$S9C_FILES" | wc -l)" -eq 0 ]'
    assert shape_probe in sec
    # Its FATAL names the multiple-commands consequence:
    assert "contains NEWLINES" in norm
    assert "its own COMMAND" in norm
    # (c) the shell self-test's two conjuncts (a broken newline test that
    # never fires is worse than no test — #2314's second launch attempt):
    assert "[ \"$(printf 'a b' | wc -l)\" -eq 0 ]" in sec
    assert "[ \"$(printf 'a\\nb' | wc -l)\" -eq 1 ]" in sec
    assert "splice-shape assertion is itself broken" in norm
    # Ordering: the shape check PRECEDES the launcher:
    assert sec.index(shape_probe) < sec.index("--junitxml=/tmp/step9c-junit-issue-<N>.xml")
    # The count check is RETAINED beside it (D4 — shape and cardinality are
    # independent failures):
    assert '[ "${S9C_GOT:-0}" -eq "${S9C_N:-0}" ]' in sec
    # NEGATIVE, FILE-WIDE: the broken case/glob form — a pattern wrapping a
    # command substitution that emits a bare newline — appears NOWHERE in the
    # SKILL (command substitution strips trailing newlines, so that pattern
    # degenerates to a match-everything glob; the trap comment PARAPHRASES it):
    assert "$(printf '\\n')" not in text


def test_splice_shape_assertion_actually_discriminates():
    """BEHAVIORAL, not a text pin: extract the assertion from the recipe and
    EXECUTE it under bash — it must fail loud (nonzero exit, FATAL on
    stderr) on a newline-joined list and stay silent on a space-joined
    list; the silently-swallowed newline is exactly the #2314 failure.
    Third leg: the broken case/glob form (constructed locally — it must not
    exist in SKILL.md) matches BOTH shapes, the regression that makes the
    readable-looking form unusable. This test is the committed fail-loud
    backing for #2317 §1."""
    assertion = _extract_splice_shape_assertion(_section_9c(_text()))
    # Negative control — a space-joined list passes silently:
    ok = _run_bash(f'S9C_FILES="tests/test_a.py tests/test_b.py"\n{assertion}\n')
    assert ok.returncode == 0, (ok.returncode, ok.stderr)
    assert "FATAL" not in ok.stderr
    # The #2314 shape — a newline-joined list FAILS LOUD:
    bad = _run_bash(f'S9C_FILES="tests/test_a.py\ntests/test_b.py"\n{assertion}\n')
    assert bad.returncode != 0, "newline-bearing <files> must refuse the launch"
    assert "FATAL" in bad.stderr
    assert "NEWLINES" in bad.stderr
    # The broken glob form fires on BOTH shapes (command substitution strips
    # the trailing newline -> empty string -> match-everything pattern), so
    # it cannot discriminate — constructed here because SKILL.md must not
    # carry it:
    broken_pat = '*"$(printf ' + "'\\n'" + ')"*'
    for value in ("tests/test_a.py tests/test_b.py", "tests/test_a.py\ntests/test_b.py"):
        r = _run_bash(
            f'S9C_FILES="{value}"\ncase "$S9C_FILES" in {broken_pat}) exit 7;; *) exit 0;; esac\n'
        )
        assert r.returncode == 7, (value, r.returncode, "broken glob must match everything")


# --- #2296: mapped-invariant BASELINE legs run off the shared repo root --------


def test_mapped_baseline_leg_runs_off_shared_root():
    """#2296 (A1/A4/A5 + the §5.3 sed and §5.5 residual-prose invariants):
    the Step 10d mapped-invariant BASELINE pytest runs on a detached sparse
    scratch tree cut at the resolved landing base (`step9c_baseline.py
    mapped-baseline`) — never with cwd inside the shared repo root, which
    every fleet commit's pre-commit stash cycle reverts repo-wide (#2015;
    measured kill + false-NEW classification: #2288 RUN1). Span-scoped to
    the two BASELINE legs: form (iii)'s GATED leg legitimately keeps its
    repo-root cwd (the surgical payload lands in the root tree), so a
    file-wide ban on a root-scoped pytest would be wrong."""
    text = _text()
    # (1) + (2): both BASELINE legs are the helper call; neither runs pytest
    # directly (span = leg comment anchor -> the next leg / checkout anchor).
    shared_base = _span(
        text,
        "# BASELINE leg — a DETACHED SPARSE SCRATCH tree",
        "# GATED leg — worktree copy",
    )
    surgical_base = _span(
        text,
        "# BASELINE leg — base-pinned scratch (#2296",
        "xargs -r -a /tmp/issue-<N>-additive-files.txt",
    )
    for name, span in (("shared", shared_base), ("surgical", surgical_base)):
        assert '"$TG_S9B" mapped-baseline' in span, (
            f"[{name}] baseline leg must invoke the mapped-baseline helper"
        )
        # Bootstrap resolution (#2296): the helper must NOT be pinned to a
        # hardcoded copy. The round that ADDS `mapped-baseline` has it only on
        # the branch, so a hardcoded "$REPO_ROOT/..." invocation dies argparse
        # exit 2 -> empty rc -> TG_CRASH, and the change can never pass through
        # the gate it introduces (measured on #2296's own first Step 10d run:
        # a 0-byte tg-baseline.txt that reads as a green baseline unless you
        # check the helper's stderr).
        assert 'uv run python "$TG_S9B" mapped-baseline --help' in span, (
            f"[{name}] the bootstrap helper-resolution probe must survive"
        )
        # #2348: the probe must ALSO require the classify-new-nodes subcommand,
        # so a root copy predating it routes to the worktree fallback — the
        # SAME bootstrap doctrine as mapped-baseline: without this, the round
        # that ADDS the subcommand (and every root copy older than it) would
        # drive a classify call that dies argparse exit 2 into the loud-WARN
        # status-quo arm on every landing until the root catches up.
        assert 'uv run python "$TG_S9B" classify-new-nodes --help' in span, (
            f"[{name}] the bootstrap probe must cover the classify-new-nodes subcommand (#2348)"
        )
        # Quote-form-agnostic (reviewer Minor, round 2): the earlier literal
        # was keyed to the double-quoted spelling, so an UNQUOTED-path
        # hardcode would have evaded the pin.
        assert not re.search(r'step9c_baseline\.py"?\s+mapped-baseline', span), (
            f"[{name}] baseline leg must not hardcode a helper copy — resolve "
            "via $TG_S9B (root preferred, worktree fallback)"
        )
        # The resolution must stay AUDITABLE: a fallback-driven baseline is
        # auditable-not-guaranteed (rc= and the FAILED set are driver-produced),
        # so the distinguishable source token is load-bearing, not cosmetic.
        assert "TG_S9B_SRC=FALLBACK-worktree" in span, (
            f"[{name}] the fallback must announce itself with a distinguishable token"
        )
        assert "uv run pytest" not in span, (
            f"[{name}] baseline leg must not run pytest directly (A1): the "
            "shared root is reverted repo-wide by the #2015 stash cycle"
        )
        assert 'cd "$REPO_ROOT"' not in span, f"[{name}] baseline leg cwd must not be the root"
        # Fail-closed rc parse (a missing/unparseable rc= line is crash-class):
        assert "TG_BASE_RC=0; TG_CRASH=yes; }   # fail CLOSED" in span, (
            f"[{name}] the missing-rc fail-closed arm must survive"
        )
    # (3) A4: the GATED legs survive untouched — shared runs the $WT copy,
    # surgical the ROOT copy (payload lands in the root tree there).
    shared_gated = _span(text, "# GATED leg — worktree copy", "TG basetemp")
    assert '( cd "$WT" && timeout --kill-after=30s ${TG_T}s' in shared_gated
    assert 'uv run pytest "${TG_TESTS[@]}"' in shared_gated
    surgical_gated = _span(
        text,
        "# TG GATED leg (#1147) — the root tree now carries the payload",
        "# TG basetemp reaped after BOTH legs",
    )
    assert '( cd "$REPO_ROOT" && timeout --kill-after=30s ${TG_T}s' in surgical_gated
    assert 'uv run pytest "${TG_TESTS[@]}"' in surgical_gated
    # (4) The <TREE> sed carries the scratch clause with a never-matching
    # default, in BOTH subtraction pipelines — omitting it inverts the
    # verdict (every both-trees red would read NEW):
    assert text.count('sed -e "s|${TG_SCRATCH:-/__eps_no_scratch__}|<TREE>|g"') == 2
    # The crash-class fold survives in both blocks (rc>1 on either leg):
    assert (
        text.count('if [ "$TG_RC" -gt 1 ] || [ "$TG_BASE_RC" -gt 1 ]; then TG_CRASH=yes; fi') == 2
    )
    # (5) The falsified one-directional-dirt residual claims are gone; the
    # ONE legitimate `always-dirty shared root` survivor is the
    # sync_repo_root.py autostash paragraph (NOT a mapped-baseline claim):
    assert "can only ENLARGE" not in text
    assert "dirty-root baseline" not in text
    assert text.count("always-dirty shared root") == 1
    survivor_idx = text.index("always-dirty shared root")
    survivor_ctx = text[max(0, survivor_idx - 300) : survivor_idx + 300]
    assert "autostash" in survivor_ctx, (
        "the surviving 'always-dirty shared root' mention must be the "
        "sync_repo_root.py autostash paragraph, not a baseline-leg claim"
    )


# --- #2310: completion-read list-shape attribution ----------------------------


def test_step9c_completion_read_attributes_list_shape():
    """The completion-read carries a list-shape attribution branch (#2310):
    rc=126 conjoined with per-path Permission-denied lines is the
    newline-splice signature, and the verdict must name the LIST SHAPE as
    the cause instead of reporting an opaque mass failure. Defense-in-depth
    behind the #2317 pre-launch splice-shape guard (a stale worktree recipe
    copy bypasses that guard; the completion-read is the last reader)."""
    sec = _section_9c(_text())
    # Detection predicate: rc=126 conjunct + path-scoped Permission-denied grep.
    attr_idx = sec.index('"${PYTEST_RC:-}" = "126"')
    assert "Permission denied" in sec[attr_idx : attr_idx + 400]
    # The attribution names cause + remedy (wrap-tolerant via _norm):
    n = _norm(sec)
    assert "LIST-SHAPE failure, not a test failure" in n
    # Placement: inside the rc-exists branch — after the rc read and the
    # zero-collected guard, before the BASETEMP reap:
    rc_read_idx = sec.index("PYTEST_RC=$(cat /tmp/step9c-rc-issue-<N>)")
    zero_idx = sec.index("no tests ran|collected 0 items")
    reap_idx = sec.index("# BASETEMP reap — INSIDE the rc-exists branch")
    assert rc_read_idx < attr_idx, "attribution must follow the rc-file read"
    assert zero_idx < attr_idx, "attribution must follow the zero-collected guard"
    assert attr_idx < reap_idx, "attribution must precede the BASETEMP reap"
    # Existing pinned guards untouched (acceptance criterion 4):
    assert "no tests ran|collected 0 items" in sec
    assert "[ ! -f /tmp/step9c-rc-issue-<N> ]" in sec
