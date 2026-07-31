#!/usr/bin/env bash
# PreToolUse(Bash) guard: block branch-switching AND working-tree reverts in
# the SHARED repo-root tree.
#
# The repo root (/home/thomasjiralerspong/explore-persona-space) is the
# canonical commit target for scripts/task.py and every concurrent VM Claude
# session — they all assume the working tree is on `main`. Running
# `git checkout -b` / `git switch` here moves the branch out from under those
# concurrent committers: their commits land on the feature branch, and a
# concurrent `git add <file> && git commit` sweeps THIS session's uncommitted
# edits to <file> into the wrong commit. A working-tree REVERT here (`git
# restore`, a pathspec / bare-path / force `git checkout`, `git clean -f`,
# `git reset --hard`) is just as destructive: it silently discards CONCURRENT
# sessions' uncommitted edits and untracked files (#897). A branch MERGE here
# (`git merge <ref>`) is in the same destructive class (#1128): a conflicting
# merge strands conflict markers in the shared tree until aborted, and even a
# clean/ff merge lands branch commits on root main outside the sanctioned
# landing path (gh pr merge / scratch worktree). A root REBASE / CHERRY-PICK
# (#1193) is the same class — conflict state stranded in the shared tree,
# history rewritten under concurrent committers. A root REVERT / AM (#1234)
# completes the family — the same sequencer/conflict-stranding class, commits
# landed on root main outside the sanctioned landing paths.
#
# Incident 2026-06-01: an infra session ran `git checkout -b fix/sweep-ckpt-persist`
# in the repo root; a concurrent marker-leakage session's CLAUDE.md commit then
# bundled the infra session's Upload-Policy paragraph, and task #459 state landed
# on the feature branch.
# Incident 2026-07-01 (#815): a #778 analyzer's improvised repo-root
# `git reset --hard` clobbered concurrent siblings #812/#813's task state.
# Incident 2026-07-02 (#841): a concurrent destructive working-tree op on the
# shared root reverted the #841 analyzer's uncommitted body.md mid-task and
# deleted untracked pre-registration + figure files.
# Incident 2026-07-08 (#1090 -> #1128): a branch merge run at the SHARED repo
# root conflicted on 2 files, leaving conflict markers in the shared tree for
# ~70s until aborted — a concurrent session staging those files in that
# window would have swept markered content into its commit.
#
# Fix: do feature/infra branch AND destructive work in a dedicated worktree:
#     bash scripts/new_worktree.sh .claude/worktrees/<name> <branch>
#     git -C .claude/worktrees/<name> ...
#
# #1554 (#1530 F5): worktree-scoped LOCAL-main merge fence. The two worktree
# allow paths — the per-clause `git -C <path>` waiver and the `cd <worktree>
# &&` latch — deliberately pass ALL worktree-internal git ops, including
# `git -C <worktree> merge main`, which fast-forwards the branch onto the
# possibly-stale, UNPUSHED local `main` tip and imports root-only commits
# (#1530 imported 4). Two narrow driver-loop arms now decline a clause-initial
# worktree-scoped merge of the BARE `main` ref only: Arm A intercepts
# `git -C <.claude/worktrees/ path | $WT spelling> ... merge ... main` BEFORE
# the path-blind -C waiver; Arm B declines the same bare-main merge shape when
# the live cd-latch was armed by a WORKTREE cd (the `scoped_wt` bit; /tmp
# latches keep their disposition byte-identical). `origin/main`, raw-sha, and
# "$MAIN_SHA" merges pass unchanged (the worktree fast-forward recipe in the
# deny text is the sanctioned form). Deliberate override:
# EPM_ALLOW_WORKTREE_LOCAL_MAIN_MERGE=1 — session env or inline command
# prefix; the inline form is a command-wide SUBSTRING match (inherited
# verbatim from the EPM_ALLOW_ROOT_PULL sibling idiom), so a command merely
# QUOTING the override string disarms the fence for that command. Residuals:
# gap (xx) below.
#
# #1861: exit-guarded worktree cd => STICKY scope; name-generalized $VAR
# latch; arming-separator restriction + cd-clause scope invalidation.
# (i) `cd <worktree> || exit N` and `cd <worktree> || { ...; exit N; }`
# grant a STICKY scope to every clause PAST the provably-exiting guard tail:
# either the cd succeeded (cwd IS the worktree) or the shell exited before
# any later clause ran. Activation is DEFERRED past the terminator clause's
# index, so the OR-tail's own clauses run UNSCOPED — they execute exactly on
# the cd-failure path, at the repo root (a gated git op inside the guard
# group still classifies and blocks). A `return` tail is rejected (at top
# level — the only context these hook-scanned command strings execute in —
# bash prints an error and CONTINUES at the root); a terminator whose own
# following separator is PIPE/BG is rejected (`|| exit 1 | op` runs the exit
# in a pipeline subshell; `|| exit 1 & op` backgrounds the whole and-or
# list — both leave the parent running at the root on cd failure). The #1554
# Arm B bare-local-main merge fence applies under sticky scope unchanged.
# (ii) The #1058 `WT=` + `cd "$WT"` latch is generalized to ANY variable
# name (assoc array `_wt_names`, same three arming proof obligations);
# DISARM is deliberately broader than arming — declare/local/typeset/
# readonly-prefixed and `NAME+=` assignment shapes count as reassignment
# and unbind the name. (iii) Fail-closed hardening shipped in the same diff:
# an OR- or PIPE-preceded cd never arms ANY latch (`a || cd X && op` runs op
# with the cd skipped when a succeeded; `a | cd X` cds in a pipeline
# subshell), and ANY later cwd-changing clause — cd/pushd/popd, tolerating
# leading `(`/`{`/whitespace and builtin/command prefixes — voids BOTH the
# plain latch and the sticky scope (invalidation is broader than arming: a
# paren-prefixed cd invalidates but never arms). Residuals, accepted
# (non-adversarial threat model, #1861): (a) quote-blind Case-B acceptance —
# the splitter does not parse quotes, so an exit-shaped FRAGMENT inside
# quoted brace-group text can satisfy the group recognizer and grant sticky
# (same class as residual (viii)); (b) interpreter-payload cds — after a
# sticky grant, `bash -c 'cd <root>; <gated op>'` splits quote-blind and the
# payload's op fragment classifies as a sticky-scoped clause while the
# subprocess executes it at the payload cd's target (narrow: requires a
# prior exit-guarded worktree cd in the SAME command; documented, not
# refused).
#
# THREAT MODEL / scope (#897): this hook gates ONLY Claude Bash tool calls.
# Git run inside Python subprocesses (sync_repo_root.py carries its own
# internal deny checker), cron wrappers, pod-side scripts, and SSH-MCP remote
# commands never pass PreToolUse; non-git destruction (`rm`, `mv`, `>`
# truncation, `sed -i`) is out of scope by design. CWD-BLINDNESS: the hook
# never consults the caller's actual $PWD — a session whose cwd genuinely IS a
# worktree running a bare `git restore .` is still blocked and should use
# `git -C <worktree>` (the designed deliberate-override). Bash("ssh ...")
# edge (#1098): a SINGLE-STATEMENT remote command string carrying a gated git
# verb (`ssh pod-779 'git checkout HEAD -- <file>'` — the gotchas.md
# diverged-pod recovery, executed in the pod's own /workspace clone) is
# WAIVED per-clause by the driver-loop ssh/grep waiver below, under its
# fail-closed refusal ladder; grep/egrep/fgrep/rg pattern arguments get the
# same waiver — including (#1538) in pipeline-PRODUCER position, when every
# downstream pipe-connected consumer clause is a VERIFIED read-only text
# filter (the _pipe_chain_is_readonly_sink() walker: allowlisted stdin->
# stdout words, no expansion / redirect / write-exec channel, chain ends on
# a non-PIPE non-BG seam; the ssh arm's pipe refusal stays consumer-
# independent). The grep-family waiver has never had a final-token /
# trailing-argument condition — a trailing file path after the pattern is
# waived by the base arm. A MULTI-STATEMENT SINGLE-QUOTED remote string in the canonical
# shape — the quoted payload is the clause's FINAL token, and the whole
# candidate passes the 8-arm refusal predicate of the
# mask_ssh_payload_separators() pre-pass (#1413: clean tail, no quote or
# latch-arming vocabulary before it, no expansion/redirect/local-exec/
# repo-path/WT= token) — has its intra-payload separators masked BEFORE the
# split, so the merged clause reaches the same waiver whole (closes the
# founding #779 false block). A clause-initial `gcloud compute ssh
# <instance> --command=...` head gets BOTH mechanisms too (#1463; the
# optional literal `timeout <N>` wrapper is accepted on BOTH remote-exec
# heads as of #1859): gcloud is a thin wrapper
# around the local ssh(1) binary whose --command payload executes ON the
# GCE instance, so the waiver + mask treat it exactly like the ssh head
# under the identical refusal arms (founding incident #825; details gap
# (xix)). Every OTHER multi-statement shape —
# double-quoted payloads, redirect-carrying payloads, trailing tokens after
# the closing quote, non-timeout-wrapped/variable/abs-path ssh, quoted or
# latch-vocabulary prefixes — still mis-splits on the quoted-separator
# trade-off and its tail clauses still classify — remediation unchanged:
# `git -C /workspace/... <verb>` inside the remote string, a pod-side
# script, or the SSH MCP (which bypasses the Bash hook
# entirely). Never waived, deliberately: an ssh clause naming the shared-repo
# path in any covered spelling (literal / $HOME/ / ~/ + the repo basename),
# and ANY waived-word clause in pipeline-producer / background position or
# carrying a non-/dev/null output redirect (`> f` — the same-call
# write-then-execute channel, closed round 2).
#
# Contract: reads the PreToolUse JSON on stdin, blocks (exit 2 + stderr fed
# back to Claude) only when a branch-CHANGING git command would move the
# repo-root tree off `main`. Exit 2 is the documented PreToolUse blocking
# exit code; any OTHER non-zero is non-blocking (stderr goes to the user and
# the tool call PROCEEDS) — code.claude.com/docs/en/hooks: "If your hook is
# meant to enforce a policy, use exit 2."
# Fail-soft: any ambiguity / parse failure exits 0 (never traps the user).
# Deny-event sidecar (#1528): every deny best-effort-appends ONE JSON row
# {ts, guard, arm, len, head, clause_head} to EPM_GUARD_DENY_SIDECAR (default
# $REPO/.claude/cache/guard-deny-events.jsonl); bounded redacted heads only,
# and every append failure is swallowed — deny/allow, exit codes, and the
# stderr message are never affected.
#
# <!-- known limitation -->
# Every detector scans the RAW command string — the guard does NOT strip
# quoted arguments before parsing. A quoted git-verb literal buried in
# ANOTHER command's argument therefore trips the guard: e.g.
# `task.py post-marker <N> epm:X --note "... git switch ..."` is blocked
# because the note text matches `git ... switch`, and a note/-m string
# carrying a full `git restore .` / `git clean -fd` / `git reset --hard` /
# `git merge <branch>` (#1128) / `git rebase <branch>` / `git cherry-pick
# <sha>` (#1193) / `git revert <sha>` / `git am <path>` (#1234) command
# literal trips the #897/#1128/#1193/#1234 detectors
# the same way. The workaround is to
# pass such note text via `--file <path.md>` instead of `--note`, and commit
# messages via `git commit -F <file>`. The `git -C <path>` per-clause waiver
# is PATH-BLIND for merge exactly as for every fenced verb (#1128/#1193):
# `git -C <repo-root-path> merge <ref>` passes the hook (pre-existing
# parity); the block message's "NEVER point -C at the repo root" line is the
# stated control. One NARROWING of the raw scan (#1058):
# heredoc BODIES destined for NON-SHELL consumers are stripped before parsing
# by the strip_heredoc_bodies() pre-pass below when provably inert — every
# opener validated, no shell-consumer / command-runner word on the opener
# line, no shell-out spelling in the body (argv-LIST-form call opens with
# non-shell heads carved out of that scan as of #1621; shell-headed lists
# and `shell=True` residuals still refuse), and (for an UNQUOTED tag, whose
# body bash expands at feed time) no expansion syntax beyond plain `${NAME}`
# references (#1501) — and as of #1621 the checkout-detach clause no longer
# matches the verb token inside a `--no-checkout` / `--checkout` FLAG —
# so document text that merely MENTIONS a gated form no longer false-blocks;
# the quoted `--note` literal above is NOT a heredoc and stays blocked (that
# limitation is unchanged). A SECOND NARROWING (#1098): a clause whose
# command word is `ssh` (remote execution) or `gcloud compute ssh` (remote
# execution on a GCE instance) — either optionally `timeout <num>`-wrapped
# (#1463 gcloud, #1859 ssh) — or `grep`/`egrep`/`fgrep`/`rg`
# (read-only pattern) is waived per-clause by the driver-loop waiver under
# fail-closed refusals; quoted git-verb literals under OTHER command words
# (`--note`/`-m` strings) still block with the same `--file`/`-F`
# remediation. A THIRD NARROWING (#1566): balanced SINGLE-QUOTED argument
# payloads of a clause-initial python-script invocation whose path token
# ends in `task.py` (optional `uv run` / `python` prefix; the boundary is
# ANY `*task.py`-suffixed script, so `codex_task.py` / `file_infra_task.py`
# are covered too — a future in-repo `*task.py` helper that shells out its
# argv must revisit the mask_taskpy_arg_payloads() head whitelist as a
# visible design decision, never rely on it silently) are masked to a
# neutral sentinel by the mask_taskpy_arg_payloads() pre-pass below under a
# fail-closed P1-P6 refusal predicate mirroring the ssh mask's R1-R8: a
# single-quoted span is inert argv DATA to a non-shell-consumer executable,
# so the canonical `task.py ... --note '<prose naming a git op>'` shape no
# longer false-blocks. Double-quoted payloads, dollar / backslash /
# backquote / redirect shapes, dirty quote prefixes, and latch-vocabulary
# text all refuse (byte-identical -> today's disposition) and keep the
# `--file` workaround. The #897 detectors use a TIGHT
# `git <verb>` bigram anchor (`git [flags] restore|clean|reset|checkout`), so
# plain-English "restore"/"clean"/"reset" inside a `-m` message (e.g.
# `git commit -m "restore defaults"`) does NOT trip — only a full
# `git <verb>` command literal does. EXCEPTION (round-2 concern id
# header-tight-anchor-claim-overbroad): the bare-pathspec EXISTENCE PROBE in
# the branch-arg classifier rides the LEGACY loose anchor
# (`\bgit\b[^;&|]*\bcheckout\b`), so prose containing `checkout <path>` where
# `<path>` names a REAL repo file trips WITHOUT a `git checkout` bigram —
# `git commit -m "fix checkout CLAUDE.md handling"` is BLOCKED (fails
# closed; test-pinned). Same remediation: `git commit -F <file>` / `--file`.
# A quote-strip
# pre-pass was tried (round 1 of #796) and reverted: stripping quoted spans
# BEFORE parsing silently hid REAL quoted git refs (`git checkout "HEAD~1"`,
# `git switch "main"`) from the detectors — a leak of the exact class this
# guard exists to block, and a false positive on quoted return-to-main. A
# shell-syntax-aware strip is not safe to do in a bash regex, so the raw-scan
# behavior (correct on git refs, over-eager on note-text literals) is the
# deliberate trade-off. See #796 round-2 report.
#
# COMMENT TAILS (#897 round 2) are the one exception to the raw scan: bash
# never EXECUTES an unquoted `#` comment tail, but reading it let trailing
# comment text WAIVE an executed destructive command (`git restore . # git -C
# /tmp status` exited 0 — fail-OPEN, the opposite direction from every other
# raw-scan trade-off). Each clause therefore has its whitespace-anchored ` #`
# tail STRIPPED before any latch / waiver / gate / classification read. The
# strip does not shell-parse quotes (deliberately — quote-parity detection
# would let an odd apostrophe before the `#` disable the strip and resurrect
# the spoof), so a QUOTED argument containing ` # ` also truncates: harmless
# for non-gated verbs (`git commit -m "see #841"` loses its tail and stays
# allowed), fail-closed for allows (truncation cannot ADD an allow token),
# and fail-open only for the residual-gap-(viii) shape (`git clean "x # y"
# -f` — the force flag after a quoted ` # ` becomes invisible).
#
# <!-- known residual gaps (#897 + #1058, accepted + documented) -->
# (i)   `git checkout main <path>` (a pathspec after the `main` allow-arm with
#       no `--`) still leaks — a naive detector would false-positive on
#       redirections (`git checkout main 2>/dev/null`); closable post-v1 by
#       applying the bare-pathspec existence probe to the token after the
#       allow-arm.
# (ii)  `git reset --soft/--merge/--keep` (branch-pointer moves / partial tree
#       writes) are not gated.
# (iii) Command substitution `$(git clean -fd)` at TOP LEVEL remains unparsed
#       (pre-existing #804 limitation; quoted separators fail CLOSED).
#       Heredoc bodies are no longer in this gap: the fail-closed
#       strip_heredoc_bodies() pre-pass (#1058) handles them, and it REFUSES
#       to hide an UNQUOTED-tag body carrying expansion syntax beyond plain
#       `${NAME}` references (#1501) — `$(` / backtick / non-plain `${...}`
#       forms; bash expands such bodies at feed time, so
#       those lines still reach the detectors.
# (iv)  Quoted-glob pathspecs (`git checkout '*.md'`) and exotic pathspec
#       magic (`git checkout ':/'`) — the existence probe sees the unexpanded
#       literal and fails soft.
# (v)   `git clean -i` (interactive; near-inert non-TTY),
#       `git -c clean.requireForce=false clean`, `git -c alias.*` smuggling,
#       and `git checkout-index -f -a` (plumbing) are ungated — ~zero
#       accidental probability under the cooperative-agent threat model.
# (vi)  `git stash` stays ALLOWED as the blessed safe alternative
#       (sync_repo_root.py's autostash depends on it). Incident responders:
#       a stash produces the same "file snapped back to HEAD" symptom with
#       the data recoverable via `git stash list`.
# (vii) An ABSOLUTE-path bare pathspec inside the repo (`git checkout
#       /home/.../explore-persona-space/CLAUDE.md`) evades the existence
#       probe — `cat-file -e "HEAD:/abs/..."` fails and `[ -e "$REPO/$arg" ]`
#       concatenates the repo prefix onto the already-absolute path — so the
#       revert is ALLOWED (fail-open) while git reverts the file
#       (round-2 concern id abs-path-bare-pathspec-residual-unnamed;
#       test-pinned). Closable post-v1 by stripping a `$REPO/` prefix from
#       the arg before probing.
# (viii) A QUOTED argument containing whitespace-anchored ` # ` hides
#       everything after it from the detectors (the round-2 comment-tail
#       strip cuts at the first whitespace-anchored `#` without
#       shell-parsing quotes), so `git clean "x # y" -f` is allowed while
#       bash executes a force clean. ~zero accidental probability under the
#       cooperative-agent threat model (requires a quoted argument
#       containing ` # ` AND a destructive flag after it); the alternative
#       (quote-parity comment detection) re-opens the comment-tail
#       allow-spoof on any odd-apostrophe prefix, which is the worse
#       direction.
# (ix)  (#1058) A heredoc body written to a NON-`.sh` file and later EXECUTED
#       (`cat > /tmp/x <<'EOF' ... EOF; bash /tmp/x`, typically across two
#       Bash calls) is no longer caught by the raw scan once the body strips.
#       Accepted: Write-tool-equivalent (the Write tool creates the same file
#       unhooked), a deliberate two-step, ~zero accidental probability; the
#       `.sh`-redirect refusal in the strip ladder covers the likely
#       accidental spelling.
# (x)   (#1058) The deliberate-construction $WT-latch family, NARROWED by the
#       separator-gated arming (the accidental defensive-check shape
#       `[ -d x ] && WT=<worktree>; cd "$WT" && git ...` BLOCKS — an
#       AND-preceded assignment never arms). What remains is deliberate-only:
#       (x-a) a quoted-prose `; WT=<worktree-path>` fragment that the
#       quoted-separator mis-split turns into an armed clause-initial
#       assignment (verified rc=0; requires a quoted string containing
#       `; WT=<worktree-path>` AND a `cd "$WT"` AND a gated git verb in ONE
#       call — the same quoted-separator raw-scan trade-off as the
#       known-limitation paragraph above, cooperative-model accepted); and
#       (x-b) a `WT` value exported from the user's own shell profile —
#       outside the harness contract that shell state never persists across
#       Bash tool calls, which the latch's same-call-assignment proof relies
#       on; and (x-c) a `WT=<worktree>` assignment line inside a compound
#       body the shell never executes (a function DEFINITION body, a
#       false-branch `if`/`then` body) — the clause splitter is a flat
#       stream with no compound-construct tracking, so an NL-preceded
#       assignment inside such a body arms the latch although bash only
#       defines/skips it; requires deliberately wrapping a worktree
#       assignment in a never-run compound AND a `cd "$WT"` AND a gated git
#       verb in one call — the same flat-stream compound blindness the
#       literal cd-latch has always had (parity), cooperative-model
#       accepted.
# (xi)  (#1058) argv-form `subprocess.run(["git", ...])` inside a stripped
#       body — explicitly PRE-EXISTING: the raw scan never saw the argv form
#       before this change either (no `git <verb>` bigram), and
#       python-subprocess git is out of the threat model by design (see the
#       THREAT MODEL paragraph above). (#1621) Check (f) is now CONSISTENT
#       with this declaration: argv-list-form call TEXT with non-shell heads
#       no longer refuses the strip (delete-from-copy carve). Deliberate-
#       construction residuals of the carve, accepted per this class:
#       variable / f-string argv heads, splat-of-split heads, and truthy
#       non-True shell kwargs (`shell=1`) — none has an ACCIDENTAL path.
# (xii) (#1058) A variable-consumer opener (`RUNNER=bash; $RUNNER <<EOF ...`)
#       and a while-read-loop executor (`while read x; do $x; done <<EOF ...`)
#       hide the shell-consumer word behind a variable / loop body, so the
#       strip fires while bash-via-variable executes the body.
#       Deliberate-only constructions, accepted (a `$`-first-word opener
#       refusal was considered and rejected: it would false-refuse every
#       legitimate `$PYTHON - <<'PY'`-style consumer for zero
#       accidental-risk reduction). Same family: build-tool / interpreter
#       stdin consumers with dash-file flags (`make -f -`, `cmake -P -`,
#       `awk -f -`) and interpreter bodies whose shell-out spelling evades
#       the finite body-refusal list — a denylist cannot enumerate every
#       stdin-executing program; feeding a destructive git recipe to such a
#       consumer at the repo root is a deliberate construction, accepted
#       (add further FAIL-CLOSED consumer words as they surface).
# (xiii) (#1058, narrowed #1501) Fail-closed FALSE-POSITIVE notes for the
#       strip (a no-strip keeps CURRENT behavior — the command blocks only
#       when a gated form ALSO appears). Plain `${NAME}` references in an
#       unquoted-tag body are FIXED as of #1501 — check (g) deletes them
#       from a scan COPY before the expansion-syntax refusal, so they no
#       longer refuse the strip. Remaining FPs: non-plain `${...}` forms
#       (`${V:-w}` fallbacks, `${V@P}` transforms, `${a[i]}` subscripts,
#       `${!v}` indirection, `${1}` positional, unclosed `${`) stay
#       fail-closed — remediation: quote the heredoc tag; a bare-dot jq
#       filter (`jq . <<J`) matching the standalone-dot source form (quoted
#       `jq '.'` unaffected); and prose like "the system (Linux) ..."
#       matching the `system *\(` body refusal. (#1621) Check-(f) FPs that
#       remain fail-closed after the argv-list carve: a BARE shell-out-word
#       prose mention (`subprocess` outside a call open — the M4b two-line
#       value-indirection block requires it), a prose `shell=True` mention,
#       non-subprocess paren-bracket-shell text matching the pre-deletion
#       shell-head arm, and a MULTI-LINE argv call (`run(` at line end, list
#       on the next line) — remediation for all: the Write tool.
#       Value-borne vectors that
#       need attacker-controlled variable VALUES (`${var@P}` prompt
#       expansion under default promptvars; arithmetic-subscript injection)
#       remain refused via the non-plain `${` arm and the `$(`-substring
#       match respectively — nothing newly allowed has an execution path,
#       so no new gap entry. The escaped-dollar shape `\${NAME}` is a
#       DELIBERATE member of the newly-allowed set (the deletion regex
#       matches the `${NAME}` substring after the backslash; sound because
#       under an unquoted tag `\$` suppresses expansion entirely — literal
#       text; pinned by the M1m allow fixture).
# (xiv) (#1098) The ssh/grep-family clause waiver's residuals, BOTH sides.
#       Fail-closed FALSE POSITIVES (harmless shapes that stay blocked):
#       multi-statement remote-string FPs are CLOSED for the CANONICAL shape
#       by the mask_ssh_payload_separators() pre-pass (#1413:
#       single-quoted, payload is the clause's final token, ladder-clean,
#       latch-clean, quote-clean prefix — `ssh pod 'cd /workspace/x && git
#       reset --hard'` now merges and waives); the REMAINING multi-statement
#       FPs keep today's disposition, each with the standing remediation
#       (`git -C /workspace/... <verb>` inside the remote string, a pod-side
#       script, or the SSH MCP): double-quoted payloads (`\"`/expansion/
#       nesting make the parse inexact); redirect-carrying payloads (`>`,
#       `<`, `2>&1` — R2's blanket refusal, and the fd-dup `&` still
#       mis-splits as BG); trailing tokens after the closing quote
#       (`2>/dev/null`, `-v`, a second quoted arg — R1); non-timeout-wrapped /
#       absolute-path / variable ssh (`nohup ssh ...`, `/usr/bin/ssh`,
#       `$SSHCMD ...` — not clause-initial `ssh`; the literal
#       `timeout <num>` wrapper is accepted as of #1859, gap (xix));
#       IPv6-bracket hosts
#       (`ssh [::1] '...'` — `[` stops the head scan); a comment-`#` inside
#       the payload (the per-clause comment-tail strip truncates the merged
#       clause — truncation cannot un-waive or create a latch match, but the
#       shape is not guaranteed to merge); ANY quoted text before the ssh
#       candidate (R5's strict any-quote refusal — quoted-prefix compounds
#       keep today's disposition); latch vocabulary in payload or prefix
#       (`cd /tmp/`, `cd ...worktrees...`, `WT=` — the R6/R7/R8 cost
#       residuals). WT-latch arm-SUPPRESSION flips are fail-CLOSED by R8
#       (exotic shapes where a payload fragment's clause-initial `WT=`
#       arming would have disappeared post-refusal go allow->block —
#       acceptable direction). The former internal-`&&`-only latch-arming
#       payload fail-open (`ssh pod 'cd /tmp/x && git reset --hard'`'s
#       FIRST fragment matching the then-unanchored cd-latch grep) is
#       CLOSED by #1443: the latch greps are clause-initial-anchored, so
#       the shape fails CLOSED (blocks — joining the R6/R7/R8 cost-residual
#       FP class, same standing remediations); a non-gated tail (a remote
#       `git fetch`) allows via the verb gate. Other pre-existing FPs, unchanged:
#       `${VAR}` in remote strings, incl. single-quoted
#       forms bash would not expand (the deliberate `${` over-match);
#       ssh clauses naming the shared-repo path in a covered spelling;
#       here-string literals (`grep -q x <<<"...gated..."` — the R8b
#       raw-scan-parity class — and ssh stdin here-strings); and ANY
#       ssh/gcloud-arm clause in pipeline-producer OR background position
#       (`ssh pod '...' | tail`, `ssh pod '...' &`, and `ssh pod '...'
#       2>&1 ...` — the fd-dup's single & mis-splits as BG, hiding a
#       following pipe from the lookahead, so BG refuses too — all stay
#       blocked; remediation: `git -C /workspace/... <verb>` inside the
#       remote string, which the pipe-blind `-C` waiver allows). The
#       GREP-FAMILY pipeline-producer FP class is RETIRED for verified
#       chains as of #1538: a grep/egrep/fgrep/rg pattern clause piped
#       into a chain of allowlisted read-only text filters (head tail wc
#       cat cut tr nl sort uniq grep egrep fgrep rg — the
#       _pipe_chain_is_readonly_sink() walker; GP1-GP7) now waives.
#       PRODUCER-vs-CONSUMER ASYMMETRY (deliberate, pre-existing on the
#       producer side): the unpiped grep-family producer arm keeps its
#       single --pre refusal (cond (5)) while walker CONSUMERS get the
#       fuller per-word channel set (--pre/--hostname-bin/-z/--search-zip,
#       sort -o/--output/--compress-program/-T/--temporary-directory,
#       uniq positional output) — do NOT misread the walker's channel
#       list as producer-side coverage. NEW residual FPs the walker
#       deliberately keeps refusing (each pinned GPN*): the fd-dup
#       mis-split `grep '<gated>' f 2>&1 | head` (GPN18), path-spelled /
#       env-prefixed consumers (`/usr/bin/head` — GPN17), sed / awk /
#       jq / pager consumers (GPN16/GPN21), trailing `&` chains (GPN11),
#       quoted consumer args carrying a separator (the mis-split ends
#       the walk early — fail-closed), quoted consumer command words
#       (GPN23), and ANY `$` or ANY `#` anywhere in a consumer clause
#       (GPN19/GPN20/GPN22 — a bare $VAR could carry a write/exec flag
#       past the static scan, and a comment strip inside a REFUSAL scan
#       would invert the fail-closed direction, so both refuse on the
#       UNSTRIPPED text; a legitimately-commented or variable-
#       parameterized read-only chain refuses and falls through to
#       classification; remediation identical: drop the comment /
#       variable or the pipe, or bound output with `grep -m N`); and ANY
#       waived-word clause carrying a `>`/`>>` output redirect whose
#       target is not exactly /dev/null (round-2 arm, cond (3b): closes
#       the same-call write-then-execute channel `ssh h 'echo <gated>'
#       > /tmp/x; bash /tmp/x`; refused consumer-independently and in
#       EVERY position incl. nextsep=END, so the harmless conveniences
#       refuse too — `grep '<gated>' . -r > results.txt`, a REMOTE-side
#       redirect inside the quoted ssh string (`ssh pod 'git checkout
#       HEAD -- f > /tmp/log'` — the raw scan cannot tell it from a
#       local one), a literal `>` inside a grep PATTERN, a
#       /dev/null redirect flush against the closing quote
#       (`'... 2>/dev/null'`), and an ordinary QUOTED /dev/null target
#       (`> "/dev/null"` / `> '/dev/null'` — the raw strip matches only
#       the unquoted spelling, so the quoted form refuses fail-closed);
#       remediation: drop the redirect, target
#       /dev/null unquoted with a whitespace/EOL boundary (`2>/dev/null`
#       stays waived), or `git -C`). Also
#       here: `git grep '<gated>'`
#       clauses stay blocked (clause-initial word is `git`, not
#       grep-family) — remediation: plain `grep`. QUALIFIER resolved by
#       #1443: the "tail clauses still classify" sentence is now exact for
#       latch-vocab FIRST statements too — the anchored driver latch never
#       arms on a payload fragment (`ssh pod 'cd /tmp/ && git clean -fd'`),
#       so the gated tail classifies and BLOCKS fail-closed (it was rc=0
#       via the then-unanchored latch, not via this waiver).
#       Deliberate-only accepted FAIL-OPENS (~zero accidental probability,
#       cooperative-agent model, same register as gaps (v)/(x)/(xii)):
#       `ssh -F <config> host` whose ProxyCommand/LocalCommand lives in
#       the config FILE (no on-command-line token; inline `-o` spellings
#       ARE refused); NON-CANONICAL shared-repo path spellings beyond the
#       three covered globs (doubled slash `/home/...//<repo>`,
#       $USER-composed paths, a `cd /home/...` + relative `--git-dir`
#       split across the mis-split boundary); variable-indirection
#       same-VM targeting (`ssh $HOST '...'` with the repo path held in a
#       remote-side variable — resolving remote host identity / variable
#       values is outside a raw-scan design by construction); and the
#       CROSS-CALL write-then-execute form ONLY (write the recipe in one
#       tool call, execute it in the next — a per-call hook cannot see
#       across calls by construction; the gap-(xii) heredoc-to-file strip
#       is the named cross-call writer). The SAME-CALL redirect channel
#       is NOT in this register: it is CLOSED by the round-2 cond (3b)
#       refusal above, and a same-call gated write under a NON-waived
#       command word (`echo git reset --hard > /tmp/x`) was always
#       blocked by the raw scan. The heredoc
#       asymmetry is deliberate: a single-clause remote git op is waived
#       while a heredoc body fed to `ssh host bash` stays blocked (C8) —
#       shellish() strips only provably-inert DATA, and a body handed to a
#       remote shell IS executed.
# (xv)  (#1128) `git commit` during an in-progress root merge COMPLETES the
#       merge (the ungated equivalent of the blocked `git merge --continue`)
#       — gating `commit` would fence the shared root's primary purpose, so
#       this stays open by design. Sanctioned recovery for an in-progress
#       root merge is `git merge --abort` (allowed). (#1193) `git commit`
#       mid-conflicted root CHERRY-PICK completes the pick exactly as it
#       completes a merge (the rebase analogue is moot — a conflicted rebase
#       detaches HEAD, so the on-main gate already exits 0). Also: a
#       parenthesized NO-ARG `(git merge)` misses the trailing space/EOL
#       anchor — a git usage error anyway ("fatal: No commit specified"),
#       ~zero risk.
# (xvi) (#1128) Merge-SEMANTICS ops under other command words stay ungated:
#       `git pull --no-rebase` / a config-override pull performs exactly the
#       fenced merge (mitigated by the shared .git/config pins
#       `pull.rebase=merges` + `rebase.autoStash=true`; root syncs route
#       through sync_repo_root.py). (#1193) The rebase family is now fenced
#       by its own detectors below; `git pull --rebase[=merges]` (the
#       sanctioned root-sync form) has command word `pull` and stays outside
#       the tight anchors by design. (#1201) The pull lane is now fenced by
#       its own sibling guard, scripts/guard_repo_root_pull.sh; this guard's
#       scope is unchanged.
# (xvii) (#1193) Rebase-family residuals: (a) CLOSED by #1234: `git revert` /
#       `git am` now carry their own fence arms (same tight anchor + per-verb
#       --abort/--quit allow; see the #1234 block below). Retained for
#       lineage. (b) The anchored PER-VERB allow-arm
#       kills the immediate-flag spoof (pinned by tests R12/R13/CP9); the
#       residual is the raw-scan parity class shared with the merge fence —
#       the guard defends against accidents, not adversaries. (c) `git
#       rebase -h` / `--help` at root false-blocks (parity with `git merge
#       --help` today); remediation: `man git-rebase`, which stays allowed
#       (`git-rebase` has no `git ` + space bigram). (d) The parenthesized
#       NO-ARG `(git rebase)` slips the trailing space/EOL anchor, and the
#       (xv) "(git merge) ~zero risk" rationale does NOT transfer verbatim —
#       a bare rebase with a configured upstream genuinely RUNS; with-arg
#       paren forms still block, so only the exact no-arg-inside-parens
#       shape slips (~nil accident probability; named, not fixed).
#       (e) `git -c <k=v> -C <worktree> rebase ...` false-blocks (the
#       per-clause waiver requires `-C` immediately after `git`) — identical
#       parity with merge/reset today. (f) The new verbs activate gap
#       (xiv)'s piped-grep FP class — NARROWED by #1538 to UNVERIFIABLE
#       consumer chains only (`grep 'git rebase ...' file | head` now
#       waives via the read-only-sink walker, GP5 pins the rebase-vocab
#       idiom; the non-piped grep clause stays waived — RA17).
# (xviii) (#1234) Revert/am residuals: (a) `git am --show-current-patch`
#       (read-only) false-blocks — strict abort/quit-only parity with
#       #1128/#1193 keeps the five fences auditable as one family; recovery
#       for an in-progress root am is `--abort` (pinned AM8). (b) The am
#       allow-anchor's quoted-prose spoof ("... am --abort ..." inside a
#       quoted arg satisfies the allow while a real `git am` runs) and the
#       flag-chain tight-anchor FP — valid global-flag-chain + quoted-prose
#       shapes can also match: `git --no-pager log --since "9 am today"` IS
#       valid git and tight-matches (the flag chain consumes `log` as
#       `--no-pager`'s value); accepted accidents-not-adversaries FP class,
#       bounce-only failure direction, pinned by one block test — are the
#       raw-scan parity class shared with (xvii)(b). (c) The new verbs
#       activate gap (xiv)'s piped-grep FP class — NARROWED by #1538 to
#       UNVERIFIABLE consumer chains only (`grep 'git am ...' f | head`
#       now waives via the read-only-sink walker; the non-piped grep
#       clause stays waived — RVA15), and
#       `git am --help` / `git revert --help` false-block ((xvii)(c)
#       parity); remediation `man git-am` / `man git-revert`, which stay
#       allowed (no `git ` + space bigram). (d) The parenthesized NO-ARG
#       `(git am)` slips the `( +|$)` anchor — the (xvii)(d) analog; unlike
#       a bare no-upstream rebase, a bare `am` genuinely READS PATCHES FROM
#       STDIN and runs, so only the exact no-arg-inside-parens shape slips
#       (~nil accident probability; named, not fixed). (e) Root-am-state
#       tension: a hook-exempt subprocess (sync_repo_root.py's internal git,
#       cron wrappers) can still create root am state whose COMPLETION this
#       fence blocks at the Bash surface — sync_repo_root.py's stale-am
#       refusal names `git am --abort` as the sanctioned root resolution;
#       FINISHING the session belongs to its owner via a `git -C <path>`-
#       scoped `am --continue`. (f) `git apply` and `git stash pop|apply`
#       also mutate the shared tree and remain ungated — zero incident
#       demand; a separate candidate if demanded (the old (xvii)(a)
#       pattern, rolled forward).
# (xix) (#1463/#1859) remote-exec timeout-wrapper residuals: the waiver
#       cond (1) head + the mask candidate head accept an optional literal
#       `timeout <num>[.frac][smhd]?` wrapper on the `gcloud compute ssh`
#       head (#1463) AND the bare `ssh` head (#1859 — the demand #1463
#       pre-registered arrived: two timeout-wrapped-ssh false blocks on the
#       #1769 failover critical path; the former N12/NM5 asymmetry pins
#       flipped to positive fixtures). The literal-word + literal-numeric
#       wrapper is the ONLY tolerated one on EITHER head. Fail-closed FPs
#       that KEEP BLOCKING, both
#       mechanisms: env-prefix / nohup / abs-path / variable wrappers and
#       `timeout` FLAG forms (`--signal=`, `-k` — GN8/GN14 gcloud-side,
#       N36-N43 ssh-side), release tracks
#       (`gcloud beta|alpha compute ssh` — GN9), other gcloud remote-exec
#       subcommands (`compute tpus tpu-vm ssh`), trailing tokens/flags after
#       the quoted payload (mask R1 — put --command last; GN11), in-payload
#       `<`/`>` redirects (R2 / cond (3b); GN12 — bound output with `| tail`
#       or `|& tail` INSIDE the payload, pipes/amps DO mask), outer local
#       pipes / fd-dups (`2>&1 | tail -N` — the verbatim #825 shape, GN1),
#       double-quoted multi-statement payloads (GN15), and quoted /
#       latch-vocabulary prefixes (R5-R7; GN13). Deliberate-only accepted
#       fail-opens (inherit the ssh arm register, gap (xiv)):
#       `--ssh-flag=-F<config>` whose ProxyCommand lives in the config FILE
#       (the token refusal scans clause text only), non-canonical shared-repo
#       path spellings, and the path-blind `git -C` per-clause waiver firing
#       BEFORE this arm (GS8 parity — pre-existing, unchanged). The
#       comment-`#`-inside-payload truncation note carries over from (xiv)
#       (safe direction: the mask R-arms scanned the full pre-strip text).
# (xx)  (#1554) Worktree-local bare-main merge fence residuals. Fail-OPEN
#       (accepted — the same class as the -C waiver's existing gaps; each
#       keeps today's ALLOW disposition):
#       (a) `git -C "$OTHER_VAR" merge main` with a non-WT variable —
#       prior-call variable bindings are invisible by design ($WT is
#       special-cased as the SKILL.md-conventional worktree variable and
#       blocked unconditionally, the fail-closed safe direction);
#       (b) wrapper-prefixed forms (`timeout 60 git -C <wt> merge main`,
#       env-assignment prefixes) evade the ^-anchored clause head — the
#       identical accepted trade the #1443 cd-latch anchoring made;
#       (c) a GLOBAL git flag before -C (`git --no-pager -C <wt> merge main`,
#       `git -c k=v -C <wt> ...`) evades Arm A's `^git +-C` head and is
#       waived by the unanchored path-blind -C waiver — fail-open toward
#       today's behavior;
#       (d) a /tmp-latched compound whose SECOND clause is a `-C <worktree>`
#       merge (`cd /tmp/x && git -C <wt> merge main`) takes the Arm-B else
#       `continue` (scoped_wt=0) without ever reaching Arm A — the inverse
#       of the fail-closed compound below;
#       (e) single-dash-flag merge spellings (`merge -q main`, `-m <msg>`
#       argument-taking forms) evade the --long-flag-only groups (no
#       prescription channel ever emitted them), and local-lineage refs
#       (`main~1`, `main^`) evade the bare-ref tail — same family;
#       (f) a quoted worktree path containing a SPACE defeats the `[^ ]*`
#       path token (fleet worktree names are space-free);
#       (g) remote-side merges inside ssh payloads (non-worktree /workspace
#       paths) stay waived downstream exactly as today;
#       (h) SIBLING VERBS are deliberately OUT OF SCOPE: `git -C <wt> rebase
#       main` / `git -C <wt> reset --hard main` are the same contamination
#       class but still ride the path-blind -C waiver ungated — this fence
#       covers the merge verb only (the recorded #1530 prescription-channel
#       forms); do NOT read the fence as covering them.
#       Fail-CLOSED (accepted): the exotic `cd <worktree> && git -C /tmp/x
#       merge main` compound declines via Arm B (the escape hatch covers a
#       deliberate need); a clause-initial literal spelling inside a masked
#       single-quoted ssh payload targeting a REMOTE .claude/worktrees/ path
#       could decline — no such remote layout exists in this fleet.
#       DRIFT VECTOR (watched channel): the primary prescription channel is
#       .claude/agent-memory/implementer/feedback_ff_worktree_to_main_before_edit.md
#       — if a future edit adds a `timeout`-style wrapper to its MERGE line
#       (it already wraps the fetch line), Arm A goes silent on that channel
#       via residual (b).
#
# Compound-command parsing is a best-effort CLAUSE SPLIT (#804): the command is
# split on `;` / `&&` / `||` / `|` / `&` / raw newline (two-char separators
# matched first) into clauses, each classified independently so a later
# safe/return-to-main clause can no longer mask an earlier dangerous one; a
# `cd <worktree|/tmp>` latch propagates ONLY across `&&` (where bash GUARANTEES
# the `cd` succeeded before the RHS runs, so the cwd persists forward). The
# latch does NOT propagate across:
#   - `;` (SEQ): bash runs the RHS regardless of the `cd` exit code; a FAILED
#     `cd` (e.g. a missing target) leaves the cwd unchanged (repo root), so the
#     git clause runs off-worktree. Fail-closed (#804 round 2): reset the latch
#     on `;` rather than trust a `cd` we cannot prove succeeded.
#   - a raw NEWLINE (NL): a multi-line command runs each line unconditionally,
#     exactly like `;` — bash does NOT short-circuit on a `cd` exit code across
#     a newline, so a FAILED `cd` on line N leaves line N+1 running in the
#     unchanged cwd (repo root). Treated as `;`: reset the latch on NL. Before
#     #804 round 3 raw newlines produced records with no leading sentinel, so
#     `sep` inherited the STALE value (an `AND` after a `&&` clause) and the
#     `cd` latch leaked ACROSS the newline — `cd <missing> && git status\n
#     git switch feature` returned rc=0. The sed pre-pass now emits an explicit
#     `NL` sentinel for each raw newline so it resets the latch like `;`.
#   - `||` (OR): the RHS runs ONLY on `cd` FAILURE, cwd unchanged (repo root).
#   - `|` (PIPE): each pipeline segment is its own subshell, so the LHS `cd`'s
#     cwd change dies with it and the git segment runs in the parent's cwd.
#   - `&` (BG): the LHS runs in a background subshell (its own cwd) while the
#     RHS runs in the foreground parent's UNCHANGED cwd (repo root); BOTH
#     execute, so a `git switch feature & git switch main` runs the dangerous
#     LHS in the repo-root tree while the allow-arm RHS masks it.
# The split is NOT a full shell parse: command substitution `$(git switch ...)`
# and separators embedded inside a quoted arg are not handled (a quoted
# `;`/`|`/`&` is treated as a real separator, the same raw-scan trade-off as
# the `--note` literal above); heredoc BODIES are stripped BEFORE the split by
# the #1058 pre-pass when provably inert, and an unstrippable heredoc keeps
# its body lines in the split, failing CLOSED. A mis-split of that class fails
# CLOSED (blocks), the safe direction for a guard.
#
# A SECOND cd-latch (#1058) covers the SKILL.md-conventional `cd "$WT"` form
# beside the literal-path latch: a BARE, unconditionally-executed
# clause-initial `<NAME>=<...>.claude/worktrees/<...>` assignment binds the
# name in `_wt_names` in stream order (#1861 generalized the original
# WT-only `wt_bound` flag to any variable name), after which `cd "$NAME"`
# (exact-arg, no
# `..`) latches the SAME `scoped` machinery as the literal latch (so it
# inherits the `&&`-only propagation + reset semantics above verbatim). The
# assignment proof is load-bearing, not cosmetic: bash `cd ""` SUCCEEDS as a
# no-op (verified 2026-07-05, bash 5.1.16), so with an UNSET WT
# `cd "$WT" && git ...` runs the git clause in the UNCHANGED cwd (the repo
# root) — a bare `cd "$WT"` latch would be fail-open. Three proof
# obligations, each verified live:
#   - BARE assignment (end anchor): `WT=<wt> true` is a per-command
#     temporary env assignment that does NOT persist past that command; any
#     trailing word after the RHS refuses arming.
#   - ARMING separators are START / `;` (SEQ) / raw newline (NL) ONLY — the
#     separators under which bash executes the assignment UNCONDITIONALLY in
#     the parent shell. These are deliberately a DIFFERENT set from the
#     `&&`-only PROPAGATION separators of the latch itself: PROPAGATION
#     needs proof the `cd` SUCCEEDED (only `&&` gives it), ARMING needs
#     proof the assignment RAN (an `&&`/`||`-preceded assignment is
#     runtime-conditional and may be skipped; a `|`-preceded one runs in a
#     pipeline subshell and dies with it; BG stays non-arming as fail-closed
#     conservatism — a BG-preceded clause does run in the parent, but there
#     is zero incident demand). A future reader must NOT assume latch-style
#     `&&` semantics for arming, nor arming-style `;`/NL semantics for
#     propagation.
#   - Reassignment DISARMS: any later clause-initial `WT=` that does not
#     re-prove the full pattern (non-worktree RHS, trailing command word,
#     conditional/subshell separator) makes the earlier arming proof stale;
#     a bare unconditional worktree re-assignment re-arms.
#
# Bash line continuations (`\<CR?><NL>`) are normalized to a single space at the
# top of the guard before any parsing (#804 round 4). Bash strips a
# backslash-newline before execution, joining the two physical lines into one
# logical command, so `git \<NL>checkout -bfoo` runs as `git checkout -bfoo`;
# without the normalization the raw-scan guard saw `git ` and `checkout -bfoo`
# as separate lines (the newline splitter fired) and missed the joined
# `git checkout` invocation entirely (a leak of the exact class this guard
# blocks). The normalization is a no-op on any command without a `\<NL>`.
set -u

REPO=/home/thomasjiralerspong/explore-persona-space
REPO_BASE=${REPO##*/}   # basename (explore-persona-space) for the #1098 waiver path globs

# --- deny-event sidecar (#1528) ---------------------------------------------
# Best-effort forensic record of every deny: one JSON row appended to
# EPM_GUARD_DENY_SIDECAR (default $REPO/.claude/cache/guard-deny-events.jsonl).
# NEVER affects the deny/allow decision, exit codes, or the stderr message —
# every failure (missing dir, unwritable path, disk full) is swallowed by the
# braced append group below. No full command text is recorded: bounded
# printable-ASCII redacted heads only (#1501 A-11).
DENY_SIDECAR="${EPM_GUARD_DENY_SIDECAR:-$REPO/.claude/cache/guard-deny-events.jsonl}"

_deny_head() {  # stdin -> control chars to space, printable ASCII only,
                # opaque [A-Za-z0-9_-] runs >=20 masked to 4-char prefix + ***,
                # THEN truncated to 120 chars. Masking runs BEFORE the final
                # truncate (on a 400-char pre-cut bounding sed cost) so a
                # secret-shaped token straddling the 120-char boundary cannot
                # leak a partial fragment (#1528 r1 concern 2).
  tr '\n\r\t' '   ' | tr -cd ' -~' | cut -c1-400 \
    | sed -E 's/([A-Za-z0-9_-]{4})[A-Za-z0-9_-]{16,}/\1***/g' | cut -c1-120
}

log_deny() {  # $1 = arm label ($blocked), $2 = full command, $3 = blocking clause
  # Defaulted ${N-} expansions: under `set -u` an unbound expansion inside the
  # braced group ABORTS the script (the `|| true` rescues command failures,
  # not expansion errors), which would fail the guard OPEN. Defense in depth.
  local arm_in="${1-}" cmd_in="${2-}" clause_in="${3-}"
  {
    mkdir -p "$(dirname "$DENY_SIDECAR")"
    jq -cn --arg ts "$(date -u +%FT%TZ)" \
       --arg arm "$(printf '%s' "$arm_in" | _deny_head)" \
       --argjson len "${#cmd_in}" \
       --arg head "$(printf '%s' "$cmd_in" | _deny_head)" \
       --arg clause_head "$(printf '%s' "$clause_in" | _deny_head)" \
       '{ts:$ts, guard:"repo_root_branch", arm:$arm, len:$len,
         head:$head, clause_head:$clause_head}' >> "$DENY_SIDECAR"
  } 2>/dev/null || true
}
# --- end deny-event sidecar ---------------------------------------------------

cmd=$(jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0
[ -n "$cmd" ] || exit 0

# Normalize Bash line continuations (`\<CR?><NL>` -> space) BEFORE any parsing.
# Bash strips these pre-execution (joining the two physical lines into one), but
# the raw-scan guard would otherwise see `git ` and `checkout -bfoo` as separate
# lines (the newline splitter fires) and miss the joined `git checkout -bfoo`
# invocation. #804 round 4 fix for `guard-backslash-continuation-bypass`. The
# `sed -zE` uses NUL-delimited whole-input so `\n` is literal; `\\\r?\n` matches
# a backslash + optional CR + newline, replaced by a single space.
cmd=$(printf '%s' "$cmd" | sed -zE 's/\\\r?\n/ /g')

# (#1058) Strip NON-SHELL-CONSUMER heredoc BODIES before any parsing. The
# newline splitter otherwise treats each heredoc body line as an executable
# clause, so DOCUMENT text that merely MENTIONS a gated form (a python
# heredoc, a cat-written note file) false-blocks. Bash hands a heredoc body
# to the consumer as STDIN DATA — it is executed only when the consumer is a
# shell interpreter, EXCEPT that an UNQUOTED-tag body is expanded by bash at
# feed time ($(...) / `...` run REGARDLESS of consumer). Fail-closed ladder
# (any doubt -> NO strip, current behavior): strip only when EVERY opener in
# the command (a) has a word-shaped tag ([A-Za-z_][A-Za-z0-9_]*), (b) is the
# only opener on its physical line, (c) has a terminator line exactly == tag
# (leading tabs allowed for <<-), (d) the opener LINE carries NO
# shell-consumer / command-runner word
#     (bash|sh|zsh|ksh|dash|eval|source|ssh|xargs|parallel|sudo|su, or a
#     standalone `.` source command),
# (e) the opener LINE does not redirect into a *.sh path, (f) NO body line
# names a shell-out spelling (os.system / subprocess / Popen / check_call /
# check_output / getoutput / bare `system(` / `from os import` /
# `shell=True`) — EXCEPT that argv-LIST-form call opens with non-shell first
# elements (`subprocess.run(["git", ...`) are deleted from a per-line scan
# COPY before the refusal scan (#1621: class-(xi) argv-form python-subprocess
# git is out of the threat model, so its TEXT must not refuse the strip),
# with two fail-closed arms kept: an argv list whose first element names a
# shell (bare / path-qualified / env head) refuses PRE-deletion, and a
# `shell=True` residual refuses post-deletion — and (g) for
# an UNQUOTED, unescaped tag (<<EOF — bash EXPANDS the body at feed time) NO
# body line carries expansion syntax beyond plain ${NAME} parameter
# references: $( / backtick / every non-plain ${...} form refuse; plain
# ${NAME} spans are deleted from a scan COPY before the refusal (#1501 —
# ${V@P} was live-verified to execute value-borne command substitution
# under default promptvars, so ONLY the plain form is allowlisted); a
# quoted/escaped tag (<<'EOF', <<"EOF", <<\EOF) suppresses expansion and
# skips check (g).
# Here-strings (<<<) are masked before detection and never match.
strip_heredoc_bodies() {
  printf '%s' "$1" | awk '
    function shellish(line) {
      if (line ~ /(^|[^A-Za-z0-9_.])(bash|sh|zsh|ksh|dash|eval|source|ssh|xargs|parallel|sudo|su)([^A-Za-z0-9_]|$)/) return 1
      # Standalone-dot source form (`. /dev/stdin <<EOF`): a `.` COMMAND WORD
      # (start/separator/space before, whitespace after). A dot glued to a
      # word (path tails: foo.sh, python3.11) never matches; a bare-dot jq
      # filter (`jq . <<J`) DOES match -> fail-closed no-strip, documented.
      if (line ~ /(^|[;&|() \t])\.[ \t]/) return 1
      return 0
    }
    # Returns 1 and sets TAG/DASH/QUOTED when line has exactly one valid
    # opener; 0 = no opener; -1 = opener present but unstrippable (fail to
    # no-strip).
    function opener(line,   tmp, rest, op) {
      tmp = line
      gsub(/<<</, "\x02", tmp)
      if (tmp !~ /<</) return 0
      if (match(tmp, /<<-?[ \t]*\\?[\x22\x27]?[A-Za-z_][A-Za-z0-9_]*[\x22\x27]?/) == 0) return -1
      op = substr(tmp, RSTART, RLENGTH)
      rest = substr(tmp, RSTART + RLENGTH)
      if (rest ~ /<</) return -1
      if (shellish(line)) return -1
      if (line ~ /(>|>>)[ \t]*[^ \t]*\.sh[\x22\x27]?([ \t]|$)/) return -1
      # (#1058 r2) Tag QUOTING decides body-expansion semantics: bash
      # suppresses parameter/command substitution in the body ONLY when the
      # tag is quoted or escaped (<<'\''EOF'\'', <<"EOF", <<\EOF). QUOTED=0
      # (bare <<EOF) arms the pass-1 expansion-syntax refusal (check g). A
      # trailing-quote-only form (<<EOF'\'' — a bash syntax error anyway)
      # classifies as UNQUOTED, the stricter direction.
      QUOTED = (op ~ /^<<-?[ \t]*(\\|\x22|\x27)/) ? 1 : 0
      TAG = op
      sub(/^<<-?[ \t]*\\?[\x22\x27]?/, "", TAG)
      sub(/[\x22\x27]$/, "", TAG)
      DASH = (op ~ /^<<-/) ? 1 : 0
      return 1
    }
    function terminator_at(j,   b) {
      b = buf[j]
      if (DASH) sub(/^\t+/, "", b)
      return (b == TAG)
    }
    BEGIN { n = 0 }
    { buf[n++] = $0 }
    END {
      # PASS 1: validate; on ANY unstrippable opener emit input unchanged.
      i = 0; ok = 1
      while (i < n) {
        r = opener(buf[i])
        if (r == -1) { ok = 0; break }
        if (r == 1) {
          j = i + 1
          while (j < n && !terminator_at(j)) {
            # (f) A body that itself SHELLS OUT (os.system / subprocess /
            # Popen / bare system( / from-import form) may execute git
            # despite a non-shell consumer -> refuse to strip (fail-closed;
            # harmless unless a gated form also appears). (#1621) argv-LIST
            # call opens with non-shell heads are carved out first — see the
            # deletion below.
            fcopy = buf[j]
            # (#1621) NEW arm 1: an argv LIST whose first element names a
            # shell refuses BEFORE the deletion below (closes
            # Popen(["bash","-c",...]) under the carve). Covers bare,
            # path-qualified ("/bin/bash") and "env" heads, plus the literal
            # backslash-n bracket-gap spelling — the gap tolerance mirrors
            # the deletion regex so no shape the carve strips can dodge this
            # arm (r2, methodology MF-1).
            if (fcopy ~ /\(((\\n)|[ \t])*\[((\\n)|[ \t])*\\?[\x22\x27]([^\x22\x27]*\/)?(bash|sh|zsh|ksh|dash|env)\\?[\x22\x27]/) { ok = 0; break }
            # (#1621) argv-list-form call opens are deleted from the scan
            # COPY only (the #1501 delete-from-copy pattern): class-(xi)
            # argv-form python-subprocess git is out of the threat model and
            # never classifies (no `git <verb>` bigram — comma-separated
            # list tokens never satisfy the clause regexes), so its TEXT
            # must not refuse the strip. Tolerates literal backslash-n +
            # blanks between paren and bracket (plan-embedded snippet
            # spelling, #1621). Pass-1 refusal still emits buf[] byte-
            # identical; deletion can never MASK a refusal (the deleted span
            # carries no other refusal token, and shell-headed lists were
            # already refused by the arm above).
            gsub(/(subprocess\.[A-Za-z_]+|Popen|check_call|check_output)\(((\\n)|[ \t])*\[/, "", fcopy)
            # (#1621) NEW arm 2: shell=True joins the refusal list (closes
            # run(["<single-string>"], shell=True) under the carve).
            if (fcopy ~ /(os\.system|subprocess|Popen|check_call|check_output|getoutput|system *\(|from +os +import|shell[ \t]*=[ \t]*True)/) { ok = 0; break }
            # (g) UNQUOTED-tag body: bash performs command/parameter
            # substitution at feed time, so $(...) / `...` in the body
            # EXECUTE regardless of consumer -> refuse to strip. Non-plain
            # ${...} forms refuse too (parameter expansion can nest command
            # substitution, ${x:-$(cmd)}; ${V@P} executes value-borne
            # command substitution under default promptvars). Plain ${NAME}
            # spans are provably inert — same feed-time semantics as bare
            # $NAME, which check (g) has never refused — and are deleted
            # BEFORE the refusal scan, from a COPY only (#1501): pass-1
            # refusal must emit buf[] byte-identical (the print loop
            # below), and an in-place gsub would hand mutated text to the
            # downstream detectors (fail-open when a deleted span sat
            # inside gated text — pinned by the M1L fixture). Deletion can
            # never MASK a refusal: a plain span contains no paren and no
            # backtick and cannot overlap a $( occurrence (the char after
            # $ in a span is {). An escaped \$( also matches (bash would
            # NOT expand it, but \\$( WOULD — refusing both is the simple
            # fail-closed read).
            if (!QUOTED) {
              scan = buf[j]
              gsub(/\$\{[A-Za-z_][A-Za-z0-9_]*\}/, "", scan)
              if (scan ~ /\$\(|\$\{|\x60/) { ok = 0; break }
            }
            j++
          }
          if (!ok) break
          if (j >= n) { ok = 0; break }      # unterminated -> no strip
          i = j + 1
          continue
        }
        i++
      }
      if (!ok) { for (k = 0; k < n; k++) print buf[k]; exit }
      # PASS 2: emit, dropping each body + terminator (opener line kept).
      i = 0
      while (i < n) {
        r = opener(buf[i])
        print buf[i]
        if (r == 1) {
          j = i + 1
          while (j < n && !terminator_at(j)) j++
          i = j + 1
          continue
        }
        i++
      }
    }'
}
cmd=$(strip_heredoc_bodies "$cmd")

# (#1413) Mask separators inside the balanced SINGLE-QUOTED final argument of
# a clause-initial `ssh` clause, so a MULTI-STATEMENT remote command string
# (`ssh pod 'cd /w && git fetch && git checkout FETCH_HEAD -- f'`) is no
# longer mis-split by the quoted-separator trade-off into a tail clause that
# lost its ssh command word (the residual-(xiv) false-positive class;
# founding incident #779). As of #1463 a SECOND candidate head — clause-
# initial `gcloud compute ssh` — runs the SAME scan, the SAME R1-R8
# predicate, and the SAME separator-only rewrite (founding incident #825;
# residuals in gap (xix)); as of #1859 BOTH heads accept an optional
# literal `timeout <num>[.frac][smhd]?` prefix (the only tolerated
# wrapper — #1769 false blocks were the ssh-side demand). The two head regexes
# here and in the driver-loop waiver cond (1) below are PARITY-pinned: if
# they ever drifted, a mask-accepted-but-ladder-refused merged clause still
# carries its git-verb text and CLASSIFIES — false block, never a leak.
# Single-quoted spans are the ONE place a
# conservative parse is EXACT: bash single-quoted strings cannot contain a
# quote at all, so '<anything-but-quote>' is the true parse — no escapes, no
# nesting (double-quoted payloads admit all three and stay a documented
# residual).
#
# Masking fires ONLY when the whole prospective clause passes an 8-arm
# fail-closed refusal predicate; ANY refusal returns the input BYTE-IDENTICAL
# (today's behavior). The arms (monotonicity invariant: a masked clause is
# waived by the EXISTING #1098 ladder before classify_clause, and can neither
# arm, ride, disarm, nor suppress-a-reset-of the driver's cd-scope / WT
# latches — so every clause that still reaches classify_clause, and every
# latch transition, is identical to today except the intended removal class):
#   R1  tail after the closing quote is whitespace-only up to ; / && / || /
#       newline / end-of-string — never a bare | or & (the ladder's PIPE/BG
#       refusal set), never a trailing token (`2>/dev/null`, `-v`, a second
#       quoted arg — those shapes keep today's disposition).
#   R2  no $( / ${ / backtick / < / > / \x01 anywhere in the candidate — a
#       strict SUPERSET of ladder conds (3)/(3b) restricted to the clause
#       (blanket < covers <( / <<< / << / plain input redirects; \x01 is
#       splitter-sentinel hygiene). Bare $VAR stays maskable (ladder parity).
#   R3  no ProxyCommand / LocalCommand / KnownHostsCommand token (ssh
#       executes all three LOCALLY — ladder cond (4) parity).
#   R4  no shared-repo path in any covered spelling (literal / $HOME/ / ~/ +
#       basename — cond (4) parity; the never-waived class stays never-waived).
#   R5  no quote char (single OR double) seen anywhere BEFORE the candidate,
#       outside previously-ACCEPTED candidates' consumed quote pairs — the
#       scanner is quote-context-blind, so a pre-opened quote could make the
#       scanned "payload" live LOCAL code between two strings (the
#       local-code-swallow hole); strict any-quote refusal, because an
#       apostrophe-parity check misses the pre-opened-double-quote variant.
#   R6  no cd-latch-arming vocabulary in the candidate — `cd` + /tmp/ or
#       .claude/worktrees/ (RETAINED as defense-in-depth + disposition pin,
#       #1443: the driver latch greps below are clause-initial-anchored, and
#       a masked merged clause is `ssh`-initial so they can never match it;
#       NM22/NM23 pin blocked dispositions — relaxing R6 is a non-goal; also
#       precludes regex-WIDENING latch matches across former statement bounds).
#   R7  no cd-latch-arming vocabulary in the PREFIX — guarantees scoped == 0
#       at candidate entry, so the mask's removal of intra-payload separators
#       (which today RESET `scoped` at each mis-split boundary) can never
#       suppress a load-bearing latch reset and skip a local gated tail.
#   R8  no `WT=` text in the candidate — a mis-split payload fragment's
#       clause-initial `WT=` arms/disarms the `_wt_names` binding TODAY
#       (#1861: formerly the WT-only `wt_bound` flag); masking would
#       suppress those transitions (WT-latch state isolation).
# The replacement token ` __EPM_SSH_SEP__ ` is space-padded word characters:
# it cannot form a separator, a refusal pattern, a repo-path spelling, a
# latch-regex match, or glue adjacent tokens into a `git <verb>` bigram.
# Sentinel collision with payload text is harmless — the mask is
# guard-internal; bash receives the ORIGINAL command either way.
mask_ssh_payload_separators() {
  printf '%s' "$1" | awk -v repo="$REPO" -v repo_base="$REPO_BASE" '
    BEGIN { nrec = 0 }
    { rec[nrec++] = $0 }
    END {
      # Re-join records with the newlines awk consumed (a trailing newline is
      # stripped either way by the caller command substitution — status quo
      # at every existing normalization stage).
      s = ""
      for (r = 0; r < nrec; r++) s = s (r ? "\n" : "") rec[r]
      n = length(s)
      i = 1; atstart = 1; out = ""; saw_quote = 0
      while (i <= n) {
        headlen = 0
        if (atstart) {
          # Candidate heads: clause-initial `ssh ` (#1413) or clause-initial
          # `gcloud compute ssh ` (#1463), EITHER with an optional literal
          # `timeout <num>[.frac][smhd]?` wrapper (#1463 gcloud-arm, #1859
          # ssh-arm — the ONLY tolerated wrapper). Head-regex PARITY with
          # the driver-loop waiver cond (1) below is an invariant;
          # MASK-vs-WAIVER drift is fail-closed (a masked-but-unwaived
          # merged clause keeps its git-verb text and still classifies ->
          # false block, never a leak). NOTE the different drift direction
          # INSIDE the driver loop: the cond (1) outer head and its inner
          # ssh/gcloud-vs-grep discriminator must move in LOCKSTEP — drift
          # THERE routes a remote-exec clause to the grep-family arm and is
          # fail-OPEN (see the cond (1) comment below).
          if (match(substr(s, i), /^(timeout[ \t]+[0-9]+(\.[0-9]+)?[smhd]?[ \t]+)?(gcloud[ \t]+compute[ \t]+)?ssh[ \t]/)) headlen = RLENGTH
        }
        if (headlen > 0) {
          # Candidate. Scan the head (options/host): letters, digits, and
          # the option/host punctuation set ONLY — every quote, $, backslash,
          # redirect, paren, bracket, and separator char stops the scan, so
          # `ssh $(evil)`, `ssh "h"`, `ssh [::1]`, `ssh h <<EOF` never parse
          # as candidates (refused -> byte-identical -> today\047s behavior).
          j = i + headlen
          while (j <= n && substr(s, j, 1) ~ /[A-Za-z0-9_.@:%=+,\/ \t-]/) j++
          if (substr(s, j, 1) == "\047") {
            cq = index(substr(s, j + 1), "\047")   # single quotes: exact parse
            if (cq > 0) {
              payload = substr(s, j + 1, cq - 1)
              after = j + 1 + cq                   # first char past closing quote
              t = after
              while (t <= n && substr(s, t, 1) ~ /[ \t]/) t++
              # R1: whitespace-only tail ending in ; / && / || / NL / EOS.
              tail_ok = (t > n)
              if (!tail_ok) {
                c1 = substr(s, t, 1); c2 = substr(s, t, 2)
                tail_ok = (c1 == ";" || c1 == "\n" || c2 == "&&" || c2 == "||")
              }
              cand = substr(s, i, after - i)       # head + quoted payload
              pfx  = substr(s, 1, i - 1)           # everything before candidate
              lc   = tolower(cand)
              if (tail_ok &&
                  saw_quote == 0 &&
                  index(cand, "$(") == 0 && index(cand, "${") == 0 &&
                  index(cand, "\140") == 0 && index(cand, "<") == 0 &&
                  index(cand, ">") == 0 && index(cand, "\001") == 0 &&
                  index(lc, "proxycommand") == 0 &&
                  index(lc, "localcommand") == 0 &&
                  index(lc, "knownhostscommand") == 0 &&
                  index(cand, repo) == 0 &&
                  index(cand, "$HOME/" repo_base) == 0 &&
                  index(cand, "~/" repo_base) == 0 &&
                  !(cand ~ /cd[ \t]/ && (index(cand, "/tmp/") > 0 ||
                                         index(cand, ".claude/worktrees/") > 0)) &&
                  !(pfx ~ /cd[ \t]/ && (index(pfx, "/tmp/") > 0 ||
                                        index(pfx, ".claude/worktrees/") > 0)) &&
                  index(cand, "WT=") == 0) {
                # arms in order: R1 tail | R5 prefix quote-state | R2 token
                # set | R3 local-exec options | R4 repo-path spellings | R6
                # candidate latch vocab | R7 prefix latch vocab | R8 WT-latch
                # isolation. ACCEPTED: the ONLY rewrite the function ever
                # performs is this separator substitution inside the payload.
                gsub(/[;&|\n]+/, " __EPM_SSH_SEP__ ", payload)
                out = out substr(s, i, j - i) "\047" payload "\047"
                i = after; atstart = 0
                # The two consumed quotes do NOT set saw_quote: by R5 they
                # are bash\047s own balanced pair, so later candidates in a
                # multi-candidate command (`ssh h1 \047..\047 && ssh h2
                # \047..\047`) still compose.
                continue
              }
            }
          }
          else if (substr(s, j, 1) == "\042") {
            # (#1710) Arm 1: DOUBLE-quoted ssh payload. Byte-identical to
            # single-quoted parse ONLY when the payload carries no
            # expansion / escape / backtick token — R9 refuses on `$`,
            # `\140` (backtick), and `\\`. A refuse falls through to the
            # head-copy-verbatim path exactly like the single-quoted
            # refused branch.
            cq = index(substr(s, j + 1), "\042")
            if (cq > 0) {
              payload = substr(s, j + 1, cq - 1)
              after = j + 1 + cq                   # first char past closing quote
              t = after
              while (t <= n && substr(s, t, 1) ~ /[ \t]/) t++
              # R1: whitespace-only tail ending in ; / && / || / NL / EOS.
              tail_ok = (t > n)
              if (!tail_ok) {
                c1 = substr(s, t, 1); c2 = substr(s, t, 2)
                tail_ok = (c1 == ";" || c1 == "\n" || c2 == "&&" || c2 == "||")
              }
              cand = substr(s, i, after - i)       # head + quoted payload
              pfx  = substr(s, 1, i - 1)           # everything before candidate
              lc   = tolower(cand)
              if (tail_ok &&
                  saw_quote == 0 &&
                  index(cand, "$(") == 0 && index(cand, "${") == 0 &&
                  index(cand, "\140") == 0 && index(cand, "<") == 0 &&
                  index(cand, ">") == 0 && index(cand, "\001") == 0 &&
                  index(lc, "proxycommand") == 0 &&
                  index(lc, "localcommand") == 0 &&
                  index(lc, "knownhostscommand") == 0 &&
                  index(cand, repo) == 0 &&
                  index(cand, "$HOME/" repo_base) == 0 &&
                  index(cand, "~/" repo_base) == 0 &&
                  !(cand ~ /cd[ \t]/ && (index(cand, "/tmp/") > 0 ||
                                         index(cand, ".claude/worktrees/") > 0)) &&
                  !(pfx ~ /cd[ \t]/ && (index(pfx, "/tmp/") > 0 ||
                                        index(pfx, ".claude/worktrees/") > 0)) &&
                  index(cand, "WT=") == 0 &&
                  # R9: no expansion / escape / backtick in the payload
                  # body. A double-quoted bash string with none of these
                  # is byte-identical to the same content single-quoted.
                  index(payload, "$") == 0 &&
                  index(payload, "\140") == 0 &&
                  index(payload, "\\") == 0) {
                gsub(/[;&|\n]+/, " __EPM_SSH_SEP__ ", payload)
                out = out substr(s, i, j - i) "\042" payload "\042"
                i = after; atstart = 0
                # ssh-mask parity: consumed pair does NOT set saw_quote.
                continue
              }
            }
          }
          # REFUSED: copy the head word(s) verbatim — minus the single
          # trailing whitespace char the head regex consumed, which the char
          # path re-emits — and fall back to the char path; the refused
          # candidate\047s own quotes then set saw_quote, so any LATER
          # candidate in this command also refuses (conservative
          # composition; nested-candidate resume inside a quoted region is
          # structurally refused by R5).
          out = out substr(s, i, headlen - 1); i += headlen - 1; atstart = 0
          continue
        }
        c = substr(s, i, 1)
        out = out c
        if (c == "\047" || c == "\042") saw_quote = 1
        if (c ~ /[;|&\n]/) atstart = 1
        else if (c !~ /[ \t]/) atstart = 0
        i++
      }
      printf "%s", out
    }'
}
cmd=$(mask_ssh_payload_separators "$cmd")

# (#1566) Mask the BODIES of balanced SINGLE-QUOTED argument payloads of a
# clause-initial `*task.py` python-script invocation (`uv run python
# scripts/task.py post-marker ... --note '<prose>'` and siblings: --title /
# --origin-prompt / set-goal's positional payload — the pass is
# flag-agnostic within a recognized clause) to the neutral sentinel
# __EPM_ARG_PAYLOAD__ BEFORE the trigger-literal pre-filter below, so
# marker/metadata prose that merely NAMES a git-mutation op no longer
# false-blocks. A single-quoted bash string is inert argv DATA to a
# non-shell-consumer executable — bash never executes it, and single-quoted
# spans admit no escapes or nesting (the exact-parse property the ssh mask's
# comment establishes above) — so the safety argument is per-span,
# independent of which flag or positional slot precedes it. The load-bearing
# boundary is the CLAUSE HEAD, not the flag: the head whitelist admits only
# a python-script invocation whose path token ends in `task.py` (never a
# shell consumer like `bash -c` / `eval` / `ssh` / `xargs`, whose quoted
# args ARE executable code). NOTE the boundary is ANY `*task.py`-suffixed
# script — `codex_task.py` / `file_infra_task.py` are covered too; a future
# in-repo `*task.py` helper that shells out its argv must revisit this head
# whitelist as a visible design decision, never rely on it silently.
#
# Masking fires ONLY when the whole prospective clause passes a fail-closed
# refusal predicate; ANY refusal returns the input BYTE-IDENTICAL (today's
# disposition, `--file` workaround). The arms (P-series, mirroring the ssh
# mask's R1-R8 above; monotonicity: the ONLY rewrite is single-quoted span
# BODIES inside accepted candidates, so every disposition-relevant literal
# outside payloads survives, and P4/P6 guarantee the latch trajectory of
# every surviving clause is identical to today's):
#   P1  head shape — clause-initial optional `uv run`, optional
#       `python[0-9.]*`, a safe-charset path token ending in `task.py`,
#       then a word-shaped subcommand token. A quoted or dollar-bearing
#       path (e.g. a "$REPO_ROOT/..." spelling) cannot match — refused.
#   P2  safe charset in the NON-payload span — between head, quoted spans,
#       and clause end, ONLY [A-Za-z0-9_.@:%=+,~/ \t-] is allowed. A strict
#       SUPERSET of the ssh mask's R2: additionally excludes ALL `$`
#       (ANSI-C $'...' quoting processes escaped quotes, so a `$` adjacent
#       to a quote breaks the exact-parse guarantee — blanket exclusion is
#       the simplest sound rule), all backslashes (escaped-quote hazard),
#       `"` (double-quoted payloads refuse => byte-identical => the
#       existing known-limitation pins keep their disposition), `#`, globs,
#       parens/braces, redirects, backticks, and the \x01 splitter
#       sentinel.
#   P3  exact single-quote payload parse — at `'` the span runs to the next
#       `'` (exact by bash single-quote semantics; the record re-join below
#       means a span may contain a literal newline); no closing quote =>
#       refuse. The BODY is replaced by the sentinel, the quotes are kept.
#       Multiple spans per clause each mask (the multi-flag case); consumed
#       pairs do NOT set saw_quote (ssh-mask parity), so later candidates
#       in a compound still compose.
#   P4  latch/WT vocabulary isolation (R6+R8 parity, applied to the WHOLE
#       candidate INCLUDING payload bodies): refuse on cd-latch vocabulary
#       (`cd` + /tmp/ or .claude/worktrees/) or `WT=` anywhere in the
#       candidate. Keeps every latch-arming shape at today's disposition.
#       COUPLING NOTE: this vocabulary mirrors the driver-loop latch greps
#       (the `^cd +[^;&|]*\.claude/worktrees/` / `^cd +/tmp/` /
#       `^(export +)?WT=` greps below) — a future latch-vocabulary addition
#       there updates P4/P6 in the same edit (same maintenance class as
#       R6/R8).
#   P5  clean prefix quote-state (R5 parity, the critical arm): refuse when
#       ANY quote char (single OR double) was seen before the candidate
#       outside previously-accepted candidates' consumed pairs — a
#       pre-opened quote could make the scanned "payload" live LOCAL code
#       (the local-code-swallow hole); strict any-quote refusal because an
#       apostrophe-parity check misses the pre-opened-double-quote variant.
#   P6  latch-clean prefix (R7 parity, deliberately STRICTER than a
#       verbatim R7/R8 copy — R8 checks the candidate only): refuse if the
#       PREFIX carries the P4 vocabulary (cd-latch vocab OR `WT=`) —
#       guarantees latch state 0 at candidate entry, so removing
#       payload-internal separators (which today create latch-resetting
#       mis-split boundaries) can never suppress a load-bearing latch reset
#       or a WT arming/disarming transition.
# Arms that deliberately do NOT transfer from R1-R8: R3 (ssh local-exec
# options — no ssh head here), R4 (repo-path spellings — a repo path in
# inert argv is prose; refusing it would re-block the common "note names
# the repo path" case), and R1's payload-must-be-final-token (trailing
# tokens after a payload are more inert argv, still bounded by the P2
# charset walk — a redirect/quote/dollar in the tail refuses via P2).
# The sentinel __EPM_ARG_PAYLOAD__ is word characters inside kept quotes:
# it cannot form a separator, a trigger literal, a latch match, or glue a
# `git <verb>` bigram. Collision with genuine payload text is harmless —
# the mask is guard-internal; bash receives the ORIGINAL command either
# way. Fail-soft: the gated call below uses the same command-substitution
# shape as the two passes above — an awk failure yields an empty cmd and
# the pre-filter exits 0; additionally every NPB block-pin fixture in
# tests/test_guard_repo_root_branch.py contains `task.py`, so a broken awk
# body (empty output => exit 0) flips those pinned exit-2 tests and
# surfaces in the suite.
mask_taskpy_arg_payloads() {
  printf '%s' "$1" | awk '
    BEGIN { nrec = 0 }
    { rec[nrec++] = $0 }
    END {
      # Re-join records with the newlines awk consumed (ssh-mask parity; a
      # trailing newline is stripped either way by the caller command
      # substitution — status quo at every existing normalization stage).
      s = ""
      for (r = 0; r < nrec; r++) s = s (r ? "\n" : "") rec[r]
      n = length(s)
      i = 1; atstart = 1; out = ""; saw_quote = 0
      while (i <= n) {
        headlen = 0
        if (atstart) {
          # P1: optional `uv run`, optional `python[0-9.]*`, a safe-charset
          # path token ending in task.py, then a word-shaped subcommand.
          if (match(substr(s, i), /^(uv[ \t]+run[ \t]+)?(python[0-9.]*[ \t]+)?[A-Za-z0-9_.\/~-]*task\.py[ \t]+[A-Za-z0-9_-]+[ \t]/)) headlen = RLENGTH
        }
        if (headlen > 0) {
          # Candidate. Walk the clause: safe chars (P2) advance; a
          # SINGLE- or DOUBLE-quoted span (P3, #1710 extends P3 to double
          # quotes under a P7 no-expansion refusal) is recorded exactly; a
          # separator or end-of-string completes the candidate; anything
          # else refuses.
          j = i + headlen
          nspans = 0
          ok = 1
          while (j <= n) {
            c = substr(s, j, 1)
            if (c == "\047") {
              cq = index(substr(s, j + 1), "\047")
              if (cq == 0) { ok = 0; break }   # P3: no closing quote
              nspans++
              sp_open[nspans] = j              # index of the opening quote
              sp_len[nspans] = cq - 1          # payload body length
              sp_quote[nspans] = "\047"        # single quote type
              j = j + 1 + cq                   # first char past closing quote
              continue
            }
            if (c == "\042") {
              # (#1710) Double-quoted span. Refused later by P7 if the
              # body carries any expansion / escape / backtick token; the
              # body must be byte-identical-to-single-quoted content
              # (no `$`, no backtick, no `\\`) for the double-quoted span
              # to admit the P3 exact-parse property.
              cq = index(substr(s, j + 1), "\042")
              if (cq == 0) { ok = 0; break }   # P3: no closing quote
              nspans++
              sp_open[nspans] = j              # index of the opening quote
              sp_len[nspans] = cq - 1          # payload body length
              sp_quote[nspans] = "\042"        # double quote type
              j = j + 1 + cq                   # first char past closing quote
              continue
            }
            if (c ~ /[;&|\n]/) break           # clause end: candidate complete
            if (c ~ /[A-Za-z0-9_.@:%=+,~\/ \t-]/) { j++; continue }   # P2
            ok = 0; break                      # P2: unsafe char -> refuse
          }
          cand = substr(s, i, j - i)           # head + spans + tail
          pfx  = substr(s, 1, i - 1)           # everything before candidate
          # (#1710) P7: every double-quoted span body must carry NO
          # expansion / backtick / backslash. A double-quoted span whose
          # body contains any of `$`, backtick, or `\\` is executable
          # data and refuses the candidate. Single-quoted spans are
          # unaffected (exact-parse by shell semantics).
          p7_ok = 1
          for (k = 1; k <= nspans; k++) {
            if (sp_quote[k] == "\042") {
              spanbody = substr(s, sp_open[k] + 1, sp_len[k])
              if (index(spanbody, "$") > 0 ||
                  index(spanbody, "\140") > 0 ||
                  index(spanbody, "\\") > 0) {
                p7_ok = 0; break
              }
            }
          }
          if (ok && p7_ok && nspans > 0 &&
              saw_quote == 0 &&
              !(cand ~ /cd[ \t]/ && (index(cand, "/tmp/") > 0 ||
                                     index(cand, ".claude/worktrees/") > 0)) &&
              index(cand, "WT=") == 0 &&
              !(pfx ~ /cd[ \t]/ && (index(pfx, "/tmp/") > 0 ||
                                    index(pfx, ".claude/worktrees/") > 0)) &&
              index(pfx, "WT=") == 0) {
            # arms in order: P2/P3 (walk above) | P5 prefix quote-state |
            # P4 candidate latch vocab + WT | P6 prefix latch vocab + WT |
            # P7 double-quoted body no-expansion (#1710).
            # ACCEPTED: the ONLY rewrite the function ever performs is
            # replacing each span BODY with the sentinel (quotes kept).
            pos = i
            for (k = 1; k <= nspans; k++) {
              out = out substr(s, pos, sp_open[k] - pos) sp_quote[k] "__EPM_ARG_PAYLOAD__" sp_quote[k]
              pos = sp_open[k] + sp_len[k] + 2
            }
            out = out substr(s, pos, j - pos)
            i = j; atstart = 0
            # Consumed pairs do NOT set saw_quote (ssh-mask parity): by P5
            # they are bash\047s own balanced pairs, so later candidates in
            # a compound still compose.
            continue
          }
          # REFUSED: copy the head verbatim — minus the single trailing
          # whitespace char the head regex consumed, which the char path
          # re-emits — and fall back to the char path; the refused
          # candidate\047s own quotes then set saw_quote, so any LATER
          # candidate in this command also refuses (conservative
          # composition, ssh-mask parity).
          out = out substr(s, i, headlen - 1); i += headlen - 1; atstart = 0
          continue
        }
        c = substr(s, i, 1)
        out = out c
        if (c == "\047" || c == "\042") saw_quote = 1
        if (c ~ /[;|&\n]/) atstart = 1
        else if (c !~ /[ \t]/) atstart = 0
        i++
      }
      printf "%s", out
    }'
}
# Cheap literal gate: the awk pass spawns only for task.py-mentioning
# commands (the hook runs on EVERY Bash call); zero added cost otherwise.
case "$cmd" in *task.py*) cmd=$(mask_taskpy_arg_payloads "$cmd") ;; esac

# (#1710) Mask the BODIES of balanced SINGLE- or DOUBLE-quoted string
# literals of a clause-initial `python[3][.<M>]? -c` / `uv run
# python[...] -c` invocation to the neutral sentinel
# __EPM_PYTHON_C_LITERAL__ BEFORE the trigger-literal pre-filter below,
# so a helper-script `python -c` payload whose Python string LITERAL
# merely quotes a destructive-git phrase as inert prose (e.g. a
# fingerprint helper that hashes a bug description) no longer
# false-blocks. A Python string literal is INERT DATA to the local
# shell — bash never executes it, and single-quoted spans admit no
# escapes (ssh mask exact-parse property). But a `python -c` payload
# IS executable code by construction, so the refusal ladder is
# STRICTER than the ssh + taskpy masks: the string is admitted ONLY
# when it carries no shell-out / subprocess / function-call vocabulary
# (see C4-C10 refusal arms in-function).
mask_python_c_string_literals() {
  printf '%s' "$1" | awk '
    BEGIN { nrec = 0 }
    { rec[nrec++] = $0 }
    END {
      # Re-join records with the newlines awk consumed (ssh/taskpy-mask
      # parity; a trailing newline is stripped by the caller command
      # substitution either way).
      s = ""
      for (r = 0; r < nrec; r++) s = s (r ? "\n" : "") rec[r]
      n = length(s)
      i = 1; atstart = 1; out = ""; saw_quote = 0
      while (i <= n) {
        headlen = 0
        if (atstart) {
          # C1: head is `python[0-9.]*[ \t]+-c[ \t]+`, optionally
          # prefixed by `uv[ \t]+run[ \t]+`. Head-regex parity with
          # taskpy mask line 1151 for the python token.
          if (match(substr(s, i), /^(uv[ \t]+run[ \t]+)?python[0-9.]*[ \t]+-c[ \t]+/)) headlen = RLENGTH
        }
        if (headlen > 0) {
          j = i + headlen
          # C2: accept a SINGLE- or DOUBLE-quoted string literal at j.
          q = substr(s, j, 1)
          if (q != "\047" && q != "\042") {
            # No quoted arg at all — refuse; fall through to char path.
            out = out substr(s, i, headlen - 1); i += headlen - 1; atstart = 0
            continue
          }
          cq = index(substr(s, j + 1), q)
          if (cq == 0) {
            # C3: no closing quote — refuse.
            out = out substr(s, i, headlen - 1); i += headlen - 1; atstart = 0
            continue
          }
          payload = substr(s, j + 1, cq - 1)
          after   = j + 1 + cq                   # first char past closing quote
          # C-tail: whitespace-only tail ending in ; / && / || / NL / EOS
          # (ssh mask R1 parity).
          t = after
          while (t <= n && substr(s, t, 1) ~ /[ \t]/) t++
          tail_ok = (t > n)
          if (!tail_ok) {
            c1 = substr(s, t, 1); c2 = substr(s, t, 2)
            tail_ok = (c1 == ";" || c1 == "\n" || c2 == "&&" || c2 == "||")
          }
          # C4-C10 refusal ladder: payload must be provably INERT
          # prose. A `python -c` string is executable code, so the
          # burden of proof for "inert" is HIGHER than the ssh mask.
          bad = 0
          if (!tail_ok) bad = 1
          if (saw_quote != 0) bad = 1
          # C4: no backslash anywhere (Python `\xNN` / `\n` / `\uNNNN`
          # escapes resolve at runtime — a payload with any `\\` is
          # not provably inert).
          if (index(payload, "\\") > 0) bad = 1
          # C5: no backtick (shell subprocess).
          if (index(payload, "\140") > 0) bad = 1
          # C6: no `$` (any expansion / `${VAR}` / `$(cmd)` heredoc).
          if (index(payload, "$") > 0) bad = 1
          # C7-C10: no shell-out / subprocess vocabulary.
          if (index(payload, "subprocess") > 0) bad = 1
          if (index(payload, "os.system") > 0) bad = 1
          if (index(payload, "os.popen") > 0) bad = 1
          if (index(payload, "Popen") > 0) bad = 1
          if (index(payload, "commands.getoutput") > 0) bad = 1
          if (index(payload, "pty.spawn") > 0) bad = 1
          if (index(payload, "check_output") > 0) bad = 1
          if (index(payload, "check_call") > 0) bad = 1
          # Conservative: any function-call shape refuses. A `run(x)`
          # / `call(x)` / `foo(x)` invocation cannot be proved inert
          # by awk-level parsing (residual gap xiii). Fail-closed.
          if (payload ~ /[A-Za-z_][A-Za-z_0-9.]*\(/) bad = 1
          if (bad) {
            # REFUSED: copy the head verbatim — minus the single
            # trailing whitespace char the head regex consumed, which
            # the char path re-emits — and fall back to the char path
            # (ssh/taskpy-mask parity).
            out = out substr(s, i, headlen - 1); i += headlen - 1; atstart = 0
            continue
          }
          # ACCEPTED: replace the payload BODY with the neutral
          # sentinel (surrounding quotes kept, quote-char preserved).
          out = out substr(s, i, j - i) q "__EPM_PYTHON_C_LITERAL__" q
          i = after; atstart = 0
          # Consumed pair does NOT set saw_quote (ssh/taskpy-mask
          # parity): by construction bash treats it as a balanced
          # pair; later candidates in a compound still compose.
          continue
        }
        c = substr(s, i, 1)
        out = out c
        if (c == "\047" || c == "\042") saw_quote = 1
        if (c ~ /[;|&\n]/) atstart = 1
        else if (c !~ /[ \t]/) atstart = 0
        i++
      }
      printf "%s", out
    }'
}
# Cheap literal gate: the awk pass spawns only for `-c`-mentioning
# commands (parity with taskpy mask's `*task.py*` gate above); zero
# added cost otherwise. The tight C1 head regex inside then rejects
# `bash -c` / `sh -c` / `ssh -c <cipher>` / `chmod -c` / other non-
# python `-c` shapes and falls through unchanged.
case "$cmd" in *" -c "*) cmd=$(mask_python_c_string_literals "$cmd") ;; esac

# Only consider git checkout/switch/restore/clean/reset/merge/rebase/
# cherry-pick/revert/am invocations at all (loose pre-filter — a cheap skip,
# not a classifier; the tight per-verb anchors live in classify_clause).
# `\bmerge\b` deliberately matches `merge-base` here (the boundary fires
# before `-`) — harmless, the tight #1128 detector never fires on it; it does
# NOT match `--rebase=merges` / `mergetool` / `--merges` / `--merged` (no
# trailing boundary). (#1193) `\brebase\b` DOES match inside
# `--rebase=merges` / `pull.rebase=merges` / `rebase.autoStash=true` (`-`,
# `.`, `=` are non-word chars, so both boundaries fire) — harmless by the
# same merge-base argument: the tight #1193 detector requires subcommand
# position + a `( +|$)` verb terminator, so none of these can reach a block;
# the shift is only WHICH exit path allows them (classifier instead of
# pre-filter skip). `\bcherry-pick\b` does NOT match the ubiquitous prose
# token `cherry-picked` (no trailing word boundary — `k` is followed by `e`),
# so commit messages / marker notes carrying it never even pass the
# pre-filter; it DOES match `git log --cherry-pick` (the boundary fires
# between `-` and `c`) — harmless at the tight detector (`log` breaks the
# subcommand chain). `git log --cherry` and the plumbing command `git cherry`
# do not match the alternation at all (the literal requires the full
# `cherry-pick`).
# (#1234) `\brevert\b` does NOT match the ubiquitous prose forms `reverted` /
# `reverting` (no trailing word boundary), so commit messages / marker notes
# carrying them never even pass the pre-filter; bare prose `revert` DOES pass
# whenever `git` appears earlier in the command and is disposed of by the
# tight #1234 detector (subcommand position: `git commit -m "revert foo"`
# never matches — `commit` is a non-dash token that breaks the flag chain).
# `\bam\b` is a common English word ("I am ...") and DOES pass this loose
# gate under the same condition — deliberate and harmless by the same
# argument: the tight detector requires `git [dash-flag [value]]* am`, so
# `git commit -m "I am done"` can never reach a block; it does NOT match
# `--amend`/`amend` (no trailing boundary after `am`), `team`/`spam`/`gram`
# (no leading boundary), so `git commit --amend` still exits at this
# pre-filter. Accepted fail-closed residual: a dash-flag-then-value-then-`am`
# shape DOES match the tight anchor — a nonsense `git -m "I am here"` (`-m`
# is not a git global flag) but ALSO valid flag-chain git like
# `git --no-pager log --since "9 am today"` — the accepted
# accidents-not-adversaries FP class, register (xviii)(b).
echo "$cmd" | grep -qE '\bgit\b.*\b(checkout|switch|restore|clean|reset|merge|rebase|cherry-pick|revert|am)\b' || exit 0

# Split the raw command into (separator, next-separator, clause) TRIPLES,
# PRESERVING which separator precedes each clause AND (#1098) which separator
# FOLLOWS it — a buffered one-record lookahead whose $nextsep field lets the
# driver-loop ssh/grep waiver refuse pipeline-producer position. The final
# clause reports nextsep=END; an EMPTY record (a trailing separator) flushes
# the buffered clause with the empty record's separator as its nextsep, so a
# trailing separator's identity is never lost (fail-closed: a trailing
# `| <empty>` still reports PIPE). The sed pre-pass, sentinel vocabulary, and
# clause-strip semantics are unchanged. Two-char separators (&& ||) are matched
# BEFORE the single-char ones (; | &) so `&&`/`||` are not mis-split into two
# single-char clauses (the `&&` substitution runs before the single `&` rule,
# so a `&&` is consumed as AND and never re-matched as a bare `&`). Each
# separator run is replaced by a newline + a \x01-delimited sentinel token
# (START | SEQ | AND | OR | PIPE | BG | NL); the first clause carries the
# implicit START. Best-effort: a separator inside a quoted arg is treated as a
# real separator (same trade-off as the raw-scan known-limitation above).
#
# `sed -z` treats the WHOLE input as one NUL-delimited record so a literal
# newline in $1 is matchable. The raw-NEWLINE -> `\n\x01NL\x01` substitution
# runs FIRST — before any separator substitution has inserted its own `\n` —
# so it only tags the raw input newlines and can NOT re-mangle the structural
# `\n` the &&/||/;/|/& rules insert afterwards (none of those rules matches
# `\n` or the already-inserted `\x01NL\x01`). Without the NL sentinel a raw
# newline produced a record with NO leading sentinel, so awk's `sep` inherited
# the STALE value from the previous line (an `AND` after a `&&` clause) and the
# `cd` scope latch leaked across the newline (#804 round 3).
split_and_label() {
  printf '%s' "$1" \
    | sed -zE 's/\n/\n\x01NL\x01/g; s/\|\|/\n\x01OR\x01/g; s/&&/\n\x01AND\x01/g; s/;/\n\x01SEQ\x01/g; s/\|/\n\x01PIPE\x01/g; s/&/\n\x01BG\x01/g' \
    | awk 'BEGIN{RS="\n"; sep="START"; have=0}
           { line=$0
             if (match(line, /^\x01(OR|AND|SEQ|PIPE|BG|NL)\x01/)) {
               sep=substr(line, 2, RLENGTH-2); line=substr(line, RLENGTH+1)
             }
             gsub(/^[ \t]+|[ \t]+$/, "", line)
             if (!length(line)) {          # empty clause: FLUSH the buffered one
               if (have) { print psep "\t" sep "\t" pline; have=0 }
               next                        # with the EMPTY record sep as nextsep
             }                             # (fail-closed: a trailing `| <empty>`
                                           # still reports PIPE, never a lost
                                           # separator)
             if (have) print psep "\t" sep "\t" pline
             psep=sep; pline=line; have=1 }
           END { if (have) print psep "\tEND\t" pline }'
}

# (#1538) TRUE iff every downstream pipe-connected clause after record $1
# is a verified read-only text filter, and the chain terminates on a
# non-PIPE, non-BG seam. Fail-closed: any unclassifiable shape -> 1.
# Operates on the driver's buffered record arrays (_seps/_nextseps/_clauses,
# file-global — populated just before the driver loop below).
_pipe_chain_is_readonly_sink() {
  local j=$(( $1 + 1 )) c
  while :; do
    [ "$j" -lt "$_nrec" ] || return 1        # trailing '|' / ran off records
    [ "${_seps[$j]}" = PIPE ] || return 1    # defensive: seam must be PIPE
    c=${_clauses[$j]}   # (#1538 v3) RAW clause text — deliberately NO comment
    # strip: the driver-loop comment-tail strip INVERTS fail-closed direction
    # inside a REFUSAL scan (deleting text can delete a refusal trigger — a
    # quoted space-hash inside a consumer arg truncates the scanned text
    # before a write/exec flag). Instead, ANY '#' in a consumer refuses
    # outright. This also refuses comment-only consumers — a deliberate
    # asymmetry vs the driver, which SKIPS a clause-initial comment: the
    # walker refuses what the driver skips (fail-closed).
    case "$c" in *'#'*) return 1 ;; esac
    [ -n "$c" ] || return 1
    # (a) clause-initial allowlisted read-only filter word (bare word only;
    #     path-spelled /usr/bin/head, env-prefixed, or quoted words refuse)
    echo "$c" | grep -qE '^(head|tail|wc|cat|cut|tr|nl|sort|uniq|grep|egrep|fgrep|rg)([[:space:]]|$)' || return 1
    # (b) (#1538 v3) NO '$' of ANY kind in a consumer — command substitution,
    #     brace expansion, AND bare $NAME (a variable can carry a write/exec
    #     channel flag past the static (d) scan) — plus backtick / procsub /
    #     here-string. Deliberate strict superset of the waiver's cond (3).
    echo "$c" | grep -qE '\$|`|<\(|>\(|<<<' && return 1
    # (c) cond-(3b) parity: no output redirect post-/dev/null-strip
    echo "$c" | sed -E 's@[0-9]*>>?[[:space:]]*/dev/null([[:space:]]|$)@ @g' | grep -q '>' && return 1
    # (d) per-word write/exec channels (VM --help scan, fact-check 2026-07-19)
    if echo "$c" | grep -qE '^(grep|egrep|fgrep|rg)([[:space:]]|$)'; then
      # --pre / --hostname-bin execute programs (rg); -z/--search-zip runs
      # fixed-name decompressors (rg) — refused for the whole family
      # (fail-closed; GNU grep's read-only -z --null-data is a rare-use FP).
      # (#1538 v3) NO trailing anchor on the short-flag bundle — mirrors the
      # sort branch — so a mid-bundle spelling (the gated letter followed by
      # more bundled letters) still catches (GPN8e).
      echo "$c" | grep -qE '(^|[[:space:]])(--(pre|hostname-bin)(=|[[:space:]])|-[A-Za-z]*z|--search-zip)' && return 1
    elif echo "$c" | grep -qE '^sort([[:space:]]|$)'; then
      # -o/--output writes a file with no '>' — incl. bundled (-ro) and
      # glued (-o/tmp/x) short forms; --compress-program executes an
      # arbitrary named program; -T/--temporary-directory writes spill
      # files to an arbitrary dir (refused fail-closed)
      echo "$c" | grep -qE '(^|[[:space:]])(-[A-Za-z]*[oT]|--output|--compress-program|--temporary-directory)' && return 1
    elif echo "$c" | grep -qE '^uniq([[:space:]]|$)'; then
      # uniq's SECOND positional arg is an OUTPUT file; piped usage needs
      # no positional args -> refuse any non-option token (fail-closed;
      # 'uniq -c' passes, 'uniq - /tmp/x' and 'uniq f.txt' refuse)
      echo "$c" | sed -E 's/^uniq//' | tr -s '[:space:][:blank:]' '\n\n' | grep -qE '^[^-]' && return 1
    fi
    case "${_nextseps[$j]}" in
      PIPE) j=$((j + 1)) ;;   # walk the next pipe stage
      BG)   return 1 ;;       # fd-dup mis-split ('2>&1') / true background
      *)    return 0 ;;       # END/SEQ/AND/OR/NL terminate the chain
    esac
  done
}

# Classify a SINGLE clause. Echoes the `blocked` reason (empty string = allow).
# This is the pre-#804 whole-command detector body, applied per-clause: `$c`
# holds one clause. The `[^;&|]*` anchors inside the detectors are no-ops
# per-clause (a clause has no separators) but are kept verbatim so the function
# stays correct even if fed a whole string — zero behavior change.
classify_clause() {
  local c="$1"
  local blocked=""

  # git switch <branch> / git switch -c <branch>  (switch is branch-only).
  # Allow only `git switch main` (bare or quoted — the arg regex tolerates an
  # optional surrounding quote, so `git switch "main"` also passes the allow-arm).
  # The allow-arm ANCHORS `main` to the FULL switch arg: `main` must be followed
  # by an optional trailing quote AND then end-of-string or a shell delimiter
  # (whitespace / `;` / `&` / `|`). A bare `\bmain\b` word boundary would ALSO
  # match before a `-` / `/` / `.` (all non-word chars), so `git switch
  # main-adjacent` / `main/foo` / `main.x` (and their quoted forms) would slip
  # through the allow-arm and LEAK a branch-switch off main. `main_x` still
  # blocks: `_` is a word char so `\bmain` never matched there either, but the
  # explicit terminator makes the intent unambiguous. Concern id:
  # switch-main-prefix-allowarm-leak (#796 round 3).
  if echo "$c" | grep -qE '\bgit\b[^;&|]*\bswitch\b'; then
    if ! echo "$c" | grep -qE '\bswitch\b +(-c +|-C +)?["'"'"']?main["'"'"']?( *($|[;&|]))'; then
      blocked="git switch"
    fi
  fi

  # git checkout -b/-B <branch>  (branch creation). The trailing class matches a
  # space (bare `-b feature`), end-of-clause (bare `-b`), OR a glued branch-name
  # char (`-bfoo`, `-B123`, `-b-x`, `-b.y`, `-b/z`) — the `(-b|-B)\b`
  # word-boundary form missed the glued `-bfoo` (`f` is a word char, no boundary
  # after `b`) and leaked branch creation off main. Concern id:
  # checkout-glued-shortflag-b-leak (#804 / #796 round 3).
  if echo "$c" | grep -qE '\bgit\b[^;&|]*\bcheckout\b +(-b|-B)([[:alnum:]_./[:space:]-]|$)'; then
    blocked="git checkout -b"
  fi

  # git checkout --detach [<ref>]  /  git switch --detach|-d <ref>  — explicit
  # detach. Fires independent of the positional arg: for `checkout --detach abc`
  # the first post-keyword token is the flag, not the ref, so the arg-classifier
  # below would miss it. The switch pattern also catches `git switch -d main`
  # (a detach AT main), which the branch-only switch detector above lets through
  # on the `main` allow-arm.
  # (#1621) The `[^-]` class before the verb token excludes HYPHEN-preceded
  # spellings — the verb inside a `--no-checkout` / `--checkout` FLAG (ERE \b
  # matches between `-` and a word char), so the documented scratch-worktree
  # recipe `git worktree add --no-checkout --detach <path> origin/main` no
  # longer false-blocks (incident 552fa84d). Every real spelling keeps a
  # space/tab before the verb and still blocks; a quote-wrapped verb never
  # matched this clause anyway (the ` +` after the verb breaks on the quote —
  # pre-existing parity, unchanged).
  if echo "$c" | grep -qE '\bgit\b[^;&|]*[^-]\bcheckout\b +(-{1,2})detach\b'; then
    blocked="git checkout --detach"
  fi
  if echo "$c" | grep -qE '\bgit\b[^;&|]*\bswitch\b +(--detach\b|-d\b)'; then
    blocked="git switch --detach"
  fi

  # ---- #897 working-tree-revert detectors -------------------------------
  # Shared TIGHT subcommand anchor: `git` + optional flag[+value] tokens +
  # verb (mirrors workflow_lint.py's _GIT_RESET_HARD_RE flag-group). The
  # legacy loose `\bgit\b[^;&|]*\bverb\b` form would newly block plain-English
  # "restore"/"clean"/"reset" in `-m` messages (`git commit -m "restore
  # defaults"`); the tight anchor only fires on a real `git <verb>` bigram
  # (the raw-scan known limitation then covers only FULL quoted command
  # literals — same contract as the checkout/switch detectors).

  # (#897) Working-tree revert via restore. Allowed ONLY when --staged (long
  # form) is present AND no --worktree / -W anywhere after `restore` —
  # index-only restore leaves the working tree untouched. Everything else
  # (bare `restore .`, explicit-path restore, --source forms, -S short form)
  # blocks, fail-closed: on the SHARED root any working-tree restore can
  # discard a CONCURRENT session's uncommitted edits (incident #841).
  if echo "$c" | grep -qE '\bgit +(-[^ ]+( +[^ ]+)?( +|$))*restore\b'; then
    if ! { echo "$c" | grep -qE '\brestore\b[^;&|]* --staged\b' \
           && ! echo "$c" | grep -qE '\brestore\b[^;&|]*( --worktree\b| -[A-Za-z]*W)'; }; then
      blocked="git restore (working-tree revert)"
    fi
  fi

  # (#897) Working-tree revert via checkout pathspec: a standalone ` -- `
  # separator (git checkout [<ref>] -- <path>), a bare `.`/`./...` positional
  # after checkout, OR a --pathspec-from-file(=|<space>) pathspec source
  # (`git checkout HEAD --pathspec-from-file=/tmp/files` is a real Git
  # pathspec mechanism the standalone-`--` form misses). `--detach`/`-b`
  # never match (the `--` needs a following space/EOL; the dot needs a
  # preceding space). This FLIPS the previous explicit-allow of `.` and the
  # `--` skip — the #841 revert class.
  if echo "$c" | grep -qE '\bgit +(-[^ ]+( +[^ ]+)?( +|$))*checkout\b[^;&|]*( --( |$)| \.($| |/)| --pathspec-from-file(=| +))'; then
    blocked="git checkout <pathspec> (working-tree revert)"
  fi

  # (#897) Force-checkout: `git checkout -f|--force <anything>` discards
  # uncommitted edits even when the target is the CURRENT branch
  # (`git checkout -f main` prints "Already on 'main'" and silently reverts
  # dirty tracked files). There is no legitimate unqualified force-checkout
  # at the repo root (branch switches are already blocked; the `-C` waiver
  # covers deliberate scoped use). The `git switch` side is already strict
  # (its allow-arm admits only bare/quoted `main`), so this closes the
  # checkout asymmetry.
  if echo "$c" | grep -qE '\bgit +(-[^ ]+( +[^ ]+)?( +|$))*checkout\b[^;&|]*( -[A-Za-z]*f\b| --force\b)'; then
    blocked="git checkout --force (working-tree revert)"
  fi

  # (#897) git clean with a force flag (-f anywhere in a short cluster, or
  # --force) deletes untracked files fleet-wide (#841 lost 3 pre_reg + 4
  # images). Dry-run (-n) and bare `git clean` (refuses without force) stay
  # allowed.
  if echo "$c" | grep -qE '\bgit +(-[^ ]+( +[^ ]+)?( +|$))*clean\b'; then
    if echo "$c" | grep -qE '\bclean\b[^;&|]*( -[A-Za-z]*f| --force\b)'; then
      blocked="git clean --force (deletes untracked files)"
    fi
  fi

  # (#897) Runtime `git reset --hard` — the #778/#815 incident class; the
  # workflow_lint check covers doc text only, this is the runtime tooth.
  # Tolerates flags/refs between reset and --hard (`git reset origin/main
  # --hard`).
  if echo "$c" | grep -qE '\bgit +(-[^ ]+( +[^ ]+)?( +|$))*reset\b[^;&|]* --hard(=|\b)'; then
    blocked="git reset --hard"
  fi

  # ---- #1128 branch-merge fence ------------------------------------------
  # `git merge <ref>` on the shared root: a conflicting merge strands
  # conflict markers in the shared tree until aborted (#1090: ~70s window a
  # concurrent `git add && git commit` could sweep), and even a clean/ff
  # merge lands branch commits on root main outside the Step 10d landing
  # path. TIGHT anchor: `merge` must be followed by whitespace/end-of-clause
  # — NOT `\b`, which fires before `-` and would trip `git merge-base`
  # (run BARE at the root by the diff-size-budget sizing recipe and the
  # Step 10d ancestry probes). Allow-arm: `--abort` / `--quit` are the
  # sanctioned in-progress-merge RECOVERY (the #1090 session recovered via
  # abort; fail-soft: never trap a user mid-recovery) — the flag must
  # IMMEDIATELY follow the verb (`git merge --abort` accepts no ref, so
  # the anchored form costs zero FPs and a quoted `-m "… --abort …"`
  # message cannot spoof the allow (raw scan reads quoted args; the
  # loose `[^;&|]*` form would fail open there). BLOCKED fail-closed:
  # `--continue` (it COMPLETES exactly the root merge commit this fence
  # prevents; recovery is abort — residual gap (xv) names the ungated
  # `git commit` equivalent) and `--ff-only` (cannot conflict, but still
  # lands branch commits on root main outside the landing path; worktree
  # ff-syncs use fetch + `git -C <worktree> merge --ff-only origin/main`
  # — never local main, whose unpushed root commits contaminate the
  # branch (#1530) — root syncs use sync_repo_root.py).
  if echo "$c" | grep -qE '\bgit +(-[^ ]+( +[^ ]+)?( +|$))*merge( +|$)'; then
    if ! echo "$c" | grep -qE '\bmerge +--(abort|quit)\b'; then
      blocked="git merge (branch merge on the shared root)"
    fi
  fi

  # ---- #1193 rebase-family fence (sibling of the #1128 merge fence) -------
  # `git rebase <ref>` / `git cherry-pick <ref>` on the shared root: a
  # conflicting run strands conflict state in the shared tree exactly like
  # the #1090 root merge, and a clean run rewrites/lands commits on root
  # main outside the sanctioned landing paths (gh pr merge --rebase /
  # scratch worktree / sync_repo_root.py — a bare `git rebase` with a
  # configured upstream genuinely RUNS, so end-of-clause blocks too).
  # TIGHT anchor: verb followed by whitespace/end-of-clause — NOT `\b`,
  # which fires before `.`/`=` and would trip `git -c rebase.autoStash=true
  # pull` at flag-value position. Allow-arm mirrors #1128: --abort/--quit
  # IMMEDIATELY after the verb (sanctioned recovery; fail-soft), ONE ARM PER
  # VERB — a combined `(rebase|cherry-pick)` allow would open a cross-verb
  # quoted-arg spoof (a quoted argument naming the OTHER verb + --abort
  # would satisfy it while the real verb runs). BLOCKED fail-closed:
  # --continue and --skip (both COMPLETE the in-progress operation on the
  # root tree — the M5 decision, mirrored; recovery is abort). Note `git
  # pull --rebase[=merges]` never reaches this anchor: its subcommand is
  # `pull` (the sanctioned root-sync form stays open).
  if echo "$c" | grep -qE '\bgit +(-[^ ]+( +[^ ]+)?( +|$))*rebase( +|$)'; then
    if ! echo "$c" | grep -qE '\brebase +--(abort|quit)\b'; then
      blocked="git rebase (history rewrite on the shared root)"
    fi
  fi
  if echo "$c" | grep -qE '\bgit +(-[^ ]+( +[^ ]+)?( +|$))*cherry-pick( +|$)'; then
    if ! echo "$c" | grep -qE '\bcherry-pick +--(abort|quit)\b'; then
      blocked="git cherry-pick (commit replay onto the shared root)"
    fi
  fi
  # ---- end #1193 rebase-family fence ---------------------------------------

  # ---- #1234 revert/am fence (completeness siblings of the #1193 family) --
  # `git revert <commit>` / `git am <mbox>` on the shared root: the same
  # conflict-stranding class — a conflicting run strands sequencer/am state
  # + conflict markers in the shared tree (#1090 class), and a clean run
  # lands commits on root main outside the sanctioned landing paths.
  # `git revert -n/--no-commit` still mutates index+tree (blocked, CP5
  # parity); bare `git revert` errors in git but blocks fail-closed at zero
  # cost (M7/CP2 parity); bare `git am` reads patches from STDIN and
  # genuinely runs, so end-of-clause blocks are load-bearing there.
  # TIGHT anchor: verb followed by whitespace/end-of-clause. Prose safety:
  # `git commit -m "revert foo"` / `-m "I am done"` never match — `commit`
  # is a non-dash token that breaks the flag chain (see the pre-filter
  # comment's #1234 boundary analysis). Allow-arm mirrors #1128/#1193:
  # --abort/--quit IMMEDIATELY after the verb, ONE ARM PER VERB (a combined
  # `(revert|am)` allow would open the R13 cross-verb quoted-arg spoof;
  # `am`'s short allow-anchor additionally admits a quoted prose
  # "... am --abort ..." spoof — the raw-scan accidents-not-adversaries
  # residual, register (xviii)(b)). BLOCKED fail-closed: --continue/--skip
  # (both COMPLETE the in-progress op on the root tree — the M5 decision,
  # mirrored) and `am --show-current-patch` (read-only but rare; strict
  # abort/quit-only parity keeps the allow surface auditable — register
  # (xviii)(a); recovery for an in-progress root am is --abort).
  if echo "$c" | grep -qE '\bgit +(-[^ ]+( +[^ ]+)?( +|$))*revert( +|$)'; then
    if ! echo "$c" | grep -qE '\brevert +--(abort|quit)\b'; then
      blocked="git revert (revert commit onto the shared root)"
    fi
  fi
  if echo "$c" | grep -qE '\bgit +(-[^ ]+( +[^ ]+)?( +|$))*am( +|$)'; then
    if ! echo "$c" | grep -qE '\bam +--(abort|quit)\b'; then
      blocked="git am (mailbox patch apply onto the shared root)"
    fi
  fi
  # ---- end #1234 revert/am fence --------------------------------------------
  # ---- end #897 detectors ------------------------------------------------

  # git checkout <existing-branch>  — NOT a pathspec form (no `--`; those are
  # blocked by the #897 pathspec detector above, whose reason wins because
  # the `--` guard below skips this classifier), arg is a real local branch
  # ref, and not `main`. Extended: a non-branch arg that resolves to a
  # commit-ish (sha / tag / origin/<branch> / HEAD~N / HEAD@{N}) DETACHES HEAD
  # and is blocked too; a non-commit-ish arg that names a REAL tracked path /
  # file is the classic bare-pathspec discard idiom (`git checkout
  # tasks/running/841/body.md` — the exact #841 op) and is blocked by the
  # existence probe in the `*)` arm. A flag-prefixed detach (`checkout -f
  # <sha>`, `-q <sha>`, `-p <sha>`, `-m <sha>`) is caught by re-scanning the
  # post-`checkout` tokens: skip known safe short-flags, then classify the
  # first positional. `-b`/`-B` are NOT skipped here (branch creation is
  # already blocked above); `main`/`-` are left as ALLOW; `.` stays a case
  # no-op here because the #897 pathspec detector above already blocked it.
  if echo "$c" | grep -qE '\bgit\b[^;&|]*\bcheckout\b' \
     && ! echo "$c" | grep -qE 'checkout\b[^;&|]*--'; then
    arg=$(echo "$c" | sed -nE 's/.*\bcheckout\b +([^ ;&|]+).*/\1/p')
    # Flag-prefixed detach: skip leading safe short-flags to reach the first
    # positional (e.g. `checkout -f <sha>` -> classify `<sha>`).
    if echo "$arg" | grep -qE '^-[fqpm]$'; then
      rest=$(echo "$c" | sed -nE 's/.*\bcheckout\b +(.*)/\1/p')
      # shellcheck disable=SC2086  # word-splitting is intentional here
      set -- $rest
      while [ $# -gt 0 ]; do
        case "$1" in
          -f|-q|-p|-m) shift ;;  # safe short-flag — skip to the next token
          *) arg="$1"; break ;;  # first positional
        esac
      done
    fi
    # Strip a single layer of surrounding quotes so a QUOTED ref classifies as
    # its bare form: `git checkout "HEAD~1"` -> HEAD~1 (detaches -> block),
    # `git checkout "main"` -> main (allow). Quoted refs are shell-equivalent to
    # unquoted ones; without this strip the quoted arg would either miss the
    # `main` allow-arm (false positive) or, before the round-1 quote-strip was
    # reverted, be erased entirely (leak). Only trailing/leading `"` or `'`.
    arg=${arg#[\"\']}
    arg=${arg%[\"\']}
    case "$arg" in
      # `.` and `-f`/`--force` are unreachable for BLOCKING purposes here —
      # the #897 pathspec / force detectors above already set `blocked` for
      # them; they stay listed so this classifier never re-classifies them
      # with a wrong reason. `main`/`-` remain genuine allows.
      ""|-b|-B|-f|--force|main|-|.) : ;;
      --*) : ;;                           # a flag (e.g. --detach handled above)
      *)
        if git -C "$REPO" show-ref --verify --quiet "refs/heads/$arg"; then
          blocked="git checkout $arg"                        # branch switch
        elif git -C "$REPO" rev-parse --verify --quiet "$arg^{commit}" >/dev/null 2>&1; then
          blocked="git checkout $arg (detaches HEAD)"         # sha/tag/remote-ref/HEAD~N
        elif git -C "$REPO" cat-file -e "HEAD:$arg" 2>/dev/null || [ -e "$REPO/$arg" ]; then
          # (#897) bare-pathspec existence probe: `git checkout <path>` with
          # NO ref, NO `--`, NO dot — the classic pre-`git restore` discard
          # idiom (the exact #841 op). An arg that resolves to NO branch, NO
          # commit-ish, and NO tracked/existing path keeps the status-quo
          # allow (git itself errors on it), so unresolvable variables /
          # redirection tokens gain no new false-positive class. Quoted-glob
          # pathspecs stay a NAMED residual gap (header block, item iv).
          blocked="git checkout $arg (working-tree revert of a tracked path)"
        fi
        ;;
    esac
  fi

  echo "$blocked"
}

# Drive classify_clause over the (separator, next-separator, clause) triples
# (every consumer below reads $sep/$clause with unchanged values and ordering;
# $nextsep is consumed only by the #1098 waiver). A `cd <worktree|/tmp>`
# latches `scoped` forward ONLY across `&&` (bash GUARANTEES the `cd` succeeded
# before the RHS runs there, so the cwd persists), so a git clause after it runs
# in the scoped cwd and is allowed. The latch RESETS across every OTHER separator
# — `;` (SEQ), `||` (OR), `|` (PIPE), `&` (BG), and a raw newline (NL) — where
# bash does NOT guarantee the `cd` took effect for the following clause (verified
# bash semantics 2026-07-01: `cd X && pwd` prints X; `cd X ; pwd` prints X ONLY on
# `cd` success — a FAILED `cd`, e.g. a missing target, leaves the ORIGINAL cwd —
# and `cd X || pwd` / `cd X | pwd` / `cd X & pwd` / a `cd X<newline>pwd` all print
# the ORIGINAL cwd on `cd` failure). Resetting on `;` / NL fails CLOSED (#804
# rounds 2/3): the guard cannot prove a `;`- or newline-preceding `cd` succeeded,
# so it declines to scope across it. The first blocking clause wins.
scoped=0
scoped_wt=0   # (#1554) whether the live scoped-latch was armed by a WORKTREE cd
# (#1861) STICKY scope — granted only by a provably exit-guarded WORKTREE cd
# (`cd <wt> || exit N` / `cd <wt> || { ...; exit N; }`, recognized by
# _cd_guard_tail_exits below): once ACTIVE, the reset branch RESTORES scope
# across every separator instead of clearing it (either the cd succeeded, or
# the shell exited before any later clause ran). Activation is DEFERRED via
# sticky_arm_at (the terminator clause's index): the OR-tail's own clauses
# execute exactly when the cd FAILED (cwd = repo root) and stay unscoped.
sticky=0
sticky_wt=0
sticky_arm_at=-1
sticky_pending_wt=0
# (#1861) Name-generalized $VAR cd-latch state: _wt_names[<name>]=1 records a
# same-command bare/export worktree-literal assignment to <name> (replaces
# the #1058 single-name `wt_bound` flag).
declare -A _wt_names=()
blocked=""
# (#1554) Worktree-local bare-main merge fence — shared definitions. Escape
# hatch (sibling convention: EPM_ALLOW_ROOT_PULL, guard_repo_root_pull.sh
# L81-112): honored as session env AND as an inline prefix on the command
# itself (a command-wide substring match — see the header note).
_wt_lm_allow=0
[ "${EPM_ALLOW_WORKTREE_LOCAL_MAIN_MERGE:-0}" = "1" ] && _wt_lm_allow=1
case "$cmd" in *EPM_ALLOW_WORKTREE_LOCAL_MAIN_MERGE=1*) _wt_lm_allow=1 ;; esac
# (#1554) bare-local-main merge ref tail: optional --long-flags on both sides
# of the ref, optional redirect tokens after; NOT origin/main (the char before
# `main` must be whitespace/quote — `origin/main` has `/` there).
_WT_LM_TAIL='merge( +--[A-Za-z-]+(=[^ ]*)?)* +["'"'"']?main["'"'"']?( +--[A-Za-z-]+(=[^ ]*)?| +[0-9]*[<>]+[^ ]*)*( *($|[;&|]))'
# Arm A: clause-initial `git -C <worktree-path | $WT spelling>` + the tail.
_WT_LM_ARM_A='^git +-C +([^ ]*\.claude/worktrees/[^ ]*|["'"'"']?\$(WT\b|\{WT\})[^ ]*) +(-[^ ]+( +[^ ]+)? +)*'"$_WT_LM_TAIL"
# Arm B: clause-initial bare `git ... merge ... main` (worktree-latch-gated).
_WT_LM_ARM_B='^git +(-[^ ]+( +[^ ]+)? +)*'"$_WT_LM_TAIL"
# (#1538) Buffer the (sep, nextsep, clause) triples so the grep-family
# pipe waiver can look ahead at downstream pipe-connected consumer
# clauses (_pipe_chain_is_readonly_sink above). read -r keeps
# trailing-field semantics: a clause containing a literal tab still
# lands whole in _c (the last var takes the remainder). Loop BODY is
# byte-identical to the former streaming `while read` form (`continue`/
# `break` semantics preserved by `for`); under `set -u` an empty array
# makes `"${!_clauses[@]}"` a no-op loop, matching the former
# zero-record behavior.
_seps=(); _nextseps=(); _clauses=()
while IFS=$'\t' read -r _s _n _c; do
  _seps+=("$_s"); _nextseps+=("$_n"); _clauses+=("$_c")
done < <(split_and_label "$cmd")
_nrec=${#_clauses[@]}

# (#1861) Exit-guard tail recognizer — called when a WORKTREE cd clause's
# next separator is OR. Verifies the OR-tail PROVABLY terminates the shell:
#   Case A: a bare `exit [N]` clause;
#   Case B: a brace group whose final clause before the closing bare `}` is
#           an unconditionally-reached (SEQ/NL-preceded) `exit [N]`;
# and that the terminator is not defused by a following PIPE/BG separator
# (`|| exit 1 | op` runs the exit in a pipeline subshell; `|| exit 1 & op`
# backgrounds the whole and-or list). `return` is deliberately NOT accepted:
# at top level — the only context these hook-scanned command strings execute
# in — a bare `return` errors and the shell CONTINUES at the root. On
# success prints the TERMINATOR clause's index (the `exit` clause for Case
# A; the `}` clause for Case B) and returns 0; any other shape returns 1
# (fail-closed). Operates on the driver's buffered record arrays
# (_seps/_nextseps/_clauses/_nrec). The splitter is quote-blind, so a
# group-internal redirect like `>&2` splits into fragment clauses — the
# recognizer keys on the last-before-`}` clause precisely so those fragments
# don't matter (quote-blindness residual (a) in the #1861 header note).
_cd_guard_tail_exits() {
  local i=$(( $1 + 1 )) j last c
  local exit_re='^exit( +[0-9]+)?[[:space:]]*$'
  local cwd_word_re='^[[:space:]({]*((builtin|command)([[:space:]]+-[A-Za-z]+)*[[:space:]]+)*(cd|pushd|popd)([[:space:]]|$)'
  [ "$i" -lt "$_nrec" ] || return 1
  [ "${_seps[$i]}" = OR ] || return 1      # defensive: seam must be OR
  c=${_clauses[$i]}
  # Case A: bare `|| exit [N]` — the terminator is the exit clause itself.
  if echo "$c" | grep -qE "$exit_re"; then
    case "${_nextseps[$i]}" in
      SEQ|NL|AND|OR|END) printf '%s' "$i"; return 0 ;;
    esac
    return 1
  fi
  # Case B: `|| { ...; exit [N]; }` — the terminator is the closing bare `}`.
  case "$c" in
    '{'|'{ '*) : ;;
    *) return 1 ;;
  esac
  case "${c#\{}" in *'{'*) return 1 ;; esac   # nested `{` in the opener
  echo "$c" | grep -qE "$cwd_word_re" && return 1   # `{ cd ...` opener
  last=$i
  j=$(( i + 1 ))
  while [ "$j" -lt "$_nrec" ] && [ "$j" -le $(( i + 10 )) ]; do
    c=${_clauses[$j]}
    if [ "$c" = '}' ]; then
      case "${_nextseps[$j]}" in
        SEQ|NL|AND|OR|END) : ;;
        *) return 1 ;;      # `}` feeding PIPE/BG defuses the terminator
      esac
      if [ "$last" -eq "$i" ]; then
        # Single-clause group (`{ exit 1` + `}`): the exit rides the opener.
        echo "${_clauses[$last]}" \
          | grep -qE '^\{[[:space:]]+exit( +[0-9]+)?[[:space:]]*$' || return 1
      else
        # The clause immediately before `}` must be an exit reached
        # UNCONDITIONALLY (SEQ/NL-preceded — `{ foo && exit 1; }` refuses).
        echo "${_clauses[$last]}" | grep -qE "$exit_re" || return 1
        case "${_seps[$last]}" in SEQ|NL) : ;; *) return 1 ;; esac
      fi
      printf '%s' "$j"
      return 0
    fi
    # Group-internal refusals (fail-closed): nested `{`, and any
    # cwd-changing command word (mirrors the driver invalidation regex).
    case "$c" in *'{'*) return 1 ;; esac
    echo "$c" | grep -qE "$cwd_word_re" && return 1
    last=$j
    j=$(( j + 1 ))
  done
  return 1                  # no closing `}` within the 10-clause scan bound
}

for _idx in "${!_clauses[@]}"; do
  sep=${_seps[$_idx]}; nextsep=${_nextseps[$_idx]}; clause=${_clauses[$_idx]}
  # (#1861) Deferred sticky activation: the exit-guard recognizer recorded
  # the terminator clause's index in sticky_arm_at; every clause PAST it is
  # provably scoped (the cd succeeded, or the shell already exited). The
  # tail/group-internal clauses themselves (_idx <= sticky_arm_at) keep the
  # UNSCOPED classification — they run on the cd-failure path, at the root.
  if [ "$sticky_arm_at" -ge 0 ] && [ "$_idx" -gt "$sticky_arm_at" ]; then
    sticky=1
    sticky_wt=$sticky_pending_wt
    sticky_arm_at=-1
  fi
  # Reset the latch unless the separator BEFORE this clause is && — a `cd`
  # only reliably scopes a following git clause when bash guarantees it ran
  # first (the && short-circuit). ; / || / | / & / a raw newline (NL) do NOT
  # carry the latch (NL is not AND, so this consolidated check resets it).
  # (#1861) Under an ACTIVE sticky grant the reset RESTORES scope instead of
  # clearing it (sticky=0 preserves the pre-#1861 behavior byte-identically).
  if [ "$sep" != AND ]; then
    scoped=$sticky
    scoped_wt=$sticky_wt
  fi

  # (#897) A clause whose first non-space char is `#` is a bash comment — bash
  # never executes it, so classifying it can only false-positive (executed
  # SKILL.md fences carry comments that SPELL gated forms, e.g. the Step 10d
  # additive-checkout fence's `git checkout issue-<N> -- <path>` comment). A
  # comment containing a separator is mis-split and its TAIL clause still
  # classifies — that mis-split fails CLOSED (blocks), the safe direction.
  # (The awk splitter already stripped leading whitespace from each clause.)
  case "$clause" in "#"*) continue ;; esac

  # (#897 round 2) Strip the unquoted comment TAIL of the clause before ANY
  # latch / waiver / gate / classification read. Bash never EXECUTES text
  # after a whitespace-anchored `#`, but the raw scan previously READ it —
  # so trailing comment text could SPOOF the `-C` waiver, the restore
  # `--staged` allow-arm, or a `cd`-latch: `git restore . # git -C /tmp
  # status` exited 0 while bash executed the destructive revert (the round-2
  # BLOCKER class, concern id comment-tail-waiver-spoof). Stripping the tail
  # makes every downstream read see (an approximation of) exactly what bash
  # executes: comment text can never GRANT an allow, a comment SPELLING a
  # gated form no longer false-blocks (`git clean -n # -f` is a dry-run and
  # now allows), and the greedy checkout-arg extraction can no longer be
  # steered by comment text. This is NOT the reverted #796 quote-strip:
  # quoted spans are PRESERVED — only a whitespace-anchored ` #` tail is cut,
  # without shell-parsing quotes, so a QUOTED ` # ` argument also truncates
  # (see the header known-limitation paragraph + residual gap (viii); a `#`
  # glued to non-space text, e.g. `path#frag`, never matches). Truncation is
  # fail-closed for allows (removing text cannot add an allow token) and can
  # err fail-open ONLY in the quoted-` # ` residual (viii).
  clause=$(printf '%s' "$clause" | sed -E 's/[[:space:]]#.*$//')
  [ -n "$clause" ] || continue

  # (#1861) Scope invalidation — fail-closed, deliberately BROADER than
  # arming: ANY clause whose command word (tolerating a leading run of
  # `(` / `{` / whitespace and optional builtin/command prefixes) is
  # cd/pushd/popd voids BOTH the plain latch and the sticky scope — the cwd
  # proof is stale once a later clause may change directories. The
  # worktree//tmp/$VAR arms below then RE-ARM as appropriate; those arms
  # stay ^cd-anchored, so a paren-prefixed `(cd ...` invalidates but never
  # arms (subshell), and a brace-group `{ cd ...` invalidates but never arms
  # (it runs in the PARENT shell — the cd persists — but the group's
  # reachability is unprovable here).
  if echo "$clause" \
     | grep -qE '^[[:space:]({]*((builtin|command)([[:space:]]+-[A-Za-z]+)*[[:space:]]+)*(cd|pushd|popd)([[:space:]]|$)'; then
    sticky=0; sticky_wt=0; sticky_arm_at=-1; sticky_pending_wt=0
    scoped=0; scoped_wt=0
  fi

  # A CLAUSE-INITIAL `cd` into a worktree / /tmp latches scope forward ONLY
  # across a following `&&` clause. Latch and continue — this clause runs the
  # `cd`, not a git command, and it must NOT scope EARLIER clauses (those were
  # classified before it). The greps are ^-anchored (#1443): the splitter
  # strips leading whitespace from every clause (L833), so a legit post-split
  # `&& cd /tmp/x` fragment is clause-initial by construction, while quoted
  # latch-vocab text mid-clause — an ssh payload fragment, echo'd prose, or a
  # superstring like `cdx .claude/worktrees/` — never arms. (`cd<TAB>` never
  # armed pre- or post-anchor: `cd +` matches spaces only — pre-existing.)
  # (#1861) Arming-separator restriction (fail-closed): an OR-preceded cd is
  # not provably executed in the parent shell (`a || cd X && op` runs op
  # with the cd SKIPPED when a succeeded) and a PIPE-preceded cd runs in a
  # pipeline subshell — neither arms any latch. (START/SEQ/NL/AND/BG all run
  # the cd in the parent shell before the following clause is reached.)
  if echo "$clause" | grep -qE '^cd +[^;&|]*\.claude/worktrees/'; then
    if [ "$sep" != OR ] && [ "$sep" != PIPE ]; then
      scoped=1
      scoped_wt=1   # (#1554) latch armed by a WORKTREE cd
      # (#1861) Exit-guarded cd => sticky grant (WORKTREE arms only): when
      # the NEXT separator is ||, a provably-exiting guard tail proves every
      # clause past the terminator runs with cwd inside the worktree.
      if [ "$nextsep" = OR ] && _t=$(_cd_guard_tail_exits "$_idx"); then
        sticky_pending_wt=1
        sticky_arm_at=$_t
      fi
    fi
    continue
  elif echo "$clause" | grep -qE '^cd +/tmp/'; then
    if [ "$sep" != OR ] && [ "$sep" != PIPE ]; then
      scoped=1
      scoped_wt=0   # (#1554) /tmp latch — disposition byte-identical to before
      # (#1861) NO sticky for /tmp latches: every observed firing is
      # worktree-side; smaller blast radius to leave it (plan §6).
    fi
    continue
  fi

  # (#1058; #1861 name-generalized) A `<NAME>=<...>.claude/worktrees/<...>`
  # BARE-ASSIGNMENT clause (optionally `export`-prefixed; NOTHING after the
  # RHS) arms the $NAME latch
  # for LATER clauses — and ONLY when its preceding separator proves the
  # assignment executes unconditionally in the parent shell: START, `;`
  # (SEQ), or a raw newline (NL). Three arming refusals, each verified live
  # (bash 5.1.16):
  #   - assignment PREFIX (`WT=... true`): a per-command temporary env
  #     assignment that does NOT persist — the end-of-clause anchor
  #     ([[:space:]]*$ after the RHS) rejects any trailing command word;
  #   - AND/OR-preceded assignment (`[ -d x ] && WT=...`): runtime-
  #     conditional — when skipped, `cd "$WT"` is a `cd ""` repo-root no-op
  #     and the git clause would run UNSCOPED at the root;
  #   - PIPE-preceded assignment (`true | WT=...`): runs in a pipeline
  #     subshell, does not persist. (A BG-preceded clause DOES run in the
  #     parent shell, but stays non-arming as fail-closed conservatism —
  #     zero incident demand.)
  # Any OTHER clause-initial assignment to the same name (non-worktree RHS,
  # trailing command word, conditional/subshell separator) DISARMS the
  # latch — a reassignment makes the earlier arming proof stale.
  # Clause-initial only: a `WT=` fragment buried in quoted prose (an echo /
  # --note argument) never matches the ^ anchor and neither arms nor
  # disarms. (#1861) The latch is keyed by VARIABLE NAME (`_wt_names`
  # assoc array — formerly the WT-only `wt_bound` flag), and DISARM is
  # deliberately BROADER than arming: declare/local/typeset/readonly-
  # prefixed (with option flags) and `NAME+=` assignment shapes count as
  # reassignment and unbind the name; only the strict bare/`export `
  # worktree-literal end-anchored shape under an unconditional separator
  # (re-)binds it.
  if echo "$clause" \
     | grep -qE '^((export|declare|local|typeset|readonly)([[:space:]]+-[A-Za-z]+)*[[:space:]]+)?[A-Za-z_][A-Za-z0-9_]*\+?='; then
    _an=$(echo "$clause" \
      | sed -nE 's/^((export|declare|local|typeset|readonly)([[:space:]]+-[A-Za-z]+)*[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)\+?=.*/\4/p')
    if [ -n "$_an" ]; then
      unset "_wt_names[$_an]"
      case "$sep" in
        START|SEQ|NL)
          if echo "$clause" | grep -qE '^(export +)?[A-Za-z_][A-Za-z0-9_]*=[^;&|[:space:]]*\.claude/worktrees/[^;&|[:space:]]*[[:space:]]*$'; then
            _wt_names[$_an]=1
          fi
          ;;
      esac
    fi
  fi

  # (#1058; #1861 name-generalized) `cd "$NAME"` — the SKILL.md-conventional
  # worktree-variable form — latches ONLY when an EARLIER clause in this
  # SAME command bound NAME to a `.claude/worktrees/` path (shell state
  # never persists across Bash tool calls, so a non-empty $NAME implies a
  # same-call assignment). The
  # assignment check is LOAD-BEARING, not cosmetic: bash `cd ""` SUCCEEDS as
  # a no-op (verified 2026-07-05, bash 5.1.16), so with an UNSET NAME
  # `cd "$NAME" && git ...` runs the git clause in the UNCHANGED cwd (the
  # repo root) — a bare `cd "$NAME"` latch would be fail-open. With the
  # assignment present, every quoting variant is safe under the && latch:
  # expanded forms cd into the worktree; a single-quoted literal `cd '$WT'`
  # fails (no such dir) and && short-circuits the git clause. A `..`
  # anywhere in the cd arg never latches; whole-arg forms $NAME, ${NAME},
  # $NAME/..., ${NAME}/... only; the #1861 OR/PIPE arming-separator
  # restriction applies as on the literal arms (fail-closed).
  if [ "${#_wt_names[@]}" -gt 0 ]; then
    cdarg=$(echo "$clause" | sed -nE 's/^cd +([^;&|]+)[[:space:]]*$/\1/p' | tr -d '\042\047')
    case "$cdarg" in
      *..*) : ;;
      \$*)
        _vn=$(printf '%s\n' "$cdarg" \
          | sed -nE 's@^\$(\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))(/.*)?$@\2\3@p')
        if [ -n "$_vn" ] && [ -n "${_wt_names[$_vn]:-}" ] \
           && [ "$sep" != OR ] && [ "$sep" != PIPE ]; then
          scoped=1
          scoped_wt=1   # (#1554) $NAME latch is a worktree latch by construction
          # (#1861) Exit-guarded cd => sticky grant (same as the literal arm).
          if [ "$nextsep" = OR ] && _t=$(_cd_guard_tail_exits "$_idx"); then
            sticky_pending_wt=1
            sticky_arm_at=$_t
          fi
          continue
        fi
        ;;
    esac
  fi
  if [ "$scoped" -eq 1 ]; then
    # (#1554 Arm B) A worktree-scoped clause merging the LOCAL main branch is
    # the #1530 contamination class (stale/unpushed root-main commits import
    # into the branch). origin/main + raw-sha merges pass; /tmp latches keep
    # the pre-#1554 disposition byte-identical (scoped_wt=0 -> plain continue).
    if [ "$scoped_wt" -eq 1 ] && [ "$_wt_lm_allow" -ne 1 ] \
       && echo "$clause" | grep -qE "$_WT_LM_ARM_B"; then
      blocked="cd <worktree> && git merge main (LOCAL-main merge imports unpushed root commits, #1530; use fetch + git merge --ff-only origin/main — recipe below; deliberate override: EPM_ALLOW_WORKTREE_LOCAL_MAIN_MERGE=1)"
      break
    fi
    continue          # this clause runs in a scoped cwd
  fi

  # (#1554 Arm A) Worktree-scoped LOCAL-main merge fence — must run BEFORE the
  # path-blind -C waiver below (which would waive exactly this shape).
  # ^-anchored per the #1443 convention: quoted latch/verb vocabulary
  # mid-clause (grep patterns, echo'd prose, ssh payloads) never matches.
  if [ "$_wt_lm_allow" -ne 1 ] && echo "$clause" | grep -qE "$_WT_LM_ARM_A"; then
    blocked="git -C <worktree> merge main (LOCAL-main merge imports unpushed root commits, #1530; use fetch + git -C <worktree> merge --ff-only origin/main — recipe below; deliberate override: EPM_ALLOW_WORKTREE_LOCAL_MAIN_MERGE=1)"
    break
  fi

  # `git -C <path>` scopes ONLY this clause (per-invocation) — allow it.
  echo "$clause" | grep -qE '\bgit +-C +' && continue

  # not a git checkout/switch/restore/clean/reset/merge/rebase/cherry-pick/
  # revert/am clause at all -> skip (loose gate — a cheap skip; the tight
  # anchors live in classify_clause; kept in sync with the whole-command
  # pre-filter above, whose comment carries the per-verb boundary analysis
  # incl. the #1193/#1234 rebase/cherry-pick/revert/am notes)
  echo "$clause" | grep -qE '\bgit\b.*\b(checkout|switch|restore|clean|reset|merge|rebase|cherry-pick|revert|am)\b' || continue

  # (#1098) ssh REMOTE-COMMAND / grep-family PATTERN-ARGUMENT clause waiver.
  # An `ssh <host> '<remote cmd>'` clause executes its command string on the
  # REMOTE host (a pod's own /workspace clone), never in this VM's repo-root
  # working tree — the same operation class the THREAT MODEL paragraph
  # already scopes out for SSH-MCP remote commands (incident 2026-07-06:
  # `ssh pod-779 'git reset --hard origin/main'` false-blocked; the #779
  # session detoured through a pod-side script). A grep/egrep/fgrep/rg
  # clause is read-only w.r.t. git — its pattern argument is data. Waive
  # (skip classification of) such a clause ONLY when ALL of:
  #   (1) the clause's COMMAND WORD is ssh|grep|egrep|fgrep|rg (^-anchored;
  #       the awk splitter already stripped leading whitespace). A MID-clause
  #       ssh/grep word never waives — `git -c core.sshCommand=ssh reset
  #       --hard` is a LOCAL destructive op and still classifies. Wrapped
  #       forms OTHER than the literal timeout prefix (`nohup ssh ...`,
  #       `/usr/bin/ssh`, `$SSHCMD ...`, env-prefix heads) are NOT
  #       clause-initial `ssh` and keep blocking (fail-closed residual FP,
  #       gap (xiv)). As of #1463 a clause-initial
  #       `gcloud compute ssh` word sequence ALSO satisfies cond (1), and
  #       as of #1859 BOTH remote-exec heads accept an optional literal
  #       `timeout <num>[.frac][smhd]?` prefix — the ONLY tolerated
  #       wrapper; `timeout` FLAG forms (`--signal=`, `-k`) never match
  #       (every real #825 fleet invocation was timeout-wrapped, and the
  #       #1769 failover path hit the same false block on the bare `ssh`
  #       head — gap (xix)). The gcloud head
  #       routes through the ssh branch of arm (4): gcloud is a thin
  #       wrapper that shells out to the LOCAL ssh(1) binary and forwards
  #       --ssh-flag / `-- SSH_ARGS` to it, so the same local-exec
  #       smuggling channels apply verbatim (SDK 576.0.0 help). Release
  #       tracks (`gcloud beta|alpha compute ssh`) and every other wrapper
  #       are NOT clause-initial matches and keep blocking (gap (xix)).
  #       Head-regex PARITY with the mask candidate head above is an
  #       invariant (drift is fail-closed — see the mask design comment).
  #   (2) pipeline-producer / background position, PER SUB-ARM (#1538 —
  #       formerly one shared consumer-independent refusal): the clause's
  #       FOLLOWING separator (the $nextsep field the splitter emits) is
  #       read against the arm's own policy. PIPE risk: a waived
  #       producer's stdout can feed a LOCAL shell consumer
  #       (`ssh host 'echo git reset --hard' | bash`,
  #       `grep 'git reset --hard' f | bash`) whose own clause carries no
  #       gated text and clears the loose gate — the round-1 Codex
  #       methodology blocker; any widening MUST therefore inspect the
  #       downstream consumer chain (consumer-independent widening is
  #       forbidden). SSH/GCLOUD arm: refuses BOTH PIPE and BG,
  #       consumer-independently, UNCHANGED — a remote command's stdout
  #       is arbitrary remote-generated text. GREP-FAMILY arm: nextsep
  #       PIPE is waived IFF _pipe_chain_is_readonly_sink() POSITIVELY
  #       verifies every downstream pipe-connected consumer (allowlisted
  #       read-only filter word, no `$`/`#`/backtick/procsub/here-string,
  #       no output redirect, no per-word write/exec channel flag, chain
  #       terminating on a non-PIPE non-BG seam — fail-closed: anything
  #       unverifiable falls through to classification); nextsep BG still
  #       refuses unconditionally. BG (implementation-round fail-closed
  #       widening, live-probed): an fd-dup redirection's single `&`
  #       (`2>&1`) is mis-split as a BG separator by the raw sed pre-pass,
  #       so `ssh host '...' 2>&1 | bash` reports nextsep=BG on its
  #       producer clause — the PIPE hides one record downstream; refusing
  #       BG closes that hole, and a TRUE background producer
  #       (`ssh pod '...' & ...`) costs only a residual FP. The ssh-arm
  #       refusal + the grep-family BG refusal remain strictly
  #       status-quo-preserving; the grep-family verified-PIPE branch is
  #       a strict SUPERSET of the former allow set, confined to chains
  #       the walker proves read-only (residual FPs + the
  #       producer-vs-consumer channel-set asymmetry documented in gap
  #       (xiv); the `git -C` remediation pipes fine — the -C waiver is
  #       pipe-blind).
  #   (3) NO locally-executing expansion / redirection syntax anywhere in
  #       the clause: $( / ${ / backtick / <( / >( / <<< .
  #       `ssh host "$(git reset --hard)"` and `grep -f <(git clean -fd) x`
  #       EXECUTE the gated text LOCALLY at expansion time. The live-clause
  #       scan keeps the BLANKET `${` refusal — quotes are not parsed in
  #       live clauses (#796 revert), so it is deliberately STRICTER than
  #       heredoc check (g), which as of #1501 deletes provably-inert plain
  #       ${NAME} spans first: a plain ${VAR} HERE (even single-quoted,
  #       which bash would NOT expand locally) still refuses the waiver.
  #       Bare `$VAR` (no brace, no
  #       paren) never executes a command and is NOT refused. Live
  #       clauses — unlike heredoc BODIES — undergo process substitution,
  #       hence <( / >( over check (g). `<<<` (here-string) feeds DATA and
  #       is refused anyway: it preserves the existing must-block pin
  #       R8b-here_string_full_literal_parity (a grep here-string carrying
  #       a full gated literal blocks today and MUST keep blocking — the
  #       round-1 statistics blocker), keeping raw-scan parity for
  #       here-string literals at zero incident cost.
  #   (3b) NO local file OUTPUT REDIRECT — `>`/`>>` with optional fd digits
  #       (`2>`, `1>>`) — unless its target is exactly /dev/null followed
  #       by whitespace/end-of-clause. Round-2 fail-closed arm (concern id
  #       redirect-file-producer-failopen): without it a waived producer
  #       could write gated text to a LOCAL file a later same-call clause
  #       executes (`ssh host 'echo git reset --hard' > /tmp/x; bash
  #       /tmp/x` — nextsep=SEQ/AND/NL/END, so cond (2) never fires; the
  #       consumer clause carries no gated literal and clears the loose
  #       gate). The check is a strip-then-scan: redirects targeting
  #       exactly /dev/null are stripped (a discard-only sink can never be
  #       re-read or executed — keeps the `2>/dev/null` sweep convenience
  #       waivable), then ANY remaining `>` refuses the waiver,
  #       consumer-independently and REGARDLESS of position (incl.
  #       nextsep=END — no same-call consumer exists there, but refusing
  #       is strictly status-quo-preserving: every redirect-carrying gated
  #       producer shape is rc=2 on main today, so the refusal costs only
  #       un-waived convenience; fail-closed residual FPs in gap (xiv):
  #       `> results.txt` capture, a remote-side redirect inside the
  #       quoted ssh string, a literal `>` in a grep PATTERN, and a
  #       /dev/null redirect flush against the closing quote —
  #       `'... 2>/dev/null'` — whose boundary is `'`, not whitespace).
  #       Pure `>`/`>>`/`N>`/`N>>` spellings reach this arm, and so does
  #       the `<>` read-write redirect (its `>` survives the pre-pass and
  #       fails closed harmlessly unless targeting exactly /dev/null,
  #       where the strip discards it — a discard sink either way): every
  #       `&`-carrying redirect (`&>`, `&>>`, `2>&1`, `>&2`, `|&`) is
  #       mis-split by the sed pre-pass into a BG/PIPE separator (cond (2)
  #       refuses), `>|` exposes a PIPE, and `>(` is refused by (3).
  #   (4) [ssh only] no ProxyCommand/LocalCommand/KnownHostsCommand token
  #       (ssh executes all three LOCALLY, in this cwd — KnownHostsCommand
  #       since OpenSSH 8.4) and no shared-repo path in ANY covered
  #       spelling — literal $REPO, $HOME/<repo-basename>, ~/<repo-basename>
  #       — anywhere in the clause: an ssh-to-this-VM remote string
  #       operating on the shared root stays blocked (option (b), widened
  #       after the round-1 critics showed `--work-tree=$HOME/...` dodges the
  #       literal-only glob; bare $HOME is deliberately not expansion-refused,
  #       so the path spellings must be). ${HOME}/... forms are already
  #       refused by (3)'s ${ arm. Non-canonical spellings (doubled slash,
  #       $USER-composed, variable-indirection `ssh $HOST` with a remote
  #       $REPO) stay deliberate-only accepted fail-opens — gap (xiv).
  #   (5) [grep-family only] no --pre token, = or space form (rg --pre
  #       executes a preprocessor command per file, locally).
  # NO LATCH: the waiver covers THIS clause only — it reads the clause's
  # own command word, so no arming/propagation separator proof is needed
  # (unlike the cd/WT latches, no state crosses clauses). A single-quoted
  # multi-statement remote string in the CANONICAL shape (clause-initial
  # `ssh` or the #1463 `gcloud compute ssh` head, EITHER with an optional
  # literal timeout wrapper — #1463 gcloud, #1859 ssh; payload is the
  # clause's final token; quote-, latch- and
  # ladder-clean per the R1-R8 predicate) is merged into ONE clause by the
  # mask_ssh_payload_separators() pre-pass (#1413) BEFORE the split, so it
  # reaches this waiver whole — the pre-pass grants nothing itself; the
  # merged clause still walks every refusal above. Any OTHER
  # multi-statement remote string (double quotes, trailing tokens,
  # redirects, non-timeout-wrapped ssh, quoted or latch-vocabulary
  # prefixes) is still
  # mis-split by the quoted-separator trade-off and its TAIL clause
  # (which lost the ssh command word) still classifies — fail-closed,
  # residual gap (xiv); remediation unchanged: single-statement
  # `git -C /workspace/... <verb>` inside the remote string (the -C waiver
  # above already allows it), a pod-side script, or the SSH MCP.
  # (#1859) The cond (1) outer head below and the inner ssh/gcloud-vs-grep
  # discriminator (the first grep inside the waiver body) share the
  # timeout-qualified remote-exec alternation and MUST move in LOCKSTEP: a
  # clause matching the OUTER head but missing the INNER discriminator
  # falls to the grep-family arm — a WIDER waiver with no ProxyCommand /
  # shared-repo-path checks — so drift BETWEEN these two regexes is
  # fail-OPEN (unlike mask-vs-waiver head drift, which is fail-closed).
  if echo "$clause" | grep -qE '^((timeout[[:space:]]+[0-9]+([.][0-9]+)?[smhd]?[[:space:]]+)?(gcloud[[:space:]]+compute[[:space:]]+)?ssh[[:space:]]|(grep|egrep|fgrep|rg)[[:space:]])'; then
    if ! echo "$clause" | grep -qE '\$\(|\$\{|`|<\(|>\(|<<<' \
       && ! echo "$clause" \
            | sed -E 's@[0-9]*>>?[[:space:]]*/dev/null([[:space:]]|$)@ @g' \
            | grep -q '>'; then
      if echo "$clause" | grep -qE '^((timeout[[:space:]]+[0-9]+([.][0-9]+)?[smhd]?[[:space:]]+)?(gcloud[[:space:]]+compute[[:space:]]+)?ssh)[[:space:]]'; then
        # ssh/gcloud arm: consumer-independent PIPE/BG refusal UNCHANGED —
        # a remote command's stdout is arbitrary remote-generated text; the
        # #1538 widening is grep-family only.
        if [ "$nextsep" != PIPE ] && [ "$nextsep" != BG ]; then
          if ! echo "$clause" | grep -qiE 'proxycommand|localcommand|knownhostscommand'; then
            case "$clause" in
              *"$REPO"*|*'$HOME/'"$REPO_BASE"*|*'~/'"$REPO_BASE"*) : ;;
                                # shared-root spelling -> classify (blocks)
              *) continue ;;    # remote-host git op -> waive this clause
            esac
          fi
        fi
      elif ! echo "$clause" | grep -qE '(^|[[:space:]])--pre(=|[[:space:]])'; then
        if [ "$nextsep" != PIPE ] && [ "$nextsep" != BG ]; then
          continue              # read-only pattern argument -> waive (unchanged)
        elif [ "$nextsep" = PIPE ] && _pipe_chain_is_readonly_sink "$_idx"; then
          continue              # (#1538) piped into a VERIFIED read-only
                                # sink chain -> waive; anything the walker
                                # cannot classify falls through -> classify
        fi
      fi
    fi
  fi

  reason=$(classify_clause "$clause")
  if [ -n "$reason" ]; then blocked="$reason"; break; fi   # first block wins
done

[ -n "$blocked" ] || exit 0

# Only protect the on-main state. If the repo-root tree is already off main,
# the horse has bolted — don't trap the user trying to recover.
cur=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
[ "$cur" = main ] || exit 0

echo "BLOCKED: '$blocked' would move the SHARED repo-root tree off main / detach HEAD / destroy uncommitted working-tree state. The repo root is the canonical commit target for scripts/task.py and every concurrent VM session (all assume HEAD==main); a branch switch here hijacks concurrent commits, and a working-tree revert (restore / checkout-pathspec / clean -f / reset --hard) silently discards CONCURRENT sessions' uncommitted edits (incidents 2026-06-01, #815, #841), and a branch MERGE here can strand conflict markers in the shared tree that a concurrent commit sweeps (#1090), and a REBASE / CHERRY-PICK here mutates root history or strands the same conflict state (#1193), and a REVERT / AM here lands commits on root main or strands the same sequencer/conflict state (#1234). Do branch/destructive work in a worktree instead:
  bash scripts/new_worktree.sh .claude/worktrees/<name> <branch> && git -C .claude/worktrees/<name> ...
NEVER point -C at the repo root itself for a destructive op — for repo-root recovery use: uv run python scripts/sync_repo_root.py
This guard matches COMMAND TEXT, not cwd — a worktree-internal op after 'cd <worktree>' in a compound is still blocked; use the git -C <worktree> form instead of cd'ing (incident #1143, 2026-07-08).
Three compliant worktree compose shapes ARE recognized (#1058/#1861): (1) per-clause git -C <worktree path> <op>; (2) the &&-chain: WT=<path under .claude/worktrees/>; cd \"\$WT\" && <op> (any variable name, e.g. WORKTREE); (3) the exit-guard: cd \"\$WT\" || exit 1 (or || { echo FATAL >&2; exit 1; }) with <op> on later ;/newline-separated clauses. An OR/PIPE-preceded cd, a non-exiting guard tail ('|| echo oops', '|| return 1'), or ANY later cd/pushd/popd clause voids the scope — recompose with git -C instead.
To LAND a branch onto main: gh pr merge <PR> --rebase (server-side, the /issue Step 10d path), or a scratch worktree: git worktree add --detach /tmp/<name> origin/main && git -C /tmp/<name> merge <branch> && git -C /tmp/<name> push origin HEAD:main.
To recover an in-progress root merge/rebase/cherry-pick/revert/am: git merge --abort / git rebase --abort / git cherry-pick --abort / git revert --abort / git am --abort (all allowed; --quit likewise). For a worktree fast-forward: git -C <worktree> fetch origin +refs/heads/main:refs/remotes/origin/main, then git -C <worktree> merge --ff-only origin/main (NEVER local main — its unpushed root commits contaminate the branch, #1530).
For marker-note text mentioning git commands, use --file <path.md> instead of --note; for commit messages, use git commit -F <file>. As of #1566 the canonical SINGLE-QUOTED task.py argument shape is masked (allowed): a clause-initial uv run python .../task.py invocation whose quoted note/title/prompt text sits in an otherwise plain clause no longer false-blocks — so a residual block on task.py argument text means a non-canonical shape (double quotes, dollar or backslash or backquote forms, redirects, a quoted or latch-vocabulary prefix); use the --file route for those.
For composing a doc/report via heredoc whose body carries backticks, command substitution, or non-plain parameter forms (\${VAR:-default}, \${VAR@P}, \${1}) alongside git-verb text: quote the heredoc tag (<<'EOF' — bash never expands a quoted-tag body, and it strips cleanly); exactly-plain \${VAR} references (letters/digits/underscore only, nothing else inside the braces) are fine even under an unquoted tag (#1501). For a body naming shell-out spellings (subprocess / os.system / ...) or fed to a python/interpreter stdin consumer, use the Write tool instead — it covers EVERY composition class (quoting the tag does NOT lift those refusals). As of #1621 argv-LIST-form call opens with a non-shell first element (subprocess.run([\"git\", ...) no longer refuse the strip; bare word mentions, string-form calls, shell=True residuals, and shell-name argv heads still do — the Write tool remains the remediation for those.
NOTE: this deny blocked your ENTIRE compound command — earlier clauses did NOT run either; regenerate any files/state those clauses were meant to produce before retrying the safe form (incident class #813/#1056).
For a POD-side remote git op, a single-statement ssh <host> 'git <verb> ...' remote command is allowed (#1098), and a SINGLE-QUOTED multi-statement remote string is allowed when the quoted payload is the clause's final token and nothing quote- or latch-ambiguous precedes it (#1413); a literal 'timeout <N>' wrapper on the ssh head is tolerated too (#1859); other shapes (double quotes, redirects, trailing tokens, non-timeout-wrapped ssh — nohup/env/abs-path/variable heads and timeout FLAG forms — quoted/latch-vocabulary prefixes) still need git -C /workspace/<repo> <verb> inside the remote string, a pod-side script, or the SSH MCP.
For a grep/rg PATTERN clause naming git verbs: the unpiped clause is waived, and piping into plain read-only text filters (head/tail/wc/cat/cut/tr/nl/sort/uniq/grep) is waived too (#1538) — so a residual block on a piped grep means the consumer chain was NOT verifiable (an off-allowlist / path-spelled / quoted consumer word, a redirect or write/exec flag, or any $ / # in a consumer clause); drop the pipe, remove the $/#/flag, or bound output with grep -m N instead.
For a GCE-side remote git op, the same two shapes are allowed with a gcloud compute ssh <instance> --command='...' head (#1463; an optional literal 'timeout <N>' wrapper is tolerated): keep --command the clause's FINAL token, no in-payload < or > redirects (bound output with | tail INSIDE the single-quoted payload — pipes mask fine), and no trailing local pipe / fd-dup ('2>&1 | tail -N' stays blocked); or put git -C /workspace/<clone> <verb> inside the payload, which is allowed regardless (path-blind -C waiver)." >&2
log_deny "$blocked" "$cmd" "${clause:-}"
exit 2
