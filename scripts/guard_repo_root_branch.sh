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
# same waiver. A MULTI-STATEMENT remote string mis-splits on the
# quoted-separator trade-off and its tail clauses still classify —
# remediation unchanged: `git -C /workspace/... <verb>` inside the remote
# string, a pod-side script, or the SSH MCP (which bypasses the Bash hook
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
# line, no shell-out spelling in the body, and (for an UNQUOTED tag, whose
# body bash expands at feed time) no `$(`/backtick/`${` expansion syntax —
# so document text that merely MENTIONS a gated form no longer false-blocks;
# the quoted `--note` literal above is NOT a heredoc and stays blocked (that
# limitation is unchanged). A SECOND NARROWING (#1098): a clause whose
# command word is `ssh` (remote execution) or `grep`/`egrep`/`fgrep`/`rg`
# (read-only pattern) is waived per-clause by the driver-loop waiver under
# fail-closed refusals; quoted git-verb literals under OTHER command words
# (`--note`/`-m` strings) still block with the same `--file`/`-F`
# remediation. The #897 detectors use a TIGHT
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
#       to hide an UNQUOTED-tag body carrying expansion syntax
#       (`$(` / `${` / backtick) — bash expands such bodies at feed time, so
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
#       THREAT MODEL paragraph above).
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
# (xiii) (#1058) Fail-closed FALSE-POSITIVE notes for the strip (a no-strip
#       keeps CURRENT behavior — the command blocks only when a gated form
#       ALSO appears): plain `${VAR}` references in an unquoted-tag body
#       (the `${` refusal is deliberately over-broad — `${x:-$(cmd)}` nests
#       command substitution); a bare-dot jq filter (`jq . <<J`) matching the
#       standalone-dot source form (quoted `jq '.'` unaffected); and prose
#       like "the system (Linux) ..." matching the `system *\(` body
#       refusal.
# (xiv) (#1098) The ssh/grep-family clause waiver's residuals, BOTH sides.
#       Fail-closed FALSE POSITIVES (harmless shapes that stay blocked):
#       multi-statement remote strings with the gated verb in a non-first
#       statement (`ssh pod 'cd /workspace/x && git reset --hard'` —
#       mis-split; the tail clause lost the ssh command word); wrapped /
#       absolute-path / variable ssh (`timeout N ssh ...`, `/usr/bin/ssh`,
#       `$SSHCMD ...`); `${VAR}` in remote strings, incl. single-quoted
#       forms bash would not expand (the deliberate `${` over-match);
#       ssh clauses naming the shared-repo path in a covered spelling;
#       here-string literals (`grep -q x <<<"...gated..."` — the R8b
#       raw-scan-parity class — and ssh stdin here-strings); and ANY
#       waived-word clause in pipeline-producer OR background position
#       (`ssh pod '...' | tail`, `ssh pod '...' &`, and `ssh pod '...'
#       2>&1 ...` — the fd-dup's single & mis-splits as BG, hiding a
#       following pipe from the lookahead, so BG refuses too — all stay
#       blocked; remediation: `git -C /workspace/... <verb>` inside the
#       remote string, which the pipe-blind `-C` waiver allows); and ANY
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
#       grep-family) — remediation: plain `grep`. QUALIFIER for register
#       exactness: the "tail clauses still classify" sentence is not
#       universally true — a FIRST remote statement matching the
#       UNANCHORED cd-latch (`ssh pod 'cd /tmp/ && git clean -fd'`) is
#       rc=0 TODAY via the pre-existing latch, not via this waiver;
#       pre-existing behavior, unchanged by this diff.
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
#       (xiv)'s piped-grep FP class (`grep 'git rebase ...' file | head`
#       blocks now that the loose gates match the verbs; the non-piped grep
#       clause stays waived — RA17).
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
#       activate gap (xiv)'s piped-grep FP class (`grep 'git am ...' f |
#       head` blocks; the non-piped grep clause stays waived — RVA15), and
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
# clause-initial `WT=<...>.claude/worktrees/<...>` assignment arms a
# `wt_bound` flag in stream order, after which `cd "$WT"` (exact-arg, no
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
# check_output / getoutput / bare `system(` / `from os import`), and (g) for
# an UNQUOTED, unescaped tag (<<EOF — bash EXPANDS the body at feed time) NO
# body line carries expansion syntax ($( / ${ / backtick); a quoted/escaped
# tag (<<'EOF', <<"EOF", <<\EOF) suppresses expansion and skips check (g).
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
            # harmless unless a gated form also appears).
            if (buf[j] ~ /(os\.system|subprocess|Popen|check_call|check_output|getoutput|system *\(|from +os +import)/) { ok = 0; break }
            # (g) UNQUOTED-tag body: bash performs command/parameter
            # substitution at feed time, so $(...) / `...` in the body
            # EXECUTE regardless of consumer -> refuse to strip. ${ refuses
            # too (parameter expansion can nest command substitution,
            # ${x:-$(cmd)}) — a fail-closed over-match on plain ${VAR}
            # references, documented. An escaped \$( also matches (bash
            # would NOT expand it, but \\$( WOULD — refusing both is the
            # simple fail-closed read).
            if (!QUOTED && buf[j] ~ /\$\(|\$\{|\x60/) { ok = 0; break }
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
  if echo "$c" | grep -qE '\bgit\b[^;&|]*\bcheckout\b +(-{1,2})detach\b'; then
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
  # ff-syncs use `git -C <worktree> merge --ff-only main`, root syncs use
  # sync_repo_root.py).
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
wt_bound=0
blocked=""
while IFS=$'\t' read -r sep nextsep clause; do
  # Reset the latch unless the separator BEFORE this clause is && — a `cd`
  # only reliably scopes a following git clause when bash guarantees it ran
  # first (the && short-circuit). ; / || / | / & / a raw newline (NL) do NOT
  # carry the latch (NL is not AND, so this consolidated check resets it).
  if [ "$sep" != AND ]; then
    scoped=0
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

  # A `cd` into a worktree / /tmp latches scope forward ONLY across a following
  # `&&` clause. Latch and continue — this clause runs the `cd`, not a git
  # command, and it must NOT scope EARLIER clauses (those were classified
  # before it).
  if echo "$clause" | grep -qE 'cd +[^;&|]*\.claude/worktrees/' \
     || echo "$clause" | grep -qE 'cd +/tmp/'; then
    scoped=1
    continue
  fi

  # (#1058) A `WT=<...>.claude/worktrees/<...>` BARE-ASSIGNMENT clause
  # (optionally `export`-prefixed; NOTHING after the RHS) arms the $WT latch
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
  # Any OTHER clause-initial WT= assignment (non-worktree RHS, trailing
  # command word, conditional/subshell separator) DISARMS the latch — a
  # reassignment makes the earlier arming proof stale. Clause-initial only:
  # a `WT=` fragment buried in quoted prose (an echo / --note argument)
  # never matches the ^ anchor and neither arms nor disarms.
  if echo "$clause" | grep -qE '^(export +)?WT='; then
    wt_bound=0
    case "$sep" in
      START|SEQ|NL)
        if echo "$clause" | grep -qE '^(export +)?WT=[^;&|[:space:]]*\.claude/worktrees/[^;&|[:space:]]*[[:space:]]*$'; then
          wt_bound=1
        fi
        ;;
    esac
  fi

  # (#1058) `cd "$WT"` — the SKILL.md-conventional worktree variable — latches
  # ONLY when an EARLIER clause in this SAME command bound WT to a
  # `.claude/worktrees/` path (shell state never persists across Bash tool
  # calls, so a non-empty $WT implies a same-call assignment). The
  # assignment check is LOAD-BEARING, not cosmetic: bash `cd ""` SUCCEEDS as
  # a no-op (verified 2026-07-05, bash 5.1.16), so with an UNSET WT
  # `cd "$WT" && git ...` runs the git clause in the UNCHANGED cwd (the repo
  # root) — a bare `cd "$WT"` latch would be fail-open. With the assignment
  # present, every quoting variant is safe under the && latch: expanded
  # forms cd into the worktree; a single-quoted literal `cd '$WT'` fails
  # (no such dir) and && short-circuits the git clause. A `..` anywhere in
  # the cd arg never latches (fail-closed).
  if [ "$wt_bound" -eq 1 ]; then
    cdarg=$(echo "$clause" | sed -nE 's/^cd +([^;&|]+)[[:space:]]*$/\1/p' | tr -d '\042\047')
    case "$cdarg" in
      *..*) : ;;
      \$WT|\${WT}|\$WT/*|\${WT}/*)
        scoped=1
        continue
        ;;
    esac
  fi
  [ "$scoped" -eq 1 ] && continue          # this clause runs in a scoped cwd

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
  #       forms (`timeout 240 ssh ...`, `nohup ssh ...`, `/usr/bin/ssh`,
  #       `$SSHCMD ...`) are NOT clause-initial `ssh` and keep blocking
  #       (fail-closed residual FP, gap (xiv)).
  #   (2) the clause is NOT in pipeline-producer / background position: its
  #       FOLLOWING separator (the $nextsep field the splitter now emits)
  #       is neither PIPE nor BG. PIPE: a waived producer's stdout can feed
  #       a LOCAL shell consumer (`ssh host 'echo git reset --hard' | bash`,
  #       `grep 'git reset --hard' f | bash`) whose own clause carries no
  #       gated text and clears the loose gate — the round-1 Codex
  #       methodology blocker. BG (implementation-round fail-closed
  #       widening, live-probed): an fd-dup redirection's single `&`
  #       (`2>&1`) is mis-split as a BG separator by the raw sed pre-pass,
  #       so `ssh host '...' 2>&1 | bash` reports nextsep=BG on its
  #       producer clause — the PIPE hides one record downstream; refusing
  #       BG closes that hole, and a TRUE background producer
  #       (`ssh pod '...' & ...`) costs only a residual FP. Refusing on ANY
  #       following pipe/BG (consumer-independent) is strictly
  #       status-quo-preserving: piped / `&`-carrying shapes are rc=2
  #       today, so the refusal costs only the un-waived convenience
  #       (`... | tail`, `... 2>&1`), documented as residual FPs in gap
  #       (xiv) with the `git -C` remediation (which pipes fine — the -C
  #       waiver is pipe-blind).
  #   (3) NO locally-executing expansion / redirection syntax anywhere in
  #       the clause: $( / ${ / backtick / <( / >( / <<< .
  #       `ssh host "$(git reset --hard)"` and `grep -f <(git clean -fd) x`
  #       EXECUTE the gated text LOCALLY at expansion time. `${` is the
  #       same fail-closed over-match as heredoc check (g) — a plain
  #       ${VAR} (even single-quoted, which bash would NOT expand locally)
  #       refuses the waiver, because the guard deliberately does not
  #       shell-parse quotes (#796 revert). Bare `$VAR` (no brace, no
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
  # (unlike the cd/WT latches, no state crosses clauses). A quoted
  # multi-statement remote string (`ssh pod 'cd /w && git reset --hard'`)
  # is mis-split by the quoted-separator trade-off and its TAIL clause
  # (which lost the ssh command word) still classifies — fail-closed,
  # residual gap (xiv); remediation unchanged: single-statement
  # `git -C /workspace/... <verb>` inside the remote string (the -C waiver
  # above already allows it), a pod-side script, or the SSH MCP.
  if echo "$clause" | grep -qE '^(ssh|grep|egrep|fgrep|rg)[[:space:]]'; then
    if [ "$nextsep" != PIPE ] && [ "$nextsep" != BG ] \
       && ! echo "$clause" | grep -qE '\$\(|\$\{|`|<\(|>\(|<<<' \
       && ! echo "$clause" \
            | sed -E 's@[0-9]*>>?[[:space:]]*/dev/null([[:space:]]|$)@ @g' \
            | grep -q '>'; then
      if echo "$clause" | grep -qE '^ssh[[:space:]]'; then
        if ! echo "$clause" | grep -qiE 'proxycommand|localcommand|knownhostscommand'; then
          case "$clause" in
            *"$REPO"*|*'$HOME/'"$REPO_BASE"*|*'~/'"$REPO_BASE"*) : ;;
                              # shared-root spelling -> classify (blocks)
            *) continue ;;    # remote-host git op -> waive this clause
          esac
        fi
      elif ! echo "$clause" | grep -qE '(^|[[:space:]])--pre(=|[[:space:]])'; then
        continue              # read-only pattern argument -> waive this clause
      fi
    fi
  fi

  reason=$(classify_clause "$clause")
  if [ -n "$reason" ]; then blocked="$reason"; break; fi   # first block wins
done < <(split_and_label "$cmd")

[ -n "$blocked" ] || exit 0

# Only protect the on-main state. If the repo-root tree is already off main,
# the horse has bolted — don't trap the user trying to recover.
cur=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
[ "$cur" = main ] || exit 0

echo "BLOCKED: '$blocked' would move the SHARED repo-root tree off main / detach HEAD / destroy uncommitted working-tree state. The repo root is the canonical commit target for scripts/task.py and every concurrent VM session (all assume HEAD==main); a branch switch here hijacks concurrent commits, and a working-tree revert (restore / checkout-pathspec / clean -f / reset --hard) silently discards CONCURRENT sessions' uncommitted edits (incidents 2026-06-01, #815, #841), and a branch MERGE here can strand conflict markers in the shared tree that a concurrent commit sweeps (#1090), and a REBASE / CHERRY-PICK here mutates root history or strands the same conflict state (#1193), and a REVERT / AM here lands commits on root main or strands the same sequencer/conflict state (#1234). Do branch/destructive work in a worktree instead:
  bash scripts/new_worktree.sh .claude/worktrees/<name> <branch> && git -C .claude/worktrees/<name> ...
NEVER point -C at the repo root itself for a destructive op — for repo-root recovery use: uv run python scripts/sync_repo_root.py
This guard matches COMMAND TEXT, not cwd — a worktree-internal op after 'cd <worktree>' in a compound is still blocked; use the git -C <worktree> form instead of cd'ing (incident #1143, 2026-07-08).
To LAND a branch onto main: gh pr merge <PR> --rebase (server-side, the /issue Step 10d path), or a scratch worktree: git worktree add --detach /tmp/<name> origin/main && git -C /tmp/<name> merge <branch> && git -C /tmp/<name> push origin HEAD:main.
To recover an in-progress root merge/rebase/cherry-pick/revert/am: git merge --abort / git rebase --abort / git cherry-pick --abort / git revert --abort / git am --abort (all allowed; --quit likewise). For a worktree fast-forward: git -C <worktree> merge --ff-only main.
For marker-note text mentioning git commands, use --file <path.md> instead of --note; for commit messages, use git commit -F <file>.
NOTE: this deny blocked your ENTIRE compound command — earlier clauses did NOT run either; regenerate any files/state those clauses were meant to produce before retrying the safe form (incident class #813/#1056).
For a POD-side remote git op, a single-statement ssh <host> 'git <verb> ...' remote command is allowed (#1098); a multi-statement remote string mis-splits — put git -C /workspace/<repo> <verb> inside the remote string, or use a pod-side script / the SSH MCP." >&2
exit 2
