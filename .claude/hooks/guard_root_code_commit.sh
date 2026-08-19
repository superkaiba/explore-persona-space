#!/usr/bin/env bash
# PreToolUse(Bash) guard (#1500): refuse a repo-root `git commit` whose pending
# payload includes UNCERTIFIED code paths (scripts/**.py, src/**, tests/**.py).
# Certification = a fresh content-hash-bound line written by
# scripts/inline_lint_gate.py on a PASSING Step 9a-ter inline payload lint gate
# run (SKILL.md Step 9a-ter § Inline payload lint gate, #1388/#1460).
#
# WHY commit-time, not push-time: scripts/auto_push_main.sh (cron */2 + Stop
# hook) pushes local main whenever ahead, so a Bash-tool push gate is bypassed
# within ~2 min of the commit landing; and push-time payload attribution on the
# shared root batches other sessions' commits. The commit is the only
# interceptable choke point for Bash-tool-driven inline rounds (plan #1500 §11).
#
# TWO-LAYER PREDICATE (plan #1500 §4.2):
#   Layer 1 (command text, cheap): detect a plausible repo-root `git commit`
#     clause. Round 3: the clause split + token scan are STRING-LITERAL-AWARE
#     — an awk char-scan pre-pass (mask_and_split, adapted from
#     scripts/guard_repo_root_branch.sh mask_ssh_payload_separators(),
#     #1185/#1413) masks every string-literal span to \001 on a SCAN copy, so
#     separators/newlines inside a quoted commit message no longer split the
#     clause, message text contributes no tokens/flags/waivers, and genuine
#     tokens outside literals still parse (concern
#     quoted-message-seams-defeat-clause-scan). Decision semantics kept from
#     the pull-guard lineage (#1201/#1250 line-continuation normalize +
#     command-position verb anchor, #804 cd-latch across `&&` only); cd/-C
#     TARGETS are extracted from the RAW copy (quoted "$WT"/root spellings
#     keep their content). Also collects commit-clause pathspecs AND
#     `git add`-clause paths chained in the same command (the compound
#     `git add X && git commit -m ...` idiom stages NOTHING at PreToolUse
#     time — the hook fires BEFORE execution).
#   Layer 2 (repo state, authoritative): classify the pending payload from
#     `git diff --cached` (staged) ∪ `-a` worktree-modified ∪ the Layer-1 text
#     paths, filtered to the gated glob; block iff any gated path lacks a
#     fresh cert line bound to the LANDING content (worktree hash for
#     `-a`/pathspec/add-clause shapes — those commit worktree content; staged
#     blob sha only for plain staged commits). A Layer-1 match with no gated
#     pending payload ALLOWS, so text-level false matches (e.g. a
#     commit-message line beginning `git commit`) cannot block on their own.
#
# NO heredoc blanket-allow (deliberate deviation from the guard siblings): the
# canonical commit-message shape `git commit -m "$(cat <<'EOF'...)"` embeds a
# heredoc, so a blanket allow would exempt most real commits; Layer 2
# neutralizes the mention-FP risk instead.
#
# FAIL MODES: stdin-parse failure -> OPEN (A16 sibling parity — a guard bug
# must never wedge the fleet); payload-classification git failure on a
# CONFIRMED root-commit clause -> CLOSED (#458/#1147 class); blanket
# `git add -A|.|--all` chained to a root commit -> CLOSED (landing set
# unknowable at hook time; blanket root staging is independently banned by
# CLAUDE.md § Concurrent repo-root committers) — EXCEPT the path-limited
# `git add -A|--all -- <explicit pathspec>` form executed AT the repo root
# with no cd/--chdir prefix (issue #1977): its pathspec BOUNDS the landing
# set, which Layer 2 resolves per file via a cwd-gated scoped `git status
# --porcelain=v1 --untracked-files=all` and feeds into classification like
# any other add-clause path; EVERY ambiguity (non-root/unknown cwd, any
# pre-`--` token beyond {-A,--all}, masked/opaque/blanket-equivalent
# candidates, a C-quoted porcelain path, a failed git read) keeps the
# CLOSED block; missing/stale/mismatched cert -> CLOSED (that IS the
# block).
#
# <!-- known limitations -->
# - WORKTREE-CWD ALLOW (issue #2066; supersedes the old CWD-BLIND limitation):
#   a commit command with NO retarget evidence (no cd-to-root / unproven cd,
#   no root-spelling or unextractable `git -C` — classify_cmd's
#   retarget_evidence flag) and NONE of the conservative-screen classes in
#   its raw text (see WT_CWD_ALLOW_SCREEN_ERE: directory/repo-pointing flags
#   long AND short — --chdir / --work-tree / --git-dir / -C-bearing
#   clusters — the core.worktree config spelling, repository-pointing
#   GIT_*= assignments incl. GIT_COMMON_DIR=, the cd WORD in any spelling
#   incl. escaped/quoted/flag-intervened forms, pushd/popd, the source
#   word) is ALLOWED without reading the root index when the hook-input cwd
#   PROVABLY sits inside a linked worktree of this repo (rev-parse toplevel
#   != root AND git-common-dir == root/.git) — parity with the
#   `git -C "$WT"` waiver; worktree commits are gated at Step 10d, not
#   here. Residuals, all fail closed toward today's behavior: an
#   unproven/unknown/root-subdir cwd still classifies against the ROOT
#   index; a cwd inside an UNRELATED repo stays conservative (common-dir
#   mismatch); any screen token anywhere in the command — even inside a
#   commit message — disables the allow; runtime-only content (variable
#   expansion, eval'd strings built at runtime, `.`-sourced file contents)
#   is invisible to the text screen and the `.` source SPELLING is
#   deliberately unscreened (accidents-not-adversaries) — both can in
#   principle retarget past the gate; pre-existing Layer-1 bypass classes
#   (masked-literal / unmatchable-lead commit spellings) remain out of
#   scope in both directions. Kill switch:
#   EPM_ROOT_CODE_COMMIT_DISABLE_WT_CWD_ALLOW=1 restores the pre-#2066
#   cwd-blind behavior (Layer 2 reads the ROOT's index for any non-root cwd).
# - Shared-index race: another session staging gated files concurrently can
#   false-block an innocent commit (rare; block direction is safe). Since
#   #1620 a root-cwd pathspec-limited commit scopes the staged read to its
#   pathspecs (pathspec SCOPING engages only when the hook-input cwd provably
#   equals the root — the WORKTREE-CWD ALLOW note above covers the
#   provably-worktree bare-commit case; other non-root cwds stay
#   conservative);
#   quoted/spacey pathspecs and non-root-cwd commits stay conservatively
#   whole-index. Since #1928 a strictly-recognized redirect token on the
#   commit clause (bare `>`/`>>`/`<`/`&>`/`&>>` + its target word, fd-dups,
#   attached clean-literal targets) is excluded from the candidate stream
#   so plain output redirections no longer defeat scoping; a QUOTED/spacey
#   pathspec combined with a redirect still fails rawtail token-count
#   parity and stays conservatively whole-index (fail-closed residual).
#   Since #2046 a strictly-recognized here-doc / here-string OPENER token on
#   the commit clause (bare `<<`/`<<-`/`<<<` + its delimiter word, or the
#   attached-delimiter forms incl. the closed-quoted spellings) is likewise
#   excluded, so a heredoc-fed `-F /dev/stdin` commit no longer defeats
#   scoping. Residuals, both fail closed: (i) a QUOTED/spacey pathspec
#   combined with an opener fails the same rawtail token-count parity and
#   stays whole-index; (ii) the ADD-clause second pass keeps opener tokens
#   opaque (out of incident scope — the exemption stays narrow).
# - Provably-root cd prefix (#2357): pathspec SCOPING also engages when the
#   command carries a CANONICAL ABSOLUTE root `cd` ($GUARD_REPO absolute
#   [+ trailing /] ONLY — the r2 arming set) AND that cd is the LAST
#   RECOGNIZED cwd-moving record before every commit clause (r3+r4): any
#   recognized cwd-moving record ANYWHERE in the armed chain after the
#   canonical cd DISARMS the widening — a non-canonical cd (r3), and (r4)
#   every record matching the cwd-mover family (pushd/popd, source/`.`,
#   eval, builtin/command wrappers, quoted/escaped cd spellings — each
#   also recognized (r5, widened r6 #2371) behind ZERO-OR-MORE leading
#   legal prefixes in ANY interleaving: plain NAME=value AND append
#   NAME+=value assignment pairs plus the `time` keyword wrapper (bare
#   or `time -p`) before the mover word, matched on the raw AND the
#   masked record lead so a QUOTED assignment value cannot hide the
#   prefix) or the
#   retarget vocabulary (WT_CWD_ALLOW_SCREEN_ERE: `git -C`/`env -C`
#   short dir wrappers, --chdir/--work-tree/--git-dir, GIT_DIR=-family
#   assignments — even as a mention inside message text: conservative by
#   design) — only a chain whose LAST recognized cwd-moving record before
#   every commit is the literal absolute-root cd scopes; a later canonical
#   arming cd re-establishes the base. UNRECOGNIZED movers remain a named
#   residual of the widening: mid-word-quoted spellings (`pu'sh'd`), a
#   mover held in a variable, a mover reached through a PREDEFINED shell
#   function or alias (the invoked name carries no mover text — accepted
#   text-unprovable indirection), and an assignment prefix whose VALUE
#   the masker cannot flatten (a bare command-substitution value with
#   internal whitespace, an array-form value) ride an armed chain
#   undetected (fail-open for
#   the widening ONLY — never wider than the pre-#2357 root-cwd behavior);
#   compound/subshell contexts stay refused by D4 regardless.
#   NOT-REACHABLE sourcing paths (r6 #2371, execution-confirmed
#   2026-08-18, GNU bash 5.1.16; pinned in tests, distinct from the
#   residual set above): `env . x` cannot source (external env cannot
#   run the `.` builtin) and stays SCOPED — unrecognized AND unreachable,
#   not a gap; non-keyword-position `time` (`NAME=v time . x`) runs
#   EXTERNAL time and cannot source, but MATCHES the r6 grammar and
#   disarms (deliberate block-direction over-tightening). The symbolic
#   ALIAS root spellings (~/explore-persona-space[/],
#   $HOME/explore-persona-space[/]) NEVER arm the widening AND, riding an
#   armed chain, DISARM it (r3): they keep their legacy non-poisoning
#   classification (cd_nonroot stays 0 — root-cwd behavior unchanged) but
#   are fail-closed for the widening, because their runtime destination
#   depends on $HOME and symlink resolution, neither provable from the
#   command text. The arming cd's own separator is START/SEQ/NL
#   (g3-matching; AND excluded), with NO token after the target, next
#   separator neither & nor |, no commit clause scanned BEFORE it, EVERY
#   separator strictly between the cd and EVERY commit clause equal to &&
#   (chain dominance: a commit that runs proves the cd executed and
#   succeeded, so the pathspec-resolution base at the commit is the root),
#   and NO compound/subshell-context record anywhere in the command.
#   Fail-closed residuals (each stays conservatively whole-index):
#   symbolic-alias root spellings (the ~/ and $HOME/ forms above —
#   legacy-kept non-poisoning, never arming, and DISARMING when they ride
#   an armed chain, r3); any recognized cwd-moving record between the
#   armed canonical cd and a commit (riding alias / subdir / relative /
#   variable cd, and the r4 non-cd mover family — the disarm above; the
#   disarm also fires on a mover AFTER the final commit clause:
#   deliberate over-tightening, see the disarm site);
#   variable / $(..) / relative / subdir / quote-broken cd targets;
#   AND/OR/BG/PIPE-separated cds (own separator); trailing-token cds
#   (`cd <root> junk`); any non-AND separator between the cd and a commit —
#   commits reached via `;`, newline, `||`, `|`, or `&` from the cd never
#   scope; commit-before-cd orderings; compound-context commands INCLUDING
#   parenthesized-subshell openers (`(cd <root> ...)` — the record
#   lead-strip would otherwise hide the paren from the cd arm while a
#   subshell cd never moves the parent cwd); `env --chdir` AND the short
#   pre-verb `-C <dir>` wrapper spellings incl. the exact-root `git -C`
#   form, whose waiver is refused (scope_unsafe, r4);
#   quoted-spacey pathspecs (rawtail parity, #1928); double-slash or
#   `/./`-bearing root spellings outside the pinned set; and the
#   path-limited `git add --all --` exemption keeps its root-cwd-no-cd
#   requirement (deliberately NOT widened).
# - Compound gated-add + non-gated-pathspec commit (`git add scripts/x.py &&
#   git commit -- docs/y.md`) still conservatively blocks: Layer-1 add-clause
#   text paths stay ADDITIVE under the #1620 pathspec-scoped read too.
# - `git -C .` from the root stays waived (path-blind sibling-parity residual);
#   the literal root SPELLINGS ($REPO absolute, ~/explore-persona-space,
#   $HOME/explore-persona-space) are NOT waived — one notch stronger than the
#   siblings' fully path-blind -C waiver.
# - `git revert` / `git cherry-pick` are not matched (verb anchor is `commit`)
#   — a deliberate residual convenient for emergency repair of a landed
#   offender; accidents-not-adversaries disposition.
# - Deletions of gated paths are exempt (no content to certify); the worktree
#   pipeline remains the sanctioned channel for deletions.
# - A byte-perfect forged cert line is indistinguishable by construction —
#   same agent-trust boundary as Step 10d's #1082 procedural ban: guards
#   target forgetting, not adversaries. NEVER hand-write the cert file.
# - String-literal spans are MASKED on the scan copy (round 3): quoted
#   message text can no longer split the clause, contribute pathspec/flag
#   tokens, or waive via a `git -C` mention; genuine trailing shell comments
#   are dropped by the masker (a gated path named in a real comment is no
#   longer scanned — comments are not command arguments). Masker residuals,
#   each analyzed toward a safe direction or an absurd shape: an UNTERMINATED
#   quote masks the remainder (bash rejects such a command — it never runs);
#   bare (unquoted) $(..)/backtick interiors stay unmasked and their
#   separators still split (pre-round-3 parity; a nested `git commit` inside
#   a substitution was never verb-anchored); only ONE pending heredoc per
#   line is tracked; a leading fd-dup/redirect before `git` (e.g.
#   `2>&1 git commit`) no longer classifies (the old fd-dup strip sed
#   text-normalized it away; unrealistic lead shape).
# - The `git -C` waiver is LEAD-ANCHORED (round 3): only the -C flag of the
#   lead git invocation can waive. A quoted -C target recovers its raw token
#   from the FIRST `git -C <word>` occurrence in the raw lead — under a
#   quoted spacey env-assignment prefix carrying its own `git -C` text that
#   recovery can read the wrong token (doubly-pathological shape); an
#   unextractable target refuses the waiver (block direction).
#
# Escape hatch: EPM_ALLOW_ROOT_CODE_COMMIT=1 — session env or inline prefix.
# Legitimate uses (record the reason in an epm:progress note): a MODIFIED
# payload file whose red is genuinely pre-existing but the helper refused
# conservatively; emergency fleet repair.
#
# Contract: reads the PreToolUse JSON on stdin, exit 0 allow, exit 2 (blocking,
# stderr fed back to Claude) refuse; any OTHER non-zero is non-blocking.
# Test overrides (hermetic tmp repos): EPM_ROOT_CODE_COMMIT_REPO,
# EPM_INLINE_CERT_PATH, EPM_INLINE_CERT_MAX_AGE_S, EPM_CERT_REHASH_DELAY_S
# (settle delay in seconds before the cert-retry re-hash pass, default 2;
# tests set 0 and/or PATH-shim `sleep` — #1857).
#
# Self-test: bash .claude/hooks/guard_root_code_commit.sh --self-test
set -u

REPO=/home/thomasjiralerspong/explore-persona-space
GUARD_REPO="${EPM_ROOT_CODE_COMMIT_REPO:-$REPO}"
CERT="${EPM_INLINE_CERT_PATH:-/tmp/eps-inline-lint-cert-v1.txt}"
MAX_AGE="${EPM_INLINE_CERT_MAX_AGE_S:-21600}" # 6 h

# Command-position anchors (#1250 parity): the clause LEAD must BE a git
# invocation whose verb is commit/add — optionally behind env assignments, the
# CLOSED wrapper set, shell compound keywords, bare -flag tokens, and a
# timeout-style duration token. An unlisted lead word (uv, bash, echo, ...)
# makes the predicate unmatchable. Composed from shared parts (round 3) so the
# lead-anchored `git -C` waiver ERE cannot drift from the verb anchors.
WRAP_UNIT_ERE='([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+|(nohup|setsid|sudo|env|time|timeout|command|exec|eval|if|elif|then|else|while|until|do|!)[[:space:]]+|-[^[:space:]]+[[:space:]]+|[0-9]+([.][0-9]+)?[smhd]?[[:space:]]+)*'
GIT_FLAGS_ERE='(-[^[:space:]]+([[:space:]]+[^[:space:]]+)?([[:space:]]+|$))*'
COMMIT_CMD_ERE='^'"${WRAP_UNIT_ERE}"'git[[:space:]]+'"${GIT_FLAGS_ERE}"'commit([[:space:]]|$)'
ADD_CMD_ERE='^'"${WRAP_UNIT_ERE}"'git[[:space:]]+'"${GIT_FLAGS_ERE}"'add([[:space:]]|$)'
# Lead-anchored `git -C <target>` detector (round 3 fix, concern
# quoted-message-seams-defeat-clause-scan): the -C must belong to the
# LEAD-anchored git invocation (its flag span, before the verb) — a string
# mention of `git -C` mid-clause (commit message quoting a remediation line)
# can no longer waive the clause.
DASHC_LEAD_ERE='^'"${WRAP_UNIT_ERE}"'git[[:space:]]+(-[^[:space:]]+([[:space:]]+[^[:space:]]+)?[[:space:]]+)*-C[[:space:]]+'
GATED_PATH_ERE='^(scripts/.*\.py|src/.+|tests/.*\.py)$'
# Worktree-cwd allow gate conservative screen (issue #2066; WIDENED in the
# round-2 code-review fix — blocker `wt-allow-screen-spelling-gaps`): any
# occurrence, anywhere in the RAW command, of a retarget construct
# classify_cmd does not model disables the allow gate. Derived in the
# FAIL-CLOSED direction: coarse over-matching tokens beat exact spellings,
# because a false positive (a screen token inside a commit message / path)
# merely restores today's block, while a false negative opens a root-landing
# allow. Alternatives, in order:
#   1. long-form directory/repo-pointing flags: --chdir (env), --work-tree /
#      --git-dir (git) — substrings, cover both = and separate-word forms;
#   2. core.worktree — the -c config spelling of the work-tree retarget;
#   3. repository-pointing env assignments: GIT_DIR= / GIT_WORK_TREE= /
#      GIT_INDEX_FILE= / GIT_COMMON_DIR=;
#   4. the cd WORD in ANY spelling: junk-tolerant word match — backslash-
#      escaped, quoted, $-prefixed cd all carry non-alnum junk the class
#      absorbs — with hard token boundaries on both sides, so a
#      flag-intervened invocation (a keyword + flags before the cd word) is
#      caught by the cd word ITSELF (subsumes the round-1
#      command/builtin-anchored alternative) while cdn / abcd / cd_helper
#      stay unmatched;
#   5. pushd / popd — bare substrings (evasion-immune: quoting or escaping
#      the word still leaves the substring in the raw text);
#   6. the source word — belt for sourced-script cwd changes;
#   7. short-form -C-bearing flag clusters (env's -C DIR, bundled -iC, ...;
#      also over-matches git -C / git commit -C — free: those fall back to
#      today's behavior).
# KNOWN RESIDUALS (named per plan §3's "the header must name residuals"):
# content reaching the shell only at runtime — variable expansion, eval'd
# strings built at runtime, `.`-sourced FILE CONTENTS — is invisible to any
# text screen; the `.` source SPELLING is deliberately NOT screened (a lone
# dot token would kill the gate for the common blanket-add-dot commit shape;
# accidents-not-adversaries, header parity). Both stay named in the header
# known-limitations block.
WT_CWD_ALLOW_SCREEN_ERE='--chdir|--work-tree|--git-dir|core\.worktree|GIT_(DIR|WORK_TREE|INDEX_FILE|COMMON_DIR)=|(^|[^[:alnum:]_])[^[:alnum:][:space:]_]*cd[^[:alnum:][:space:]_]*([^[:alnum:]_]|$)|pushd|popd|(^|[^[:alnum:]_])source([[:space:]]|$)|(^|[[:space:]])-[A-Za-z]*C([[:space:]=]|$)'
# #2357 r4 (concerns non-cd-cwd-mover-rides-armed-chain /
# unmodeled-cwd-mover-survives-scope-base): lead-anchored cwd-mover FAMILY
# for the per-record armed-chain disarm in classify_cmd. Tolerates adjacent
# quote/backslash chars so the `\cd`, `'cd'`, `"source"`, `'.'` spellings —
# which bash quote-removal still executes as the builtin — match at the
# record lead; `./script` does NOT match (the trailing class requires
# whitespace/EOL right after the word, and a `/` follows here). `cd` itself
# is listed ONLY for the quoted/escaped spellings: a plain-lead `cd` record
# is classified by the dedicated cd arm above the mover check and never
# reaches it. `eval` matches regardless of its argument (an eval'd string
# can move the cwd invisibly); `builtin`/`command` match regardless of the
# wrapped word (conservative — a `command`-prefixed record in an armed
# commit chain is rare and the disarm only narrows the #2357 widening).
# Paired at the disarm site with a raw-record grep of the
# WT_CWD_ALLOW_SCREEN_ERE retarget vocabulary above (catches mid-record
# movers: `git -C` / `env -C`, GIT_DIR=, --work-tree, a cd inside an eval
# argument). The dot-source spelling lives HERE and not in the screen ERE:
# lead-anchored, it cannot fire on the blanket-add-dot token the screen's
# residual note protects.
# #2357 r5 (reconcile v4 — the same two concerns re-opened): a record
# carrying leading legal ASSIGNMENT PREFIXES (NAME=value pairs before the
# command word — bash still executes the suffixed mover) defeated the `^`
# anchor, and dot-source is the ONE mover family with no whole-record
# screen fallback (the lone dot stays deliberately unscreened above), so
# an assignment-prefixed dot-source record rode an armed chain undisarmed
# — the executed r4 false-allow. Fix: tolerate ZERO-OR-MORE NAME=value
# prefixes before the WHOLE family alternation (uniform — the prefixed
# cd/pushd/popd/source/dot/eval/builtin/command spellings all disarm
# identically; the group is zero-width, so every pre-r5 match is
# unchanged). The value class [^[:space:]]* is exact on THIS (raw) copy
# only for unquoted values; the disarm site therefore ALSO greps the
# MASKED lead, where a QUOTED value's interior (spaces included) is
# spaceless filler — see the disarm-site note. The `builtin`/`command`
# family members double as the wrapper path for dot-source: a
# builtin-wrapped source record matches at the wrapper word itself.
# #2371 r6: the r5 prefix group widens to an alternated ATOM set, repeated
# zero-or-more in ANY interleaving before the mover family: (a) the
# assignment atom tolerates an optional literal `+` before `=` (append
# assignment, NAME+=value — bash executes the suffixed mover exactly as
# for NAME=value; the masked-lead arm covers quoted append values the
# same way as r5); (b) a second atom recognizes the `time` keyword
# wrapper (bare or `time -p`) — `time . x` sources in the CURRENT shell
# (keyword position), and the group's mandatory trailing whitespace gives
# `time` word-boundary safety (`timeout ...` cannot match). Execution-
# confirmed NOT-REACHABLE sourcing paths (2026-08-18, GNU bash 5.1.16;
# pinned in tests, not grammar gaps): `env . x` cannot source (external
# env cannot run the `.` builtin — unrecognized AND unreachable, stays
# scoped), and non-keyword-position `time` (`NAME=v time . x`) runs
# EXTERNAL time which cannot source — it nonetheless MATCHES the extended
# grammar (assignment atom then wrapper atom) and disarms: deliberate
# block-direction over-tightening, consistent with the standing "a record
# that merely mentions mover vocabulary also disarms" contract.
CWD_MOVER_LEAD_ERE='^(([A-Za-z_][A-Za-z0-9_]*[+]?=[^[:space:]]*|time([[:space:]]+-p)?)[[:space:]]+)*["'\''\\]*(cd|pushd|popd|source|\.|eval|builtin|command)["'\''\\]*([[:space:]]|$)'
FILL=$'\001' # masker filler byte for string-literal interiors (never IFS, never a separator)
# cd VARIABLE-target shape (issue #1676): exactly $NAME / ${NAME}, optionally
# followed by a literal /suffix — the only unproven-target family eligible for
# the provable same-command-assignment resolution arm (resolve_cd_var).
CD_VAR_TGT_ERE='^[$]([{][A-Za-z_][A-Za-z0-9_]*[}]|[A-Za-z_][A-Za-z0-9_]*)(/.*)?$'

# mask_and_split <command>: shell-quote-aware normalizer + clause splitter
# (round 3; concern quoted-message-seams-defeat-clause-scan). Awk char-scan
# adapted from scripts/guard_repo_root_branch.sh mask_ssh_payload_separators()
# (#1185/#1413) — generalized from "separators inside one ssh payload" to ALL
# string-literal spans. Emits THREE lines per clause record:
#   1: separator tag that PRECEDED the record (START|AND|OR|SEQ|PIPE|BG|NL)
#   2: MASKED clause — every char inside a string literal (single-quoted;
#      double-quoted incl. $(..) substitution, backtick spans, and heredoc
#      bodies nested inside it; $'..' ANSI-C), every backslash-escaped char,
#      and every genuine-comment char replaced/dropped via \001. Record splits
#      happen ONLY at unquoted separators/newlines, so a `;`/`&&`/`|`/newline
#      inside a commit message can no longer split the clause (round-2
#      Major 1). The lead/verb/waiver decisions AND the token/flag/pathspec
#      scan all read THIS copy — message-string content contributes no tokens
#      and no `git -C` waiver, and genuine tokens outside literals survive
#      verbatim.
#   3: RAW clause (in-literal newlines -> \001 to keep the 3-line protocol) —
#      the decision copy for cd/-C TARGET extraction only, where a quoted
#      target ("$WT", "$REPO"-spellings) must keep its content.
# Bare (unquoted) $(..)/backtick interiors are deliberately NOT masked
# (parity with the prior scanner: their content stays scannable; separators
# inside them still split). Top-level heredoc bodies are DROPPED (data, not
# clause text); one pending heredoc per line is tracked. A masker failure
# yields no records -> Layer 1 finds no clause -> fail-OPEN (A16 parity: a
# guard parse bug must never wedge the fleet).
mask_and_split() {
  printf '%s\n' "$1" | LC_ALL=C awk '
    function nl1(ch) { return ch == "\n" ? F : ch }
    function nl2(str,   t) { t = str; gsub(/\n/, F, t); return t }
    function emit(mc, rc) { mbuf = mbuf mc; rbuf = rbuf rc }
    function flushrec() {
      gsub(/\n/, F, mbuf); gsub(/\n/, F, rbuf) # stride belt: 1 line per field
      print pending; print mbuf; print rbuf
      mbuf = ""; rbuf = ""; prev = ""
    }
    # Parse a heredoc opener at position p ("<<"): records the delimiter in
    # hd_pending and returns the index just past it, or p when unparseable.
    function parse_heredoc_delim(p,   j, qd, k, d) {
      j = p + 2
      if (substr(s, j, 1) == "-") j++
      while (j <= n && substr(s, j, 1) ~ /[ \t]/) j++
      qd = substr(s, j, 1)
      if (qd == "\047" || qd == "\042") {
        k = index(substr(s, j + 1), qd)
        if (k == 0) return p
        d = substr(s, j + 1, k - 1)
        if (d == "" || index(d, "\n") > 0) return p
        hd_pending = d
        return j + 1 + k
      }
      k = j
      while (k <= n && substr(s, k, 1) ~ /[^ \t\n;&|()<>]/) k++
      d = substr(s, j, k - j)
      if (d == "") return p
      hd_pending = d
      return k
    }
    BEGIN { F = "\001"; nrec = 0 }
    { rec[nrec++] = $0 }
    END {
      s = ""
      for (r = 0; r < nrec; r++) s = s (r ? "\n" : "") rec[r]
      n = length(s)
      q = ""       # "" unquoted | s single | d double | a ANSI-C | c comment
      subst = 0    # $( depth inside a double-quoted span
      iq = ""      # quote state inside that substitution: "" | s | d
      bt = 0       # backtick span inside a double-quoted span
      hd_pending = ""; hd_active = 0; hd_top = 0; hd_delim = ""
      mbuf = ""; rbuf = ""; prev = ""; pending = "START"
      i = 1
      while (i <= n) {
        c = substr(s, i, 1); c2 = substr(s, i, 2)
        if (hd_active) {
          # Heredoc body: whole lines until the terminator line (<<- form
          # tolerates leading tabs). Top-level bodies are dropped;
          # in-substitution bodies are masked into the enclosing span.
          p = index(substr(s, i), "\n")
          if (p == 0) { line = substr(s, i); adv = length(line) }
          else { line = substr(s, i, p - 1); adv = p }
          tline = line; sub(/^\t+/, "", tline)
          if (!hd_top) {
            fl = line; gsub(/./, F, fl)
            emit(fl, fl)
            if (p > 0) emit(F, F)
          }
          i += adv
          if (tline == hd_delim) hd_active = 0
          continue
        }
        if (q == "s") {
          if (c == "\047") { q = ""; emit(c, c); prev = c; i++; continue }
          emit(F, nl1(c)); i++; continue
        }
        if (q == "a") { # $\047..\047 ANSI-C: backslash escapes honored
          if (c == "\\" && i < n) { emit(F F, nl2(substr(s, i, 2))); i += 2; continue }
          if (c == "\047") { q = ""; emit(c, c); prev = c; i++; continue }
          emit(F, nl1(c)); i++; continue
        }
        if (q == "c") { # genuine comment: dropped from BOTH copies
          if (c == "\n") { q = ""; continue } # newline handled by the U branch
          i++; continue
        }
        if (q == "d") {
          if (bt) {
            if (c == "\\" && i < n) { emit(F F, nl2(substr(s, i, 2))); i += 2; continue }
            if (c == "\140") bt = 0
            emit(F, nl1(c)); i++; continue
          }
          if (subst > 0) {
            if (iq == "s") {
              if (c == "\047") iq = ""
              emit(F, nl1(c)); i++; continue
            }
            if (iq == "d") {
              if (c == "\\" && i < n) { emit(F F, nl2(substr(s, i, 2))); i += 2; continue }
              if (c == "\042") iq = ""
              emit(F, nl1(c)); i++; continue
            }
            if (c == "\\" && i < n) { emit(F F, nl2(substr(s, i, 2))); i += 2; continue }
            if (c == "\047") { iq = "s"; emit(F, c); i++; continue }
            if (c == "\042") { iq = "d"; emit(F, c); i++; continue }
            if (substr(s, i, 3) == "<<<") { emit(F F F, "<<<"); i += 3; continue }
            if (c2 == "<<") {
              j = parse_heredoc_delim(i)
              if (j > i) {
                seg = substr(s, i, j - i); fl = seg; gsub(/./, F, fl)
                emit(fl, nl2(seg)); i = j; continue
              }
              emit(F F, "<<"); i += 2; continue
            }
            if (c == "(") { subst++; emit(F, c); i++; continue }
            if (c == ")") { subst--; emit(F, c); i++; continue }
            if (c == "\n") {
              if (hd_pending != "") { hd_active = 1; hd_top = 0; hd_delim = hd_pending; hd_pending = "" }
              emit(F, F); i++; continue
            }
            emit(F, nl1(c)); i++; continue
          }
          if (c == "\\" && i < n) { emit(F F, nl2(substr(s, i, 2))); i += 2; continue }
          if (c2 == "$(") { subst = 1; emit(F F, "$("); i += 2; continue }
          if (c == "\140") { bt = 1; emit(F, c); i++; continue }
          if (c == "\042") { q = ""; emit(c, c); prev = c; i++; continue }
          emit(F, nl1(c)); i++; continue
        }
        # ---- unquoted ----
        if (c == "\\") {
          if (c2 == "\\\n") { emit(" ", " "); prev = " "; i += 2; continue } # line continuation
          if (substr(s, i, 3) == "\\\r\n") { emit(" ", " "); prev = " "; i += 3; continue }
          if (i < n) { emit("\\" F, nl2(substr(s, i, 2))); prev = F; i += 2; continue } # escaped char: 1-char literal
          emit(c, c); prev = c; i++; continue
        }
        if (c == "\047") {
          q = (prev == "$") ? "a" : "s"
          emit(c, c); prev = c; i++; continue
        }
        if (c == "\042") { q = "d"; emit(c, c); prev = c; i++; continue }
        if (c == "#" && (prev == "" || prev == " " || prev == "\t")) { q = "c"; continue }
        if (substr(s, i, 3) == "<<<") { emit("<<<", "<<<"); prev = "<"; i += 3; continue }
        if (c2 == "<<") {
          j = parse_heredoc_delim(i)
          if (j > i) { seg = substr(s, i, j - i); emit(seg, seg); prev = substr(seg, length(seg), 1); i = j; continue }
          emit("<<", "<<"); prev = "<"; i += 2; continue
        }
        if (c2 == "&&") { flushrec(); pending = "AND"; i += 2; continue }
        if (c2 == "||") { flushrec(); pending = "OR"; i += 2; continue }
        if (c == ";") { flushrec(); pending = "SEQ"; i++; continue }
        if (c2 == "|&") { flushrec(); pending = "PIPE"; i += 2; continue }
        if (c == "|") { flushrec(); pending = "PIPE"; i++; continue }
        if (c == "&") {
          if (c2 == "&>") { emit(c, c); prev = c; i++; continue }  # &> / &>> redirect
          if (prev == ">") { emit(c, c); prev = c; i++; continue } # fd-dup 2>&1
          flushrec(); pending = "BG"; i++; continue
        }
        if (c == "\n") {
          flushrec(); pending = "NL"
          if (hd_pending != "") { hd_active = 1; hd_top = 1; hd_delim = hd_pending; hd_pending = "" }
          i++; continue
        }
        emit(c, c); prev = c; i++
      }
      flushrec()
    }'
}

# cd_latch_verdict <tgt>: shared 3-valued commit-binding classification of a
# cd TARGET string (issue #1676 fix (a) factoring). Factors the latch pattern
# list out of the literal-cd arm so the resolved-variable arm re-enters the
# SAME list — single point of truth: resolution can never latch a pattern the
# literal arm would not, and a root-spelling RHS maps to `root` (never
# latched). Sets the global cd_verdict: latch (provably non-root), root (a
# root spelling), unproven (relative / variable / empty — never trusted,
# fail closed). NO new latch pattern may be added here without re-review of
# BOTH callers. The absolute root arm compares against $GUARD_REPO (#2046):
# it defaults to $REPO, so production is bit-identical, while hermetic test
# repos (EPM_ROOT_CODE_COMMIT_REPO) gain cd-to-their-own-root coverage; the
# literal ~/ and $HOME/ spellings stay production spellings by design.
cd_latch_verdict() {
  case "$1" in
    *.claude/worktrees/*) cd_verdict=latch ;;                # a worktree IS its own tree
    "$GUARD_REPO" | "$GUARD_REPO"/*) cd_verdict=root ;;      # root or a subdir (git walks up)
    '~/explore-persona-space' | '~/explore-persona-space/'*) cd_verdict=root ;;
    '$HOME/explore-persona-space' | '$HOME/explore-persona-space/'*) cd_verdict=root ;;
    /* | '~' | '~/'* | '$HOME/'*) cd_verdict=latch ;;        # absolute/~-anchored, not the root
    *) cd_verdict=unproven ;;                                # relative/variable/empty: unproven
  esac
}

# path_sane_component <s>: 0 iff <s> is a plausible LITERAL path fragment for
# the variable-resolution arm (issue #1676 gates 4/5): non-empty; no
# whitespace; no \001 fill byte; no backtick / $( substitution; no shell
# metachars < > ( ) & | ;; no remaining quote char or backslash (a
# mixed-quote or ANSI-C $'..' value expands to something OTHER than the
# scanned text); no `..` path segment (deliberately STRICTER than the
# literal-cd arm — strictly-tightening additions are free on a new inference
# path). Embedded plain $VAR references may remain: parity with the literal
# arm's pattern list — a variable-leading result re-enters the unproven
# branch of cd_latch_verdict.
path_sane_component() {
  [ -n "$1" ] || return 1
  case "$1" in
    *[[:space:]]* | *"$FILL"* | *'`'* | *'$('*) return 1 ;;
    *'<'* | *'>'* | *'('* | *')'* | *'&'* | *'|'* | *';'*) return 1 ;;
    *'"'* | *"'"* | *'\'*) return 1 ;;
  esac
  case "/$1/" in */../*) return 1 ;; esac
  return 0
}

# compound_context_present: 0 iff ANY record in the caller's recs opens or
# continues a compound statement — resolve_cd_var's g7 set (if/while/for/case/
# function keywords, name() defs, bare `{` group openers), PLUS (#2357 r1
# MF-3) a parenthesized-subshell OPENER: a record whose whitespace-stripped
# masked text begins with `(`. Rationale for the `(` widening: classify_cmd's
# record lead-strip removes leading `[({]` BEFORE the cd arm, so a subshell's
# inner cd reads as a plain exact-root cd there while a subshell cd can NEVER
# move the parent shell's cwd — `(cd <root> && true) && git commit -- p` is
# an all-AND chain whose commit executes at the parent's unmoved cwd, the one
# bypass chain dominance alone admits. THIS loop's record read strips
# whitespace ONLY, so the `(` survives here and the record-level check is
# well-posed. Reads the caller's recs/n via bash dynamic scoping (the
# resolve_cd_var / classify_candidate mechanism). Factored out of
# resolve_cd_var's g7 (#1857 landing_sha precedent) so its call site and the
# #2357 post-loop verdict cannot diverge; the `(` widening is strictly
# TIGHTENING for resolve_cd_var (fail-closed direction).
compound_context_present() {
  local j m
  for ((j = 0; j + 2 < n; j += 3)); do
    m=$(printf '%s' "${recs[j + 1]}" | sed -E 's/^[[:space:]]+//')
    if printf '%s' "$m" | grep -qE \
      '^(if|then|elif|else|fi|while|until|for|do|done|case|esac|function)([[:space:]]|$)|^[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\(\)|^\{([[:space:]]|$)|^\('; then
      return 0
    fi
  done
  return 1
}

# resolve_cd_var <name> <suffix> <cd_record_index>: variable-resolution arm of
# the cd-latch (issue #1676 fix (a) — incident #1644: a worktree-bound
# `cd "$WT" && ... && git commit` compound was classified as a root commit).
# Resolves a cd target that is exactly a $NAME/${NAME} expansion (optional
# literal /suffix) from a PROVABLE same-command assignment, so the resolved
# string can re-enter cd_latch_verdict. Reads the caller's `recs`/`n` via
# bash dynamic scoping (the classify_candidate mechanism below). Bash runs
# each Bash tool call in a FRESH shell, so a same-command whole-clause
# assignment is the only command-text-provable way $NAME can be set; an
# inherited-environment value is unprovable by construction and stays
# unproven. Resolution succeeds ONLY when all SEVEN certification gates hold;
# ANY failure returns 1 with a reason token in the global resolve_reason
# (consumed by the block-path cd-diag lines) and the target stays unproven —
# today's fail-closed behavior:
#   g7 compound-context: NO record in the command may open/continue a
#      compound statement (if/while/for/case/function keywords, name() defs,
#      bare `{` group openers; factored to compound_context_present, #2357,
#      which additionally refuses parenthesized-subshell `(` openers —
#      strictly tightening here) — the masker tracks quote/heredoc state only
#      (no brace/keyword depth), so a compound-BODY line surfaces as its own
#      NL-tagged record and would otherwise read as unconditionally executed
#      while being conditional (or dead) at runtime;
#   g1 whole-clause assignment anchor: a MASKED record that IS the assignment
#      (`[export] NAME=<value>`), leading whitespace stripped ONLY (a `(`/`{`
#      wrapped or env-prefix `NAME=x cmd` assignment never matches — neither
#      persists past its own clause);
#   g2 exactly one such record in the WHOLE command, PRECEDING the cd record
#      (two+ = last-write-wins scan ambiguity; after the cd = unset at cd
#      time);
#   g3 unconditional position: the assignment record's own separator tag is
#      START/SEQ/NL, and the NEXT record's separator is neither BG nor PIPE
#      (a trailing `&` or `|` runs the assignment in a subshell — it does
#      not persist);
#   g6 mutation belt (narrow): no record mutates NAME via unset / declare /
#      typeset / read / printf -v or a NAME+= append (command-wide belt;
#      deliberate-evasion indirection — eval, source, arithmetic — stays out
#      of the hook family's threat model: the logged EPM_ALLOW_ROOT_CODE_COMMIT
#      escape is the sanctioned deliberate path);
#   g4 path-sane literal RHS, extracted from the RAW record (the masked copy
#      fills quoted content), one surrounding quote pair stripped;
#   g5 path-sane literal suffix (additionally: no $ — a suffix is never a
#      nested expansion).
# On success sets the global resolved_tgt (RHS + suffix). Single hop by
# construction: a variable-leading RHS re-enters cd_latch_verdict's unproven
# branch — no recursion, no second resolution.
resolve_cd_var() {
  local vname="$1" vsuffix="$2" cd_idx="$3"
  local j m raw_a rhs sq rhs_ere
  local asg_idx=-1 asg_count=0
  resolved_tgt="" resolve_reason=""

  # g7 — compound-context refusal (whole command; reason wins over g1-g6).
  # Factored to compound_context_present (#2357 D4); the reason token stays
  # at THIS call site and the return polarity is preserved. The helper's
  # `(`-opener widening is strictly tightening here (fail-closed direction).
  if compound_context_present; then
    resolve_reason=compound-context
    return 1
  fi

  # g1 + g2 — whole-clause assignment anchor, exactly one, preceding the cd.
  for ((j = 0; j + 2 < n; j += 3)); do
    m=$(printf '%s' "${recs[j + 1]}" | sed -E 's/^[[:space:]]+//')
    if printf '%s' "$m" | grep -qE "^(export[[:space:]]+)?${vname}=[^[:space:]]*[[:space:]]*$"; then
      asg_count=$((asg_count + 1))
      asg_idx=$j
    fi
  done
  if [ "$asg_count" -eq 0 ]; then
    resolve_reason=no-assignment
    return 1
  fi
  if [ "$asg_count" -gt 1 ]; then
    resolve_reason=multiple-assignments
    return 1
  fi
  if [ "$asg_idx" -ge "$cd_idx" ]; then
    resolve_reason=no-assignment # no assignment PRECEDES the cd
    return 1
  fi

  # g3 — unconditional position + next-separator subshell check.
  case "${recs[asg_idx]}" in
    START | SEQ | NL) : ;;
    *)
      resolve_reason=conditional-assignment
      return 1
      ;;
  esac
  if [ $((asg_idx + 3)) -lt "$n" ]; then
    case "${recs[asg_idx + 3]}" in
      BG)
        resolve_reason=backgrounded-assignment
        return 1
        ;;
      PIPE)
        # `NAME=x | cmd` runs the assignment inside a pipeline subshell — it
        # never persists (strictly-tightening addition beyond the plan's BG
        # check; pinned by self-test B34).
        resolve_reason=pipelined-assignment
        return 1
        ;;
    esac
  fi

  # g6 — mutation belt.
  for ((j = 0; j + 2 < n; j += 3)); do
    m="${recs[j + 1]}"
    printf '%s' "$m" | grep -qE "(^|[^A-Za-z0-9_])${vname}([^A-Za-z0-9_]|$)" || continue
    if printf '%s' "$m" | grep -qE \
      '(^|[[:space:]])(unset|declare|typeset|read)([[:space:]]|$)|(^|[[:space:]])printf[[:space:]]+-v([[:space:]]|$)' \
      || printf '%s' "$m" | grep -qE "(^|[[:space:]])${vname}[+]="; then
      resolve_reason=mutation-belt
      return 1
    fi
  done

  # g4 — path-sane literal RHS from the RAW record (gate-1 matched the MASKED
  # copy; quoted content only survives on the raw copy).
  raw_a=$(printf '%s' "${recs[asg_idx + 2]}" | sed -E 's/^[[:space:]]+//')
  sq="'"
  rhs_ere="^(export[[:space:]]+)?${vname}=(\"[^\"]*\"|${sq}[^${sq}]*${sq}|[^[:space:]]+)[[:space:]]*\$"
  if [[ $raw_a =~ $rhs_ere ]]; then
    rhs="${BASH_REMATCH[2]}"
  else
    resolve_reason=dynamic-rhs # raw shape diverges from the masked anchor
    return 1
  fi
  rhs=$(printf '%s' "$rhs" | sed -E "s/^\"(.*)\"\$/\\1/; s/^'(.*)'\$/\\1/")
  if ! path_sane_component "$rhs"; then
    resolve_reason=dynamic-rhs
    return 1
  fi

  # g5 — suffix sanity (only when the cd target carried a /suffix).
  if [ -n "$vsuffix" ]; then
    case "$vsuffix" in
      *'$'*)
        resolve_reason=dynamic-rhs
        return 1
        ;;
    esac
    if ! path_sane_component "$vsuffix"; then
      resolve_reason=dynamic-rhs
      return 1
    fi
  fi

  resolved_tgt="${rhs}${vsuffix}"
  return 0
}

# classify_cmd <command>: Layer 1. Sets globals root_commit / has_dash_a /
# add_all_chained / text_paths (newline-separated gated-prefix tokens) /
# add_pathspecs (newline-separated post-`--` candidates of EXEMPTED
# path-limited `git add -A|--all -- <pathspec>` clauses, issue #1977 —
# resolved by the Layer-2 cwd-gated `git status` read).
# classify_candidate <tok>: second-pass commit-clause pathspec candidate
# classifier (issue #1620). Mutates the caller's clause state via bash dynamic
# scoping: clause_opaque / commit_pathspecs / n_cand / commit_has_pathspec.
# Reject (opaque) any token the hook cannot treat as a LITERAL pathspec:
# masked/quoted/backslash-bearing tokens, unexpanded shell tokens (MF-2 —
# $VAR / $(..) / backtick / parens / leading ~), and redirection-shaped
# tokens; plain glob tokens (* ? []) stay clean candidates (git evaluates
# them). Fallback direction: opaque => today's whole-index check.
classify_candidate() {
  case "$1" in
    *"$FILL"* | *[\"\'\\]*) clause_opaque=1 ;;
    *'$'* | *'`'* | *'('* | *')'* | '~'* | *'<'* | *'>'*) clause_opaque=1 ;;
    *)
      commit_pathspecs="$commit_pathspecs
$1"
      n_cand=$((n_cand + 1))
      commit_has_pathspec=1
      ;;
  esac
}

# redirect_tok_kind <tok>: commit-clause redirect-token classifier (issue
# #1928). Redirections are consumed by the SHELL and are never git pathspec
# arguments, so excluding a STRICTLY-recognized redirect grammar from the
# candidate stream is semantically exact — not a heuristic relaxation.
# Echoes exactly one of:
#   pair — bare operator that consumes the NEXT word (optional-fd `>`, `>>`,
#          `<`, and the `&>` / `&>>` forms);
#   self — self-contained: fd-dup (`2>&1`, `>&2`, `2>&-`), or an operator
#          with an ATTACHED clean-literal target — the SAME literal test
#          classify_candidate applies;
#   no   — everything else: FILL/quote/backslash-bearing tokens, the
#          process-substitution family `>(..)` / `<(..)`, the here-doc /
#          here-string operator family `<<` / `<<-` / `<<<`, and attached
#          targets carrying `$` / backtick / parens / leading `~`.
# The EREs are anchored ^...$; the operator classes admit neither a `(`
# after the operator nor a second `<` beyond the single input-redirect form
# (pinned by the r14/r15 refuse tests). Fallback direction: `no` keeps
# today's opaque -> whole-index -> block path via classify_candidate.
redirect_tok_kind() {
  local tok="$1" tgt
  case "$tok" in
    *"$FILL"* | *[\"\'\\]*)
      echo no
      return
      ;;
  esac
  if printf '%s\n' "$tok" | grep -qE '^([0-9]*(>>|>|<)|&>>|&>)$'; then
    echo pair
    return
  fi
  if printf '%s\n' "$tok" | grep -qE '^[0-9]*>&([0-9]+|-)$'; then
    echo self
    return
  fi
  if printf '%s\n' "$tok" | grep -qE '^([0-9]*(>>|>|<)|&>>|&>)[^<>&|()]+$'; then
    tgt=$(printf '%s\n' "$tok" | sed -E 's/^([0-9]*(>>|>|<)|&>>|&>)//')
    case "$tgt" in
      *'$'* | *'`'* | '~'*) echo no ;;
      *) echo self ;;
    esac
    return
  fi
  echo no
}

# heredoc_tok_kind <tok>: commit-clause here-doc / here-string OPENER-token
# classifier (issue #2046). Openers are consumed by the SHELL and are never
# git pathspec arguments, so excluding a STRICTLY-recognized opener grammar
# from the candidate stream is semantically exact — the #1928 redirect
# argument. Soundness anchor: the masker emits a TOP-LEVEL opener VERBATIM
# into the masked copy (the parse_heredoc_delim seg, quotes included; the
# literal `<<<`), while string-literal content masks to \001 fill — so a
# masked-clause token carrying a literal `<<` / `<<<` PREFIX can only
# originate from genuine opener syntax (a fully-quoted token masks its
# operator too and never keeps the spelling). Echoes exactly one of:
#   pair — bare operator (`<<`, `<<-`, `<<<`): consumes the NEXT word (the
#          space-separated delimiter / here-string word — the same word
#          parse_heredoc_delim consumed on the raw text);
#   self — operator + ATTACHED remainder: an unquoted delimiter drawn from
#          clean literal chars only, or a CLOSED quoted delimiter
#          (`<<'D'` / `<<"D"`, optional `-`, non-empty, no same-kind quote
#          or fill inside — exactly the masked shape the top-level opener
#          emit produces); any non-empty attachment after `<<<` (the
#          here-string word is stdin data by construction);
#   no   — everything else: tokens without the literal opener prefix
#          (masked string literals included), unterminated / spacey quoted
#          delimiters (a spacey quoted delimiter word-splits and its pieces
#          stay opaque), and delimiter shapes carrying $ / backtick / quote
#          / backslash / fill / redirect metachars — each keeps today's
#          opaque -> whole-index -> block path via classify_candidate.
# redirect_tok_kind stays byte-untouched: its heredoc-family->`no` contract
# is pinned by its doc block + the test-side grammar pin (rd15b); this
# classifier runs BEFORE it in both commit-clause token arms. Fallback
# direction: `no` keeps today's opaque -> whole-index -> block path.
heredoc_tok_kind() {
  local tok="$1" rem inner
  case "$tok" in
    '<<<')
      echo pair
      return
      ;;
    '<<<'*)
      echo self
      return
      ;;
    '<<' | '<<-')
      echo pair
      return
      ;;
    '<<'*) : ;;
    *)
      echo no
      return
      ;;
  esac
  rem="${tok#<<}"
  rem="${rem#-}"
  case "$rem" in
    "'"?*"'")
      inner="${rem#\'}"
      inner="${inner%\'}"
      case "$inner" in
        *"'"* | *"$FILL"*) echo no ;;
        *) echo self ;;
      esac
      return
      ;;
    '"'?*'"')
      inner="${rem#\"}"
      inner="${inner%\"}"
      case "$inner" in
        *'"'* | *"$FILL"*) echo no ;;
        *) echo self ;;
      esac
      return
      ;;
  esac
  case "$rem" in
    '' | *"$FILL"* | *[\"\'\\]* | *'$'* | *'`'* | *'<'* | *'>'* | *'&'* | *'|'* | *'('* | *')'* | *';'*)
      echo no
      ;;
    *) echo self ;;
  esac
}

classify_cmd() {
  local cmd="$1"
  root_commit=0 has_dash_a=0 add_all_chained=0 text_paths=""
  # Pathspec-scoping state (issue #1620): set by the second per-clause token
  # pass below; consumed by the Layer-2 cwd gate + scoped read.
  commit_has_pathspec=0 pathspec_opaque=0 commit_bare_clause=0 scope_unsafe=0
  cd_nonroot=0 commit_pathspecs="" add_pathspecs=""
  # Provably-root-cd scope-base state (issue #2357): armed by the exact-root
  # cd arm below, broken by any non-AND separator (chain dominance, D3a),
  # ordered by the commit-verb bits (D3b), DISARMED by any later
  # non-canonical cd on the armed chain (r3, concern
  # mutable-symbolic-root-proof: the base is provable only while the LAST
  # cwd-moving cd before every commit clause is the canonical absolute
  # root — a riding alias/subdir/other cd clears cd_root_seen and sets
  # cd_base_disarmed; a later canonical arming cd re-establishes both),
  # reduced to cd_root_base in the post-loop verdict; consumed by the
  # Layer-2 scope gate as an OR-alternative to cwd_ok.
  cd_root_seen=0 cd_and_chain=0 commit_before_root_cd=0 commit_off_chain=0 cd_root_base=0
  cd_base_disarmed=0
  # Unproven-cd tracking (issue #1676 fix (b)): one entry per cd clause whose
  # FINAL verdict (after any resolution attempt) is unproven — consumed by
  # the block path's cd-diag lines; never read on the allow path.
  cd_unproven=""
  # Root-retarget evidence (issue #2066): set to 1 whenever ANY clause carries
  # a construct that can re-anchor a later commit at the root or at an
  # unprovable target — a cd whose FINAL verdict is root or unproven, or a
  # `git -C` waiver REFUSAL (root spelling / unextractable target). Consumed
  # ONLY by the worktree-cwd allow gate: evidence present => the
  # bare-clause-only proof does not hold, the gate never fires (today's
  # fail-closed behavior). No existing flag's semantics change.
  retarget_evidence=0

  local triplets
  triplets=$(mask_and_split "$cmd")

  local -a recs
  mapfile -t recs <<< "$triplets"

  local n=${#recs[@]} i sep masked raw lead raw_lead tgt cpfx mtgt ctgt latched=0 verb
  local vname vsuffix tgt_trail nsep
  for ((i = 0; i + 2 < n; i += 3)); do
    sep=${recs[i]} masked=${recs[i + 1]} raw=${recs[i + 2]}
    # #2357 chain-break (D3a): any non-AND separator after an armed root cd
    # breaks the cd->commit AND-dominance chain. FIRST statement of the loop
    # body — before the comment-strip, the empty-lead continue, and every
    # case dispatch — so no record class can skip it (the cd arm itself ends
    # in `continue`). A re-arming exact-root cd re-sets the chain in its own
    # arm below (its SEQ/NL separator breaks the old chain here first).
    if [ "$cd_root_seen" = 1 ] && [ "$sep" != AND ]; then cd_and_chain=0; fi
    [ "$sep" = AND ] || latched=0

    # Genuine-comment strip, belt only: the masker already drops comment
    # chars, and an in-literal `#` is \001 on the masked copy — so this sed
    # can never discard in-message tokens (the round-2 Major class).
    masked=$(printf '%s' "$masked" | sed -E 's/(^|[[:space:]])#.*$//')
    lead=$(printf '%s' "$masked" | sed -E 's/^[[:space:]{(]+//')
    [ -n "$lead" ] || continue
    raw_lead=$(printf '%s' "$raw" | sed -E 's/^[[:space:]{(]+//')

    # cd-latch ARM (pull-guard #804 semantics): detection on the MASKED lead;
    # TARGET from the RAW lead (a quoted worktree/root target keeps its
    # content). Latch only provably NON-root targets; unproven targets stay
    # unlatched (fail closed). Verdict via the shared cd_latch_verdict list
    # (issue #1676): a $NAME-shaped unproven target gets ONE provable
    # same-command-assignment resolution attempt (resolve_cd_var, fix (a));
    # every still-unproven target is recorded for the block-path cd-diag
    # lines (fix (b)).
    if printf '%s' "$lead" | grep -qE '^cd([[:space:]]|$)'; then
      tgt=$(printf '%s' "$raw_lead" | sed -E 's/^cd[[:space:]]*//' | awk '{print $1}' \
        | sed -E "s/^[\"']//; s/[\"']\$//")
      # #2357: the post-target remainder of the SAME raw lead (everything
      # after the first whitespace-delimited word). Non-empty => the cd
      # carries a trailing token — bash cd then fails deterministically
      # ("too many arguments") while a SEQ/OR-chained commit still runs
      # (r1 MF-2) — which refuses ARMING only; cd_nonroot is untouched for
      # an exact-root trailing-token cd (today's behavior at root cwd).
      tgt_trail=$(printf '%s' "$raw_lead" \
        | sed -E 's/^cd[[:space:]]*//; s/^[^[:space:]]+[[:space:]]*//')
      cd_latch_verdict "$tgt"
      resolve_reason=not-a-variable
      if [ "$cd_verdict" = unproven ] && [[ $tgt =~ $CD_VAR_TGT_ERE ]]; then
        vname="${BASH_REMATCH[1]}"
        vname="${vname#\{}"
        vname="${vname%\}}"
        vsuffix="${BASH_REMATCH[2]:-}"
        if resolve_cd_var "$vname" "$vsuffix" "$i"; then
          cd_latch_verdict "$resolved_tgt"
          # A resolved-but-variable-leading RHS stays unproven (single hop).
          [ "$cd_verdict" = unproven ] && resolve_reason=dynamic-rhs
        fi
      fi
      latched=0
      [ "$cd_verdict" = latch ] && latched=1
      # #2066: a cd whose FINAL verdict is root (provable root retarget) or
      # unproven (unprovable target, fail closed) is retarget evidence for
      # the worktree-cwd allow gate; a latch verdict (provably non-root) is
      # not — a cd-latched sibling worktree never lands a commit at root.
      [ "$cd_verdict" = root ] && retarget_evidence=1
      if [ "$cd_verdict" = unproven ]; then
        retarget_evidence=1
        cd_unproven="$cd_unproven
target=$(printf '%.80s' "$tgt") reason=$resolve_reason"
      fi
      # MF-1 (ii), issue #1620: any cd whose target is not an EXACT root
      # spelling (repo subdir, relative, unproven, or latched-away) moves the
      # pathspec-resolution base — disable scoping for the whole command.
      # Operates on the ORIGINAL target, deliberately NOT the resolved one
      # (issue #1676 must-ask: scoping-off is the conservative direction and
      # moot under a latch — a latched chain sets no root_commit). The
      # absolute arm compares against $GUARD_REPO (#2046): production
      # bit-identical (GUARD_REPO defaults to $REPO); hermetic test repos
      # keep scoping on for a cd to their own root.
      case "$tgt" in
        "$GUARD_REPO" | "$GUARD_REPO"/)
          # #2357: CANONICAL ABSOLUTE root cd — the ONLY arming spellings
          # (r2 fix, concern mutable-symbolic-root-proof): an absolute
          # literal equal to $GUARD_REPO is the one target the COMMAND TEXT
          # proves lands at the root that the armed Layer-2 cert check
          # resolves pathspecs against (root-pinned `git -C "$GUARD_REPO"`).
          # Arm only when ALL of:
          #  (1) own separator in {START, SEQ, NL} — g3-matching
          #      (resolve_cd_var g3). AND is EXCLUDED: an AND-separated cd
          #      can itself be SKIPPED by an earlier failure in its && chain
          #      while a later SEQ/OR-separated commit still runs at the
          #      hook cwd (r1 MF-1).
          #  (2) NO token after the target on this record (r1 MF-2, the
          #      tgt_trail computation above).
          #  (3) next record's separator not BG/PIPE: a trailing & or |
          #      backgrounds/subshells the cd — the parent shell's cwd never
          #      moves. (Chain dominance, D3/D5, subsumes this; kept as a
          #      cheap local belt-and-suspenders refusal.)
          # Chain dominance (D3) + the compound/subshell-context refusal
          # (D4, compound_context_present) complete the predicate in the
          # post-loop verdict (D5). Arming RE-SETS cd_and_chain=1: the chain
          # is measured from the MOST RECENT armed cd (a fresh `; cd <root>`
          # re-establishes the base — its SEQ record already broke any prior
          # chain via the D3a line above) and CLEARS cd_base_disarmed (r3):
          # a canonical arming cd supersedes any earlier non-canonical
          # disarm — the base is measured from the LAST canonical cd.
          case "$sep" in
            START | SEQ | NL)
              if [ -z "$tgt_trail" ]; then
                nsep=""
                [ $((i + 3)) -lt "$n" ] && nsep=${recs[i + 3]}
                case "$nsep" in
                  BG | PIPE) : ;;
                  *) cd_root_seen=1 cd_and_chain=1 cd_base_disarmed=0 ;;
                esac
              fi
              ;;
          esac
          ;;
        '~/explore-persona-space' | '~/explore-persona-space/' \
          | '$HOME/explore-persona-space' | '$HOME/explore-persona-space/')
          # #2357 r2+r3 (concern mutable-symbolic-root-proof): symbolic
          # ALIAS root spellings keep their legacy NON-poisoning
          # classification (cd_nonroot stays 0 — today's behavior at root
          # cwd unchanged) and NEVER arm the scope base: the tilde form
          # resolves through $HOME (mutable, reassignable in the same
          # command) and both forms through symlinks at RUNTIME, so the
          # command text cannot prove the cd lands at the root the armed
          # cert check's root-pinned pathspec resolution assumes. r3: an
          # alias cd RIDING an already-armed canonical chain additionally
          # DISARMS the widening — the r2 no-op keep left cd_root_seen=1,
          # so a canonical-cd-prefixed alias chain scoped pathspecs at the
          # root while the shell's real resolution base was the alias's
          # runtime destination (verified false-allow, reconciler r2). A
          # later canonical arming cd re-establishes the base (its arm
          # clears cd_base_disarmed). cd_nonroot stays untouched — the
          # disarm gates ONLY the #2357 widening, never the legacy
          # root-cwd path. DELIBERATE: also fires on an alias cd AFTER the
          # final commit (fail-closed over-tightening — see the r4 mover
          # disarm note; do not "fix" as a bug).
          if [ "$cd_root_seen" = 1 ]; then
            cd_root_seen=0 cd_base_disarmed=1
          fi
          ;;
        *)
          cd_nonroot=1
          # r3 belt: any OTHER non-canonical cd riding an armed chain also
          # disarms the widening. cd_nonroot=1 already kills the Layer-2
          # gate globally for these spellings; the disarm keeps the
          # cd_root_base predicate locally sound (the LAST cwd-moving cd
          # before every commit must be the canonical absolute root)
          # without leaning on that distant AND-term. DELIBERATE: also
          # fires on a cd AFTER the final commit (fail-closed
          # over-tightening — see the r4 mover disarm note).
          if [ "$cd_root_seen" = 1 ]; then
            cd_root_seen=0 cd_base_disarmed=1
          fi
          ;;
      esac
      continue
    fi

    # #2357 r4 (concerns non-cd-cwd-mover-rides-armed-chain /
    # unmodeled-cwd-mover-survives-scope-base): PER-RECORD conservative
    # invalidation for cwd/repository movers NOT spelled as a plain-lead
    # `cd`. The r3 disarm arms above fire only on records the `^cd`
    # dispatch recognizes; a successful cwd-moving construct spelled any
    # other way (pushd/popd, prefixed/escaped/quoted builtin-cd spellings,
    # eval, source/., a short `-C <dir>` wrapper) moved the real
    # pathspec-resolution base while cd_root_seen stayed 1 — the executed
    # r3 false-allow class. Trigger set = the lead-anchored mover family
    # (CWD_MOVER_LEAD_ERE — covers the `.`-source lead the screen ERE
    # deliberately cannot spell) UNION the hook's own retarget vocabulary
    # (WT_CWD_ALLOW_SCREEN_ERE), both grepped on the RAW record text —
    # quoted/escaped mover spellings are filler on the masked copy.
    # Deliberately conservative, fail closed for the widening only: a
    # commit message that merely MENTIONS mover vocabulary also disarms
    # (scoping degrades to origin/main's whole-index read; sibling
    # precedent: scripts/guard_repo_root_branch.sh's cd/pushd/popd sticky
    # scope invalidation). Armed-only (cd_root_seen=1): a mover BEFORE the
    # arming cd never disarms (the canonical cd establishes the base in
    # its own arm), legacy root-cwd behavior is untouched (cd_nonroot is
    # NOT set here), and a later canonical arming cd re-establishes the
    # base. DELIBERATE over-tightening (r3 Minor, kept): the disarm also
    # fires on a mover AFTER the final commit clause — converting some
    # sound `git commit -- p; pushd x` chains from scoped-allow to
    # whole-index — because narrowing it to pre-commit movers would trade
    # the D5 end-state belt (cd_root_seen=1 && cd_base_disarmed=0) for
    # per-commit snapshots, an allow-direction risk; do not "fix" this as
    # a bug. (r5, reconcile v4) The mover-family grep runs on BOTH lead
    # copies: the RAW lead catches quoted/escaped MOVER words (which are
    # filler on the masked copy), and the MASKED lead catches QUOTED
    # assignment VALUES in the r5 prefix group — a quoted value's
    # interior (spaces included) is spaceless filler on the masked copy,
    # so the NAME=value prefix completes there while the raw text hides
    # it; (r6 #2371) the masked-lead arm covers quoted APPEND-assignment
    # values (NAME+='a b' . x) identically. The union only ADDS disarm
    # triggers, and a disarm is block-direction END TO END as of r7 (#2371
    # rev 2): the scope=0 fallback it routes into reads the worktree
    # superset for pathspec-form commits and binds landing content (see
    # the whole-index fallback + landing_sha) — pre-r7 that fallback read
    # staged names only, so a recognized disarm could PERMIT an
    # uncertified worktree edit landing via a cwd-independent pathspec
    # (concern disarm-fallback-drops-worktree-pathspecs). NAMED
    # RESIDUAL (fail-open for the widening, at-or-below origin/main
    # behavior everywhere else): mover spellings the text cannot
    # recognize — mid-word-quoted forms (`pu'sh'd`), a mover held in a
    # variable ($X where X=pushd), a mover reached through a PREDEFINED
    # shell function or alias (the invoked name carries no mover text —
    # accepted text-unprovable indirection), and an assignment prefix
    # whose VALUE the masker cannot flatten (a bare command-substitution
    # value with internal whitespace, an array-form value) — ride an
    # armed chain undetected; compound/subshell contexts stay refused by
    # D4 regardless.
    if [ "$cd_root_seen" = 1 ]; then
      if printf '%s' "$raw_lead" | grep -qE -e "$CWD_MOVER_LEAD_ERE" \
        || printf '%s' "$lead" | grep -qE -e "$CWD_MOVER_LEAD_ERE" \
        || printf '%s' "$raw" | grep -qE -e "$WT_CWD_ALLOW_SCREEN_ERE"; then
        cd_root_seen=0 cd_base_disarmed=1
      fi
    fi

    # `git -C <path>` waiver — LEAD-ANCHORED (round-3 fix): detection on the
    # MASKED lead via DASHC_LEAD_ERE, so a message-string `git -C` mention
    # never waives the clause; waive UNLESS the -C target token literally
    # spells the repo root (one notch stronger than the siblings' path-blind
    # waiver; `git -C .` stays waived — pinned sibling-parity residual). A
    # masked (quoted) target recovers its raw token from the FIRST
    # `git -C <word>` in the RAW lead; an unextractable target REFUSES the
    # waiver (falls through to classification — block direction).
    if printf '%s' "$lead" | grep -qE "$DASHC_LEAD_ERE"; then
      cpfx=$(printf '%s' "$lead" | grep -oE "$DASHC_LEAD_ERE" | head -n1)
      ctgt=""
      if [ -n "$cpfx" ]; then
        mtgt=$(printf '%s' "${lead#"$cpfx"}" | awk '{print $1}')
        case "$mtgt" in
          *"$FILL"*)
            ctgt=$(printf '%s' "$raw_lead" \
              | grep -oE 'git[[:space:]]+-C[[:space:]]+[^[:space:]]+' | head -n1 \
              | sed -E 's/^git[[:space:]]+-C[[:space:]]+//')
            ;;
          *) ctgt=$mtgt ;;
        esac
        ctgt=$(printf '%s' "$ctgt" | sed -E "s/^[\"']//; s/[\"']\$//")
      fi
      # Waiver REFUSAL arms set retarget_evidence (#2066): a root-spelling
      # -C is a provable root retarget; an unextractable target is
      # unprovable — both disable the worktree-cwd allow gate (fail closed).
      case "$ctgt" in
        "$REPO" | "$REPO"/) retarget_evidence=1 ;; # root spelling: waiver REFUSED, classify below
        '~/explore-persona-space' | '~/explore-persona-space/') retarget_evidence=1 ;;
        '$HOME/explore-persona-space' | '$HOME/explore-persona-space/') retarget_evidence=1 ;;
        '') retarget_evidence=1 ;; # unextractable target: waiver REFUSED (fail toward classification)
        *) continue ;; # worktree / other-repo / `.` target: waived
      esac
    fi

    # Verb classification (command-position anchored, on the MASKED lead —
    # a quoted env-assignment value with spaces/separators no longer breaks
    # the wrapper-prefix match).
    verb=""
    if printf '%s' "$lead" | grep -qE "$COMMIT_CMD_ERE"; then
      verb=commit
    elif printf '%s' "$lead" | grep -qE "$ADD_CMD_ERE"; then
      verb=add
    fi
    [ -n "$verb" ] || continue
    [ "$latched" = 1 ] && continue

    if [ "$verb" = commit ]; then
      root_commit=1
      # #2357 (D3b) base-order + chain-dominance bits: a commit clause
      # scanned BEFORE any armed root-cd executes against the (unproven)
      # hook cwd; a commit whose chain back to the armed cd carries ANY
      # non-AND separator can run even when the cd was skipped or failed
      # (r1 MF-1). A chain break AFTER a commit record does not
      # retroactively uncover it — later separators cannot change where an
      # earlier commit ran.
      if [ "$cd_root_seen" = 0 ]; then
        commit_before_root_cd=1
      elif [ "$cd_and_chain" = 0 ]; then
        commit_off_chain=1
      fi
    fi

    # Token scan (noglob: a literal `scripts/*.py` token must never expand
    # against the hook's cwd). Runs on the MASKED clause: string-literal
    # content is \001 filler, so message text can contribute neither a
    # pathspec nor a `-a`-shaped flag (round-2 class); genuine unquoted
    # tokens survive verbatim. Quote-adjacent and mask-bearing tokens are
    # excluded (plan §4.2: ^(scripts|src|tests)/[^[:space:]"']+).
    local tok
    set -f
    # shellcheck disable=SC2086
    for tok in $masked; do
      case "$verb:$tok" in
        # The blanket-add latch (add:-A | add:--all | add:.) moved to the
        # dedicated per-ADD-clause post-scan below (issue #1977): same
        # trigger set, same latch — deferred only so the sanctioned
        # path-limited `git add -A|--all -- <pathspec>` shape can exempt.
        commit:--all) has_dash_a=1 ;;
        commit:--*) : ;;
        commit:-[a-zA-Z]*)
          case "$tok" in *a*) has_dash_a=1 ;; esac
          ;;
      esac
      case "$tok" in
        scripts/* | src/* | tests/*)
          case "$tok" in
            *[\"\']* | *"$FILL"*) : ;; # quote-/mask-bearing token: not a pathspec
            *) text_paths="$text_paths
$tok" ;;
          esac
          ;;
      esac
    done
    set +f

    # SECOND, dedicated token pass per COMMIT clause (issue #1620): collect
    # per-clause pathspec candidates + the scope-eligibility bits for the
    # Layer-2 scoped staged read. The scan above stays byte-identical; this
    # pass only ever NARROWS via the Layer-2 gate (any ambiguity => the bits
    # force the whole-index fallback, block direction).
    if [ "$verb" = commit ]; then
      local after_ddash=0 skip_next=0 saw_verb=0 n_cand=0 clause_opaque=0
      local pd_masked=0 pd_skip=0 rawtail rtok nrec
      local -a pd_toks=()
      set -f
      # shellcheck disable=SC2086
      for tok in $masked; do
        if [ "$saw_verb" = 0 ]; then
          # Pre-verb cwd-changing wrapper (env --chdir=DIR git commit ...)
          # moves the pathspec-resolution base: never scope. r4 (#2357,
          # concern unmodeled-cwd-mover-survives-scope-base): the SHORT
          # per-invocation directory-flag spellings get the same treatment
          # — `-C <dir>` / `-C<dir>` (git's dir wrapper; also env's -C) and
          # a cluster ending in C (`-iC`). Parity with --chdir: never scope
          # regardless of target, the exact-root spelling included (its
          # waiver is REFUSED above, so it is the one -C form that reaches
          # this scan; forcing the whole-index read there is fail closed).
          case "$tok" in --chdir* | -C* | -[A-Za-z]*C) scope_unsafe=1 ;; esac
          [ "$tok" = commit ] && saw_verb=1
          continue
        fi
        if [ "$after_ddash" = 1 ]; then
          if [ "$pd_skip" = 1 ]; then
            pd_skip=0
            continue
          fi
          # Heredoc/here-string opener interception (issue #2046), post-`--`
          # twin of the positional arm below; runs BEFORE the #1928 redirect
          # interception (redirect_tok_kind echoes `no` for the whole opener
          # family by pinned design, so the order is semantically free — the
          # opener family simply never reaches it). NOTE: the rawtail-parity
          # recovery below still counts opener tokens in $raw, so a masked
          # (quoted) pathspec + heredoc fails parity and stays opaque
          # (accepted fail-closed residual; see the known-limitations
          # header).
          case "$(heredoc_tok_kind "$tok")" in
            pair)
              pd_skip=1
              continue
              ;;
            self) continue ;;
          esac
          # Redirect interception (issue #1928), post-`--` twin of the
          # positional arm below. NOTE: the rawtail-parity recovery below
          # still counts redirect tokens in $raw, so a masked (quoted)
          # pathspec + redirect fails parity and stays opaque (accepted
          # fail-closed residual; see the known-limitations header).
          case "$(redirect_tok_kind "$tok")" in
            pair)
              pd_skip=1
              continue
              ;;
            self) continue ;;
          esac
          pd_toks+=("$tok")
          case "$tok" in *"$FILL"* | *[\"\'\\]*) pd_masked=1 ;; esac
          continue
        fi
        if [ "$skip_next" = 1 ]; then
          skip_next=0
          continue
        fi
        case "$tok" in
          --) after_ddash=1 ;;
          --include | --interactive | --patch | --all | --pathspec-from-file | --pathspec-from-file=*)
            scope_unsafe=1 ;; # these land content beyond an explicit pathspec
          -m | -F | -C | -c | -t | --message | --file | --template | --author | --date | --fixup | --squash | --cleanup | --trailer | --reuse-message | --reedit-message)
            skip_next=1 ;; # known arg-taking flag: consume its separate word
          --amend | --signoff | --no-signoff | --no-verify | --verify | --quiet | --verbose | --dry-run | --status | --no-status | --allow-empty | --allow-empty-message | --reset-author | --branch | --porcelain | --long | --short | --null | --edit | --no-edit | --only | --gpg-sign | --no-gpg-sign | --untracked-files | --*=*)
            : ;; # known no-separate-arg flag (or attached =arg): ignore
          --*) scope_unsafe=1 ;; # UNKNOWN long flag: may consume the next word — never guess
          -[a-zA-Z]*)
            case "$tok" in *a* | *i* | *p*) scope_unsafe=1 ;; esac # -a re-stage / -i include / -p patch
            case "$tok" in *m | *F | *C | *c | *t) skip_next=1 ;; esac # cluster ending in an arg-taking letter
            ;;
          *)
            # Heredoc/here-string opener interception (issue #2046), then
            # redirect interception (issue #1928): a strictly-recognized
            # opener or redirect token is shell syntax, never a pathspec —
            # drop it (self) or also consume its separate delimiter/target
            # word (pair); every other token keeps today's candidate path
            # unchanged.
            case "$(heredoc_tok_kind "$tok")" in
              pair) skip_next=1 ;;
              self) : ;;
              no)
                case "$(redirect_tok_kind "$tok")" in
                  pair) skip_next=1 ;;
                  self) : ;;
                  *) classify_candidate "$tok" ;; # positional token = candidate pathspec
                esac
                ;;
            esac
            ;;
        esac
      done
      set +f
      # Post-`--` candidates. Clean set => classify normally. A mask/quote-
      # bearing candidate => raw-after-`--` recovery (L371-377 precedent):
      # take the substring after the LAST whitespace-delimited `--` word in
      # $raw (pathspecs trail the message; no pathspec token is `--`), strip
      # ONE surrounding quote pair per token, and gate on token-count parity
      # (a spacey quoted path is ONE masked token but >=2 raw tokens).
      if [ "$after_ddash" = 1 ] && [ "${#pd_toks[@]}" -gt 0 ]; then
        if [ "$pd_masked" = 0 ]; then
          for tok in "${pd_toks[@]}"; do classify_candidate "$tok"; done
        else
          rawtail=$(printf '%s' "$raw" | awk \
            '{ for (i = NF; i >= 1; i--) if ($i == "--") { for (j = i + 1; j <= NF; j++) print $j; exit } }')
          nrec=0
          [ -n "$rawtail" ] && nrec=$(printf '%s\n' "$rawtail" | grep -c '.')
          if [ "$nrec" -ne "${#pd_toks[@]}" ]; then
            clause_opaque=1 # spacey quoted path / unparseable raw tail
          else
            while IFS= read -r rtok; do
              [ -n "$rtok" ] || continue
              rtok=$(printf '%s' "$rtok" | sed -E "s/^\"(.*)\"\$/\\1/; s/^'(.*)'\$/\\1/")
              classify_candidate "$rtok"
            done <<EOF_RAWTAIL
$rawtail
EOF_RAWTAIL
          fi
        fi
      fi
      # Clause close-out: a commit clause with zero clean candidates is BARE
      # (a `--` with nothing after it included — git commits the staged
      # index); any opaque candidate poisons scoping for the whole command.
      [ "$n_cand" -eq 0 ] && commit_bare_clause=1
      [ "$clause_opaque" = 1 ] && pathspec_opaque=1
    fi

    # SECOND, dedicated token pass per ADD clause (issue #1977): the blanket
    # latch (formerly the unconditional `add:-A | add:--all | add:.` arm in
    # the first token pass) now defers to this pass so the path-limited
    # `git add -A|--all -- <explicit pathspec>` form can exempt — its
    # pathspec BOUNDS the landing set, which Layer 2 resolves per file via a
    # cwd-gated scoped `git status` (the candidates collected here). The
    # TRIGGER set is unchanged ({-A, --all, .} as a bare token anywhere in
    # the clause — byte-parity with the removed arm); the EXEMPTION requires
    # ALL of: pre-`--` tokens drawn ONLY from the closed allowlist
    # {-A, --all} (MF-2: a pre-`--` positional — `.` included — is a live
    # pathspec the scoped read would under-enumerate, and flags like -f
    # stage ignored files `git status` cannot see), a literal `--`, and
    # >=1 clean post-`--` candidate with ZERO rejections. Anything else
    # latches add_all_chained exactly as before (FAIL CLOSED). Structural
    # template: the #1620 commit-clause pass above.
    if [ "$verb" = add ]; then
      local a_saw_verb=0 a_after_ddash=0 a_saw_blanket=0 a_eligible=1 a_ncand=0
      local a_cands=""
      set -f
      # shellcheck disable=SC2086
      for tok in $masked; do
        # #1991: recognize blanket-equivalent spellings symmetric with the L910 post-`--` rejection arm.
        # Backslash-escaped star spellings (`\*`, `\*\*`) mask via L304 to `\`+$FILL(+`\`+$FILL),
        # so pattern-match them by that masked shape (a literal `*`/`**` token never survives the
        # unquoted shell as-is — the shell would glob it — so this masked shape is the only
        # reachable form of an author-intended blanket-star spelling here).
        case "$tok" in
          -A | --all | . | ./ | .// | :/) a_saw_blanket=1 ;;
          "\\"$FILL | "\\"$FILL"\\"$FILL) a_saw_blanket=1 ;;
        esac
        if [ "$a_saw_verb" = 0 ]; then
          # Pre-verb cwd-changing wrapper (env --chdir=DIR git add ...)
          # moves the pathspec-resolution base — exemption-INELIGIBLE
          # (mirror of the commit pass's scope_unsafe arm, incl. the r4
          # short -C dir-wrapper spellings — same parity as the commit
          # pass). A pre-verb blanket-shaped token is not the sanctioned
          # shape either.
          case "$tok" in --chdir* | -C* | -[A-Za-z]*C | -A | --all | .) a_eligible=0 ;; esac
          [ "$tok" = add ] && a_saw_verb=1
          continue
        fi
        if [ "$a_after_ddash" = 1 ]; then
          # Post-`--` pathspec candidate. REJECT any token the hook cannot
          # hand to `git status` as a LITERAL, repo-relative, non-blanket
          # pathspec: masked/quoted/backslash-bearing, unexpanded shell
          # ($ / backtick / parens / redirection), pathspec magic (leading
          # `:` or `~`), absolute (leading `/`), flag-shaped (leading `-`),
          # or a blanket-equivalent spelling (`.` / `..` / `./...` / bare
          # `*`, which shell-expands against the executing cwd). Plain
          # glob-BEARING tokens (scripts/*.py) stay clean — git evaluates
          # them (classify_candidate parity).
          case "$tok" in
            *"$FILL"* | *[\"\'\\]*) a_eligible=0 ;;
            *'$'* | *'`'* | *'('* | *')'* | *'<'* | *'>'*) a_eligible=0 ;;
            '~'* | :* | /* | -* | ./* | . | .. | \*) a_eligible=0 ;;
            *)
              a_cands="$a_cands
$tok"
              a_ncand=$((a_ncand + 1))
              ;;
          esac
          continue
        fi
        case "$tok" in
          --) a_after_ddash=1 ;;
          -A | --all) : ;; # the closed pre-`--` allowlist (MF-2)
          *) a_eligible=0 ;; # positional / other flag / `.`: not the sanctioned shape
        esac
      done
      set +f
      if [ "$a_saw_blanket" = 1 ]; then
        if [ "$a_eligible" = 1 ] && [ "$a_after_ddash" = 1 ] && [ "$a_ncand" -gt 0 ]; then
          add_pathspecs="$add_pathspecs$a_cands"
        else
          add_all_chained=1 # today's blanket latch (incl. the no-`--` form)
        fi
      fi
    fi
  done

  # #2357 (D5) post-loop verdict: the provably-root-cd scope base. Armed only
  # when an exact-root cd armed (D2), NO commit clause was scanned before it,
  # NO commit's chain back to it carries a non-AND separator (chain
  # dominance: commit-runs => every predecessor in the chain, the cd
  # included, executed and succeeded => the pathspec-resolution base at the
  # commit is the root — a failed or skipped cd short-circuits the whole &&
  # chain, so no armed command's commit ever runs off-base), NO
  # non-canonical cd DISARMED the base after arming without a later
  # canonical re-arm (r3: the per-record disarm sites clear cd_root_seen and
  # set cd_base_disarmed; the `== 0` AND-term here is defense-in-depth over
  # that loop-side clear — the LAST cwd-moving cd before every commit must
  # be the canonical absolute root), and NO record opens a
  # compound/subshell context (D4). Consumed by the Layer-2 scope gate as
  # an OR-alternative to cwd_ok.
  cd_root_base=0
  if [ "$cd_root_seen" = 1 ] && [ "$commit_before_root_cd" = 0 ] \
    && [ "$commit_off_chain" = 0 ] && [ "$cd_base_disarmed" = 0 ] \
    && ! compound_context_present; then
    cd_root_base=1
  fi
  return 0
}

# check_certified <path> <landing-sha>: 0 iff a fresh matching v1 cert line
# exists. Malformed lines (non-numeric epoch) never match and never crash the
# arithmetic (block direction).
check_certified() {
  local p="$1" sha="$2" now tag epoch csha cpath
  now=$(date +%s)
  while IFS=' ' read -r tag epoch csha cpath; do
    case "$epoch" in '' | *[!0-9]*) continue ;; esac
    [ "$tag" = v1 ] && [ "$cpath" = "$p" ] && [ "$csha" = "$sha" ] \
      && [ $((now - epoch)) -le "$MAX_AGE" ] && return 0
  done < <(grep -F -- " $p" "$CERT" 2>/dev/null || true)
  return 1
}

# landing_sha <path>: echo the sha of the LANDING content for a gated pending
# path, per the BINDING RULE at the per-path cert loop below (worktree hash
# for -a / pathspec / add-clause shapes; staged blob sha only for a plain
# commit of the staged set). Returns 1 for a deletion-exempt path (no content
# lands — caller skips). A failed git read echoes EMPTY (check_certified then
# never matches: block direction preserved). Reads globals scope / has_dash_a
# / text_paths / GUARD_REPO at call time; factored out of the loop (#1857) so
# the first cert pass and the cert-retry re-hash pass cannot diverge.
landing_sha() {
  local p="$1" worktree_shape=0 sha=""
  # Scoped read engaged => a pathspec commit lands WORKTREE content for every
  # pending path (BINDING RULE at the loop below; issue #1620).
  [ "${scope:-0}" = 1 ] && worktree_shape=1
  # r7 (#2371 rev 2, concern disarm-fallback-drops-worktree-pathspecs): a
  # pathspec-form commit (clean OR opaque positional candidates) lands
  # WORKTREE content exactly like -a, and under scope=0 the pathspec is
  # unresolvable — bind the worktree hash for every worktree-modified
  # pending path (block direction: staged-blob binding here validated a
  # certified OLDER staged blob while a fresher uncertified worktree edit
  # was what the pathspec commit actually landed).
  if { [ "$has_dash_a" = 1 ] || [ "$commit_has_pathspec" = 1 ] || [ "$pathspec_opaque" = 1 ]; } \
    && git -C "$GUARD_REPO" diff --name-only -- "$p" 2>/dev/null | grep -qxF -- "$p"; then
    worktree_shape=1 # -a / pathspec-form commit lands worktree content
  fi
  if printf '%s\n' "$text_paths" | grep -qxF -- "$p"; then
    worktree_shape=1 # commit pathspec / chained add-clause
  fi
  if [ "$worktree_shape" = 1 ]; then
    [ -f "$GUARD_REPO/$p" ] || return 1 # deletion via -a/pathspec: exempt
    sha=$(git -C "$GUARD_REPO" hash-object -- "$GUARD_REPO/$p" 2>/dev/null || true)
  elif git -C "$GUARD_REPO" diff --cached --name-only -- "$p" 2>/dev/null | grep -qxF -- "$p"; then
    sha=$(git -C "$GUARD_REPO" ls-files -s -- "$p" 2>/dev/null | awk '{print $2}')
    [ -n "$sha" ] || return 1 # staged DELETION: exempt
  else
    [ -f "$GUARD_REPO/$p" ] || return 1
    sha=$(git -C "$GUARD_REPO" hash-object -- "$GUARD_REPO/$p" 2>/dev/null || true)
  fi
  printf '%s' "$sha"
}

# landing_certified <path>: single certification decision for a gated pending
# path, shared by the first cert pass AND the #1857 cert-retry pass (the same
# no-divergence factoring rationale as landing_sha). rc=0 iff EVERY blob an
# executable commit clause of this record can land for <path> is certified;
# rc=1 uncertified (block); rc=2 exempt (no clause lands content).
# r8 (#2371 r3, concern cross-clause-pathspec-evidence-authorizes-bare-blob):
# the evidence flags are command-global, so a HETEROGENEOUS record — a BARE
# commit clause chained with a clause carrying -a / pathspec evidence — can
# land TWO different blobs for one path: the bare clause commits the STAGED
# blob while the sibling clause's evidence binds landing_sha to WORKTREE
# content. Certifying either blob alone authorizes the other's landing (the
# r2 guard validated a certified worktree hash while the bare clause landed
# a different, uncertified staged blob), so on such records BOTH blobs must
# be certified — requiring more certs only over-tightens (block direction);
# a fully-certified pair still permits. The staged-side requirement keys on
# the SAME evidence disjunction as the landing_sha / cert_diag / fallback
# sites conjoined with commit_bare_clause; the text_paths (chained-add)
# route is deliberately NOT part of the key — an add clause re-stages
# worktree content BEFORE the commit, so the stale staged blob cannot land
# through it (commit-before-add orderings of that shape are a pre-existing
# class outside this record family, like the trunk -a analogue was).
landing_certified() {
  local p="$1" sha="" bare_sha="" primary_exempt=0 staged_names ls_out
  sha=$(landing_sha "$p") || primary_exempt=1
  if [ "$commit_bare_clause" = 1 ] \
    && { [ "$has_dash_a" = 1 ] || [ "$commit_has_pathspec" = 1 ] \
      || [ "$pathspec_opaque" = 1 ]; }; then
    # These two reads are ROUND-INTRODUCED and load-bearing, so they fail
    # CLOSED on a git error (rc=1 blocks) — a failed read must never
    # silently degrade the dual requirement back to the r2 single-binding
    # permit (the failure-collapse class the r2 reconcile downgraded was
    # downgradable ONLY because those reads pre-existed on trunk).
    if ! staged_names=$(git -C "$GUARD_REPO" diff --cached --name-only -- "$p" 2>/dev/null); then
      return 1 # failed staged-set read on a heterogeneous record
    fi
    if printf '%s\n' "$staged_names" | grep -qxF -- "$p"; then
      if ! ls_out=$(git -C "$GUARD_REPO" ls-files -s -- "$p" 2>/dev/null); then
        return 1 # failed index read on a heterogeneous record
      fi
      # The bare clause lands the STAGED blob for a staged-modified path.
      # An empty second field here is provably a staged DELETION (p is in
      # the staged set yet absent from the index): the bare clause lands
      # no content and only the primary binding's requirement remains.
      bare_sha=$(printf '%s' "$ls_out" | awk '{print $2}')
    fi
  fi
  if [ "$primary_exempt" = 1 ] && [ -z "$bare_sha" ]; then
    return 2 # deletion-exempt on every clause shape: no content lands
  fi
  if [ "$primary_exempt" = 0 ]; then
    # A failed git read inside landing_sha echoes EMPTY; check_certified
    # never matches an empty want, so the block direction is preserved.
    check_certified "$p" "$sha" || return 1
  fi
  if [ -n "$bare_sha" ] && [ "$bare_sha" != "$sha" ]; then
    check_certified "$p" "$bare_sha" || return 1
  fi
  return 0
}

# cert_diag <path>: one stable grep-able diagnostic line per uncertified path
# (issue #1620 fix (c)); called ONLY in the block path (zero hot-path cost).
# Format:
#   cert-diag: <path> binding=<staged|worktree> want=<sha12|EMPTY>
#     staged=<sha12|-> worktree=<sha12|-> cert=<none-for-path |
#     sha-mismatch:<csha12>,age:<s>s | stale:<age>s>max_age:<MAX_AGE>s | ok>
#     cert-file:<bytes>B,mtime:<epoch>
# `none-for-path` is the lost-append/race signature; `sha-mismatch` = content
# drifted since certification; `stale` = matching sha past MAX_AGE;
# `want=EMPTY` exposes a failed git hash-object/ls-files read; `ok` would
# contradict the block UNLESS the line carries the r8 heterogeneous-record
# suffix ` bare-staged-cert=<ok|uncertified>` — on a bare+pathspec/-a record
# with divergent blobs (landing_certified's dual requirement) the block can
# come from the bare clause's staged blob while the worktree leg reads ok.
# Mirrors the per-path loop's binding + landing-sha computation; reads
# globals scope / has_dash_a / text_paths / GUARD_REPO / CERT / MAX_AGE at
# call time.
cert_diag() {
  local p="$1" binding want stg wt now tag epoch csha cpath state age
  local best_epoch="" best_sha="" certbytes="0" certmtime="-" hetero_suffix=""
  stg=$(git -C "$GUARD_REPO" ls-files -s -- "$p" 2>/dev/null | awk '{print $2}')
  [ -n "$stg" ] || stg="-"
  wt=""
  if [ -f "$GUARD_REPO/$p" ]; then
    wt=$(git -C "$GUARD_REPO" hash-object -- "$GUARD_REPO/$p" 2>/dev/null || true)
  fi
  [ -n "$wt" ] || wt="-"
  binding=staged
  [ "${scope:-0}" = 1 ] && binding=worktree
  if { [ "$has_dash_a" = 1 ] || [ "$commit_has_pathspec" = 1 ] || [ "$pathspec_opaque" = 1 ]; } \
    && git -C "$GUARD_REPO" diff --name-only -- "$p" 2>/dev/null | grep -qxF -- "$p"; then
    binding=worktree # mirrors landing_sha's r7 pathspec-form evidence key
  fi
  if printf '%s\n' "$text_paths" | grep -qxF -- "$p"; then
    binding=worktree
  fi
  if [ "$binding" = worktree ]; then
    want="$wt"
  elif git -C "$GUARD_REPO" diff --cached --name-only -- "$p" 2>/dev/null | grep -qxF -- "$p"; then
    want="$stg"
  else
    want="$wt"
  fi
  case "$want" in '' | '-') want=EMPTY ;; esac
  now=$(date +%s)
  state="none-for-path"
  while IFS=' ' read -r tag epoch csha cpath; do
    case "$epoch" in '' | *[!0-9]*) continue ;; esac
    [ "$tag" = v1 ] && [ "$cpath" = "$p" ] || continue
    if [ -z "$best_epoch" ] || [ "$epoch" -gt "$best_epoch" ]; then
      best_epoch=$epoch best_sha=$csha
    fi
  done < <(grep -F -- " $p" "$CERT" 2>/dev/null || true)
  if [ -n "$best_epoch" ]; then
    age=$((now - best_epoch))
    if [ "$best_sha" != "$want" ]; then
      state="sha-mismatch:$(printf '%.12s' "$best_sha"),age:${age}s"
    elif [ "$age" -gt "$MAX_AGE" ]; then
      state="stale:${age}s>max_age:${MAX_AGE}s"
    else
      state="ok"
    fi
  fi
  if [ -f "$CERT" ]; then
    certbytes=$(wc -c < "$CERT" 2>/dev/null | tr -d ' ' || echo '?')
    certmtime=$(stat -c %Y "$CERT" 2>/dev/null || echo '?')
  fi
  # r8 heterogeneous-record suffix: mirrors landing_certified's bare-clause
  # staged-blob requirement so a dual-requirement block stays legible even
  # when the primary (worktree) leg's cert state reads ok.
  hetero_suffix=""
  if [ "$commit_bare_clause" = 1 ] \
    && { [ "$has_dash_a" = 1 ] || [ "$commit_has_pathspec" = 1 ] \
      || [ "$pathspec_opaque" = 1 ]; } \
    && [ "$stg" != "-" ] && [ "$stg" != "$want" ] \
    && git -C "$GUARD_REPO" diff --cached --name-only -- "$p" 2>/dev/null | grep -qxF -- "$p"; then
    if check_certified "$p" "$stg"; then
      hetero_suffix=" bare-staged-cert=ok"
    else
      hetero_suffix=" bare-staged-cert=uncertified"
    fi
  fi
  printf 'cert-diag: %s binding=%s want=%.12s staged=%.12s worktree=%.12s cert=%s%s cert-file:%sB,mtime:%s\n' \
    "$p" "$binding" "$want" "$stg" "$wt" "$state" "$hetero_suffix" "$certbytes" "$certmtime"
}

run_self_test() {
  local SCRIPT FAILED=0 TMP RART RCODE CERTF STAGED_SHA
  SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' RETURN
  # Blocked self-test cases hit the #1857 cert-retry pass; zero its settle
  # delay so the self-test stays fast + wall-clock-independent.
  export EPM_CERT_REHASH_DELAY_S=0

  # Repo with artifact-only staged payload.
  RART="$TMP/art" && git init -q "$RART"
  mkdir -p "$RART/tasks" "$RART/figures"
  echo note > "$RART/tasks/t.md" && echo png > "$RART/figures/f.png"
  git -C "$RART" add tasks/t.md figures/f.png

  # Repo with a gated scripts/ file staged (plus one untracked gated file).
  RCODE="$TMP/code" && git init -q "$RCODE"
  mkdir -p "$RCODE/scripts"
  printf 'print(1)\n' > "$RCODE/scripts/issue9_fig.py"
  git -C "$RCODE" add scripts/issue9_fig.py
  printf 'print(2)\n' > "$RCODE/scripts/issue9_new.py" # untracked (compound-add case)
  STAGED_SHA=$(git -C "$RCODE" ls-files -s -- scripts/issue9_fig.py | awk '{print $2}')

  # Repo with a tracked gated file MODIFIED in the worktree, nothing staged:
  # only the commit-clause pathspec / post-message -a can carry the payload
  # (B15/B15b — the #-in-message token-loss regression, round-2 Major).
  local RMOD
  RMOD="$TMP/mod" && git init -q "$RMOD"
  mkdir -p "$RMOD/scripts"
  printf 'print(1)\n' > "$RMOD/scripts/issue9_fig.py"
  git -C "$RMOD" add scripts/issue9_fig.py
  git -C "$RMOD" -c user.email=t@t -c user.name=t commit -q -m init
  printf 'print(2)\n' > "$RMOD/scripts/issue9_fig.py" # modified, UNSTAGED

  # Repo with a FOREIGN uncertified gated file staged + an artifact staged
  # (pathspec-scoping cases B18/B19, issue #1620).
  local RFOR
  RFOR="$TMP/foreign" && git init -q "$RFOR"
  mkdir -p "$RFOR/scripts" "$RFOR/tasks"
  printf 'print(0)\n' > "$RFOR/scripts/foreign.py"
  echo note > "$RFOR/tasks/t.md"
  git -C "$RFOR" add scripts/foreign.py tasks/t.md
  # #2357 F-walk payload: a second staged uncertified gated file whose
  # BASENAME resolves cwd-relatively from the scripts/ subdir (B49/B50,
  # B53/B54).
  # No-flip: scoped allow rows ignore foreign staged files by construction,
  # and every whole-index block row already had a gated foreign file staged.
  printf 'print(1)\n' > "$RFOR/scripts/own.py"
  git -C "$RFOR" add scripts/own.py

  # Root repo + linked WORKTREE (issue #2066 worktree-cwd allow gate, cases
  # W1-W16): `git worktree add` needs a commit to branch from, so an
  # artifact-only init commit precedes the UNCERTIFIED gated staging at the
  # "root". A separate unrelated repo covers the W11 common-dir-mismatch leg.
  local RWTROOT RWT RUNREL
  RWTROOT="$TMP/wtroot" && git init -q "$RWTROOT"
  mkdir -p "$RWTROOT/scripts" "$RWTROOT/tasks"
  echo note > "$RWTROOT/tasks/t.md"
  git -C "$RWTROOT" add tasks/t.md
  git -C "$RWTROOT" -c user.email=t@t -c user.name=t commit -q -m init
  printf 'print(3)\n' > "$RWTROOT/scripts/issue9_wt.py"
  git -C "$RWTROOT" add scripts/issue9_wt.py # uncertified gated staged at "root"
  RWT="$TMP/wtroot-wt"
  git -C "$RWTROOT" worktree add -q "$RWT" >/dev/null 2>&1
  RUNREL="$TMP/unrel" && git init -q "$RUNREL"

  # Message file for the -F commit-form cases (issue #1949); the hook parses
  # only the argv shape — the file content is never read.
  local MSGF
  MSGF="$TMP/msg.txt"
  printf 'msg\n' > "$MSGF"

  CERTF="$TMP/cert.txt"

  run_case() {
    # Optional 6th arg (issue #1620): the hook-input cwd, defaulting to the
    # case's repo root (so pathspec scoping can engage in self-test cases).
    # envflag values (#2066): '' = hermetic default (both escape/kill env
    # vars scrubbed), 'env' = EPM_ALLOW_ROOT_CODE_COMMIT=1 escape hatch,
    # 'nowt' = EPM_ROOT_CODE_COMMIT_DISABLE_WT_CWD_ALLOW=1 kill switch.
    local desc="$1" expect="$2" cmdstr="$3" repo="$4" envflag="${5:-}" case_cwd="${6:-$4}"
    local rc=0
    if [ "$envflag" = env ]; then
      jq -n --arg c "$cmdstr" --arg d "$case_cwd" '{tool_input: {command: $c}, cwd: $d}' \
        | env -u EPM_ROOT_CODE_COMMIT_DISABLE_WT_CWD_ALLOW EPM_ALLOW_ROOT_CODE_COMMIT=1 \
          EPM_ROOT_CODE_COMMIT_REPO="$repo" \
          EPM_INLINE_CERT_PATH="$CERTF" bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
    elif [ "$envflag" = nowt ]; then
      jq -n --arg c "$cmdstr" --arg d "$case_cwd" '{tool_input: {command: $c}, cwd: $d}' \
        | env -u EPM_ALLOW_ROOT_CODE_COMMIT EPM_ROOT_CODE_COMMIT_DISABLE_WT_CWD_ALLOW=1 \
          EPM_ROOT_CODE_COMMIT_REPO="$repo" \
          EPM_INLINE_CERT_PATH="$CERTF" bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
    else
      jq -n --arg c "$cmdstr" --arg d "$case_cwd" '{tool_input: {command: $c}, cwd: $d}' \
        | env -u EPM_ALLOW_ROOT_CODE_COMMIT -u EPM_ROOT_CODE_COMMIT_DISABLE_WT_CWD_ALLOW \
          EPM_ROOT_CODE_COMMIT_REPO="$repo" \
          EPM_INLINE_CERT_PATH="$CERTF" bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
    fi
    if [ "$rc" -eq "$expect" ]; then
      echo "PASS (exit $rc): $desc"
    else
      echo "FAIL (got exit $rc, want $expect): $desc"
      FAILED=1
    fi
  }

  # --- must ALLOW (exit 0) ---
  run_case "A1 artifact-only staged commit" 0 'git commit -m x' "$RART"
  run_case "A2 non-git command" 0 \
    'uv run python scripts/task.py post-marker 9 epm:progress --note commit' "$RART"
  run_case "A3 git non-commit (push)" 0 'git push origin main' "$RCODE"
  run_case "A4 worktree -C commit with gated staged at root" 0 \
    'git -C "$WT" commit -m x' "$RCODE"
  run_case "A5 cd-latched worktree commit" 0 \
    'cd .claude/worktrees/issue-9 && git commit -m x' "$RCODE"
  run_case "A7a inline escape hatch" 0 'EPM_ALLOW_ROOT_CODE_COMMIT=1 git commit -m x' "$RCODE"
  run_case "A7b session env escape hatch" 0 'git commit -m x' "$RCODE" env
  run_case "A9 heredoc message mentioning a commit command (artifact-only)" 0 \
    'git commit -m "$(cat <<EOF
fix: never git commit -m at the root
EOF
)"' "$RART"
  run_case "A11 non-gated code-adjacent staged" 0 'git commit -m x' "$RART"

  # --- must BLOCK (exit 2) ---
  run_case "B1 gated staged, no cert" 2 'git commit -m x' "$RCODE"
  run_case "B6 pathspec form, no cert" 2 'git commit -m x scripts/issue9_fig.py' "$RCODE"
  run_case "B7 -C spelling the repo root: waiver refused" 2 \
    "git -C $REPO commit -m x" "$RCODE"
  run_case "B8 classification failure fails CLOSED" 2 'git commit -m x' "$TMP/notarepo"
  run_case "B12 compound add+commit of an untracked gated file" 2 \
    'git add scripts/issue9_new.py && git commit -m x' "$RCODE"
  run_case "B13 blanket add -A chained: fail CLOSED" 2 \
    'git add -A && git commit -m x' "$RART"
  run_case "B15 pathspec after #-bearing message" 2 \
    'git commit -m "task #9: fix" scripts/issue9_fig.py' "$RMOD"
  run_case "B15b post-message -a after #-bearing message" 2 \
    'git commit -m "task #9: fix" -a' "$RMOD"
  run_case "A13 artifact-only commit with #-bearing message" 0 \
    'git commit -m "task #9: docs"' "$RART"
  run_case "B16 pathspec after semicolon-bearing message" 2 \
    'git commit -m "update; refactor" scripts/issue9_fig.py' "$RMOD"
  run_case "B16b pathspec after heredoc message" 2 \
    'git commit -m "$(cat <<EOF
update; refactor && more
EOF
)" scripts/issue9_fig.py' "$RMOD"
  run_case "B16c post-message -a after semicolon-bearing message" 2 \
    'git commit -m "update; refactor" -a' "$RMOD"
  run_case "B17 -C mention inside message does not waive (staged gated)" 2 \
    'git commit -m "docs: use git -C $WT commit for worktrees"' "$RCODE"
  run_case "A14 commit-then-scripts-tool compound stays allowed" 0 \
    'git commit -m x tasks/t.md && uv run python scripts/task.py post-marker 9 epm:progress --note done' "$RART"
  run_case "B18 pathspec excludes foreign staged, no cert, root cwd" 0 \
    'git commit -m x -- tasks/t.md' "$RFOR"
  run_case "B19 dir pathspec covers staged gated, no cert" 2 \
    'git commit -m x -- scripts/' "$RFOR"

  # --- redirect tokens on the commit clause (issue #1928) ---
  run_case "A21 pathspec commit + detached redirect keeps scoping (#1928)" 0 \
    'git commit -m x -- tasks/t.md > /tmp/i1928_selftest.log 2>&1' "$RFOR"
  run_case "A22 pathspec commit + fd-dup keeps scoping (#1928)" 0 \
    'git commit -m x -- tasks/t.md 2>&1' "$RFOR"
  run_case "B39 bare commit + redirect still blocks (sweep protection, #1928)" 2 \
    'git commit -m x > /tmp/i1928_selftest.log 2>&1' "$RFOR"
  run_case "B40 pathspec naming staged gated file + redirect still blocks (#1928)" 2 \
    'git commit -m x -- scripts/foreign.py > /tmp/i1928_selftest.log 2>&1' "$RFOR"

  # --- -F message-file commit form (issue #1949) ---
  run_case "A23 -F msgfile + artifact pathspec keeps scoping (#1949)" 0 \
    "git commit -F $MSGF -- tasks/t.md" "$RFOR"
  run_case "B41 bare -F msgfile commit still blocks (sweep protection, #1949)" 2 \
    "git commit -F $MSGF" "$RFOR"

  # --- here-doc / here-string openers on the commit clause (issue #2046) ---
  run_case "A24 incident composite: cd root + -F /dev/stdin + excluding pathspec + redirect + heredoc + tail (#2046)" 0 \
    "cd $RFOR
git commit -F /dev/stdin -- tasks/t.md > /tmp/i2046_selftest.log 2>&1 <<'MSG'
docs: fold interim notes
MSG
echo \"commit rc=\$?\"; git log -1 --oneline -- tasks/t.md" "$RFOR"
  run_case "A25 minimal heredoc: -F /dev/stdin + excluding pathspec (#2046)" 0 \
    "git commit -F /dev/stdin -- tasks/t.md <<'MSG'
docs: fold interim notes
MSG" "$RFOR"
  run_case "A26 cd-to-root prefix + -m + excluding pathspec (#2046)" 0 \
    "cd $RFOR
git commit -m x -- tasks/t.md" "$RFOR"
  run_case "B42 heredoc + pathspec covering the staged gated file blocks (#2046)" 2 \
    "git commit -F /dev/stdin -- scripts/foreign.py <<'MSG'
docs: fold interim notes
MSG" "$RFOR"
  run_case "B43 bare commit + heredoc still blocks (sweep protection, #2046)" 2 \
    "git commit -F /dev/stdin <<'MSG'
docs: fold interim notes
MSG" "$RFOR"

  # --- provably-root cd prefix scope base (issue #2357) ---
  # New-arm ALLOW rows run from a NON-root case_cwd ($TMP): at the default
  # root cwd the pre-#2357 cwd_ok gate already scopes these commands, so the
  # non-root cwd is the load-bearing distinction (r1 MF-5). B49/B50/B53/B54
  # run from a root SUBDIR with a staged gated relative pathspec — the
  # c11-class bypass shape the arming refusals must keep blocked. B53/B54
  # (r2, concern mutable-symbolic-root-proof): the symbolic alias root
  # spellings are matched LITERALLY against the fixed strings (independent
  # of GUARD_REPO), stay legacy non-poisoning, and must never arm.
  run_case "A27 cd-to-root + pathspec + redirect scopes from non-root cwd (#2357)" 0 \
    "cd $RFOR && git commit -m x -- tasks/t.md > /tmp/i2357_selftest.log 2>&1" "$RFOR" '' "$TMP"
  run_case "B44 cd-to-root prefix scopes past foreign staged (#2357)" 0 \
    "cd $RFOR && git commit -m x -- tasks/t.md" "$RFOR" '' "$TMP"
  run_case "B45 cd-to-root + add+commit compound scopes (#2357)" 0 \
    "cd $RFOR && git add tasks/t.md && git commit -m x -- tasks/t.md" "$RFOR" '' "$TMP"
  run_case "B46 commit BEFORE root cd stays whole-index (#2357)" 2 \
    "git commit -m x -- tasks/t.md; cd $RFOR" "$RFOR" '' "$TMP"
  run_case "B47 OR-separated root cd never arms (#2357)" 2 \
    "true || cd $RFOR && git commit -m x -- tasks/t.md" "$RFOR" '' "$TMP"
  run_case "B48 root cd INSIDE compound body never arms (#2357 r1 MF-3)" 2 \
    "while true
do
cd $RFOR && git commit -m x -- tasks/t.md
break
done" "$RFOR" '' "$TMP"
  run_case "B49 AND-separated root cd + SEQ commit never arms (#2357 r1 MF-1)" 2 \
    "false && cd $RFOR; git commit -m x -- own.py" "$RFOR" '' "$RFOR/scripts"
  run_case "B50 own-sep PIPE root cd never arms (#2357 r1 MF-4)" 2 \
    "true | cd $RFOR && git commit -m x -- own.py" "$RFOR" '' "$RFOR/scripts"
  run_case "B51 SEQ-separated root cd scopes (allow direction, #2357)" 0 \
    "true; cd $RFOR && git commit -m x -- tasks/t.md" "$RFOR" '' "$TMP"
  run_case "B52 NL-separated root cd scopes (allow direction, #2357)" 0 \
    "true
cd $RFOR && git commit -m x -- tasks/t.md" "$RFOR" '' "$TMP"
  run_case "B53 tilde-alias root cd never arms from root-subdir cwd (#2357 r2)" 2 \
    'cd ~/explore-persona-space && git commit -m x -- own.py' "$RFOR" '' "$RFOR/scripts"
  run_case "B54 home-var-alias root cd never arms from root-subdir cwd (#2357 r2)" 2 \
    'cd $HOME/explore-persona-space && git commit -m x -- own.py' "$RFOR" '' "$RFOR/scripts"
  # r3 (concern mutable-symbolic-root-proof, riding case): a non-canonical
  # cd RIDING an armed canonical chain disarms the widening — the LAST
  # cwd-moving cd before the commit must be the canonical absolute root.
  run_case "B55 tilde-alias cd riding armed canonical chain disarms (#2357 r3)" 2 \
    "cd $RFOR && cd ~/explore-persona-space && git commit -m x -- own.py" \
    "$RFOR" '' "$RFOR/scripts"
  run_case "B56 home-var alias cd riding armed canonical chain disarms (#2357 r3)" 2 \
    "cd $RFOR && cd \$HOME/explore-persona-space && git commit -m x -- own.py" \
    "$RFOR" '' "$RFOR/scripts"
  run_case "B57 canonical-prefixed SUBDIR cd riding armed chain disarms (#2357 r3)" 2 \
    "cd $RFOR && cd $RFOR/scripts && git commit -m x -- own.py" \
    "$RFOR" '' "$RFOR/scripts"
  run_case "B58 alias-then-canonical SEQ re-arm still scopes (allow direction, #2357 r3)" 0 \
    "cd ~/explore-persona-space; cd $RFOR && git commit -m x -- tasks/t.md" \
    "$RFOR" '' "$TMP"
  run_case "B59 disarm-then-canonical-re-arm still scopes (allow direction, #2357 r3)" 0 \
    "cd $RFOR && cd ~/explore-persona-space; cd $RFOR && git commit -m x -- tasks/t.md" \
    "$RFOR" '' "$TMP"
  # r4 (concerns non-cd-cwd-mover-rides-armed-chain /
  # unmodeled-cwd-mover-survives-scope-base): NON-`cd`-spelled cwd movers
  # riding an armed canonical chain disarm the widening (B60-B64), the
  # short pre-verb -C dir wrapper on the commit clause refuses scoping
  # (B65 — canonical-root literal: the one -C form whose waiver is refused
  # so it reaches the commit scan), and a mover BEFORE the arming cd never
  # disarms (A28, allow direction).
  run_case "B60 pushd riding armed canonical chain disarms (#2357 r4)" 2 \
    "cd $RFOR && pushd scripts && git commit -m x -- own.py" \
    "$RFOR" '' "$RFOR/scripts"
  run_case "B61 backslash-escaped cd riding armed chain disarms (#2357 r4)" 2 \
    "cd $RFOR && \\cd scripts && git commit -m x -- own.py" \
    "$RFOR" '' "$RFOR/scripts"
  run_case "B62 command-prefixed cd riding armed chain disarms (#2357 r4)" 2 \
    "cd $RFOR && command cd scripts && git commit -m x -- own.py" \
    "$RFOR" '' "$RFOR/scripts"
  run_case "B63 eval'd cd riding armed chain disarms (#2357 r4)" 2 \
    "cd $RFOR && eval 'cd scripts' && git commit -m x -- own.py" \
    "$RFOR" '' "$RFOR/scripts"
  run_case "B64 dot-source riding armed chain disarms (#2357 r4)" 2 \
    "cd $RFOR && . scripts/env.sh && git commit -m x -- own.py" \
    "$RFOR" '' "$RFOR/scripts"
  run_case "B65 short -C root wrapper on commit refuses scoping (#2357 r4)" 2 \
    "git -C /home/thomasjiralerspong/explore-persona-space commit -m x -- tasks/t.md" \
    "$RFOR"
  run_case "A28 mover BEFORE arming cd still scopes (allow direction, #2357 r4)" 0 \
    "pushd scripts; cd $RFOR && git commit -m x -- tasks/t.md" \
    "$RFOR" '' "$TMP"
  # r5 (reconcile v4 — the same two concerns re-opened): leading legal
  # ASSIGNMENT PREFIXES before a mover record defeated the ^-anchored
  # family match, and dot-source has no whole-record-screen fallback —
  # the executed r4 false-allow (B66: origin/main=2 / r4=0 / r5=2). B67
  # pins the MASKED-lead arm (quoted assignment value — invisible on the
  # raw text); B68 pins the builtin-wrapped dot-source path (family-
  # covered since r4); B69 pins multi-assignment + a non-dot family
  # member (uniform prefix group). A29 pins armed-only (an assignment-
  # prefixed mover BEFORE the arming cd never disarms).
  run_case "B66 assignment-prefixed dot-source riding armed chain disarms (#2357 r5)" 2 \
    "cd $RFOR && FOO=bar . scripts/env.sh && git commit -m x -- own.py" \
    "$RFOR" '' "$RFOR/scripts"
  run_case "B67 quoted-value assignment-prefixed dot-source disarms (#2357 r5)" 2 \
    "cd $RFOR && FOO=\"a b\" . scripts/env.sh && git commit -m x -- own.py" \
    "$RFOR" '' "$RFOR/scripts"
  run_case "B68 builtin-wrapped dot-source riding armed chain disarms (#2357 r5)" 2 \
    "cd $RFOR && builtin . scripts/env.sh && git commit -m x -- own.py" \
    "$RFOR" '' "$RFOR/scripts"
  run_case "B69 multi-assignment-prefixed pushd riding armed chain disarms (#2357 r5)" 2 \
    "cd $RFOR && A=1 B=2 pushd scripts && git commit -m x -- own.py" \
    "$RFOR" '' "$RFOR/scripts"
  run_case "A29 assignment-prefixed mover BEFORE arming cd still scopes (#2357 r5)" 0 \
    "FOO=bar . scripts/env.sh; cd $RFOR && git commit -m x -- tasks/t.md" \
    "$RFOR" '' "$TMP"

  # --- path-limited `git add --all -- <pathspec>` exemption (issue #1977) ---
  run_case "A20 path-limited add --all with artifact pathspec" 0 \
    'git add --all -- tasks/t.md && git commit -m x' "$RART"
  run_case "B35 path-limited add --all naming an uncertified gated file" 2 \
    'git add --all -- scripts/issue9_fig.py && git commit -m x' "$RCODE"
  run_case "B36 blanket-equivalent candidate (.) keeps the latch" 2 \
    'git add --all -- . && git commit -m x' "$RART"
  run_case "B37 pre-ddash positional keeps the latch (MF-2)" 2 \
    'git add --all src -- tasks/t.md && git commit -m x' "$RART"
  run_case "B38 exempted shape from a repo SUBDIR cwd blocks (MF-1)" 2 \
    'git add --all -- tasks/t.md && git commit -m x' "$RART" '' "$RART/tasks"

  # --- cd-latch variable resolution (issue #1676) ---
  # A17-A19 allow the provable same-command-assignment shape; B20-B34 pin
  # every resolution-gate refusal arm (each degrades to today's unlatched
  # fail-closed behavior, now with block-path cd-diag lines).
  run_case "A17 var-assignment cd-latched worktree commit" 0 \
    'WT=.claude/worktrees/issue-9 && cd "$WT" && git commit -m x' "$RCODE"
  run_case "A18 SEQ-separated quoted assignment, braced var" 0 \
    'WT="/abs/.claude/worktrees/issue-9"; cd "${WT}" && git commit -m x' "$RCODE"
  run_case "A19 var + literal suffix" 0 \
    'WT=/abs/.claude/worktrees && cd "$WT/issue-9" && git commit -m x' "$RCODE"
  run_case "B20 unresolved var cd (the #1644 shape, no assignment) still blocks" 2 \
    'cd "$WT" && git commit -m x' "$RCODE"
  run_case "B21 two assignments: last-write-wins ambiguity refused" 2 \
    'WT=.claude/worktrees/issue-9; WT=$REPO; cd "$WT" && git commit -m x' "$RCODE"
  # B22 re-key (#2046, deliberate): the RHS is the GUARDED root ($RCODE here
  # — cd_latch_verdict compares against $GUARD_REPO), preserving the tested
  # property "a root-spelling RHS never latches" against the guarded root.
  run_case "B22 root-path RHS never latches (guarded root)" 2 \
    "WT=$RCODE && cd \"\$WT\" && git commit -m x" "$RCODE"
  # Companion (#2046): a NON-guard absolute RHS still LATCHES — the allow is
  # carried by the latch against this GATED-staged fixture, never by an
  # empty index.
  run_case "B22b non-guard absolute RHS still latches (allows via latch)" 0 \
    'WT=/abs/other-repo && cd "$WT" && git commit -m x' "$RCODE"
  run_case "B23 dynamic RHS (command substitution) refused" 2 \
    'WT=$(mktemp) && cd "$WT" && git commit -m x' "$RCODE"
  run_case "B23b dynamic RHS with args fails the whole-clause anchor" 2 \
    'WT=$(mktemp -d) && cd "$WT" && git commit -m x' "$RCODE"
  run_case "B24 AND-positioned (conditional) assignment refused" 2 \
    'true && WT=.claude/worktrees/issue-9; cd "$WT" && git commit -m x' "$RCODE"
  run_case "B25 subshell assignment refused" 2 \
    '(WT=.claude/worktrees/issue-9); cd "$WT" && git commit -m x' "$RCODE"
  run_case "B26 backgrounded assignment refused" 2 \
    'WT=.claude/worktrees/issue-9 & cd "$WT" && git commit -m x' "$RCODE"
  run_case "B27 latch persistence unchanged: SEQ after cd resets" 2 \
    'WT=.claude/worktrees/issue-9 && cd "$WT"; git commit -m x' "$RCODE"
  run_case "B28 multi-line conditional-body assignment refused (gate 7)" 2 \
    'if true; then
WT=.claude/worktrees/issue-9
fi
cd "$WT" && git commit -m x' "$RCODE"
  run_case "B29 multi-line function-body assignment refused (gate 7)" 2 \
    'f() {
WT=.claude/worktrees/issue-9
}
f; cd "$WT" && git commit -m x' "$RCODE"
  run_case "B30 suffix carrying a parent-directory segment refused (gate 5)" 2 \
    'WT=/abs/.claude/worktrees && cd "$WT/issue-9/../.." && git commit -m x' "$RCODE"
  run_case "B31 variable mutated between assignment and cd refused (gate 6)" 2 \
    'WT=.claude/worktrees/issue-9; unset WT; cd "$WT" && git commit -m x' "$RCODE"
  run_case "B32 env-prefix assignment on another command refused (gate 1)" 2 \
    'WT=.claude/worktrees/issue-9 true; cd "$WT" && git commit -m x' "$RCODE"
  run_case "B33 assignment AFTER the cd refused (gate 2 precedes)" 2 \
    'cd "$WT" && git commit -m x; WT=.claude/worktrees/issue-9' "$RCODE"
  run_case "B34 pipeline-tail assignment refused (gate 3, next-sep PIPE)" 2 \
    'WT=.claude/worktrees/issue-9 | true; cd "$WT" && git commit -m x' "$RCODE"

  # --- worktree-cwd allow gate (issue #2066) ---
  # W1/W2/W12 allow: a provably-worktree hook cwd with no retarget evidence
  # never reads the root index. W3-W11 pin every fail-closed refusal arm.
  run_case "W1 worktree cwd + bare commit (root has uncertified gated staged)" 0 \
    'git commit -m x' "$RWTROOT" '' "$RWT"
  run_case "W2 worktree cwd + blanket-add-chained commit" 0 \
    'git add -A && git commit -m x' "$RWTROOT" '' "$RWT"
  run_case "W3 worktree cwd + cd-to-root then commit" 2 \
    "cd $RWTROOT && git commit -m x" "$RWTROOT" '' "$RWT"
  run_case "W4 worktree cwd + -C-spelling-root commit" 2 \
    "git -C $REPO commit -m x" "$RWTROOT" '' "$RWT"
  run_case "W5 worktree cwd + unproven-cd then commit" 2 \
    'cd "$WT" && git commit -m x' "$RWTROOT" '' "$RWT"
  run_case "W6 worktree cwd + --work-tree=<root> retarget token" 2 \
    "git --work-tree=$RWTROOT commit -m x" "$RWTROOT" '' "$RWT"
  run_case "W7 root-SUBDIR cwd + bare commit (B38 semantics preserved)" 2 \
    'git commit -m x' "$RWTROOT" '' "$RWTROOT/tasks"
  run_case "W8 kill switch set + worktree cwd bare commit" 2 \
    'git commit -m x' "$RWTROOT" nowt "$RWT"
  run_case "W9 worktree cwd + GIT_DIR= env-assignment retarget prefix" 2 \
    "GIT_DIR=$RWTROOT/.git git commit -m x" "$RWTROOT" '' "$RWT"
  run_case "W10 worktree cwd + pushd-to-root chain before the commit" 2 \
    "pushd $RWTROOT && git commit -m x" "$RWTROOT" '' "$RWT"
  run_case "W11 unrelated-repo cwd (worktree proof common-dir leg fails)" 2 \
    'git commit -m x' "$RWTROOT" '' "$RUNREL"
  run_case "W12 worktree-SUBDIR cwd + bare commit (toplevel != root, common dir = root)" 0 \
    'git commit -m x' "$RWTROOT" '' "$RWT/tasks"
  # W13-W16 (round-2 code-review fix, blocker wt-allow-screen-spelling-gaps):
  # retarget-spelling variants inside the plan-§3 residual classes that the
  # round-1 screen missed — each was BLOCK on main, ALLOW under the round-1
  # gate; the widened screen restores the block.
  run_case "W13 worktree cwd + flag-intervened cd-builtin invocation" 2 \
    "command -p cd $RWTROOT && git commit -m x" "$RWTROOT" '' "$RWT"
  run_case "W14 worktree cwd + backslash-escaped cd word" 2 \
    "\\cd $RWTROOT && git commit -m x" "$RWTROOT" '' "$RWT"
  run_case "W15 worktree cwd + quoted cd word" 2 \
    "'cd' $RWTROOT && git commit -m x" "$RWTROOT" '' "$RWT"
  run_case "W16 worktree cwd + short-form env directory flag + trailing bare commit" 2 \
    "env -C $RWTROOT git commit -m x && git commit -m x" "$RWTROOT" '' "$RWT"

  # A6 fresh matching cert allows; B3 wrong-sha cert blocks.
  printf 'v1 %s %s scripts/issue9_fig.py\n' "$(date +%s)" "$STAGED_SHA" > "$CERTF"
  run_case "A6 fresh matching cert" 0 'git commit -m x' "$RCODE"
  printf 'v1 %s %s scripts/issue9_fig.py\n' "$(date +%s)" "0000000000000000000000000000000000000000" > "$CERTF"
  run_case "B3 wrong-blobsha cert" 2 'git commit -m x' "$RCODE"
  rm -f "$CERTF"

  # A16 fail-soft trio: empty command / malformed JSON / missing field.
  local rc=0
  jq -n '{tool_input: {command: ""}}' \
    | env -u EPM_ALLOW_ROOT_CODE_COMMIT bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
  [ "$rc" -eq 0 ] && echo "PASS (exit 0): A16a empty command" \
    || { echo "FAIL (got exit $rc, want 0): A16a empty command"; FAILED=1; }
  rc=0
  printf 'not-json' | env -u EPM_ALLOW_ROOT_CODE_COMMIT bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
  [ "$rc" -eq 0 ] && echo "PASS (exit 0): A16b malformed stdin JSON" \
    || { echo "FAIL (got exit $rc, want 0): A16b malformed stdin JSON"; FAILED=1; }
  rc=0
  jq -n '{tool_input: {}}' \
    | env -u EPM_ALLOW_ROOT_CODE_COMMIT bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
  [ "$rc" -eq 0 ] && echo "PASS (exit 0): A16c missing command field" \
    || { echo "FAIL (got exit $rc, want 0): A16c missing command field"; FAILED=1; }

  if [ "$FAILED" = 1 ]; then
    echo "self-test: FAIL" >&2
    return 1
  fi
  echo "self-test: PASS (all cases)"
  return 0
}

if [ "${1:-}" = "--self-test" ]; then
  run_self_test
  exit $?
fi

# Session-env escape hatch (no-fork case form — this hook runs on EVERY Bash
# call, so the common path must stay subprocess-free before the jq parse).
case "${EPM_ALLOW_ROOT_CODE_COMMIT:-}" in
  1 | true | TRUE | True | yes | YES | Yes) exit 0 ;;
esac

# Capture the payload ONCE (issue #1620): the Layer-2 cwd gate re-reads it
# for `.cwd`. The cmd extraction keeps its exact fail-soft contract (jq parse
# failure -> exit 0, A16 parity).
payload=$(cat)
cmd=$(printf '%s' "$payload" | jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0 # fail-soft (A16 parity)
[ -n "$cmd" ] || exit 0
case "$cmd" in *EPM_ALLOW_ROOT_CODE_COMMIT=1*) exit 0 ;; esac # inline escape hatch
# Cheap prefilters: both substrings must co-occur before any further work —
# ~all fleet traffic exits here.
case "$cmd" in *git*) ;; *) exit 0 ;; esac
case "$cmd" in *commit*) ;; *) exit 0 ;; esac

# ---- Layer 1 ----
classify_cmd "$cmd"
[ "$root_commit" = 1 ] || exit 0

# Shared block preamble (#2013): every BLOCK path below states that the commit
# did NOT happen. Remedy-only messages were read as advice while the commit was
# narrated as landed 16 s later (task #2013 driving incident).
NOT_LANDED_LINE="NOT LANDED: the commit did NOT happen. Until a retry succeeds, do NOT state (to the user, in a task marker, in a summary, or in a commit report) that anything was committed, pushed, or landed, and do NOT publish a link whose commit SHA you have not read back from the repo. Confirm a retry with a read-only git log -1 on the intended paths before claiming success."

# Payload cwd (issue #1620; read moved up for #2066): consumed by the
# worktree-cwd allow gate below, the path-limited-add resolution, and the
# Layer-2 pathspec-scoping gate. cwd_ok computation unchanged.
hook_cwd=$(printf '%s' "$payload" | jq -r '.cwd // empty' 2>/dev/null || true)
cwd_ok=0
if [ -n "$hook_cwd" ] \
  && [ "$(realpath -m -- "$hook_cwd" 2>/dev/null)" = "$(realpath -m -- "$GUARD_REPO" 2>/dev/null)" ]; then
  cwd_ok=1
fi

# ---- Worktree-cwd allow gate (issue #2066) ----
# When EVERY contribution to root_commit=1 came from a BARE clause — no
# retarget evidence (cd-to-root / unproven cd / refused `git -C` waiver;
# classify_cmd), and no conservative-screen retarget spelling anywhere in
# the raw command — the hook-input cwd is where every commit clause
# executes. If that cwd PROVABLY sits inside a linked worktree of the
# guarded repo (toplevel != root AND git-common-dir == root/.git), the
# commit lands in the worktree (or a cd-latched sibling worktree), never at
# the root, so the root-index read below would misfire on foreign root
# state: ALLOW (exit 0), parity with the `git -C "$WT"` waiver — worktree
# commits are gated at Step 10d, not here. Fail-closed: any screen hit,
# probe failure, empty probe output, or path mismatch keeps today's
# behavior; EPM_ROOT_CODE_COMMIT_DISABLE_WT_CWD_ALLOW=1 (kill switch)
# restores the pre-#2066 cwd-blind behavior wholesale.
if [ "$retarget_evidence" = 0 ] && [ -n "$hook_cwd" ] \
  && [ -z "${EPM_ROOT_CODE_COMMIT_DISABLE_WT_CWD_ALLOW:-}" ] \
  && ! printf '%s' "$cmd" | grep -qE -e "$WT_CWD_ALLOW_SCREEN_ERE"; then
  wt_top=$(git -C "$hook_cwd" rev-parse --show-toplevel 2>/dev/null || true)
  wt_common=$(git -C "$hook_cwd" rev-parse --git-common-dir 2>/dev/null || true)
  if [ -n "$wt_top" ] && [ -n "$wt_common" ]; then
    # A relative common-dir (git emits `.git` inside a plain repo) resolves
    # against the hook cwd, never the hook process's own cwd.
    case "$wt_common" in /*) : ;; *) wt_common="$hook_cwd/$wt_common" ;; esac
    guard_rp=$(realpath -m -- "$GUARD_REPO" 2>/dev/null || true)
    wt_top_rp=$(realpath -m -- "$wt_top" 2>/dev/null || true)
    wt_common_rp=$(realpath -m -- "$wt_common" 2>/dev/null || true)
    if [ -n "$guard_rp" ] && [ -n "$wt_top_rp" ] && [ -n "$wt_common_rp" ] \
      && [ "$wt_top_rp" != "$guard_rp" ] && [ "$wt_common_rp" = "$guard_rp/.git" ]; then
      exit 0
    fi
  fi
fi

# Blanket stage chained to a root commit: the landing set is unknowable at
# PreToolUse time -> FAIL CLOSED.
if [ "$add_all_chained" = 1 ]; then
  echo "BLOCKED: 'git add -A|.|--all' chained to a repo-root commit — the landing set cannot be classified at hook time, and blanket staging is banned at the shared root (CLAUDE.md § Concurrent repo-root committers). Stage by explicit path, or use the sanctioned path-limited form 'git add --all -- <explicit paths>' (run AT the repo root, no cd/--chdir prefix, no other flags/positionals before the '--'); run the inline payload lint gate on any scripts/src/tests payload (uv run python scripts/inline_lint_gate.py --issue <N> --payload-file <paths.txt>), then commit. Deliberate override: EPM_ALLOW_ROOT_CODE_COMMIT=1." >&2
  echo "$NOT_LANDED_LINE" >&2
  exit 2
fi

# ---- Layer 2: repo-state classification (authoritative; FAIL CLOSED) ----
# Pathspec scoping (issue #1620): a pathspec-limited `git commit -- <paths>`
# lands ONLY pathspec-matched worktree content (git-commit(1) "will ignore
# changes staged in the index ... for other paths"), so when every commit
# clause carries readable literal pathspecs AND the commit provably executes
# AT the repo root (git resolves pathspecs against the executing cwd, never
# $GUARD_REPO — MF-1), the staged/modified reads are scoped to those
# pathspecs. Every ambiguity falls back to the whole-index check (block
# direction: for a pathspec-form commit the fallback reads staged plus ALL
# worktree-modified files and binds worktree content — r7 #2371 rev 2;
# staged-only there was the disarm-fallback-drops-worktree-pathspecs
# permit). The hook_cwd / cwd_ok reads live ABOVE the worktree-cwd allow
# gate (#2066); their computation is unchanged.

# Path-limited `git add -A|--all -- <pathspec>` resolution (issue #1977): the
# Layer-1 add-clause post-scan deferred the blanket latch because the
# pathspec bounds the landing set — resolve that landing set here, per file,
# via a scoped `git status`. Pathspecs resolve against the EXECUTING cwd,
# never $GUARD_REPO (#1620 MF-1), so the resolution engages ONLY when the
# hook cwd provably IS the repo root and no in-command cd precedes the
# clause; every other outcome FAILS CLOSED with the blanket block (exit 2).
if [ -n "$add_pathspecs" ]; then
  if [ "$cwd_ok" != 1 ] || [ "$cd_nonroot" != 0 ]; then
    echo "BLOCKED: path-limited 'git add --all -- <paths>' chained to a repo-root commit, but the command does not provably execute AT the repo root — pathspecs resolve against the executing cwd, so the landing set cannot be classified at hook time; failing CLOSED (blanket staging is banned at the shared root, CLAUDE.md § Concurrent repo-root committers). Re-run from the repo root with no cd/--chdir prefix, or stage by explicit path. Deliberate override: EPM_ALLOW_ROOT_CODE_COMMIT=1." >&2
    echo "add-cwd-gate: cwd_ok=$cwd_ok cd_nonroot=$cd_nonroot hook_cwd=$(printf '%.120s' "$hook_cwd")" >&2
    echo "$NOT_LANDED_LINE" >&2
    exit 2
  fi
  mapfile -t apspecs < <(printf '%s\n' "$add_pathspecs" | grep -v '^$')
  if add_status=$(git -C "$GUARD_REPO" status --porcelain=v1 --untracked-files=all -- "${apspecs[@]}" 2>/dev/null); then
    case "$add_status" in
      *'"'*)
        echo "BLOCKED: path-limited 'git add --all -- <paths>' chained to a repo-root commit: the scoped status read returned a C-quoted (special-character) path — unparseable at hook time; failing CLOSED. Stage by explicit path, or override deliberately: EPM_ALLOW_ROOT_CODE_COMMIT=1." >&2
        echo "$NOT_LANDED_LINE" >&2
        exit 2
        ;;
    esac
    # `XY path` per porcelain-v1 line; the ` -> ` split applies ONLY to
    # rename/copy lines (XY contains R or C) — both sides are taken (the
    # old side is a deletion, exempted downstream by landing_sha rc=1; the
    # new side lands content). Every extracted path joins text_paths:
    # worktree binding + the pending union follow from the existing
    # machinery. An EMPTY status is a clean no-op — the add stages nothing
    # under these pathspecs, so the commit's own staged read governs.
    while IFS= read -r st_line; do
      [ -n "$st_line" ] || continue
      st_xy=${st_line:0:2}
      st_path=${st_line:3}
      case "$st_xy" in
        *R* | *C*)
          case "$st_path" in
            *' -> '*)
              text_paths="$text_paths
${st_path%% -> *}
${st_path#* -> }"
              ;;
            *)
              text_paths="$text_paths
$st_path"
              ;;
          esac
          ;;
        *)
          text_paths="$text_paths
$st_path"
          ;;
      esac
    done <<EOF_ADDSTATUS
$add_status
EOF_ADDSTATUS
  else
    echo "BLOCKED: guard_root_code_commit.sh could not resolve the path-limited add's landing set (scoped git status failed) for a repo-root commit — cannot classify the payload; failing CLOSED (#458/#1147 class). Retry, or override deliberately: EPM_ALLOW_ROOT_CODE_COMMIT=1." >&2
    echo "$NOT_LANDED_LINE" >&2
    exit 2
  fi
fi

scope=0
# #2357 (D6): cd_root_base joins cwd_ok as an OR-alternative scope BASE — it
# proves every commit clause executes AT the root (chain dominance), exactly
# the property cwd_ok proves today; every other poisoning bit stays ANDed
# unchanged. The path-limited add--all resolution gate above is deliberately
# NOT widened (root-cwd-only, #1977 — widening an EXEMPTION is a separate
# risk decision).
if { [ "$cwd_ok" = 1 ] || [ "$cd_root_base" = 1 ]; } && [ "$cd_nonroot" = 0 ] \
  && [ "$has_dash_a" = 0 ] && [ "$scope_unsafe" = 0 ] && [ "$commit_bare_clause" = 0 ] \
  && [ "$pathspec_opaque" = 0 ] && [ "$commit_has_pathspec" = 1 ] && [ -n "$commit_pathspecs" ]; then
  scope=1
fi
if [ "$scope" = 1 ]; then
  mapfile -t pspecs < <(printf '%s\n' "$commit_pathspecs" | grep -v '^$')
  # Quoted array expansion: glob tokens reach git UNEXPANDED; git evaluates
  # globs / dir pathspecs / renames natively. worktree!=HEAD implies
  # (worktree!=index) OR (index!=HEAD), so --cached UNION plain-diff covers
  # the pathspec landing set exactly (unborn-HEAD-safe, like L614 below).
  if staged=$(git -C "$GUARD_REPO" diff --cached --name-only -- "${pspecs[@]}" 2>/dev/null) \
    && mod=$(git -C "$GUARD_REPO" diff --name-only -- "${pspecs[@]}" 2>/dev/null); then
    :
  else
    scope=0 # git rejected the pathspec (bad magic / outside repo): conservative fallback
  fi
fi
if [ "$scope" = 0 ]; then
  if ! staged=$(git -C "$GUARD_REPO" diff --cached --name-only 2>/dev/null); then
    echo "BLOCKED: guard_root_code_commit.sh could not read the staged set (git diff --cached failed) for a repo-root commit — cannot classify the payload; failing CLOSED (#458/#1147 class). Retry, or override deliberately: EPM_ALLOW_ROOT_CODE_COMMIT=1." >&2
    echo "$NOT_LANDED_LINE" >&2
    exit 2
  fi
  mod=""
  # r7 (#2371 rev 2, concern disarm-fallback-drops-worktree-pathspecs): this
  # fallback read staged names only, so a record whose scope base was
  # DISARMED (recognized mover) could PERMIT an uncertified worktree edit
  # that a cwd-independent repo-top pathspec commit lands — the exact BLOCK
  # the armed scoped read enforces. A pathspec-form commit lands WORKTREE
  # content (BINDING RULE below) and its pathspec is unresolvable here
  # (unknown cwd / opaque tokens), so the fail-closed read includes EVERY
  # worktree-modified file — the conservative superset, exactly as for -a; a
  # pathspec matching fewer files only over-tightens (block direction).
  # NAMED RESIDUAL (flag-form pathspec channels: --pathspec-from-file /
  # --include / --interactive / --patch). These flags set scope_unsafe,
  # never scope, so the SCOPED read above never engages for them; whether
  # THIS widened read engages instead depends only on the clause's
  # POSITIONAL candidate stream. Argument-less spellings and the equals
  # form (--pathspec-from-file=<file>) leave no positional candidate, so
  # the read stays staged-only for them — r5-identical, armed vs disarmed
  # verdict-identical. The SEPARATE-ARGUMENT spelling
  # (--pathspec-from-file <file>) is NOT consumed by the flag arm (no
  # skip_next), so its filename argument reaches classify_candidate as a
  # positional candidate and sets commit_has_pathspec, ARMING this widened
  # read — over-tightening only (block direction: under r5 the same record
  # fell to the staged-only read, so r5→r7 moved that spelling
  # permit→block, never the reverse). Positional pathspecs riding
  # --include / --patch clauses arm it the same way.
  if [ "$has_dash_a" = 1 ] || [ "$commit_has_pathspec" = 1 ] || [ "$pathspec_opaque" = 1 ]; then
    mod=$(git -C "$GUARD_REPO" diff --name-only 2>/dev/null || true)
  fi
fi
pending=$(printf '%s\n%s\n%s\n' "$staged" "$mod" "$text_paths" \
  | grep -E "$GATED_PATH_ERE" | sort -u)
[ -n "$pending" ] || exit 0 # artifact-only / non-code commit: allow

# Cert check per gated path; deletions exempt. BINDING RULE: bind the cert to
# the LANDING content — `git commit -a` re-stages WORKTREE content of tracked
# modified files at commit time, and a pathspec commit likewise commits
# worktree content, so for any path covered by -a, named as a commit pathspec,
# or named in a chained add clause, the landing content is the WORKTREE file;
# the staged blob sha is authoritative ONLY for a plain commit of the staged
# set. r8: on a HETEROGENEOUS record (a bare commit clause chained with a
# -a / pathspec-evidence clause) BOTH the worktree AND the bare clause's
# staged blob must be certified — landing_certified holds the whole
# decision so this pass and the cert-retry pass cannot diverge (#1857).
# Space-safe iteration (while read, never for-in word-split): a gated
# path containing a space must fail toward BLOCK, never silently allow.
uncertified_nl="" # newline-joined (space-safe); the block path's space-joined form is derived below
while IFS= read -r p; do
  [ -n "$p" ] || continue
  landing_certified "$p"
  case $? in
    1) uncertified_nl="${uncertified_nl}${p}
" ;;
    2) : ;; # deletion-exempt path — skip (no clause lands content)
  esac
done <<EOF_PENDING
$pending
EOF_PENDING

[ -z "$uncertified_nl" ] && exit 0

# Cert-retry pass (#1857): ONE bounded settle-and-re-hash retry before the
# negative verdict, firing ONLY on a would-block path (the happy path never
# sleeps). A transient worktree-hash flip (concurrent writer / filesystem
# settle: the guard-time read != the cert sha, then the file settles back
# within the window — the 07-28 incident) must not block a certified commit;
# a STABLE mismatch keeps today's block verdict byte-for-byte. The retry
# re-READS the landing sha via the same landing_certified → landing_sha
# path the first pass used — it never re-BINDS to a different content
# source (the #1620 binding rule is unchanged, r8 dual requirement
# included). Delay knob: EPM_CERT_REHASH_DELAY_S (seconds, default 2; tests
# set 0 and/or PATH-shim `sleep`). `|| true`: a malformed delay makes sleep
# fail — the re-check still runs immediately, so a genuine mismatch still
# blocks (fail toward BLOCK; a failed sleep must never crash the guard into
# a non-blocking exit under a hook harness that only blocks on exit 2).
sleep "${EPM_CERT_REHASH_DELAY_S:-2}" || true
retry_uncertified_nl=""
while IFS= read -r p; do
  [ -n "$p" ] || continue
  landing_certified "$p"
  case $? in
    0) echo "cert-retry: $p recovered after re-hash (transient worktree flip)" >&2 ;;
    1) retry_uncertified_nl="${retry_uncertified_nl}${p}
" ;;
    2)
      # Deleted between passes: mirror the first pass's deletion-exempt skip
      # (no content lands for this path anymore).
      echo "cert-retry: $p exempt after re-hash (deleted between passes)" >&2
      ;;
  esac
done <<EOF_RETRY
$uncertified_nl
EOF_RETRY

[ -z "$retry_uncertified_nl" ] && exit 0

# Space-joined form the existing block path renders (rendering unchanged).
uncertified=""
while IFS= read -r p; do
  [ -n "$p" ] || continue
  uncertified="$uncertified $p"
done <<EOF_JOIN
$retry_uncertified_nl
EOF_JOIN

[ -z "${uncertified:-}" ] && exit 0

# Block-path diagnostics (issue #1620 fix (c)): one cert-diag line per
# uncertified path, interpolated right after the BLOCKED line. The same loop
# classifies each just-composed line for the #2357 (D9) stale-cert hint: a
# path is "stale-matching" iff its cert state is stale AND its want/staged/
# worktree 12-char sha fields are all equal and none is EMPTY/`-`.
diag_lines=""
all_stale_matching=1
n_uncert=0
for p in $uncertified; do
  dline=$(cert_diag "$p")
  diag_lines="$diag_lines$dline
"
  n_uncert=$((n_uncert + 1))
  case "$dline" in
    *' cert=stale:'*)
      d_want=$(printf '%s' "$dline" | sed -nE 's/.* want=([^[:space:]]+).*/\1/p')
      d_stg=$(printf '%s' "$dline" | sed -nE 's/.* staged=([^[:space:]]+).*/\1/p')
      d_wt=$(printf '%s' "$dline" | sed -nE 's/.* worktree=([^[:space:]]+).*/\1/p')
      case "$d_want" in
        '' | EMPTY | -) all_stale_matching=0 ;;
        *)
          { [ "$d_want" = "$d_stg" ] && [ "$d_want" = "$d_wt" ]; } || all_stale_matching=0
          ;;
      esac
      ;;
    *) all_stale_matching=0 ;;
  esac
done

# #2357 (D7) scope-diag: the scope-eligibility bit vector, block path only
# (zero hot-path cost; cert-diag/cd-diag precedent). The #2332-incident
# attribution took a transcript-level forensic session; this makes the next
# one a one-line read. Emitted only when scope=0 — a scoped block's
# attribution is already unambiguous.
scope_diag=""
if [ "$scope" = 0 ]; then
  scope_diag="scope-diag: scope=0 cwd_ok=$cwd_ok cd_root_base=$cd_root_base cd_root_seen=$cd_root_seen cd_base_disarmed=$cd_base_disarmed commit_before_root_cd=$commit_before_root_cd commit_off_chain=$commit_off_chain cd_nonroot=$cd_nonroot scope_unsafe=$scope_unsafe pathspec_opaque=$pathspec_opaque commit_bare_clause=$commit_bare_clause has_dash_a=$has_dash_a commit_has_pathspec=$commit_has_pathspec
"
fi

# #2357 (D9) stale-cert re-certify hint: when EVERY uncertified path is
# stale-matching (>=1 exists), the block leads with the re-certify paragraph
# BEFORE any foreign-payload attribution — the #2332 incident's block message
# attributed a stale-but-content-matching cert as a foreign uncertified
# payload, which misled remediation. A mixed set (any drifted or certless
# path) genuinely needs the full uncertified-payload remediation, so the hint
# stays silent there.
stale_para=""
if [ "$n_uncert" -ge 1 ] && [ "$all_stale_matching" = 1 ]; then
  stale_para="STALE-CERT? Every path above matches its certificate sha (want==staged==worktree)
— the cert merely exceeded max age (${MAX_AGE}s). No content drift and likely no
foreign payload: re-run the inline payload lint gate on these paths (command
above) to re-certify, then retry the commit.
"
fi

# Foreign-staged recovery paragraph (issue #1620 fix (b)): fires only on the
# whole-index (scope=0) path when >=1 uncertified path came from the shared
# staged index rather than the command line — the pathspec-limited recovery
# is named BEFORE the env-var escape hatch.
foreign=0
if [ "$scope" = 0 ]; then
  for p in $uncertified; do
    printf '%s\n' "$text_paths" | grep -qxF -- "$p" || { foreign=1; break; }
  done
fi
foreign_para=""
if [ "$foreign" = 1 ]; then
  foreign_para="FOREIGN-STAGED? Path(s) above came from the shared STAGED INDEX, not your
command line. Another session's staging? Commit ONLY your own paths from the
repo root — a pathspec-limited commit is never blocked by foreign staged
files: git commit -m \"<msg>\" -- <your paths>  (unquoted paths; run at the
repo root, or with a leading cd <absolute repo root> &&  prefix — the literal
absolute path, not a variable, with every clause up to the commit joined by
&& AND no other cwd-moving record between it and the commit: the cd to the
literal absolute root must be the LAST recognized cwd-moving record before
the commit — any later recognized cwd-moving record (a non-canonical cd,
pushd/popd, source/., eval, a -C/--chdir directory wrapper, or a mere
mention of mover vocabulary in message text) disarms the scoping (#2357) —
the guard scopes its check to the pathspec). Plain output redirections on
the commit clause are tolerated since #1928.
"
fi

# Unproven-cd diagnostics (issue #1676 fix (b), cert-diag/foreign_para
# precedent — block path only, zero hot-path cost): the blocked command
# carried >=1 cd whose target the guard could not prove, so later clauses
# were classified against the repo root. One stable grep-able cd-diag line
# per unproven target + a remediation paragraph naming the cause and the
# three provable alternatives.
cd_para=""
if [ -n "${cd_unproven:-}" ]; then
  while IFS= read -r cdline; do
    [ -n "$cdline" ] || continue
    cd_para="${cd_para}cd-diag: unproven-cd ${cdline}
"
  done <<EOF_CDUNPROVEN
$cd_unproven
EOF_CDUNPROVEN
  cd_para="${cd_para}UNPROVEN-CD? The guard could not prove where the cd target(s) above land, so
later clauses were classified against the repo root (fail closed): an unset or
externally-set variable makes a quoted cd silently no-op at the inherited cwd
(rc=0), so unproven targets are never trusted. Remediations: use
git -C \"\$WT\" ... per clause; or cd to a literal absolute worktree path; or
assign the path once in the SAME command (WT=<literal-worktree-path> &&
cd \"\$WT\" && ...), which the guard can prove.
"
fi

cat >&2 <<BLOCK_MSG
BLOCKED: repo-root commit carries UNCERTIFIED code payload:${uncertified}
${NOT_LANDED_LINE}
REMEDIATION (pick the case that matches, #2066 — details/diagnostics below):
Committing in a WORKTREE instead? Rewrite the command as
  git -C "\$WT" commit -F <msgfile> -- <paths>
(worktrees are gated at Step 10d, not here). NEVER hand-write ${CERT} (#1082 parity).
Direct-to-main code (scripts/src/tests) must pass the inline payload lint gate
first (SKILL.md Step 9a-ter § Inline payload lint gate, #1388/#1460/#1500):
  printf '%s\n' <paths> > /tmp/issue-<N>-<round-slug>-inline-payload.txt
  uv run python scripts/inline_lint_gate.py --issue <N> \\
    --payload-file /tmp/issue-<N>-<round-slug>-inline-payload.txt   # ONE background Bash (~3-8 min)
The <round-slug> makes the path ROUND-unique (e.g. r2-fu1); the bare
issue-keyed name issue-<N>-inline-payload.txt is REFUSED by the gate (#1948:
concurrent same-issue rounds clobber the shared path).
On PASS it certifies each path's exact content; re-run after any further edit.
${diag_lines}${scope_diag}${stale_para}${foreign_para}${cd_para}If your blocked command COMPOUNDED "git add ... && git commit ...", the add
never ran either — re-stage before retrying the commit (2026-07-28: a retry
without the add hit a pathspec error).
Genuinely pre-existing red on a MODIFIED payload file the gate refused, or an
emergency fleet repair: prefix the commit with EPM_ALLOW_ROOT_CODE_COMMIT=1
and record the reason in an epm:progress note.
BLOCK_MSG
exit 2
