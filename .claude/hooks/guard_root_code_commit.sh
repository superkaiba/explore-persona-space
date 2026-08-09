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
# - CWD-BLIND (pull-guard parity): a bare `git commit` issued while the Bash
#   shell's inherited cwd is a worktree matches Layer 1, but Layer 2 reads the
#   ROOT's index — it allows unless the root simultaneously has gated files
#   staged. Remediation: `git -C "$WT" commit`.
# - Shared-index race: another session staging gated files concurrently can
#   false-block an innocent commit (rare; block direction is safe). Since
#   #1620 a root-cwd pathspec-limited commit scopes the staged read to its
#   pathspecs (pathspec SCOPING engages only when the hook-input cwd provably
#   equals the root — the CWD-BLIND note above covers the bare-commit case);
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
#      bare `{` group openers) — the masker tracks quote/heredoc state only
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
  for ((j = 0; j + 2 < n; j += 3)); do
    m=$(printf '%s' "${recs[j + 1]}" | sed -E 's/^[[:space:]]+//')
    if printf '%s' "$m" | grep -qE \
      '^(if|then|elif|else|fi|while|until|for|do|done|case|esac|function)([[:space:]]|$)|^[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\(\)|^\{([[:space:]]|$)'; then
      resolve_reason=compound-context
      return 1
    fi
  done

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
  # Unproven-cd tracking (issue #1676 fix (b)): one entry per cd clause whose
  # FINAL verdict (after any resolution attempt) is unproven — consumed by
  # the block path's cd-diag lines; never read on the allow path.
  cd_unproven=""

  local triplets
  triplets=$(mask_and_split "$cmd")

  local -a recs
  mapfile -t recs <<< "$triplets"

  local n=${#recs[@]} i sep masked raw lead raw_lead tgt cpfx mtgt ctgt latched=0 verb
  local vname vsuffix
  for ((i = 0; i + 2 < n; i += 3)); do
    sep=${recs[i]} masked=${recs[i + 1]} raw=${recs[i + 2]}
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
      if [ "$cd_verdict" = unproven ]; then
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
        "$GUARD_REPO" | "$GUARD_REPO"/) : ;;
        '~/explore-persona-space' | '~/explore-persona-space/') : ;;
        '$HOME/explore-persona-space' | '$HOME/explore-persona-space/') : ;;
        *) cd_nonroot=1 ;;
      esac
      continue
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
      case "$ctgt" in
        "$REPO" | "$REPO"/) : ;; # root spelling: waiver REFUSED, classify below
        '~/explore-persona-space' | '~/explore-persona-space/') : ;;
        '$HOME/explore-persona-space' | '$HOME/explore-persona-space/') : ;;
        '') : ;; # unextractable target: waiver REFUSED (fail toward classification)
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
          # moves the pathspec-resolution base: never scope.
          case "$tok" in --chdir*) scope_unsafe=1 ;; esac
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
          # (mirror of the commit pass's scope_unsafe arm). A pre-verb
          # blanket-shaped token is not the sanctioned shape either.
          case "$tok" in --chdir* | -A | --all | .) a_eligible=0 ;; esac
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
  if [ "$has_dash_a" = 1 ] \
    && git -C "$GUARD_REPO" diff --name-only -- "$p" 2>/dev/null | grep -qxF -- "$p"; then
    worktree_shape=1 # -a re-stages worktree content
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
# contradict the block (itself diagnostic of a logic bug). Mirrors the
# per-path loop's binding + landing-sha computation; reads globals scope /
# has_dash_a / text_paths / GUARD_REPO / CERT / MAX_AGE at call time.
cert_diag() {
  local p="$1" binding want stg wt now tag epoch csha cpath state age
  local best_epoch="" best_sha="" certbytes="0" certmtime="-"
  stg=$(git -C "$GUARD_REPO" ls-files -s -- "$p" 2>/dev/null | awk '{print $2}')
  [ -n "$stg" ] || stg="-"
  wt=""
  if [ -f "$GUARD_REPO/$p" ]; then
    wt=$(git -C "$GUARD_REPO" hash-object -- "$GUARD_REPO/$p" 2>/dev/null || true)
  fi
  [ -n "$wt" ] || wt="-"
  binding=staged
  [ "${scope:-0}" = 1 ] && binding=worktree
  if [ "$has_dash_a" = 1 ] \
    && git -C "$GUARD_REPO" diff --name-only -- "$p" 2>/dev/null | grep -qxF -- "$p"; then
    binding=worktree
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
  printf 'cert-diag: %s binding=%s want=%.12s staged=%.12s worktree=%.12s cert=%s cert-file:%sB,mtime:%s\n' \
    "$p" "$binding" "$want" "$stg" "$wt" "$state" "$certbytes" "$certmtime"
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

  # Message file for the -F commit-form cases (issue #1949); the hook parses
  # only the argv shape — the file content is never read.
  local MSGF
  MSGF="$TMP/msg.txt"
  printf 'msg\n' > "$MSGF"

  CERTF="$TMP/cert.txt"

  run_case() {
    # Optional 6th arg (issue #1620): the hook-input cwd, defaulting to the
    # case's repo root (so pathspec scoping can engage in self-test cases).
    local desc="$1" expect="$2" cmdstr="$3" repo="$4" envflag="${5:-}" case_cwd="${6:-$4}"
    local rc=0
    if [ -n "$envflag" ]; then
      jq -n --arg c "$cmdstr" --arg d "$case_cwd" '{tool_input: {command: $c}, cwd: $d}' \
        | EPM_ALLOW_ROOT_CODE_COMMIT=1 EPM_ROOT_CODE_COMMIT_REPO="$repo" \
          EPM_INLINE_CERT_PATH="$CERTF" bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
    else
      jq -n --arg c "$cmdstr" --arg d "$case_cwd" '{tool_input: {command: $c}, cwd: $d}' \
        | env -u EPM_ALLOW_ROOT_CODE_COMMIT EPM_ROOT_CODE_COMMIT_REPO="$repo" \
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
# direction).
hook_cwd=$(printf '%s' "$payload" | jq -r '.cwd // empty' 2>/dev/null || true)
cwd_ok=0
if [ -n "$hook_cwd" ] \
  && [ "$(realpath -m -- "$hook_cwd" 2>/dev/null)" = "$(realpath -m -- "$GUARD_REPO" 2>/dev/null)" ]; then
  cwd_ok=1
fi

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
if [ "$cwd_ok" = 1 ] && [ "$cd_nonroot" = 0 ] && [ "$has_dash_a" = 0 ] \
  && [ "$scope_unsafe" = 0 ] && [ "$commit_bare_clause" = 0 ] \
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
  [ "$has_dash_a" = 1 ] && mod=$(git -C "$GUARD_REPO" diff --name-only 2>/dev/null || true)
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
# set. Space-safe iteration (while read, never for-in word-split): a gated
# path containing a space must fail toward BLOCK, never silently allow.
uncertified_nl="" # newline-joined (space-safe); the block path's space-joined form is derived below
while IFS= read -r p; do
  [ -n "$p" ] || continue
  if sha=$(landing_sha "$p"); then
    check_certified "$p" "$sha" || uncertified_nl="${uncertified_nl}${p}
"
  fi # landing_sha rc=1: deletion-exempt path — skip (no content lands)
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
# re-READS the landing sha via the same landing_sha function — it never
# re-BINDS to a different content source (the #1620 binding rule is
# unchanged). Delay knob: EPM_CERT_REHASH_DELAY_S (seconds, default 2; tests
# set 0 and/or PATH-shim `sleep`). `|| true`: a malformed delay makes sleep
# fail — the re-check still runs immediately, so a genuine mismatch still
# blocks (fail toward BLOCK; a failed sleep must never crash the guard into
# a non-blocking exit under a hook harness that only blocks on exit 2).
sleep "${EPM_CERT_REHASH_DELAY_S:-2}" || true
retry_uncertified_nl=""
while IFS= read -r p; do
  [ -n "$p" ] || continue
  if sha=$(landing_sha "$p"); then
    if check_certified "$p" "$sha"; then
      echo "cert-retry: $p recovered after re-hash (transient worktree flip)" >&2
    else
      retry_uncertified_nl="${retry_uncertified_nl}${p}
"
    fi
  else
    # Deleted between passes: mirror the first pass's deletion-exempt skip
    # (no content lands for this path anymore).
    echo "cert-retry: $p exempt after re-hash (deleted between passes)" >&2
  fi
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
# uncertified path, interpolated right after the BLOCKED line.
diag_lines=""
for p in $uncertified; do
  diag_lines="$diag_lines$(cert_diag "$p")
"
done

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
files: git commit -m \"<msg>\" -- <your paths>  (unquoted paths, run at the
repo root; the guard scopes its check to the pathspec). Plain output
redirections on the commit clause are tolerated since #1928.
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
${diag_lines}${foreign_para}${cd_para}Direct-to-main code (scripts/src/tests) must pass the inline payload lint gate
first (SKILL.md Step 9a-ter § Inline payload lint gate, #1388/#1460/#1500):
  printf '%s\n' <paths> > /tmp/issue-<N>-<round-slug>-inline-payload.txt
  uv run python scripts/inline_lint_gate.py --issue <N> \\
    --payload-file /tmp/issue-<N>-<round-slug>-inline-payload.txt   # ONE background Bash (~3-8 min)
The <round-slug> makes the path ROUND-unique (e.g. r2-fu1); the bare
issue-keyed name issue-<N>-inline-payload.txt is REFUSED by the gate (#1948:
concurrent same-issue rounds clobber the shared path).
On PASS it certifies each path's exact content; re-run after any further edit.
If your blocked command COMPOUNDED "git add ... && git commit ...", the add
never ran either — re-stage before retrying the commit (2026-07-28: a retry
without the add hit a pathspec error).
Committing in a WORKTREE instead? Use git -C "\$WT" commit (worktrees are
gated at Step 10d, not here). NEVER hand-write ${CERT} (#1082 parity).
Genuinely pre-existing red on a MODIFIED payload file the gate refused, or an
emergency fleet repair: prefix the commit with EPM_ALLOW_ROOT_CODE_COMMIT=1
and record the reason in an epm:progress note.
BLOCK_MSG
exit 2
