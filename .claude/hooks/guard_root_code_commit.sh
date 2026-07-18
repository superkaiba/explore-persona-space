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
# CLAUDE.md § Concurrent repo-root committers); missing/stale/mismatched
# cert -> CLOSED (that IS the block).
#
# <!-- known limitations -->
# - CWD-BLIND (pull-guard parity): a bare `git commit` issued while the Bash
#   shell's inherited cwd is a worktree matches Layer 1, but Layer 2 reads the
#   ROOT's index — it allows unless the root simultaneously has gated files
#   staged. Remediation: `git -C "$WT" commit`.
# - Shared-index race: another session staging gated files concurrently can
#   false-block an innocent commit (rare; block direction is safe).
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
# EPM_INLINE_CERT_PATH, EPM_INLINE_CERT_MAX_AGE_S.
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

# classify_cmd <command>: Layer 1. Sets globals root_commit / has_dash_a /
# add_all_chained / text_paths (newline-separated gated-prefix tokens).
classify_cmd() {
  local cmd="$1"
  root_commit=0 has_dash_a=0 add_all_chained=0 text_paths=""

  local triplets
  triplets=$(mask_and_split "$cmd")

  local -a recs
  mapfile -t recs <<< "$triplets"

  local n=${#recs[@]} i sep masked raw lead raw_lead tgt cpfx mtgt ctgt latched=0 verb
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
    # unlatched (fail closed).
    if printf '%s' "$lead" | grep -qE '^cd([[:space:]]|$)'; then
      tgt=$(printf '%s' "$raw_lead" | sed -E 's/^cd[[:space:]]*//' | awk '{print $1}' \
        | sed -E "s/^[\"']//; s/[\"']\$//")
      case "$tgt" in
        *.claude/worktrees/*) latched=1 ;;                # a worktree IS its own tree
        "$REPO" | "$REPO"/*) latched=0 ;;                 # root or a subdir (git walks up)
        '~/explore-persona-space' | '~/explore-persona-space/'*) latched=0 ;;
        '$HOME/explore-persona-space' | '$HOME/explore-persona-space/'*) latched=0 ;;
        /* | '~' | '~/'* | '$HOME/'*) latched=1 ;;        # absolute/~-anchored, not the root
        *) latched=0 ;;                                   # relative/variable/empty: unproven
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
        add:-A | add:--all | add:.) add_all_chained=1 ;;
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

run_self_test() {
  local SCRIPT FAILED=0 TMP RART RCODE CERTF STAGED_SHA
  SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' RETURN

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

  CERTF="$TMP/cert.txt"

  run_case() {
    local desc="$1" expect="$2" cmdstr="$3" repo="$4" envflag="${5:-}"
    local rc=0
    if [ -n "$envflag" ]; then
      jq -n --arg c "$cmdstr" '{tool_input: {command: $c}}' \
        | EPM_ALLOW_ROOT_CODE_COMMIT=1 EPM_ROOT_CODE_COMMIT_REPO="$repo" \
          EPM_INLINE_CERT_PATH="$CERTF" bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
    else
      jq -n --arg c "$cmdstr" '{tool_input: {command: $c}}' \
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

cmd=$(jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0 # fail-soft (A16 parity)
[ -n "$cmd" ] || exit 0
case "$cmd" in *EPM_ALLOW_ROOT_CODE_COMMIT=1*) exit 0 ;; esac # inline escape hatch
# Cheap prefilters: both substrings must co-occur before any further work —
# ~all fleet traffic exits here.
case "$cmd" in *git*) ;; *) exit 0 ;; esac
case "$cmd" in *commit*) ;; *) exit 0 ;; esac

# ---- Layer 1 ----
classify_cmd "$cmd"
[ "$root_commit" = 1 ] || exit 0

# Blanket stage chained to a root commit: the landing set is unknowable at
# PreToolUse time -> FAIL CLOSED.
if [ "$add_all_chained" = 1 ]; then
  echo "BLOCKED: 'git add -A|.|--all' chained to a repo-root commit — the landing set cannot be classified at hook time, and blanket staging is banned at the shared root (CLAUDE.md § Concurrent repo-root committers). Stage by explicit path, run the inline payload lint gate on any scripts/src/tests payload (uv run python scripts/inline_lint_gate.py --issue <N> --payload-file <paths.txt>), then commit. Deliberate override: EPM_ALLOW_ROOT_CODE_COMMIT=1." >&2
  exit 2
fi

# ---- Layer 2: repo-state classification (authoritative; FAIL CLOSED) ----
if ! staged=$(git -C "$GUARD_REPO" diff --cached --name-only 2>/dev/null); then
  echo "BLOCKED: guard_root_code_commit.sh could not read the staged set (git diff --cached failed) for a repo-root commit — cannot classify the payload; failing CLOSED (#458/#1147 class). Retry, or override deliberately: EPM_ALLOW_ROOT_CODE_COMMIT=1." >&2
  exit 2
fi
mod=""
[ "$has_dash_a" = 1 ] && mod=$(git -C "$GUARD_REPO" diff --name-only 2>/dev/null || true)
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
uncertified=""
while IFS= read -r p; do
  [ -n "$p" ] || continue
  worktree_shape=0 # 1 => landing content is the worktree file
  if [ "$has_dash_a" = 1 ] \
    && git -C "$GUARD_REPO" diff --name-only -- "$p" 2>/dev/null | grep -qxF -- "$p"; then
    worktree_shape=1 # -a re-stages worktree content
  fi
  if printf '%s\n' "$text_paths" | grep -qxF -- "$p"; then
    worktree_shape=1 # commit pathspec / chained add-clause
  fi
  if [ "$worktree_shape" = 1 ]; then
    [ -f "$GUARD_REPO/$p" ] || continue # deletion via -a/pathspec: exempt
    sha=$(git -C "$GUARD_REPO" hash-object -- "$GUARD_REPO/$p" 2>/dev/null || true)
  elif git -C "$GUARD_REPO" diff --cached --name-only -- "$p" 2>/dev/null | grep -qxF -- "$p"; then
    sha=$(git -C "$GUARD_REPO" ls-files -s -- "$p" 2>/dev/null | awk '{print $2}')
    [ -n "$sha" ] || continue # staged DELETION: exempt
  else
    [ -f "$GUARD_REPO/$p" ] || continue
    sha=$(git -C "$GUARD_REPO" hash-object -- "$GUARD_REPO/$p" 2>/dev/null || true)
  fi
  check_certified "$p" "$sha" || uncertified="$uncertified $p"
done <<EOF_PENDING
$pending
EOF_PENDING

[ -z "${uncertified:-}" ] && exit 0
cat >&2 <<BLOCK_MSG
BLOCKED: repo-root commit carries UNCERTIFIED code payload:${uncertified}
Direct-to-main code (scripts/src/tests) must pass the inline payload lint gate
first (SKILL.md Step 9a-ter § Inline payload lint gate, #1388/#1460/#1500):
  printf '%s\n' <paths> > /tmp/issue-<N>-inline-payload.txt
  uv run python scripts/inline_lint_gate.py --issue <N> \\
    --payload-file /tmp/issue-<N>-inline-payload.txt     # ONE background Bash (~3-8 min)
On PASS it certifies each path's exact content; re-run after any further edit.
Committing in a WORKTREE instead? Use git -C "\$WT" commit (worktrees are
gated at Step 10d, not here). NEVER hand-write ${CERT} (#1082 parity).
Genuinely pre-existing red on a MODIFIED payload file the gate refused, or an
emergency fleet repair: prefix the commit with EPM_ALLOW_ROOT_CODE_COMMIT=1
and record the reason in an epm:progress note.
BLOCK_MSG
exit 2
