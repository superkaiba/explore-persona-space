---
name: Codex static PASS misses quoting/gluing spelling variants in shell token-walk guards
description: When reviewing a PreToolUse shell guard (guard_*.sh) that parses command strings by word-splitting, Codex's execution-banned static review credits the tested spellings and misses quote-wrapped / glued variants of the same operand class; live-probe spelling variants before crediting a Codex PASS.
type: feedback
---

When the artifact under review is a shell guard that PARSES untrusted command
strings via a word-split token walk (`.claude/hooks/guard_*.sh`), a Codex PASS
grounded on static reading + the committed test cases does NOT cover the
spelling space: the guard's tokens keep their quote characters (the walk does
no quote removal), while the REAL shell strips quotes before the target
program sees the operand — so any per-token `case`/prefix check silently
misses `'X'` / `"X"` / glued (`-nX`, `--flag='X'`) variants of exactly the
operand class the check gates.

**Why:** #1057 r1 — Codex PASSed the two-tier guard_log_dump relief; Claude
FAILed on quote-wrapped signed counts (`tail -n '+201'`, `tail -n'+201'`,
`head -n '-201'`, `tail --lines='+201'`): the sign check at the await_n and
`-n*/--lines=*` arms saw a leading quote, `tr -dc '0-9'` still extracted the
number → relief tier granted an unbounded read of a big code/doc file that
main blocked. My live repro confirmed all 4 forms post-fix=0 / main=2 → FAIL
upheld, plan-§3 verbatim breach ("INCLUDING the signed-count forms ... still
exit 2"). Codex had credited the unquoted pin cases and never generated the
quoted spellings; its "pre-existing parser limitations" note didn't apply
(pre-fix behavior was exit 2, so the allow was this-round-introduced).

**How to apply:** on any Claude-FAIL vs Codex-PASS split over a shell-parsing
guard, do NOT weigh the static walkthrough against the empirical claim —
EXECUTE the disputed spellings yourself via the hook's stdin JSON contract
against BOTH the branch hook and `git show main:<hook>` (fixture sized past
the guard's byte threshold). Enumerate the variant family: bare, `'X'`,
`"X"`, glued `-nX`, `--flag=X`, `--flag='X'`, doubled `''X''`, ANSI-C
`$'X'`, backslash `\X`, quoted-flag composition (`'-n' -201` — the quoted
flag falls to the file arm and the operand hits a sign-blind legacy arm).
Check which token classes the guard quote-strips (grep for `${tok#\'}`) —
an arm that strips for one class (sed scripts, file names) but not another
(count operands) is the tell; a ONE-PASS strip vs doubled quotes is a
second tell. The inverse also holds: a guard arm that ALREADY behaved
identically on main for the variant (byte-unchanged) is
pre-existing-on-trunk, not a blocker.

**#1057 r3 (direction partition on "no new blocks" clauses — Claude PASS
upheld):** when a sanctioned parser fix collaterally flips main=0→wt=2 on
spellings the fix newly parses, partition by DIRECTION and by the policy
baseline, not the parser-accident baseline: a new ALLOW of a policy-refused
read is Blocking (r1/r2 pattern), but a new BLOCK of a spelling whose
canonical UNQUOTED twin main refuses BY POLICY is a sanctioned strengthening
— main's allow of the quoted variant was a parse failure, and refusing it is
the parser working correctly. Probe the false-positive class yourself (every
twin-main-ALLOWED bounded/relief spelling must stay wt=0) before crediting;
also check whether the "narrower" fix Codex prescribes would re-allow
genuinely unbounded spellings (it did — `tail '-n' +201` back to allow),
i.e. weaker protection purchased to preserve parser accidents. A reconciler's
own prior-round residual clause ("no new block of any BOUNDED read") binds
purposively against the hook's policy baseline; state the binding
interpretation explicitly in the verdict so later rounds have no ambiguity.

**#1057 r2 (the inverse split — Claude PASS vs Codex FAIL):** on a
multi-round branch, "pre-existing" means pre-BRANCH (vs main), NOT
pre-ROUND. Claude's r2 PASS found the residual spellings (`$'+201'`,
`\+201`) in its own sweep but classified them "pre-existing round-1 parse
scope" because the ROUND-2 diff didn't touch those paths — wrong baseline:
the branch was unmerged and main refused them (main=2 → wt=0 = branch-
introduced new allow). Codex's FAIL was upheld on its verified core
(`head '-n' -201` complement 5799/6000 lines; `tail -n ''+201''` unbounded)
even though 3 of its 4 Critical example spellings were mis-attributed
(main=0 too — pre-existing). Partition every disputed spelling into
main=wt (pre-existing miss, Standing-only) vs main=2→wt=0 (new allow,
Blocking); also probe the REAL shell semantics — `head -n \+201` was
main=2→wt=0 but bounded in reality (201 lines), low severity.
