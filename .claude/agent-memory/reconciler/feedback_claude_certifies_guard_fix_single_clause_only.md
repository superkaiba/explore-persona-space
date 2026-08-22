---
name: claude-certifies-guard-fix-single-clause-only
description: "Guard-fix reviews keyed on command-GLOBAL evidence flags: Claude certifies block-direction analyzing only single-clause records — probe MULTI-clause records whose clauses have different landing semantics (#2371 r2)"
metadata:
  type: feedback
---

When a guard/hook fix keys landing or binding semantics on command-GLOBAL
evidence flags (accumulated once per record across ALL clauses — the #2371
shape: `commit_has_pathspec` / `pathspec_opaque` / `has_dash_a` initialized
once in `classify_cmd`), Claude's code-reviewer verifies the fix on
SINGLE-clause records, correctly classifies the homogeneous binding flip as
"correctness alignment", and certifies "block direction end to end" — while
a HETEROGENEOUS multi-clause record (bare commit clause chained with a
pathspec/-a clause) lets one clause's evidence authorize the blob binding
for a sibling clause that lands DIFFERENT content.

**Why:** #2371 r2 — the r7 fallback repair added `commit_has_pathspec` to
the worktree-binding condition; a bare clause + artifact-pathspec clause
record with certified-worktree/uncertified-staged blobs flipped
block→permit vs the round baseline (probe P1a live rc=0 vs P1b pre-r7
reconstruction rc=2; bare-only control still rc=2). Claude re-ran the r1
reconciler probe, 301 tests green, swept all 3 evidence-key sites — but
every fixture was single-clause. Codex caught it structurally
(cross-clause-pathspec-evidence-authorizes-bare-blob, upheld BLOCKER).

**How to apply:** in any guard-fix reconcile where the diff widens a
condition reading command-global parse state, enumerate the flag's
INITIALIZATION scope vs its CONSUMER scope; if flags accumulate across
clauses, execute a two-clause probe where the clauses have different
landing semantics (bare = staged blob, pathspec/-a = worktree blob) and
divergent certified/uncertified blobs. The committed revert-pin textual
reconstruction convention (c30g2-style evidence-cond swap) gives the
baseline arm. Companion calibration: the SAME round's Codex twin
over-reached on the sibling concern (git-read failure-collapse `|| true`)
— byte-identical pre-existing at baseline+trunk, failure degrades TO the
baseline behavior = no new permit → downgrade per
[[codex-hardening-beyond-minimal-port-contract]].
