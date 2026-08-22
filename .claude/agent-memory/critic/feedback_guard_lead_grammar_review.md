---
name: guard-lead-grammar-review
description: Reviewing guard_root_code_commit.sh lead-grammar (CWD_MOVER_LEAD_ERE) extension plans — 5-min executed verification recipe (#2357 r5, #2371)
metadata:
  type: feedback
---

For any plan extending `CWD_MOVER_LEAD_ERE` (or a sibling lead-anchored grammar in
`.claude/hooks/guard_root_code_commit.sh`), do the EXECUTED verification, not a
regex read-through — all three probes run in ~5 min:

1. **Replay the proposed ERE via `grep -qE`** against (a) every crux lead the plan
   claims to newly catch, (b) the r(N−1) backward-compat leads (`FOO=bar . x`,
   bare `. x`), and (c) NEAR-MISS controls that must stay nomatch: `timeout 5 cd x`
   (word-boundary), `times`, `time git commit -m x -- p` (allow-direction: a
   time-wrapped COMMIT must not disarm), `env . x`, `NAME=1 make` (prefix alone
   never disarms — the family word is still required).
2. **Re-run the bash reachability probes yourself** (`cd sub` script + `pwd`
   read-back in /tmp): each claimed sources/does-not-source row is one line.
   #2371 facts (bash 5.1.16): `NAME+=v . x`, `time . x`, `time -p . x`,
   `time NAME=v . x`, `builtin . x`, `command . x` all SOURCE; `env . x` and
   `NAME=v time . x` do NOT (external env/time cannot run the `.` builtin —
   `time` is a keyword only at pipeline start).
3. **Masked-vs-raw arm:** simulate the masked copy by replacing a quoted value's
   interior space with `\001` filler — a quoted-value prefix must fail the RAW
   lead and match the MASKED lead (both are grepped at the disarm site, ~1119).

**Why:** the grammar is the conclusion — a wrong atom spelling flips block/permit
directly, and the executed replay is cheaper than reasoning about ERE backtracking.
**How to apply:** also check (i) the c30c-family count-assert survives (module
constant updated to the NEW group spelling, count==1, strip-whole-group still
reconstructs the r3/r4 ERE byte-for-byte — rebase-merge rewrites SHAs so textual
reconstruction, never `git show` blob pins); (ii) "N tests" claims: `def test_`
count ≠ pytest collected count (parametrization) — 208 vs 283 at #2371; (iii)
over-matching non-reachable forms (grammar matches but shell can't source) are
pinned as BLOCKED (deliberate over-tightening), never exempted. See
[[infra-plan-review-checklist]] item B (choke-point greps).
