---
name: Claude names a feature-defeating bug in "Unaddressed Cases" then excuses it
description: code-review FAIL calibration — Claude's own Unaddressed/Minor section describes the exact worst-case input that defeats the feature, then waves it off as "acceptable"; verify whether the named behavior is the disputed blocker before crediting the PASS
type: feedback
---

When the Claude code-reviewer PASSes but its own "Unaddressed Cases" /
"Style" / "Minor" section DESCRIBES a behavior that, on the feature's
worst-case input, defeats the feature's purpose — read that paragraph as a
candidate blocker, not as a disposition you can trust. Claude frequently
*notices* the bug and then mis-classes it "acceptable" in the same breath.

**Why:** #679 r1. The disk-guard active-task escalation (`#679`'s whole
point: alert on a large active `hf_dl/.../store/` the terminal-gate can't
reap) sizes the cache via `sub.bytes_freed`, which sums only `removed`. The
nested-store parity guard puts an unmirrored active cache into `skipped`
(not `removed`), so `bytes_freed == 0` → escalation suppressed on EXACTLY
the shape the feature targets. Claude's "Unaddressed Cases" wrote: "a cache
whose ONLY content is a nested-store-blocked `hf_dl` would size as 0 and
never escalate. Acceptable." That sentence IS the Codex Critical, mis-labeled
"acceptable." Codex was right; verdict FAIL.

**Two compounding tells in the same PASS, both in my standing ledger:**
1. **Test credited without tracing falsification** (cf.
   `feedback_claude_credits_test_without_tracing_falsification_path.md`,
   `feedback_claude_synthetic_fixture_smoke_masks_args_grid_bug.md`): the
   "escalation" test stubbed `clean_issue_downloads` and hand-set
   `cr.removed=[x]` / `cr.sizes_bytes={x: N>0}`, forcing `bytes_freed>0` —
   so it never exercises the real parity-skip→`skipped`→`bytes_freed==0`
   path. Claude cited "test asserts the cache survives" (deletion-safety),
   a DIFFERENT claim than escalation-firing.
2. **Spec-weakening on a DELETE path** (cf.
   `feedback_claude_misses_floor_vs_raise_divergence.md`,
   `feedback_codex_hardening_beyond_minimal_port_contract.md` inverse —
   here the hardening was REAL): plan said "per-file size match"; impl
   matched by BASENAME (threw away `RepoFile.path`, which was in hand), so an
   unrelated same-name+size HF file falsely marks generated data "mirrored"
   → `rmtree(hf_dl)` deletes it. The implementer documented the weakening in
   the docstring — the rationalizing comment is the smell.

**How to apply:** On a code-review PASS-vs-FAIL where the disputed blocker is
about a guard/alert/safety feature, (a) grep Claude's own report for the
behavior Codex flagged — if Claude NAMED it and called it acceptable, weight
toward FAIL and re-derive whether the named behavior defeats the feature on
its worst-case input; (b) for any test Claude credits, open the fixture and
check whether it STUBS the function whose interaction is the disputed bug —
a stub that hand-sets the output the bug would zero is a masked false
positive; (c) when the impl deviates from a plan's "per-file"/"per-X"
wording with a self-justifying docstring, check whether the path-faithful
data was available (it was, in `entry.path`) — an avoidable weakening on a
delete path is Real-blocking, not a nit.
