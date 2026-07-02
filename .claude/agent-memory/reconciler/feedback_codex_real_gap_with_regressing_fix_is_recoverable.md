---
name: Codex flags a real lint gap but its stated fix regresses a documented design exception
description: Alternatives/Methodology lens — when Codex's grounded REVISE finding is real but its proposed fix would re-break the exact edge case the plan was built around, the finding is implementer-recoverable (APPROVE + standing rec), not fatal
type: feedback
---

When the Codex twin issues a REVISE on the Alternatives/Methodology lens with a
grounded + mechanizable finding, VERIFY the proposed FIX against the artifact, not
just the finding. A finding can be correct while its stated fix is wrong.

**The pattern (#714 r1, plan v1):** a `--check-skill-refs` lint resolved `/skill`
refs against `{p.name for p in .claude/skills/*}` (dir existence) specifically
because `clean-results` is a live skill dir with NO `SKILL.md`. Codex correctly
flagged two real gaps: (F1) the dir-blind live set would also admit a future
non-skill dir; (F2) reusing the shared `HISTORICAL_REF_OPT_OUT` waiver token lets
one waiver silence both the script-ref AND skill-ref lints on a line. BOTH verified
on disk / in source. BUT Codex's F1 fix-as-stated (`assert d.name not in live_set
unless (d/SKILL.md).exists()`) would REGRESS the `/clean-results` resolution the
plan deliberately designed around — exactly the false-positive the plan rejected the
candidate's `glob("*/SKILL.md")` to avoid.

**Adjudication rule:** a real-but-hardening finding whose literal fix regresses a
documented plan exception is RECOVERABLE through implementer judgment, NOT fatal →
bind APPROVE, carry BOTH findings as mandatory standing recs, and in the rec spell
out the CORRECT resolution (here: known-exception set `{clean-results}` or add a
SKILL.md stub) plus an explicit "do NOT apply Codex's literal fix" so the implementer
doesn't blindly regress.

**Why recoverable, not fatal:** apply the conclusion-changing bar. F1 was a
future-hypothetical permissiveness (not a present false-PASS; green-on-main test +
closed allowlist constrain today's live set, and a non-skill dir under
`.claude/skills/` is itself a convention violation). F2's false-negative requires a
single line to carry a script citation needing the waiver AND a separate genuinely-
stale skill ref — vanishing overlap, with the repo-tree-is-clean test as backstop.
Neither changes the check's design conclusion; both are small additive refinements
within the plan's degrees of freedom.

**Also:** Codex verdicts sometimes cite STALE section numbers (it cited §3.3/§12.4;
live plan v1 numbered them §4.1/§4.5/§11). Map by SUBSTANCE, not label — a wrong
§ number is not grounds to discard a finding that resolves to a real section.
