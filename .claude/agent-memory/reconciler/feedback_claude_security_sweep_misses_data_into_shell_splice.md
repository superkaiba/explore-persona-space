---
name: claude-security-sweep-misses-data-into-shell-splice
description: Claude code-reviewer rates the security sweep CLEAN on diffs that splice unrestricted data placeholders into quoted shell source in orchestrator-prose templates — its sweep checks secrets/shell=True/pipes, not data interpolation.
metadata:
  type: feedback
---

**Rule:** when the artifact under adjudication adds or edits an
orchestrator-prose shell TEMPLATE (skill step docs, fenced Bash the /issue
orchestrator substitutes placeholders into), do not credit Claude's
"Security sweep: CLEAN" as covering injection — re-scan the diff yourself
for placeholders (`<task title>`, `<slug>`, any non-integer-constrained
`<...>`) spliced inside single/double-quoted shell source or variable
assignments. Claude's sweep vocabulary is secrets / `shell=True` / piped
push compliance / timeout fences; data-into-shell-source interpolation is
outside it.

**Why:** #2241 r1 — Claude PASS with "Security sweep: CLEAN" while the new
Step-5 block rendered `gh pr create --title "issue-<N>: <task title>"`;
titles are stored unsanitized (`set_title`, task_workflow.py:6324) and ~1%
of registry titles carry backticks/quotes. Codex caught it (over-classed as
round Critical — the class was pre-existing-live at 18-step-10d.md and
plan-verbatim, so the reconcile was PASS + deferred CONCERN), but Claude's
miss meant a single-reviewer round would have shipped it unrecorded.

**How to apply:** on any skill-doc / prose-template diff, grep the ADDED
fenced Bash for `<[a-z ]+>` placeholders and classify each: integer/enum-
constrained (safe) vs free-text task metadata (title, note, summary —
flag). A real hit is usually Real-but-non-blocking when the splice mirrors
a live trunk idiom and the input is same-trust-domain (see
[[feedback_codex_hardening_beyond_minimal_port_contract]] #2241 entry), but
it must land in the ledger, not vanish under a CLEAN sweep line.
