---
name: Codex reads contextual plan prose as unconditional requirement
description: Codex FAILs round-1 on "the plan explicitly says X must stay visible" when the sentence is a surface-contrast (dashboard vs title) + honesty rationale, and the plan's own pinned type/kill-criterion compose to the shipped behavior
type: feedback
---

Codex code-reviewer FAILs a diff citing one plan sentence as an unconditional
requirement ("The '+ plan review' annotation must stay visible on the
dashboard") while the shipped behavior drops the element under the plan's §7
kill criterion ("drop the ETA chip; ship the position bar regardless").

**Why:** the sentence's context resolves it as CONDITIONAL: (a) the immediately
preceding parenthetical contrasts the two surfaces ("title suffix omits this
for chars") — "must stay visible on the dashboard" means "on the dashboard,
unlike the title"; (b) the honesty rationale ("silently absorbing it would be
dishonest") only bites when the qualified display (the machine-ETA countdown)
is actually rendered — no countdown, nothing to be dishonest about; (c)
DECISIVELY, the plan's own pinned view type (§3.6 `TaskProgressView = { pct;
etaLabel; state; basis }` — no separate planReviewAhead field) folds the
annotation INTO the killed element, so plan-§3.6 + plan-§7 compose mechanically
to the shipped behavior; (d) the acceptance-criteria enumeration and the §10
must-ask list never name the element as independent. Codex's proposed fix (a
bare annotation with no band) was a rendering the plan never specified — it had
to invent it.

**How to apply:** when Codex's FAIL hinges on "the plan explicitly requires X
visible/present" against a fired kill/degrade criterion, (1) verify the code
mechanics (usually correct — here progress.ts:436-437 null etaLabel +
TaskProgressBar.tsx:100 early return were exact); (2) re-read the quoted plan
sentence IN CONTEXT — check whether "must stay" contrasts surfaces or
conditions; (3) check the plan's own pinned types/schemas: if the plan's type
folds the element into the killed component, the kill rule wins; (4) check
whether honesty-rationale prose still bites with the display gone; (5) check
whether sibling state labels DID survive (here blocked/overdue/stale/
waiting-on-you all render band-independently — only the chip appendage drops).
PASS + standing rec (band-independent disclosure + kill-switch-default test as
a deliberate choice). Origin: task #587 round-1 (Claude PASS / Codex FAIL,
`plan-review-ahead-hidden`).
