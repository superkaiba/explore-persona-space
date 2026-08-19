---
title: 'task.py set-body: make the ## Goal H2 guard workflow-aware so a v2 report
  body does not need --allow-goal-drop'
kind: infra
tags: []
created_at: '2026-08-08T03:23:22Z'
has_clean_result: false
origin_prompt: 'hit while posting the first workflow:v2 report body on #2162: the
  #1112 GoalH2DropError guard fires on every v2 report body because the v2 template
  deliberately has no ## Goal section, forcing --allow-goal-drop on the happy path'
workflow: v1
---
# `task.py set-body`: make the `## Goal` H2 guard workflow-aware so a v2 report body does not need `--allow-goal-drop`

## Goal

`set-body` refuses to write a `kind: experiment` body that removes a `## Goal`
H2 present in the prior body (`GoalH2DropError`, incident #1112), overridable
with `--allow-goal-drop`. That guard is correct for markdown v4 clean-results,
where `## Goal` is a required section.

It is wrong for `workflow: v2` report bodies. The v2 report template has five
required H2 sections — Motivation / TLDR / Methodology (shared) / Results /
Conclusion and next steps (`scripts/verify_report.py` `REQUIRED_SECTIONS`) —
and `## Goal` is deliberately **not** among them; the Motivation section
carries that context instead. So every v2 report body legitimately drops
`## Goal`, and the guard fires on every one of them.

Make the guard skip (or invert) for a body whose frontmatter is
`workflow: v2` — or, more precisely, key it on the body being a report-v1 body
rather than a v4 clean-result. The v4 path must be left exactly as strict as it
is today.

## Why this matters rather than being a harmless extra flag

The override trains the caller to pass `--allow-goal-drop` reflexively on
report posts. Once it is habitual it is also passed on the runs where the drop
is the #1112 accident the guard exists to catch, and the guard stops working
for the case it was built for. A guard that fires on a legitimate,
spec-mandated action every single time is worse than no guard, because it
teaches people to silence it.

## What actually happened (#2162, first v2 task)

Posting the assembled v2 report body was refused. Verified before overriding:

- `verify_report.py` `REQUIRED_SECTIONS` contains no `## Goal`.
- The assembled body has exactly the five required H2s and no others.
- The canonical Goal was never at risk: `goal:` frontmatter was intact at 1,326
  characters before and after the write, and `set-body` preserves frontmatter.
  The Goal that downstream agents read is the frontmatter field, which
  `set-goal` writes — not the body H2.
- The prior body (with its `## Goal`) was snapshotted to `original-body.md` by
  `--snapshot`, so nothing was lost.

So the drop was correct-by-spec and the override was safe here. It was still an
override on the happy path, which is the defect.

## Scope notes

- Do NOT weaken the v4 path. The check should stay byte-identical in behavior
  for any body that is not a v2 report; ideally add the branch, do not
  restructure the guard.
- Decide deliberately whether the discriminator is `workflow: v2` frontmatter
  or the presence of the `<!-- report-v1 -->` sentinel in the NEW body. The
  sentinel is the more precise signal (it describes the body being written
  rather than the task's routing), and it also covers a v2 task whose body is
  something other than a report. Prefer the sentinel, or require both.
- Consider whether the inverse guard is worth adding for the v2 path: refuse a
  report-v1 body that is MISSING one of the five required sections, so the same
  call site protects both shapes rather than just going quiet for v2. That is a
  natural extension but a judgement call — `verify_report.py` already covers it
  at the gate, so this may be redundant.
- `#2162` is the only v2 task so far, so there is exactly one real call site to
  test against; its posted body is a good fixture.
- Confidence: high that the current behavior is wrong for v2 and that the fix
  is small; moderate on the discriminator choice, which is the one real design
  decision here.
