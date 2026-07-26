---
title: 'daily-held: #1639 missing #1310 assistant-test fold'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-26T07:07:49Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 3): The #1310 round-3 assistant
  test landed on main on 2026-07-23 and is named in #1639''s own Context scope row,
  but #1639''s body carries no assistant-test result line and its Repro code list
  cites only rounds 1 and 2.'
workflow: v1
---
## Why this needs you

Filed by the `/daily` 2026-07-25 problem sweep as a **route-3 judgment call** under the
"Scientific-meaning changes" carve-out: folding a result that *dissolves* a clean-result's
headline effect is an interpretation change on a body sitting at `awaiting_promotion`.

**#1639's clean-result does not contain the #1310 assistant-test result, which landed on
`main` on 2026-07-23 and reportedly collapses the headline assistant-vs-character gap.**

## What was found

Session `63122023` @ 2026-07-25T18:07:54Z, while answering an unrelated question,
discovered that `scripts/issue1310_xpersona_assistant_test.py` had run on 2026-07-23,
was committed (`9e65fe09ad`), and was never folded into #1639. Its reported result takes
the headline 3–5× assistant-vs-character gap down to **0.04 R²**. The finding was
surfaced in chat; nothing was folded and no task was filed.

## Verified at filing (2026-07-25)

- `task.py view 1639 --json` → `status: awaiting_promotion`; title:
  *"The four fiction-character context→dialogue maps are one dominantly shared linear
  operator: a pooled…"*.
- The body's own `**Context:**` row states its scope verbatim: *"consolidating the
  #1310 cross-persona similarity rounds 1-3: cross-character battery, principled
  re-analysis, and the assistant test"* — so the assistant test is **in scope by the
  body's own account**.
- Scanned the body for any assistant-test result: **0 lines** mention "assistant"
  together with an R² value or the word "gap". Its `**Repro:**` row cites
  `issue1310_xpersona_similarity.py` @`82d85db5ee` (round 1) and
  `issue1310_xpersona_similarity_v2.py` @`9edaab4fa4` (round 2) — **round 3 / the
  assistant test is absent from the code list**.
- The artifact exists on `main`: `git log --oneline -1 9e65fe09ad` →
  *"issue #1310 round-3 assistant-test: artifacts + summary + figure (auto-harvest;
  summary_rc=0 cert_rc=0)"*; `scripts/issue1310_xpersona_assistant_test.py` present.

**One premise I could NOT verify — treat as unverified hypothesis:** the specific
"3–5× gap → 0.04 R²" figure is quoted from the session's chat text, not read from the
round-3 summary JSON. Read
`eval_results/issue_1310/` round-3 outputs before acting on the magnitude. The
*structural* fact — a landed, in-scope round absent from the body — is verified above.

## The decisions that are yours

1. **Fold or don't.** If the result holds, #1639's headline ("one dominantly shared
   linear operator") may need requalifying before promotion — that is an interpretation
   call on your text.
2. **Promotion timing.** #1639 has been sitting at `awaiting_promotion` with an
   incomplete evidence base for 2 days; decide whether it parks until the fold or
   promotes with a stated scope caveat.

## Related work already routed tonight

A route-2 workflow-fix task from this sweep proposes the *mechanical* half — an audit
pass flagging any task body whose named pending artifact/commit already exists on
`main`, so a landed-but-unfolded round cannot sit silently. That task does not touch
#1639's body.
