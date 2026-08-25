---
title: 'verify_report.py: scan the detailed companion writeup, not just the report
  body'
kind: infra
tags: []
created_at: '2026-08-08T03:48:46Z'
has_clean_result: false
origin_prompt: 'surfaced by report-verifier round 1 on #2162 (first workflow:v2 report):
  the interpretivity/banned-lexicon scan and structural checks run on the body only,
  while the report template says the same discipline applies to docs/reports/issue_<N>_detailed.md;
  the verifier had to scan it by hand'
workflow: v1
---
# `verify_report.py`: scan the detailed companion writeup, not just the report body

## Goal

`verify_report.py`'s interpretivity / banned-lexicon scan and its structural
checks run on the report BODY only. The v2 report template
(`.claude/skills/issue-v2/report-template.md` § The detailed companion writeup)
states that the companion at `docs/reports/issue_<N>_detailed.md` is 100%
agent-written and that "the interpretivity rule applies throughout it, and the
banned-lexicon scope mirrors the body's". Nothing mechanically enforces that.

Add a generation-mode check that resolves the body's `**Detailed writeup:**`
pin, materializes the companion at that SHA (`git show <sha>:<path>`), and:

1. runs the same banned-lexicon scan over it — with the Motivation copy keeping
   the body's hypothesis-framing exemption, since the companion's Motivation is
   a verbatim copy of the body's; and
2. asserts the forbidden headings are absent — no `## TLDR`, no `**Takeaways**`,
   no `## Conclusion and next steps`. The companion carries no Thomas slots at
   all, because it is regenerated wholesale on every follow-up round, so
   anything hand-written there would be silently destroyed.

## Why it matters

The companion is the full-detail layer: every figure view, the unabridged
methodology. It is longer than the body and it is the document a reader goes to
for detail. An asserted conclusion that lands there is exactly as misleading as
one in the body, and right now nothing catches it — the whole point of v2 is
that no agent asserts a conclusion.

The structural half matters for a different reason: a `**Takeaways**` block in
the companion would read as a claim slot, invite someone to fill it, and then
be wiped by the next regeneration.

## Evidence this is a real gap (#2162, the first v2 report)

The 18-check `verify_report.py` output has no companion-content check of any
kind. The `report-verifier` agent scanned the companion BY HAND in round 1 and
found 3 lexicon hits — all benign process-facts in the deviations section ("each
confirmed from artifacts", "the dispatch log confirmed num_workers=4", "the
plan's phrasing implies one fused forward"), none reading a measured value as a
finding, so the round still PASSed. But that scan was manual and therefore
depends on the reviewing agent choosing to do it.

Worth noting for whoever implements this: those 3 benign hits show a naive scan
will produce false positives on legitimate methodology prose. "confirmed" and
"implies" have honest process meanings ("confirmed from the artifact", "the
plan's phrasing implies"). Consider whether the companion's scan should report
hits as WARN rather than FAIL, or whether the lexicon needs
process-sense carve-outs. A check that FAILs on "confirmed from artifacts" will
be routed around, which is worse than a WARN that gets read.

## Scope notes

- Generation mode is the natural home (the pin and the local objects exist by
  construction there). At promote, a body whose companion pin predates a fresh
  clone may be unresolvable — degrade to WARN, mirroring the existing
  `image-pin-blob-identity` mode-split ladder rather than inventing a new one.
- A grandfathered body with no `**Detailed writeup:**` line already only WARNs;
  keep that, and skip this check entirely for those.
- Do NOT weaken any existing check.
- Two sibling `verify_report.py` tasks are already filed and are NOT superseded
  by this one: the committed-under + code-SHA asserts, and the
  stale-evidence-pin check. This is a fourth, distinct gap in the same file —
  #2162 being the first v2 report, the file is getting its first real exercise.
- Confidence: high that the gap is real and the template already requires the
  behavior; moderate on the FAIL-vs-WARN posture, which is the one judgement
  call and is worth deciding deliberately rather than defaulting to FAIL.
