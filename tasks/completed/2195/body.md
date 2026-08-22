---
title: 'verify_report.py: WARN when a report cites a repo artifact at a SHA the same
  branch has since modified'
kind: infra
tags: []
created_at: '2026-08-08T03:07:40Z'
has_clean_result: false
origin_prompt: 'surfaced by methodology-critic round 3 on #2162: the report cited
  routing_evidence.json at db5d1680a2 while a newer commit on the same branch had
  rewritten that artifact to contradict the citing sentence'
workflow: v1
---
# `verify_report.py`: warn when a report cites a repo artifact at a SHA that the same branch has since modified

## Goal

Add a **stale-evidence-pin check** to `scripts/verify_report.py`: for every
in-repo artifact a report cites at a pinned SHA, run
`git log <pin>..origin/<branch> -- <path>`. A non-empty result means the artifact
was modified on the same branch AFTER the pin, so the report is citing a
superseded version of its own evidence. Emit a WARN naming the newer commit.

## The instance that motivated it (#2162, methodology gate round 3)

The report cited `eval_results/issue_2162/judge/routing_evidence.json` at commit
`db5d1680a2` as evidence for a documented deviation. By the time the gate ran,
the same branch had a newer commit (`28d35cffc6`) that **rewrote that artifact**,
adding a `superseded_readings` list which named the report's own stated mechanism
as WRONG and replaced it.

So the report was citing, at a pin, a file whose newer version on the same branch
contradicted the sentence it was cited to support. Nothing was missing — the
pinned blob resolved fine — which is exactly why a resolves/does-not-resolve check
misses it. The reviewer caught it only by noticing the branch tip had moved and
reading the newer version.

This is a live hazard whenever evidence artifacts are iterated during report
review, which is the normal case: the report and its evidence are being written
and corrected concurrently.

## Why a WARN rather than a FAIL

An as-of pin is legitimate and often correct — a report may deliberately cite the
state at authoring time, and #2162 also carries a deliberate as-of branch pin plus
an HF revision pin that are both fine. The check cannot know whether a later
modification contradicts the citation or merely adds unrelated content. So WARN,
naming the newer commit, and let the reviewer decide.

Consider FAILing only in the narrower case where the newer version of the cited
artifact contains a marker explicitly retracting a reading (the `superseded_readings`
shape above) — but that requires a convention, so treat it as a possible follow-on
rather than part of this task.

## Relationship to the sibling task already filed

A separate task already covers two other mechanizable report-accuracy asserts
from the same review — (a) "committed under `<path>`" claims resolving to at
least one blob at the pin, and (b) code-SHA rows matching per-phase
reproducibility-card `git_commit`s. This one is a THIRD, distinct check: those two
ask "does the cited thing exist / match", this one asks "has the cited thing
changed since you cited it". Implementing them together in one pass is sensible
if both land in the same session; they are independent asserts either way.

## Scope notes

- Additive only; do not weaken existing `verify_report.py` checks.
- Scope to IN-REPO artifacts. HF revision pins are a different problem (no local
  history to compare) and upload-verification already covers HF-side identity.
- #2162 is the first `workflow: v2` task in the repo, so the report gate has run
  against exactly one report so far — prefer a shape that is cheap to extend as
  more patterns surface.
- Confidence: high on the value (one confirmed instance where it would have
  caught a blocking error before a reviewer did), moderate on the parsing shape,
  since which strings in a report count as "an artifact citation at a pin" is a
  convention question. A conservative matcher that under-fires beats one that
  WARNs on every SHA mentioned in prose.
