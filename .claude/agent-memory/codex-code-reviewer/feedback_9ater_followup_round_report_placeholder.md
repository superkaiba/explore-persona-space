---
name: 9a-ter free-analysis follow-up rounds — the brief's target_marker_kind field decides marker fetch vs placeholder
description: A 9a-ter free-analysis round may or may not have posted an epm:experiment-implementation marker; key on the brief's target_marker_kind (none → placeholder; a named version → fetch + inline + round-match)
type: feedback
---

For a 9a-ter free-analysis follow-up review round (brief says
`target_marker_kind: none`; the round is a single ANALYSIS-ONLY commit), the
Step 2-pre `task.py latest-marker --prefix epm:experiment-implementation`
fetch does NOT apply — the highest-version implementation marker belongs to a
PRIOR round of the main experiment, and inlining it would grade the wrong
round. The round's actual evidence is (a) the `epm:progress` stage-dispatch
marker with `stage=free-analysis-followup` (fetchable from events.jsonl —
inline it as the round contract) and (b) the follow-up implementer's
RETURN-TEXT report, which exists only in the orchestrator's context.

**Why:** hit on #920 r3 (2026-07-03). The fail-loud rule ("no implementation
marker → post epm:failure") would have been a false failure here; the brief
explicitly said the orchestrator holds the report.

**How to apply:** compose the prompt with a literal
`{{followup_implementer_report_body}}` placeholder inside the standard
`---BEGIN/END IMPLEMENTATION MARKER BODY---` envelope, and state
"ORCHESTRATOR ACTION REQUIRED: substitute the report before dispatch" in the
Step-4 return. Adapt the mechanical gates in the composed prompt: Step 0.5
scores the inlined report's substance (9a-ter return-text layout ≠ four-H3
marker layout — present-but-imperfect is CONCERNS); Step 0.55 is satisfied
by the task's existing smoke-arch marker (presence-ON-TASK — inline it so
Codex doesn't re-raise); Step 0.6's proof surface is the committed outputs
in the SAME commit + the report digest; diff scope is `git show <round-sha>`
only, never main...HEAD (per the concurrent-followups memory). Also inline
the open-concerns result (`list-concerns --open-only`) so Codex doesn't
shell out to task.py.

**Variant (#958 r5, 2026-07-04): a 9a-ter round CAN post a real marker.** When
the brief names `target_marker_kind: experiment-implementation (highest = vN;
fetch + inline)`, the standard Step 2-pre fetch DOES apply — the free-analysis
implementer posted a real `epm:experiment-implementation vN`. The placeholder
path above fires ONLY on `target_marker_kind: none`. Either way, round-match
the fetched marker by note body (it must name the round's commit + scope), and
inline the round's `epm:progress stage=free-analysis-followup` dispatch note as
the round contract (there is no per-round plan version; plan vK is parent
conventions only).

**Variant (#2203 full-rerun-bugfix CJK-producer review, 2026-08-11): a
follow-up ANALYSIS commit with NO marker and NO orchestrator-held report.**
When the round's target is a post-rerun analysis-producer commit landed by
the followup-run stage (latest impl marker covers an EARLIER bugfix round —
wrong round; no return-text report exists either), skip the placeholder
entirely: inline the COMMIT MESSAGE inside the standard
`---BEGIN/END IMPLEMENTATION MARKER BODY---` envelope with a first line
stating "NO IMPLEMENTATION MARKER EXISTS FOR THIS ROUND (analysis-only
producer commit; the commit itself is the evidence)". No orchestrator
substitution needed. Declare the marker gates N/A explicitly in the prompt
("do NOT emit marker-shape / smoke-run-missing — no marker contract to
score"; standing smoke-arch marker STANDS) and adapt the Blocker-tags line
to `substantive | data-access-blocked | git-provenance` only. Also: when the
brief hands a FOCUSED rubric ("focus, not the whole codebase") for a single
new file, quote the plan's few relevant rows in a
`---BEGIN/END APPROVED PLAN EXCERPTS (v<K>)---` envelope instead of the full
90KB plan, and pre-empt legitimate cross-round set differences the parity
check would trip on (parent stats had 16 phase2 arms vs the round's 24 —
parity is shape/keys/value-forms, never arm-set equality).
