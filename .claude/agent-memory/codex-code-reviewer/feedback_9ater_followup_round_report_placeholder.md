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

**Variant (#2333 cr9ater = marker v7, 2026-08-18): 9a-ter round AFTER the
interpretation gate closed on a task whose loop rounds hit cap-5 + a
post-cap greenlit r6.** Marker numbering continues task-wide (v7 after the
loop's v1–v6); frame explicitly "NOT round 7 of the loop; rounds 1–6 are
ADJUDICATED". The round contract is the `epm:progress` note whose text
LEADS "9a-ter free-analysis round DISPATCH" (fetch by exact version from
events.jsonl — there is no `stage=` token; grep the lead phrase). When the
brief hands a FOCUSED lens set, compose tight (~28 KB: contract envelope +
report placeholder + 6 anchored lenses + adapted verdict tail) — no full
rubric; adapt Blocker-tags to `substantive | data-access-blocked |
git-provenance` (the #2203 tag set) and pre-declare 0.5/0.55/0.6/4.6 N/A so
Codex cannot manufacture marker objections. PROBE the committed output
JSONs at compose time: the brief's flip shorthand can be coarser than the
artifact (brief: "q35 bstart banked → natural-opening-indeterminate"; the
JSON showed samewave NO-flip + banked-only flip) — hand the exact committed
flip flags/CIs neutrally with a verify-don't-assume instruction, and pin
`generated_at` vs JSON-commit timestamps for the regeneration-integrity
lens. Open ledger rows (all main-pipeline scope) get regression-duty-ONLY
framing + one named look-hardest row where the diff grazes a concern's
neighborhood (donor STAGING edit vs the generate-side donor-cache concern).

**Variant (#823 P-Gen ladder round, 2026-08-19): the placeholder pattern
also fires on a 9b PAID/GPU same-issue follow-up round, not only 9a-ter.**
After the round-1 implementer died to autocompact thrash and was respawned
micro-scoped, its report existed only in-session — `latest-marker` returned
the PARENT's July v5 marker (round-mismatched; never inline it) and the
race re-probe + epm:results probe + branch-events probe all confirmed no
round marker. Placeholder + "ORCHESTRATOR ACTION REQUIRED" return applies
unchanged. Three additions: (a) the smoke-arch marker CAN be round-matched
even when the impl marker is absent (the respawned implementer posted v5
naming the unit) — inline it, and route its DISCLOSED pending-harvest leg
(batch in_progress inside SLA; synthetic probes for assert/selection fns)
via Step 0.6 as present-but-limited/CONCERNS, never `smoke-run-missing`
fabrication; (b) when the brief hands a SPEC SLICE (thrash-killed prior
agent; "use the slice, not the ~120KB plan"), inline the slice as the
authoritative plan envelope AND instruct Codex not to open the full plan
or the ~111KB parent driver (bounded greps of the parent allowed for
parity claims only); (c) brief-enumerated registered requirements get a
dedicated `## Registered requirements` verdict-body section with an
INDEPENDENT-recompute duty — e.g. "confirm the asserts catch
persona(i,k)=(i+1) mod k by your own arithmetic; a shifted modulo
PRESERVES per-persona counts, so count/balance asserts alone cannot catch
it" — never trusting the smoke-arch marker's "FIRED" claims.

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

**Variant (#823 P-Gen v13 amendment round, 2026-08-20): orchestrator LANDING
NOTE as the round record — inline directly, no placeholder.** When the
implementer orphans its turn (uncommitted payload, #2041 shape) and the
ORCHESTRATOR lands the commit + records it as an `epm:progress` note (v108:
landed SHA, requirement→test table, lint-gate blob-sha certification), that
note IS the durable round record: inline it in the standard IMPLEMENTATION
envelope (no `{{placeholder}}`, no ORCHESTRATOR-ACTION return), state
"absence of a per-round impl marker is BY DESIGN", and invalidate BOTH Step
0.5 `marker-shape` AND Step 0.6 `smoke-run-missing` in the prompt + the
Blocker-tags bracket (round evidence = lint-gate PASS + committed fixtures;
coverage gaps route `substantive`). Plan handling when the brief orders
BY-PATH + no-inline but the worktree plans/ is frozen at an older version
(v10 symlink vs binding v13 on main): copy the canonical plan to /tmp and
reference BOTH the /tmp copy (primary) and the main-root absolute path
(fallback), bar worktree `tasks/` reads explicitly, give sed line-window
anchors for the binding sections, and make plan-lens BLOCKED genuine only
after BOTH paths fail. Sentinel when no impl marker exists and prior codex
sentinels are task-numbered: max existing posted codex version + 1 (here
v6), mapping stated in the return.
