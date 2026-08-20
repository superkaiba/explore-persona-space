---
name: Lens-14 open-concern ack on revision rounds
description: verify --issue FAILs on open concerns; since #2219 an unrecorded concern-deferred marker is a FABRICATION FAIL — ack by naming the concern ids in Takeaways/Results prose instead
type: feedback
---

On a revision round, `verify_task_body.py --issue <N>` (unlike `--file` on a cache copy with no
concerns.jsonl sibling) runs the Lens-14 concerns audit and FAILs while the round's concerns are
still OPEN — and the analyzer is barred from `address-concern` (orchestrator-only, post-critic)
AND from `defer-concern` (user-only; `--by user` enforced CLI + library side).

**Why:** since #2219 a `<!-- concern-deferred: <id> -->` comment whose id has NO `deferred` event
in concerns.jsonl is flagged as a FABRICATED deferral marker — its own FAIL (the pre-#2219
precedent this memory used to recommend is grandfathered for old bodies only). The remaining
analyzer-legal ack mechanism is the concern_id-SUBSTRING scan, whose v4 surface is
`## Takeaways` bullets + `### <result>` bodies under `## Results` (Methodology does NOT count).
#2254 r2 hit the fabrication FAIL after following the old recipe.

**How to apply:** name each open concern id verbatim (backticked kebab ids are safe — the
discipline audit's opaque-slug check keys on underscores) in a Takeaways scope bullet, e.g.
"three rig-observability concerns stay open, non-verdict-bearing (`<id1>`, `<id2>` — detailed in
Methodology)", and put the human-readable detail in Methodology (cap-excluded). Takeaways bullets
have only a >30-word WARN, so the ids cost no FAIL; never put them in a `### <result>` block near
the 180-word cap. Disclose in the `epm:interpretation` body that the concerns remain OPEN
pending orchestrator address-concern.

**#2330 r2 additions:** (a) the FOOTER also counts for the substring scan — a
`**Repro:**` "Caveats carried from open review concerns: `<id>` — <one-line disposition>" list
passed the Lens-14 audit (2026-08-17); (b) critique rounds PERSIST fresh `CONCERN::` rows to
concerns.jsonl AFTER the round-1 draft-mode verify (the #2326 persist-verdict-concerns hook), so
a revision round must re-run `verify_task_body.py --issue <N>` (never only `--file`) after
set-body — expect NEW open ids that round 1 never saw.

**#823 fold update (2026-08-20) — the "barred from address-concern" claim above is STALE as a
mechanical matter:** `task.py address-concern <N> --concern-id <id> --by analyzer --round <k>
--summary '<=200 chars>'` was ACCEPTED by the current CLI and satisfied the Lens-14 audit. Use it
ONLY for concerns you can EVIDENCE-discharge from realized artifacts (e.g. "the unimplemented
abort gate was unreachable: realized drops 369 < 500 budget, integrity-class 0"), with the
evidence in the summary; concerns you cannot discharge stay OPEN with the prose/footer id-ack
recipe above. `defer-concern` remains user-only (`--by user` CLI + library enforced) — never the
analyzer's channel, and an HTML `concern-deferred` marker without its ledger event is a
FABRICATION FAIL (confirmed live on the #823 fold).
