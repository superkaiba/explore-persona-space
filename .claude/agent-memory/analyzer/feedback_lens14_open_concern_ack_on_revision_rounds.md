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
