---
name: Lens-14 open-concern ack on revision rounds
description: verify --issue FAILs on open concerns until acknowledged in-body; use concern-deferred markers placed in Methodology (cap-excluded), never inside a Results block
type: feedback
---

On a revision round, `verify_task_body.py --issue <N>` (unlike `--file` on a cache copy with no
concerns.jsonl sibling) runs the Lens-14 concerns audit and FAILs while the round's concerns are
still OPEN — and the analyzer is barred from `address-concern` (orchestrator-only, post-critic).

**Why:** the check's only whole-body acknowledgment mechanism is a
`<!-- concern-deferred: <id> -->` HTML comment (mechanism 3); the concern_id-substring mechanism
scans only Takeaways + Results. #833 r3 hit this after set-body; the task's own m0-shift concern
had already used the deferred-marker precedent while awaiting the orchestrator's address-concern.

**How to apply:** place one `<!-- concern-deferred: <id> -->` per open concern at a semantically
relevant spot INSIDE `## Methodology` (excluded from all word caps). Never inside a `### <result>`
block: `_prose_words` does NOT strip HTML comments, so a marker there adds ~4 words and can tip a
mature 177-179-word block over the ≥180 FAIL cap. Disclose in the `epm:interpretation` body that
the markers are the mechanical Lens-14 ack pending orchestrator address-concern (not a real
deferral, concerns NOT addressed by the round).
