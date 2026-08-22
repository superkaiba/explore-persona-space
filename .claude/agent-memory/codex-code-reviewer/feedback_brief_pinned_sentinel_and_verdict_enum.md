---
name: brief-pinned-sentinel-and-verdict-enum
description: When the orchestrator's brief pins the marker head sentinel (e.g. v1 on a round-2 compose) or narrows the verdict enum (PASS|FAIL, no CONCERNS), follow the brief literally — it is the validator's extraction contract — and flag the convention divergence in the return
metadata:
  type: feedback
---

On #2228 r2 (2026-08-20) the brief's output contract demanded the marker
block delimited `<!-- epm:code-review-codex v1 -->` on a ROUND-2 compose
(both standing conventions would say v2: head-sentinel==revision_round per
the composer spec, and sentinel==impl-marker-version per the #2329
convention — the round's report was `epm:results v2`), and a BINARY
`**Verdict:** PASS|FAIL` line (no CONCERNS option).

Rule: the brief is the ORCHESTRATOR'S extraction/parse contract — it will
grep for the tags and verdict values it wrote. Compose exactly what the
brief demands, and FLAG the divergence in the return (name both conventions
and what was pinned) so the orchestrator can patch the sentinel at post time
if the brief was a typo. Never silently "correct" the brief to convention —
a mismatched start tag fails extraction and burns a retry dispatch.

**How to apply (binary-verdict variant):** when CONCERNS is removed from the
enum, route non-blocking findings explicitly in the prompt: Minor/Major
findings + fresh `CONCERN:: ` rows persist them; FAIL stays reserved for
substantive blockers; state that present-but-imperfect marker shape can
never be the sole FAIL ground (Step 0.7 unchanged). Already-persisted ledger
ids still get closure STATUS LINES, never re-emitted rows.

**Recurred #2253 r2 (2026-08-21), with a sharper twist:** the brief pinned
`v1` on the r2 compose while the SAME task's r1 Codex marker in events.jsonl
ALSO carries `<!-- epm:code-review-codex v1 -->` — a genuine in-history
sentinel collision, not just a convention divergence. Extraction from the
fresh OUTPUT FILE stays unambiguous (one block per file); the hazard is only
a later events.jsonl re-extraction keyed on the sentinel. Still: follow the
brief, name the collision explicitly in the return so the orchestrator can
patch the sentinel at post time if it was a typo.
Related: [[revision-round compose recipe]], [[concerns-machine-rows-2326]].
