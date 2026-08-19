---
name: feedback-lens7-no-judge-activation-experiments
description: Lens 7 for no-judge / activation-DV experiments — local recomputation only; HF liveness sub-checks advisory; fiction/story corpora audited structurally, never by loading story text
metadata:
  type: feedback
---

For experiments whose DV is activation-based or otherwise has NO judge /
no firing-rate claim, Lens 7 does not reduce to "sample firing vs
non-firing rows": scope it to LOCAL recomputation of whatever per-row
quantities the body quotes (counts, per-cell values, sample-row joins),
and mark HF-hosted sub-checks advisory (`sandbox-unverifiable —
(advisory)`), never BLOCKED, when the local artifacts carry the needed
fields.

For fiction/story corpora: never instruct Codex to load story text into
context — run a STRUCTURAL audit instead (row counts, key presence,
length distributions, id joins).

**How to apply:** when composing Lens 7 for such a task, enumerate the
local files + the exact recompute, state the advisory carve-out for HF
paths, and give the structural-audit shape for any long-form text corpus.

(Reconstructed 2026-08-18 from the MEMORY.md index hook after the original
file went missing from disk — kept deliberately minimal; re-ground it with
specifics the next time the situation recurs.)
