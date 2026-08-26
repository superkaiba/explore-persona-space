---
name: codex-side-sharded-round-compose
description: "#2569 r1: the Codex review itself can be SHARDED (3 briefs, path-scoped) — not always whole-round-unsplit; scope diff reads to shard paths, keep contract gates ON with shared-marker imperfections fenced to concerns, still strip the split token, name out-of-scope sibling shards"
metadata:
  type: feedback
---

The [[whole-round-unsplit-compose]] premise ("the Codex twin's brief is a
WHOLE-ROUND UNSPLIT review") is not universal: on #2569 r1 (2026-08-25, a
~1 MB round) the orchestrator sharded the CODEX side too — three composer
briefs, each path-scoped ("Codex shard 3 of 3", 12 named paths, 373 KB),
with the sibling shards named. Compose deltas that worked:

1. **Scope every diff read to the shard paths** (`git diff
   origin/main...HEAD -- <one path at a time>`), list compose-time per-file
   sizes with a read-depth per file (A-status round-new files may be read
   at HEAD), and BAN reviewing/FAILing on out-of-scope files — but allow
   HEAD reads of seam files (poll_pipeline.py, sibling drivers,
   convention-precedent scripts) with findings grounded in shard files.
2. **Contract gates stay ON in every shard** (impl + smoke-arch markers
   inlined as usual), but fence the shared-marker surface: present-but-
   imperfect marker issues are concerns, and only structural absence in
   the INLINED body is a `marker-shape` Critical — three shards would
   otherwise triplicate mechanical blockers on one shared marker.
3. **Still strip the Step-0 split-review paragraph** (the literal trigger
   token arms write-to-file/skip-gates behavior); path-scoping comes from
   the brief's scope block, never from that token.
4. **Brief-supplied round facts ride in two fenced blocks**: "round-specific
   facts (verify AGAINST them)" vs "verified facts (do NOT re-derive)" —
   the second names pre-existing reds (#2572/#2584-class) with the Step 0.9
   `pre-existing-on-trunk` routing so no shard burns effort or false-FAILs
   on them.
5. **Brief-pinned binary PASS|FAIL** ([[brief-pinned-sentinel-and-verdict-enum]]):
   map rubric CONCERNS to `VERDICT: PASS` + MAJOR/MINOR + `CONCERN::` rows;
   first line inside the marker block is the `VERDICT:` token.
6. **Multi-unit FINAL-unit marker on a big round**: the marker's "this
   round's diff is 2 files" sentence (final-unit increment) vs the shard's
   373 KB scope is a false-FAIL trap — pre-empt with an explicit coverage
   note + the round-matched `[unit ` progress-note envelope.

**Why:** a shard brief inherits most whole-round deltas (token strip, size
strategy, truthful plan reference) but inverts the scoping default; without
the gate fence, N shards file N copies of every shared-marker imperfection.

**How to apply:** any brief carrying "Codex shard k of N" + a named path
scope. Related: [[whole-round-unsplit-compose]],
[[brief-pinned-sentinel-and-verdict-enum]].
