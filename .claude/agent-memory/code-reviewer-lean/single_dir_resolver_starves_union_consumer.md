---
name: single-dir-resolver-starves-union-consumer
description: A coverage-preference staging resolver that returns EXACTLY ONE mirror dir starves a phase needing the UNION of two prefixes' rows; a downstream fail-open gate converts the fail-loud raise into silent feature forfeiture (#2389 R1 g3)
metadata:
  type: feedback
---

When a fork adds a phase whose input rows are produced by TWO different
uploaders into TWO different Hub prefixes (early `anchors_gate` vs full
`anchors`), and the input dir is picked by a coverage-preference resolver
that returns EXACTLY ONE dir (`_resolve_anchors_dir`: full-if-covers-gate
else gate), enumerate per phase which prefix holds each REQUIRED row class
AT THE TIME THE PHASE MUST RUN. A phase needing the union (gate rows +
sibling-leg rows) can be starved by construction in its own execution
window: whichever dir the resolver picks, the other class is missing and
the phase raises "re-run after they upload" forever — until a terminal
bulk upload merges the prefixes, long after the decision the phase feeds
has been frozen.

**Why:** #2389 R1 g3 (`issue2389_judge.phase_vllm_parity`): the vLLM-parity
gate needs parity cells' GATE rows (only in `anchors_gate` pre-P5) AND
parity-HF-REST rows (only in `anchors` pre-P5). `_resolve_anchors_dir`
picks the gate mirror in that window → `missing_hf` raise every retry →
the parity verdict structurally cannot land before run.py's rest-entry
routing freeze → the claim leg times out and the FAIL-OPEN design lets the
run proceed HF-only, silently forfeiting the user-approved item-4 vLLM
leg. The staging plan's `anchors_any` ("EITHER prefix") semantics encoded
the author's wrong model — right for gate-3 (gate rows only), wrong for
the union consumer.

**How to apply:** for every new phase in a staging-fed driver, (1) list its
required row classes and which prefix each lives in DURING the phase's
window (check the producers' upload call sites, not the terminal bulk
upload); (2) if >1 prefix is required, a single-dir resolver is a bug —
demand a union loader with filename-keyed dedup (post-terminal-upload both
mirrors hold the gate shards, so naive concat trips the duplicate-unit
assert); (3) weigh severity by what the phase FEEDS: a fail-open gate
downstream means the repeated raise degrades to silent feature loss, not a
visible wedge.

**Certifying the fix (#2389 R2 g3, PASS):** four producer-side probes, none
satisfiable from the fix's docstring: (a) name-collision ⇔ same local file —
every upload call site must preserve the local shard filename into BOTH
prefixes, else filename dedup drops distinct rows; (b) first-dir-wins order
must match the producer's upload order, so a crash between the two uploads
leaves the PREFERRED mirror fresh, never stale-preferred; (c) the union
makes the cross-mirror duplicate-unit assert newly REACHABLE — verify the
two row classes are unit-disjoint by construction at the producer (the
parity sweep's `if cid not in gate_id_set` filter), else the fix trades
retry-forever for a crash; (d) staging bug twin: `any(generator)` over
side-effectful `_stage(...)` calls SHORT-CIRCUITS after the first non-empty
prefix — the sibling mirror is never staged and the resolver's coverage
check runs on a phantom-empty dir; the fix is list-materializing
(`any([...])`), and it changes staging for EVERY phase sharing the plan
key — check the resolver's mid-window resolution flip is the designed
direction before calling it side-effect-free. Siblings: [[staging-gate-single-phase-silent-fallback]]
(per-phase loader branches), [[spend-consumer-accepts-partial-shard-set]]
(partial staged set accepted); this one is a COMPLETE staged set that no
single chosen dir exposes.
