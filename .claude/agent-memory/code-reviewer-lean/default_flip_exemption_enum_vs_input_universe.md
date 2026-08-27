---
name: default-flip-exemption-enum-vs-input-universe
description: A permissive-default→fail-closed fix must have its exemption set diffed against the FULL committed input universe (every class × realized value ids), then per-class outcomes diffed vs the parent's committed artifact — the fix's own tests never carry the missed class
metadata:
  type: feedback
---

When a fix flips a permissive default to fail-closed (e.g. "missing metadata
row defaults True" → "missing row = excluded"), the exemption list it adds is
a CLAIM about the input universe. Certify it in three steps (#2587 r2 g1,
where this found the round's only blocker):

1. Enumerate the FULL class/value universe from the committed production
   inputs, not the fix's prose or its tests — e.g. `jq` the committed bank
   manifest for every `pair_class` and its realized `value_a`/`value_b` sets,
   and the fire/metadata artifact for every keyed (axis, vid).
2. Drive the NEW predicate over each class's realized values LIVE (import the
   commit-state module, build a minimal duck-typed arg object). #2587: 7 of 8
   classes were exempt-or-covered; `query_form` (vids E/imp/stmt, 36 pairs,
   structurally uncheckable) was in neither exemption list → 0/36 fired + 36
   false "missing" + axis falsely compliance-limited, on BOTH model sides.
3. Diff the per-class outcome against the PARENT's committed output artifact
   on the same inputs (here `minpair_delta.json`: `compliance_limited=false,
   n_headline=36`) — the parity contradiction is the one-line blocker proof.

**Why:** the fix's own tests are authored from the same mental enumeration
that produced the exemption list, so the missed class is missing from the
fixture bank too (here: `_parent_pairs_and_contexts` built no query_form
pairs) — passing tests are zero evidence on exactly the failing class. Also
check BOTH arms of a two-arm gate (value-level mask AND axis-level
floor/row-presence guard): #2587 missed query_form in both, via two different
incomplete lists (`UNCHECKED_CLASSES` and a reused `GRIDLESS_CLASSES` guard).

**How to apply:** any diff whose commit message says "X with no
row/entry/check now defaults to excluded/failed, exemptions: <list>" — before
grading the exemptions, run the 3-step probe. Related: [[fork_not_inherited_list_vs_parent_gate_surface]],
[[empty_form_blindspot_falsified_by_later_unit]].
