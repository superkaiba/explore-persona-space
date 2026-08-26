---
name: suffix-mapped-sorted-grid-pairing
description: Two pair-class grids paired row-by-row via each class's OWN sorted key strings, where one class's ids are a transformed (suffixed) copy of the other's — probe order preservation live, demand an alignment assert
metadata:
  type: feedback
---

When an analysis pairs two (unit × replicate) grids by ROW POSITION and each
grid's row order is `sorted()` over its own key strings, with class B's ids a
per-component transform of class A's (e.g. `{vid}p` paraphrase suffixing:
swap vp `v1-v2` ↔ famswap vp `v1p-v2p`), the pairing is only correct if the
transform preserves lexicographic sort order — which FAILS whenever one id is
a proper prefix of another (the separator `-` sorts below letters, the suffix
char may not). #2564 r1 g5: cross-family aspect-consistency read paired the
swap grid with the famswap grid this way; alignment held only because the
frozen ids are uniformly `v1..v5`, and the builder DISCARDED the second
grid's vps (`fams_grid, _ = _grid_for(...)`), so nothing pinned it.

**Why:** a silent misalignment computes every cross-class cosine between
MISMATCHED unit pairs — a plausible-looking wrong number, no error.

**How to apply:** on any sorted-key positional grid pairing across classes:
(1) live-probe order preservation with the REAL frozen ids
(`sorted(keys) order == sorted(transformed) order` per axis, plus a proper-
prefix collision scan); (2) file a Minor demanding an explicit elementwise
assert (`b_keys == [transform(k) for k in a_keys]`) at grid-build time —
"holds today" is fragile under any future id rename. Sibling of
[[registered-gate-quantity-substituted]] (artifact field named like a plan
registration but keyed on a different predicate — the same round's
`compliance_limited` finding).
