---
name: binding-count-artifact-review
description: Review recipe for a committed binding count/dedup report artifact — 5 cheap probes certify it; residual (near-miss) set is where key-normalization bugs hide
metadata:
  type: feedback
---

A commit that adds ONLY a binding count artifact (dedup report, coverage manifest, realized-count JSON consumed by a plan trigger) is certified by 5 cheap probes, no dataset re-run needed: (1) arithmetic (totals − dropped = kept; dropped ≤ eligible ≤ total; list length/uniqueness/shape via a 10-line python check); (2) producer field-map — diff the JSON's keys+order+indent against the producing function's dict literal; (3) provenance replay — `git_commit` in the artifact must be the producing commit, `git_dirty` false, `ts` just before the commit timestamp; (4) staleness probe — `git diff <producer-sha> HEAD -- <producer.py>` to confirm no post-hoc semantics change orphaned the committed numbers; (5) trigger arithmetic — recompute the plan's decision-gate quantity (e.g. 0.7×pool vs d) from the realized counts in BOTH branches, and grep consumers for hardcoded plan-estimate figures (a test using the estimate as a sample ARG is fine; a production pin is not).

**Why:** #2388 R1 g4 — a 398-line dedup_report.json was fully certifiable this way in ~6 tool calls; the plan's ~4,218 code-train figure turned out to be the PRE-dedup estimate (realized ≈3,958), which only the branch-arithmetic recompute surfaces.

**How to apply:** the one disclosure gap to flag every time: the RESIDUAL near-miss set (eligible-but-unmatched, here 381−373=8) — key-normalization mismatches (slugify vs canonical slugs, apostrophe handling) hide exactly there, and an artifact listing only the matched drops leaves the residual un-eyeballable. Links: [[registered-gate-quantity-substituted]] (adjacent: trigger quantity must be the plan's literal one).
