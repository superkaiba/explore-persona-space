---
name: keyed-id-edge-exemption-split-straddle
description: When a dedup/superfamily graph exempts id-keyed items from its text edges, probe same-content-different-key duplicates straddling dev/test, and run the graph's own lexical criteria across the exempted boundary yourself
metadata:
  type: feedback
---

When a diff routes a population onto ID-KEYED identity (problem_id/benchmark
key) and the graph's text edges (exact/near-dup/rephrase) filter on
`problem_id is None`, THREE distinct exposures open at once — check all three
with live probes, never by argument:

1. **Cross-boundary inertness** (the disclosed one): keyed items can never
   link to free-text nodes (extraction corpora), so `n_barred == 0` is BY
   CONSTRUCTION. Probe: run the graph's OWN criteria (same thresholds, same
   normalize/shingle/jaccard helpers imported from the module) across the
   boundary yourself; a measured genuine zero downgrades this to a
   disclosure fix.
2. **Within-population duplicate keys** (the one nobody discloses): two
   dataset entries with the SAME content carry DIFFERENT keys → separate
   superfamilies → can straddle dev/test. #2658 group D: 360 exact
   normalized-stem duplicate groups among 13,204 keyed items, 175 straddling,
   193/6,538 (3.0%) sealed-test items with a dev twin; 127/175 first-pairs
   passed the graph's own composed-text merge thresholds. This is a FAIL
   (code fix + re-freeze), not a disclosure line.
3. **Split reconstruction is cheap**: singleton keyed superfamilies make
   sfid/split recomputable as sha(item_id) → verify your reconstruction
   against the committed split manifest (500/500 sanity) before quantifying.

**Why:** the swap commit's message narrated the structural zero as a
"re-measure" and the module docstring still said "overlap is MEASURED (never
asserted empty)" — the reviewer must separate honest-narration defects
(disclosure) from real leakage (straddle), and only live probes separate them.

**How to apply:** any diff adding a `problem_id`/key-first identity path to
an existing text-dedup graph, or swapping a roster from free-text onto keyed
items. Also probe rendered-block re-parsers in the same diffs (option lines
recovered by regex): continuation lines that don't match the line pattern are
SILENTLY dropped (36/12,032 MMLU-Pro option texts truncated) even when the
docstring claims the label-sequence check catches embedded newlines — it only
catches pattern-MATCHING extra lines. (Fix-round correction, re-measured vs
SOURCE options: the real pre-fix content loss was 9/12,032 ws-collapsed — the
36 counted probe-side line shapes, and its own example row was clean. Measure
reviewer counts against source, not the probe's intermediate.)

**Fix-round re-review addendum (#2658 a76241f96f4).** An exact-stem edge-2
fix verifies cleanly (my independent instrument: 360 dup groups, 0 straddle,
0/6,525 test-with-dev-twin, full rebuild parity 0 mismatches) and STILL fails
the estimand: tier the residual with the module's OWN edges-3/4 criteria run
cross-boundary on stems (prefix-filtered exact all-pairs sweep, ~8 s for
6.5k x 6.7k). Found: 131/6,525 test items (2.0%) with a dev stem at
charJ>=0.8 — 56 at charJ==1.0 (leading-400 truncation twins), 38 pairs
sharing >=90% of the stem, 600-char stems differing by ONE char, and 8
punctuation-only twins invisible to an edge-punctuation-only normalizer (run
an aggressive alnum-only normalizer as a second exact pass). Rate the tiers
separately: tokJ 0.6-0.7 token-only may be MCQ template similarity
(over-merge risk via union-find chaining — a design call for the planner),
but the charJ-1.0/prefix-90% tier is seen content, period. Anchor the ruling
in the PLAN's superfamily definition ("duplicate/near-duplicate, and
rephrase", unscoped) — a fix that faithfully implements round-1's prescribed
exact-only recipe can still be a plan deviation.
