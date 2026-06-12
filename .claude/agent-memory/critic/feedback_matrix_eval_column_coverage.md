---
name: Matrix-testbed eval-column coverage vs Goal bidirectionality
description: B×B' matrix plans can collapse the eval side to ~K outcome columns and silently lose the within-family directional cells the Goal names — check every train behavior has an expression read (column or family-mate diagonal battery) before approving
type: feedback
---

In B×B′ leakage-matrix / testbed plans, the train side enumerates ~N behavior
instances but the eval side often collapses to ~K outcome columns (K < N).
If the Goal (or a planned figure, e.g. symmetric/antisymmetric decomposition
over within-family pairs) promises "both directions of every within-family
pair", check instance-by-instance that each train behavior ALSO has an
expression read available to every other adapter in its family. The per-row
diagonal manipulation-check battery typically exists but is run only on its
own row — so broad→narrow cells (e.g. agreement-trained adapter → compliment-
writing expression) and within-family narrow↔narrow cells (e.g. bad-medical ↔
risky-financial advice) generate NO data, and the analyzer cannot recover
them post-hoc (needs eval-time generation + judging).

**Why:** Task #545 v1 (2026-06-10): Goal named "both directions of every
within-family pair" and §6 planned a sym/anti decomposition figure, but the
11-column battery had no compliment-expression or per-Turner-domain advice
columns; B3 broad→narrow (the spec's explicitly named novel pair) and all
within-B1 pairs were unmeasurable. Fix was cheap: register each row's
diagonal battery as a column run across same-family adapters + base panel.

**How to apply:** For any plan whose Goal/figures promise directional or
within-family structure, build the (train-instance × eval-read) coverage
table yourself from the plan's column list; any Goal-named cell with no
column → Must-Fix (add family-mate diagonal batteries as columns; negligible
GPU, modest judge cost).
