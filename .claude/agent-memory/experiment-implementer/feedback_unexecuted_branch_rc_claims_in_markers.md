---
name: unexecuted-branch-rc-claims-in-markers
description: Never state a tool's exit code / branch outcome in a durable marker without executing that branch — misuse-path rc claims invert the misreading they exist to prevent (#1739 P-C round)
metadata:
  type: feedback
---

Never assert the exit code (or any runtime branch outcome) of a code path you
did not execute in a durable marker, report, or handoff note. Execute the
misuse path (a synthesized wrong-shaped input takes minutes), or hedge
explicitly ("rc unverified — run the misuse path before keying on it").

**Why:** #1739 P-C round (2026-08-07) — my harvest-guidance marker claimed a
repro gate would "check 0 cells and exit 2" on P-C output. The teammate RAN it
(synthesized P-C-labeled source): the file-present/all-joins-miss path exits
rc=1; rc=2 is only the file-missing path. A harvest chain watching for
rc=2-means-ignore would have read rc=1 as a genuine reproduction failure — the
exact misreading the marker existed to prevent, inverted. The same empirical
run also exposed a worse latent defect (zero joined cells rendered the 0.0
max-delta initializer as a perfect-reproduction PASS) that pure reasoning had
missed twice.

**How to apply:** whenever a marker/report states "tool X exits N when Y" or
"branch Z fires on input W", either paste evidence of that execution (the
command + rc) or label the claim unverified. Reading the source is not
sufficient for multi-branch exit logic — run the input shape you are
describing. Same family as [[must-fix-done-claims-verified-on-disk]] (verify
claims against ground truth, not intention).
