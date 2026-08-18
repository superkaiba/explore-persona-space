---
name: exception-degraded-fingerprint-leg-inert
description: A try/except degrade branch around a resume-fingerprint leg (code-sha, manifest read) can silently fire on EVERY run via a typo'd attribute — probe happy-path variance, don't read the branch
metadata:
  type: feedback
---

A resume/fingerprint leg wrapped in a broad `except Exception` degrade branch
("git-less tree: never crash") can be inert on the HAPPY path too: #2356 R1's
`_phase_fingerprint` read `git_provenance().sha` — a nonexistent attribute (the
field is `commit_sha`) — so the AttributeError routed EVERY run through the
degrade constant and a code edit never forced a recompute. The code READS as
having a live leg; only execution shows it doesn't.

**Why:** the whole point of the except branch is to swallow environment errors,
so it also swallows programming errors in the leg itself — indistinguishable at
read time.

**How to apply:** when reviewing any fingerprint/resume key with an
exception-degraded leg, (1) run a two-value variance probe — monkeypatch the
leg's source to two different values and assert the fingerprints differ (2
minutes, settles it); (2) verify the claimed fail-pre-fix via the parent blob
(`git show <sha>^:<file>` extracted + imported ahead of HEAD — see
[[fails-pre-fix-probe-parent-commit]]), never the commit message's stash claim;
(3) sweep SIBLING drivers in the same round for the same attribute misuse
(`grep -nE 'provider_fn\(\)\.(old|new)'`) — in #2356 the other two drivers were
already correct, isolating the bug to one file. A dataclass-attribute leg is
safer verified against the class definition (`grep -n "class GitProvenance" -A12`)
than against usage precedent.
