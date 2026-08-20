---
name: fold-round-context-file-briefs
description: Same-issue follow-up FOLD reviews may brief via a /tmp context file instead of the standard Step 9a-bis fields — embed it as a scoped REVIEW CONTEXT block, adapt out run-it-yourself commands.
metadata:
  type: feedback
---

Fold-round (same-issue follow-up) clean-result reviews can arrive with a
`/tmp/*-critic-context.md` brief file marked authoritative, instead of the
standard Step 9a-bis field set. Compose normally (Step 1b `task.py find`
derivation, Step 1d pre-pass, #556 self-serve interpretation extraction when
`interpretation_marker_path` is omitted) and EMBED the context file as a
clearly-delimited "REVIEW CONTEXT FOR THIS ROUND" block between the grounding
rule and the lens definitions — keeping the confirmatory-fold scope notes
(e.g. "title/confidence deliberately unchanged — do not flag"), ground-truth
artifact paths, judge-drop disclosure expectations, and acknowledged-WARN
list, but ADAPTING OUT any run-it-yourself command the context quotes
(`task.py view <N>`, verifier invocations) since Codex must not execute repo
scripts and the Step 4 no-residue guard bans the verifier/list-concerns forms.

**Why:** #2223 fold round (2026-08-19) — the nap spawn passed
`/tmp/nap-critic-context.md` as authoritative; the fold cycle restarts the
critique loop, so with zero prior `epm:clean-result-critique-codex` markers
the head sentinel is round 1 even when the pre-fold Claude critic history
shows v2 (marker top-level version is auto max+1 per kind). Pre-fold critique
history goes into PRIOR CRITIQUE SUMMARIES as settled context, not as
re-litigable rounds.

**How to apply:** on any fold-cycle compose — determine the round from the
CODEX marker kind's own history on events.jsonl, not the Claude critic's;
summarize the pre-fold cycle in one or two lines; instruct Codex to verify
(and quote) the in-body acknowledgements for WARNs the brief says are
acknowledged rather than taking the brief's word.
