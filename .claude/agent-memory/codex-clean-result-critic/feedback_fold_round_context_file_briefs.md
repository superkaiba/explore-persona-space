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
CODEX marker kind's own history on events.jsonl, not the Claude critic's,
UNLESS the brief explicitly sets `revision_round` (an explicit brief field is
authoritative for the head sentinel; the posted top-level version is auto
max+1 either way — #823 r8, 2026-08-20); summarize the pre-fold cycle in one
or two lines; instruct Codex to verify (and quote) the in-body
acknowledgements for WARNs the brief says are acknowledged rather than taking
the brief's word.

Three additional fold-round compose patterns (#823 r8, 2026-08-20):
- **Brief-named evidence markers** (`epm:progress vN`, `epm:upload-verification`,
  `epm:analysis`): extract each note to its own /tmp file at compose time and
  pass absolute paths in the REVIEW CONTEXT block — Codex cannot read
  events.jsonl. Same shape as the #556 interpretation self-extraction.
- **Binary artifacts a focus question cites** (`.npz` etc.): Codex cannot
  parse them — extract a composer digest (keys/shapes + the derived
  load-bearing reads) and inline it as an EXTRA envelope with a
  `command: composer numpy digest of <path> (...)` metadata line; verify the
  worktree copy's blob identity against the body pin first and SAY so in the
  prompt (covers Codex when its sandbox denies git).
- **Re-adjudicating self-discharged concerns** (analyzer `address-concern`
  on its own body): inline the FULL ledger (`task.py list-concerns <N> --json`,
  no `--open-only`) as a second concerns envelope and rewrite Lens 14's
  Step-0 replacement to name BOTH envelopes; direct Codex to re-adjudicate
  each `addressed_by: analyzer` disposition on its evidence, not accept it.

Two more fold-round compose patterns (#1901 boundary fold, 2026-08-23):
- **The #556 fallback can yield a PRE-FOLD interpretation note** — a fold
  cycle may reach this gate with no fold-specific `epm:interpretation`
  marker, so the latest-marker extraction returns the PREVIOUS fold's note.
  Don't hard-fail (the marker exists) and don't pass it off as current:
  pass the extracted note with an explicit staleness line in the prompt
  ("predates this fold; experiment context only") and note it in the
  Step 5 return.
- **A Lens-14 FAIL can be born mid-cycle**: a reconciler concern raised
  AFTER the fold body landed FAILs the compose-time verifier run even
  though the analyzer never saw it. Inline the FAIL as normal envelope
  data, and add a REVIEW CONTEXT pointer to the nearest in-body prose
  (e.g. a footer advisory sentence discussing the same issue without the
  kebab-case id) so Codex adjudicates acknowledgement-vs-id under the
  lens rules — never pre-judge the disposition in the prompt.
