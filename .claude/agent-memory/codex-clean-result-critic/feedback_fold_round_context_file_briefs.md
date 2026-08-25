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

Two more fold-round compose patterns (#1901 mlp-scaling-densify fold,
2026-08-25):
- **Prior-CYCLE codex markers do not advance a new cycle's head
  sentinel**: when the codex kind's history carries v1/v2 head sentinels
  from an EARLIER fold cycle, a new fold cycle's first review is still
  head-sentinel round 1 — "own-kind history" counts only WITHIN the
  current fold cycle (the Claude critic's fold rounds restart at 1 the
  same way; observed at the #1901 generic-boundary fold). The posted
  top-level version stays auto max+1 on the kind (v3 there) — state the
  head-vs-posted offset in the Step 5 return.
- **User-directed body inline**: when the brief orders the body inlined,
  wrap the verbatim copy in its own `---BEGIN/END ...---` span so the
  Step-4 awk strips it from the no-residue greps (captured content, not
  composer instruction text) and label it the review target of record
  (status-move-proof). First grep every inline source for
  `^---(BEGIN|END) .*---$` collisions — a matching line inside
  SPEC/lens/body would corrupt the envelope strip.

Two more fold-round compose patterns (#1901 mlp-scaling-densify fold,
2026-08-25):
- **Prior-CYCLE codex markers do not advance a new cycle's head
  sentinel**: when the codex kind's history carries v1/v2 head sentinels
  from an EARLIER fold cycle, a new fold cycle's first review is still
  head-sentinel round 1 — "own-kind history" counts only WITHIN the
  current fold cycle (the Claude critic's fold rounds restart at 1 the
  same way; observed at the #1901 generic-boundary fold). The posted
  top-level version stays auto max+1 on the kind (v3 there) — state the
  head-vs-posted offset in the Step 5 return.
- **User-directed body inline**: when the brief orders the body inlined,
  wrap the verbatim copy in its own `---BEGIN/END ...---` span so the
  Step-4 awk strips it from the no-residue greps (captured content, not
  composer instruction text) and label it the review target of record
  (status-move-proof). First grep every inline source for
  `^---(BEGIN|END) .*---$` collisions — a matching line inside
  SPEC/lens/body would corrupt the envelope strip.

Two more delta/reconciler-round compose patterns (#1901 mlp-scaling-densify
r2, 2026-08-25):
- **Truncated verifier finding-lists get a composer recompute envelope**:
  when a fix's adjudication turns on WHICH sections a verifier WARN names
  and the WARN message truncates its list (check-49 prints 2 entries +
  "…"), recompute the untruncated classification at compose time with the
  verifier MODULE's own helpers (`sys.path.insert(0,"scripts"); import
  verify_task_body`; re-run the check's exact loop, print per-section
  FLAGGED/SILENCED + the operative regex) and inline it as a
  `COMPOSER ... RECOMPUTE` envelope. Present it as NEUTRAL mechanical
  data — never pre-judge the disposition (the #1901 case: the analyzer's
  `companion` clause landed in the setup beat, which check-49
  deliberately does not scan, so the fixed result still FLAGGED;
  discharge-vs-residue was left to Codex).
- **Reconciler-bound delta rounds inline the binding verdict + both body
  versions' diff**: extract the `epm:review-reconcile` note verbatim as
  its own envelope (fix list + do-not-touch rulings ARE the round's
  adjudication standard — no 15-lens inline needed), locate the two
  set-body commits bracketing the fix (`git log -- <body path>`), attest
  worktree==HEAD, and inline `git diff <r1-reviewed> <fixed>` as a
  COMPOSER DELTA DIFF envelope with the expected hunk set attested —
  including lifecycle hunks inside the span that are NOT analyzer edits
  (a `remove-tag keep-running` frontmatter hunk rode the #1901 span; an
  unattested lifecycle hunk would read as a delta-confinement violation).

Two more fold-round compose patterns (#1901 mlp-scaling-densify fold,
2026-08-25):
- **Prior-CYCLE codex markers do not advance a new cycle's head
  sentinel**: when the codex kind's history carries v1/v2 head sentinels
  from an EARLIER fold cycle, a new fold cycle's first review is still
  head-sentinel round 1 — "own-kind history" counts only WITHIN the
  current fold cycle (the Claude critic's fold rounds restart at 1 the
  same way; observed at the #1901 generic-boundary fold). The posted
  top-level version stays auto max+1 on the kind (v3 there) — state the
  head-vs-posted offset in the Step 5 return.
- **User-directed body inline**: when the brief orders the body inlined,
  wrap the verbatim copy in its own `---BEGIN/END ...---` span so the
  Step-4 awk strips it from the no-residue greps (captured content, not
  composer instruction text) and label it the review target of record
  (status-move-proof). First grep every inline source for
  `^---(BEGIN|END) .*---$` collisions — a matching line inside
  SPEC/lens/body would corrupt the envelope strip.
