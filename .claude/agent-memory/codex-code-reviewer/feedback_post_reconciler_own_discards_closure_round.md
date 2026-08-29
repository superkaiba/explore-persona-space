---
name: post-reconciler closure round fencing the twin's OWN discards
description: Round 2 after a binding reconciler FAIL that upheld 1 of the twin's 4 legs — fence the twin's own discarded/overruled legs with measured rationale + a new-evidence escape; promote the implementer's self-disclosed unverified check to the primary question; probe the worktree plan symlink for staleness
metadata:
  type: feedback
---

Shape (#2385 r2, 2026-08-29): round 1 = Claude PASS / Codex FAIL on FOUR
legs → `reconciler` BINDING FAIL that **upheld one leg, discarded two on
measurement, and overruled one rating**. Round 2 closes the upheld blocker.
The twin is reviewing the closure of its OWN partially-vindicated FAIL.

**Why:** without an explicit fence the twin predictably re-charges its own
discarded legs (the #825/#2329 post-overturn class), and without an explicit
promotion it skips the one check the implementer openly says it never ran.

**How to apply.**

1. **Fence the twin's own discards WITH the measurement, not just the
   verdict.** One bullet per discarded/overruled leg carrying the reconciler's
   evidence verbatim ("a failed process substitution emits nothing, so the
   loop runs zero iterations"; "`git rm` without `-f` refuses worktree- and
   staged-modified files — your own r1 marker conceded this"). Then the
   two-sided rule: *a bare restatement is a REVIEW DEFECT*, but if THIS
   round's change makes one newly wrong, show the new evidence from this
   diff. A fence without the escape invites an honest miss.
2. **An OVERRULED plan-adherence rating needs its ground restated.** The
   reconciler ruled the `±` wrong because the approved plan had explicitly
   adjudicated the question, so r1 implemented it verbatim. Say that, and
   point at the plan's new `**Round-1 correction**` paragraph — otherwise the
   twin reads the later fix as proof its original `±` was right.
3. **Author-neutrality line, both directions:** do not defend the prior
   finding by demanding more than the binding contract's stated terms, and do
   not wave the fix through because it answers you.
4. **Promote the implementer's self-disclosed unverified check to a numbered
   primary question.** The marker's (d) said in as many words *"Not
   re-verified this round: that the new test FAILS against the pre-fix code."*
   That is a hollow-gate question (Step 0.68, substantive, never stripped) and
   the composer's job is to hand the trace ANCHORS — extractor helper names +
   their span boundaries, the production line numbers inside/outside that
   span, the shim's exact predicate, and the pre-fix blob command
   (`git show <r1-sha>:<path>`) — while explicitly NOT resolving it. Add the
   never-fabricate clause: "if undecidable from reading, say so — a
   fabricated `verified` is worse than an honest CONCERNS."
5. **Blast-radius question for a new `exit 1` in a fenced bash block.** Ask
   what it aborts and what it strands, and hand composer greps as
   cross-checks (no `flock` / `trap` / `.lock`; the only `mktemp` is
   downstream; the fence closes at line N so the sibling arm is skipped too),
   plus the file's OWN precedents to compare against (the pre-existing
   sync-commit FATAL arm; Guard 3's no-merge-base stop in the sibling step
   doc, with line anchors).
6. **Ledger rows can be all-`raised` with no `addressed` event** on a round
   that demonstrably fixed two of them — orchestrator bookkeeping lag. Say
   "score closure on the CODE, never on the ledger's event state, and do not
   raise a finding about the missing `addressed` rows." A row the reconciler
   NARROWED ("legs `:535`/`:549` narrowed to the merge-base leg only") gets
   that narrowing quoted, or the twin re-expands it to its original scope.

**Compose-time probes that paid off here:**

- **Worktree plan symlink was STALE** — `plans/plan.md` → `v2.md` in the
  worktree while canonical was `v3.md` (the amendment landed on main after the
  branch cut, the #546 silent class). Always `readlink`/`diff -q` both copies;
  inline canonical and say "do NOT read any `tasks/.../plans/` path".
- **Two-dot was catastrophically contaminated** (60 files / −3,329 from
  main-side drift) while three-dot was clean at 7 files / 42 KB. Measure BOTH
  and name the contaminated form as banned, with the number — an unquantified
  "don't use two-dot" is weaker than "it reports 3,300 phantom deletions".
- **Cross-pin collision check on a new diagnostic string.** The new Step-5a
  FATAL message embedded a substring that is ALSO another step doc's Guard-3
  message and is pinned by a test. Composer verified the pin asserts a
  differently-prefixed, region-scoped form ⇒ clean, handed over as a
  RE-DERIVE cross-check. Worth running whenever a round adds operator-facing
  diagnostic prose that echoes an existing guard's wording.
- **Marker shape observations stated at compose time with their ceilings:**
  lettered sections present but differently labelled; `### (e)` absent with
  the addressed-claim substance living in (a)/(b)/(d); the gate-scope line
  PRESENT-BUT-TERSE (lost the `(#1288)` token and the contract fields the r1
  marker carried). Each named with "at most CONCERNS, never `marker-shape`"
  so the twin neither misses nor escalates.

See also [[feedback_revision_round_compose_recipe]],
[[feedback_concern_discharge_round_severity_fence]],
[[feedback_brief_named_concern_adjudication]].
