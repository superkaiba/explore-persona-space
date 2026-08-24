---
name: gate-block-remedy-round-compose
description: Composing a round that exists because a GATE blocked (not a reviewer FAIL) — Step 10d lint gate (#2253 r4), doc-TRIM (#2280 r3), or a PRODUCTION validity gate halting the run with a plan-amended predicate recalibration (#2476 fs r2) — gate root-cause is the round contract, amendment-fidelity both directions, stale-record probes
metadata:
  type: feedback
---

When a review round exists because the Step 10d pre-push LINT GATE returned
`block` on the round's own payload (a gate-tree artifact, e.g. the archive
pathspec missing a manifest the new check reads), compose it as a TIGHT
checklist round (validated #2253 r4, 2026-08-21, 27.7 KB):

- **No acceptance-contract envelope.** Prior rounds all PASSed; there is no
  prior FAIL verdict to inline. The round contract is the brief's claim list
  + the gate ROOT-CAUSE, framed up front: which leg failed (gated vs
  baseline — baseline-zero means the check isn't on main yet), why unfixed
  = fleet-wide breakage (#931 shape), and where the spec prescribes the
  remedy (quote the doc's own extend-the-set comment).
- **Pathspec-edit check 1 is an explicit old-vs-new TOKEN diff** with the
  dropped-cone hazard named (adding a token while dropping a cone silently
  NARROWS the gate's scan surface — worse than the bug being fixed;
  Critical `substantive`).
- **"Pin has teeth" translates to a static trace on a read-only twin:**
  hand-trace the parser against `git show <parent>:<doc>`'s pre-fix window
  (token set lacks the manifest ⇒ assertion trips), confirm post-fix pass,
  verify the `text.index` anchor literal is UNIQUE across the ASSEMBLED
  doc (issue_skill_text splices step bodies — grep the whole skills dir,
  not just the companion), and note the fail-loud shape (unspliced body ⇒
  ValueError ⇒ red, never silent green). The Claude twin carries the live
  red-pre-fix/green-post-fix run.
- **Environmental-red adjudication as its own check:** a reported one-off
  red the implementer root-caused as their own invocation env (bare
  .venv python without bin/ on PATH) gets a sound-vs-masks-a-defect
  adjudication duty — read the test, confirm the diff touches nothing it
  imports; sound ⇒ one-line disposition, never auto-flag.
- Sentinel arithmetic is trivial here (revision_round == impl marker
  version == 4, fresh in history) — but check the r2 precedent on the same
  task ([[brief-pinned-sentinel-and-verdict-enum]]) before assuming.

**Why:** the shape differs from both the fix-round (no prior-verdict
acceptance contract) and the merge-round ([[merge-reconciliation-review-compose]]);
without the gate root-cause framing Codex has no way to score whether the
remedy matches the failure.

**How to apply:** any round whose brief says the Step 10d (or inline
payload) lint gate blocked and the diff is the gate-surface remedy + pin.
See also [[revision-round-compose-recipe]].

**Doc-TRIM variant (#2280 r3, 2026-08-22 — the size-gate ratchet round; a
recurring shape since gotchas.md will hit the WARN budget again):** when
the gate-block remedy is a CONTENT TRIM of an always-on .md rule file, the
round's core deliverable is a `## Removed-span clause ledger` — every
removed/reworded span gets a disposition: ARCHAEOLOGY-OK |
LOAD-BEARING-LOST | COMPRESSED-AMBIGUOUS | SIGNATURE-LOST (batching
allowed for same-kind trivial removals). Name the TWO test-invisible loss
modes explicitly: (a) rule compressed into ambiguity (words remain,
instruction no longer actionable), (b) diagnostic SIGNATURE removed with
the fix retained (entry unfindable by the symptom text an agent would
grep — "what would an agent hitting this bug grep for?"). Compose-time
attestations that make the review cheap: (i) protected-span zero-hit probe
(`git show <sha> --unified=0 -- <file> | grep -E '^[+-][^+-]' | grep -icE
'<tokens>'` → 0) + per-token occurrence counts IDENTICAL both sides;
(ii) citation-id SET diff (`grep -oE '#[0-9]+' | sort -u` both blobs,
diff empty) — hand Codex the settled zero-loss and reserve its round for
ORPHANED citations (kept #N whose referenced clause left); (iii) entry
collapses (two entries merged) get a both-sources-preserved,
no-conflation duty. Mechanics for one-giant-line entries: word-diff is
the primary read, and blocker citations use entry LEAD PHRASE + exact
fragment in backticks — bare line numbers are useless on a 200KB
one-line-heavy file. When a merge of origin/main brought main's entries
into the branch by design (lost-update recovery), pre-resolve them as
NOT-scope-creep. Note the worktree `tasks/` status folder can CHANGE
between rounds when the branch merges main (approved/ → running/ on
#2280) — re-probe the plan path every round; the prior round's path
sentence goes stale. Step 4.5 is satisfied-by-construction (the
pre-existing gate test IS the regression guard; demand no new test).

**Production-gate recalibration variant (#2476 floor-sensitivity-sweep r2,
2026-08-24):** when the round exists because a PRODUCTION run halted on the
deliverable's own validity gate (rc=32 on a 1-of-879 fp near-tie) and the
remedy is a PLAN-AMENDED predicate recalibration (v6 max-based → v7
quantile):

- **Inline the halt-diagnosis `epm:progress` note in its own envelope** as
  the incident ground truth (realized delta distribution, the violator's
  identity/values, the decision rationale + rejected alternatives). The
  round's realized-production-distribution test pin must be CONSISTENT with
  it; the relaunch mechanics it names (`--resume-across-code-sha`, pod
  held) are the orchestrator's duty — their absence from the diff is never
  a finding.
- **Attest the amendment in the plan preamble:** compose-time `diff vN-1
  vN` proving the amendment touches ONLY the brief-named spans, plus the
  `epm:plan-verify` PASS quote. Lets Codex score fidelity against a settled
  amendment instead of re-adjudicating the plan change.
- **The review contract is amendment fidelity in BOTH directions:**
  implements the amended predicate EXACTLY (comparator boundary arithmetic:
  ≤ vs < at each tolerance, share passing at exactly the bar, n-floor
  inclusive; fixtures must pin the boundaries they claim) AND is
  not-a-loosening (the FAIL-leg pins — systematic shift + order-of-magnitude
  excursion — must still halt with rc + record-written-FIRST) AND does not
  go beyond the amendment (no extra tolerance widening, no violator-list
  cap, sibling gates/rc codes untouched).
- **The halted out-root carries a STALE OLD-SHAPE gate record** — hand the
  twin the stale-record probes explicitly: no stale-record skip on resume
  (a recorded FAIL must never satisfy resume), no re-apply of the recorded
  old-predicate verdict, no reader indexing NEW-shape keys (`predicate`,
  new fields) on the OLD record before re-write, and the re-run gate
  supersedes record-first. Regime-key neutrality (new constants
  deliberately OUT of `_regime()` so `config_hash` still matches the halted
  out-root) is a declared-decision adjudication — the impl marker's (b)
  rationale + needs-eyeball flag it; verify, don't auto-flag as drift.
- **When round 1 closed via reconciler PASS over the twin's OWN FAIL**, the
  ledger swells with the reconciler's DOWNGRADED re-raises of the twin's
  own rows: walk them as STATUS LINES (NOT-TOUCHED is the expected,
  legitimate status on a micro-diff), pair the author-neutrality line with
  the severity fence (re-raise at LEDGER severity only if this diff worsens
  the surface), and pre-resolve the reconciler's standing recommendations
  as do-not-gate — the brief scoped the round to the gate fix only.
