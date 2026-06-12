---
name: Delta-scoped amendment-round prompting
description: How to compose lens prompts when the brief scopes critique to a vN amendment delta of an already-approved plan (custom verdict line, verbatim scope note, do-not-relitigate guard)
type: feedback
---

When the orchestrator brief carries a "DELTA SCOPE NOTE (include verbatim)"
for an amendment round (e.g. #537 v5 = surgical delta over ensemble-approved
v4):

1. **Paste the scope note verbatim** as its own labeled block right after THE
   BAR, before the lens items — it governs scope and usually carries its own
   verdict-line format (e.g. `**Verdict (<lens>, delta-scoped):
   APPROVE|REVISE|REJECT**`). Use THAT line in place of the standard
   `**Rating: ...**` line in the output format; keep the epm marker tags
   unchanged (the orchestrator validates tags, not the rating-line wording).
2. **Add an explicit do-not-relitigate guard** in both the PRIOR CRITIQUES
   slot ("the vN-1 plan was ensemble-APPROVED; do not re-litigate it") and
   the PLAN TEXT intro ("critique ONLY the vN delta") — the full plan body
   still ships so Codex can see context, and without the guard it drifts
   into v4-wide findings.
3. **Scope every enumerated sub-question to the delta**, but ALWAYS include
   one cross-boundary sub-question asking whether the delta can silently
   move a previously-registered headline (e.g. new eval columns entering an
   aggregate leaderboard / CV scoring registered in the approved version) —
   that is the one place a "surgical" delta is conclusion-changing for the
   parent plan.
4. Tell Codex to give a one-line verdict per labeled sub-question (a
   dedicated `### Sub-question verdicts` section) — keeps delta rounds short
   and makes the reconciler's job mechanical on disagreement.

5. **If the brief defines verdict SEMANTICS but no verdict-line format**
   (e.g. #537 v6 round 4 gave APPROVE/REVISE/REJECT definitions only), mint
   the delta-scoped line yourself — `**Verdict (<lens>, delta-scoped):
   APPROVE | REVISE | REJECT**` — and paste the brief's semantics into the
   scope note as "Verdict definitions for this round". When the descope is
   an explicit USER OVERRIDE (e.g. seed floor), add it to the SETTLED list
   as "review the IMPLEMENTATION, never the choice" and forbid proposing
   the reverted value back.

**Why:** #537 v5 alternatives dispatch (2026-06-09): the brief overrode the
verdict-line format and capped scope to 2 new eval-only contexts; the main
leak risk was the delta columns entering the v4-registered leaderboard
aggregation (sub-question F). #537 v6 methodology dispatch (round 4,
2026-06-09): semantics-only verdict definitions + user-override seed
descope; sub-questions keyed to estimator re-basing (raw-data persistence,
residual >=2-seed assumptions, demotion propagation as the headline-leak
check).

6. **Divergence-block briefs (plan documents a contradiction in its own
   approved scope):** when the brief says the approved scope was jointly
   unsatisfiable and the planner resolved it with a documented divergence
   block, make "is the divergence LEGITIMATE (minimal edit preserving the
   approved hypothesis) and BOUNDED (nothing rides along under its cover)"
   the first two labeled sub-questions, and add a selection/circularity
   sub-question whenever the resolution introduces a data-dependent pick
   (e.g. checkpoint anchor gated on the same panel cells that form the
   PRIMARY DV — ask whether the gate ever sees the y-axis). Inline the
   fact-checker's CONFIRMED verdict on the contradiction premise so Codex
   doesn't burn effort re-deriving it. First used: #480 followup-2
   methodology (2026-06-09).

7. **Composition-of-two-executed-designs follow-ups** (amendment = design
   A's arms x design B's regime, both already run on the same issue): the
   high-value methodology sub-questions are (a) composition
   second-variable (does the cross silently change anything vs either
   executed parent), (b) ARM-SYMMETRY of every train/eval mismatch (a
   carried caveat that is symmetric across arms cancels in a paired
   d_seed; an asymmetric one is the fatal-confound class), (c) smoke
   CONTENT-vs-count adequacy (does the pre-launch smoke assert the
   composed rows' encodings/parity or only row counts — a
   mis-composition passing a count-only smoke yields clean-looking but
   wrongly-trained cells), and (d) the standard cross-boundary
   descriptive-join check. Also: when the scope mandates inheriting an
   executed parent's `--no-traj` / endpoint-only capture, pre-scope
   critic-spec item 5 (marker-dynamics trajectory) to "does its absence
   break THIS endpoint contrast" so Codex doesn't bounce the round on a
   parent-inherited, DV-identity-mandated choice. If the orchestrator
   wrote the scope marker to `/tmp/issue-<N>-followup-scope-<label>.md`,
   read and paste THAT file verbatim. First used: #464
   minimal_content_cn methodology (2026-06-10).

8. **Borrowed-construction loss-surface mismatch:** when the follow-up
   borrows another arm's data construction (e.g. #519's contrastive
   negatives, whose contrast mechanism lived in marker-only loss at one
   slot) but trains it under a different loss surface (full-sequence CE),
   add an explicit sub-question: "does the borrowed construction still
   instantiate the manipulated variable under the new loss, and is the
   plan's naming of the variable (data construction vs loss) honest enough
   for the analyzer to interpret either outcome?" Pair it with a
   near-on-policy-negatives gradient sub-question when negatives are the
   base model's own text. First used: #552 contrastive-2x2-completion v4
   methodology (2026-06-10).

9. **Gate-split / threshold-re-grounding amendments** (a registered HALT
   gate proven unsatisfiable mid-run is split into HALT / routing /
   observation classes with thresholds re-derived FROM the data they gate):
   the two load-bearing methodology sub-questions are (a) RETAINED FORCE —
   name a corruption mode the OLD gate would have HALTed on that the split
   gate now passes (especially cells demoted out of the HALT class), and
   (b) CIRCULARITY-vs-re-registration — distinguish "gate binds future
   reads (relaunch recomputes it)" from "gate is consumed once on the data
   that set its thresholds, so it can never fail"; resolve by having Codex
   read the driver's resume semantics (a provenance-skip on relaunch means
   the gate binds nothing). Also: when the amendment fixes a launch
   mechanism the Codex twin itself REVISE'd in round 1, say so in the
   prior-critiques slot ("this is YOUR lens's thread") — it reliably makes
   Codex read the actual driver script instead of accepting the prose; and
   pre-settle the CHOICE of mechanism (bash driver) while leaving its
   correctness live, or Codex proposes systemd/tmux alternatives. For new
   runtime machinery (heartbeat, pid guard, sentinel skips), reuse item
   11's smoke-verifiability residue: "ensemble re-review covers code
   review, not runtime demonstration." Watch for stale-sentinel skips
   (resume path accepting a pre-amendment artifact produced under the old
   gate schema). First used: #601 v3 round 2 (2026-06-11).

**How to apply:** Any round whose brief says "vN amendment of approved
vN-1" / includes a verbatim scope note. Also pair with the stale-tmp-files
rule — amendment rounds are exactly when stale prompt/output files from the
prior version sit at the canonical /tmp paths. For same-issue FOLLOW-UP
rounds (no plan version number), suffix the tmp paths with the followup
label instead: `/tmp/codex-critic-<N>-<followup_label>-<lens>-{prompt,output}.md`.
